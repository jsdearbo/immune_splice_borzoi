#!/usr/bin/env python3
# ===========================================================================
# Pretrained-Borzoi zero-shot baseline + per-event prediction TSVs consumed by consolidated_comparison_table.py (Appendix Table A2).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/fine_tuning/npz_testing/eval_pretrained_borzoi.py
# Data inputs: run outputs under the paths in figures/config.py (currently
#   hardcoded in this script -- see figures/README.md 'Finishing the port').
# ===========================================================================
"""
eval_pretrained_borzoi.py

Zero-shot evaluation of the pretrained Borzoi model on splicing events.

For each event the PSI proxy is a coverage ratio:
  PSI_proxy = mean_RNA_coverage_in_event / (event_cov + flanking_cov + eps)

Coverage is the mean of all 440 RNA-seq output tracks from the pretrained
mouse Borzoi model (mouse_rep0, 2609 tracks total).

Outputs: {out_dir}/pretrained_borzoi_{chrom}_summary.csv — row format
compatible with eval_scatter.plot_comparison (same columns as
baseline_run_1/summary.csv, with baseline="pretrained_borzoi").

Usage (one cell type at a time; run for B_Cell, T_Cell, Macrophage):
  python eval_pretrained_borzoi.py --cell-type B_Cell --out-dir /path/to/out

The output CSV is written idempotently: re-running for the same cell type
replaces its rows in the CSV.
"""

import math
import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import helpers from eval_scatter (safe because eval_scatter uses
# `if __name__ == "__main__"` guard).
import eval_scatter as es
import grelu.resources
import grelu.io.genome
from grelu.sequence.format import strings_to_one_hot

# ── Constants ────────────────────────────────────────────────────────────────

TARGETS_MOUSE = (
    "/mnt/data_sda/jdearborn/git_repos/edit_baskerville/"
    "borzoi/examples/targets_mouse.txt"
)

NPZ_BASE = (
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/"
    "data_labeling/results/labelled_data/"
    "single_cell_type_datasets/multitask_balanced"
)

# Each cell-type-specific NPZ has exactly 1 task (task_idx=0).
NPZ_TEST_DIRS = {
    "B_Cell":     os.path.join(NPZ_BASE, "B_Cell",     "test"),
    "T_Cell":     os.path.join(NPZ_BASE, "T_Cell",     "test"),
    "Macrophage": os.path.join(NPZ_BASE, "Macrophage", "test"),
}

INPUT_LEN = 524288
BIN_SIZE  = 32


# ── Track loading ────────────────────────────────────────────────────────────

def load_rna_indices(targets_path: str) -> np.ndarray:
    """Row indices (0-based) of RNA-seq tracks in targets_mouse.txt."""
    df = pd.read_csv(targets_path, sep="\t")
    return np.where(df["description"].str.startswith("RNA:"))[0]


# ── Coverage-ratio PSI proxy ─────────────────────────────────────────────────

def predict_coverage_ratio(model, events_df, genome, rna_indices_dev,
                            n_bins_out, crop_bp, device):
    """
    For each event in events_df, run the pretrained model and return the
    mean-RNA-coverage ratio as a PSI proxy in [0, 1].

    rna_indices_dev : LongTensor on device — indices of RNA-seq tracks.
    """
    preds = []
    pbar = tqdm(events_df.iterrows(), total=len(events_df),
                desc="Predicting", mininterval=1.0)

    for _, row in pbar:
        chrom = row["Chromosome"]
        start = int(row["Start"])
        end   = int(row["End"])

        center  = (start + end) // 2
        w_start = center - INPUT_LEN // 2
        w_end   = w_start + INPUT_LEN

        # Fetch sequence
        try:
            seq = genome.fetch(chrom, w_start, w_end).upper()
        except Exception:
            preds.append(np.nan)
            continue

        if len(seq) == INPUT_LEN + 1:
            seq = seq[:INPUT_LEN]
        if len(seq) != INPUT_LEN:
            preds.append(np.nan)
            continue

        one_hot = strings_to_one_hot([seq])
        tens = (
            torch.from_numpy(one_hot).float()
            if isinstance(one_hot, np.ndarray)
            else one_hot.float()
        ).to(device)

        # Forward pass — output (1, 2609, n_bins_out)
        with torch.no_grad():
            out = model(tens)

        # Mean over all RNA-seq tracks → (n_bins_out,)
        rna_mean = out[0, rna_indices_dev, :].mean(dim=0)

        # Event bins in output space
        out_start = w_start + crop_bp
        b_start = max(0, (start - out_start) // BIN_SIZE)
        b_end   = min(n_bins_out, math.ceil((end - out_start) / BIN_SIZE))
        b_end   = max(b_end, b_start + 1)           # at least 1 bin

        # Flanking bins (same span as event, clamped to 5–50 bins)
        flank_n  = max(5, min(50, b_end - b_start))
        fl_left  = rna_mean[max(0, b_start - flank_n) : b_start]
        fl_right = rna_mean[b_end : min(n_bins_out, b_end + flank_n)]
        flanks   = [x for x in [fl_left, fl_right] if x.numel() > 0]
        bg_cov   = torch.cat(flanks).mean() if flanks else rna_mean.mean()
        ev_cov   = rna_mean[b_start:b_end].mean()

        psi_proxy = float(ev_cov / (ev_cov + bg_cov + 1e-8))
        preds.append(psi_proxy)

    return preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-type", required=True,
                    choices=["B_Cell", "T_Cell", "Macrophage"])
    ap.add_argument("--chrom", default="chr11")
    ap.add_argument("--npz-dir", default=None,
                    help="NPZ test shard directory. Defaults to the "
                         "cell-type-specific balanced dataset test split.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for pretrained_borzoi_summary.csv")
    ap.add_argument("--events", nargs="+",
                    default=["SE", "RI", "A3SS", "A5SS", "CONST", "ALL"],
                    metavar="EVENT")
    ap.add_argument("--const-limit", type=int, default=500,
                    help="Cap constitutive events (seed 42, matching "
                         "eval_scatter.py --const-limit). No limit on alt events.")
    ap.add_argument("--task-idx", type=int, default=0,
                    help="NPZ task index for ground-truth PSI. Each "
                         "cell-type-specific NPZ has 1 task (index 0).")
    ap.add_argument("--targets-mouse", default=TARGETS_MOUSE,
                    help="Path to targets_mouse.txt (track metadata).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # NPZ directory
    npz_dir = args.npz_dir or NPZ_TEST_DIRS.get(args.cell_type)
    if not npz_dir or not os.path.isdir(npz_dir):
        log.error(f"NPZ test dir not found: {npz_dir}. Pass --npz-dir explicitly.")
        return

    # RNA-seq track indices
    log.info(f"Loading RNA-seq track indices from {args.targets_mouse}")
    rna_indices_np = load_rna_indices(args.targets_mouse)
    rna_indices_dev = torch.tensor(rna_indices_np, dtype=torch.long, device=args.device)
    log.info(f"  {len(rna_indices_np)} RNA-seq tracks")

    # Pretrained model
    log.info("Loading pretrained Borzoi (mouse_rep0) — may download on first call ...")
    model = grelu.resources.load_model(project="borzoi", model_name="mouse_rep0")
    model = model.eval().to(args.device)

    # Output geometry via dummy pass
    with torch.no_grad():
        dummy_out = model(torch.zeros(1, 4, INPUT_LEN, device=args.device))
    n_bins_out = dummy_out.shape[-1]
    crop_bp    = (INPUT_LEN - n_bins_out * BIN_SIZE) // 2
    log.info(f"  Output: {n_bins_out} bins, crop={crop_bp} bp")
    del dummy_out

    # Genome
    genome = grelu.io.genome.get_genome(es.GENOME_FASTA)

    # NPZ windows for ground-truth PSI
    log.info(f"Loading NPZ windows for {args.chrom} from {npz_dir}")
    windows = es.load_npz_windows(npz_dir, args.chrom)
    log.info(f"  {len(windows)} windows")

    # ── Load events ──────────────────────────────────────────────────────────

    # Alt events (SE / RI / A3SS / A5SS)
    log.info(f"Loading rMATS alt events for {args.cell_type} on {args.chrom} ...")
    alt_df = es.load_events_for_chrom(args.cell_type, args.chrom)
    if not alt_df.empty:
        alt_df["psi"] = es.lookup_npz_psi(windows, alt_df, args.task_idx)
        alt_df = alt_df.dropna(subset=["psi"])
        log.info(f"  {len(alt_df)} alt events with NPZ PSI")

    # CONST events
    log.info(f"Loading CONST events for {args.cell_type} on {args.chrom} ...")
    const_df = es.load_constitutive_events(args.cell_type, args.chrom)
    if not const_df.empty:
        const_df["psi"] = es.lookup_npz_psi(windows, const_df, args.task_idx)
        const_df = const_df.dropna(subset=["psi"])
        log.info(f"  {len(const_df)} CONST events with NPZ PSI (before limit)")
        if args.const_limit and len(const_df) > args.const_limit:
            const_df = const_df.sample(args.const_limit, random_state=42).reset_index(drop=True)
            log.info(f"  Subsampled to {len(const_df)} CONST events")

    # ── Run inference (once per unique event) ───────────────────────────────────
    # Alt events are predicted once on the full alt_df; stats are then computed
    # per event_type (SE/RI/A3SS/A5SS) and for ALL without re-running the model.

    if not alt_df.empty:
        log.info(f"=== Alt events (ALL): {len(alt_df)} events ===")
        alt_preds = predict_coverage_ratio(
            model, alt_df, genome, rna_indices_dev, n_bins_out, crop_bp, args.device
        )
        alt_df = alt_df.copy()
        alt_df["predicted_psi"] = alt_preds

        # Per-event predictions (not just the summary CSV below) so downstream
        # analyses (e.g. bootstrap CIs) can resample events directly, matching
        # the preds_*.tsv format already written by eval_scatter.py for the
        # fine-tuned models.
        preds_tsv = os.path.join(args.out_dir, f"preds_pretrained_{args.cell_type}_{args.chrom}.tsv")
        alt_df.to_csv(preds_tsv, sep="\t", index=False)
        log.info(f"Saved per-event predictions -> {preds_tsv}")

    if not const_df.empty:
        log.info(f"=== CONST: {len(const_df)} events ===")
        const_preds = predict_coverage_ratio(
            model, const_df, genome, rna_indices_dev, n_bins_out, crop_bp, args.device
        )
        const_df = const_df.copy()
        const_df["predicted_psi"] = const_preds

    # Build per-event-type DataFrames from cached predictions
    event_dfs = {}
    if not alt_df.empty:
        for et in ["SE", "RI", "A3SS", "A5SS"]:
            sub = alt_df[alt_df["event_type"] == et]
            if not sub.empty:
                event_dfs[et] = sub
        if len(alt_df) >= 2:
            event_dfs["ALL"] = alt_df
    if not const_df.empty and len(const_df) >= 2:
        event_dfs["CONST"] = const_df

    # ── Compute stats per event type ─────────────────────────────────────────

    all_rows = []

    for etype in args.events:
        edf = event_dfs.get(etype)
        if edf is None or len(edf) < 2:
            log.warning(f"Skipping {etype}: {len(edf) if edf is not None else 0} events")
            continue

        valid = edf.dropna(subset=["psi", "predicted_psi"])
        if len(valid) < 2:
            log.warning(f"  {etype}: too few valid predictions")
            continue

        pr, _ = pearsonr(valid["psi"].values, valid["predicted_psi"].values)
        sr, _ = spearmanr(valid["psi"].values, valid["predicted_psi"].values)
        log.info(f"  {etype}: Pearson r={pr:.3f}  Spearman r={sr:.3f}  n={len(valid)}")

        all_rows.append({
            "event_type":  etype,
            "cell_type":   args.cell_type,
            "chrom":       args.chrom,
            "split":       "test",
            "baseline":    "pretrained_borzoi",
            "pearson_r":   pr,
            "spearman_r":  sr,
            "n_samples":   len(valid),
        })

    if not all_rows:
        log.warning("No results produced.")
        return

    # ── Save (idempotent: replace existing rows for this cell type) ──────────

    out_csv = os.path.join(args.out_dir, f"pretrained_borzoi_{args.chrom}_summary.csv")
    new_df  = pd.DataFrame(all_rows)

    if os.path.isfile(out_csv):
        existing = pd.read_csv(out_csv)
        existing = existing[existing["cell_type"] != args.cell_type]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(out_csv, index=False)
    log.info(f"Saved {len(new_df)} rows ({args.cell_type}) → {out_csv}")
    log.info(f"CSV total rows: {len(combined)}")


if __name__ == "__main__":
    main()
