# ===========================================================================
# Figure 2A-C (single-cell-type prediction performance) and Appendix Fig A3 (event-level diagnostics).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/fine_tuning/npz_testing/eval_scatter.py
# Data inputs: run outputs under the paths in figures/config.py (currently
#   hardcoded in this script -- see figures/README.md 'Finishing the port').
# ===========================================================================
import os
import sys
import glob
import logging
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from types import SimpleNamespace
from scipy.stats import pearsonr, spearmanr

# Make the script portable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_fine_tuning_dir = os.path.dirname(_script_dir)
if _fine_tuning_dir not in sys.path:
    sys.path.append(_fine_tuning_dir)
if _script_dir not in sys.path:
    sys.path.append(_script_dir)

# generate_splicing_labels lives in the data_labeling directory
fine_tuning_parent_dir = os.path.dirname(_fine_tuning_dir)
project_root = os.path.dirname(fine_tuning_parent_dir)
_data_labeling_dir = f"{project_root}/Imm_cell_rMATS/src" 
if _data_labeling_dir not in sys.path:
    sys.path.append(_data_labeling_dir)

# Keep custom loss symbols importable during checkpoint unpickling.
from losses import FocalLoss, WeightedPSILoss, PSILoss, BhattacharyyaLoss, CompoundPSILoss, TaskDiversityLoss, CellTypeBhattacharyyaLoss, StrongPSILoss

from generate_splicing_labels import load_rmats_se, load_rmats_ri, load_rmats_alt_site
from lora_patch import load_finetuned_checkpoint, ConditionalHead, _unwrap_borzoi_model
import grelu.resources
from grelu.lightning import LightningModel
import grelu.io.genome
from grelu.sequence.format import strings_to_one_hot
import torch.nn as nn

# DEFAULTS
GENOME_FASTA = "/mnt/data_sda/jdearborn/genomes/mouse/mm10/fasta/GRCm38.primary_assembly.genome.fa"
RMATS_BASE = "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/results"

CELL_TYPE_MAP = {
    "B_Cell": "B_only",
    "T_Cell": "T_only",
    "Macrophage": "Macrophage_only"
}

BED_BASE_PATH = "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/data_labeling/results/PSI_labelled_bed/master_beds"

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_rmats_psi(psi_str):
    """
    Parses rMATS 'IncLevel1' format which is often '0.5,0.4,0.6' or 'NA'.
    Returns the mean of the values.
    """
    if pd.isna(psi_str) or psi_str == 'NA':
        return np.nan
    try:
        # Split by comma, convert to float, filter out NAs
        vals = [float(x) for x in str(psi_str).split(',') if x != 'NA']
        if not vals:
            return np.nan
        return np.mean(vals)
    except:
        return np.nan

def load_npz_windows(npz_dir: str, chrom: str) -> list:
    """Preload all NPZ windows for a chromosome into memory."""
    windows = []
    for sp in sorted(glob.glob(os.path.join(npz_dir, "shard_*.npz"))):
        with np.load(sp, allow_pickle=False) as data:
            chroms = data["chrom"]
            for i in range(len(chroms)):
                if chroms[i] == chrom:
                    windows.append({
                        "ws":      int(data["window_start"][i]),
                        "we":      int(data["window_end"][i]),
                        "targets": data["targets"][i],
                        "mask":    data["mask"][i],
                    })
    return windows


def lookup_npz_psi(windows: list, events_df: pd.DataFrame,
                   task_idx: int, bin_size: int = 32) -> pd.Series:
    """
    For each event, find the first NPZ window fully covering it and return
    the mask-weighted mean PSI at that event's bins. Returns NaN where no
    covering window exists or all bins are masked.
    """
    psi_vals = []
    for _, row in events_df.iterrows():
        es, ee = int(row["Start"]), int(row["End"])
        hit = next((w for w in windows if w["ws"] <= es and w["we"] >= ee), None)
        if hit is None:
            psi_vals.append(np.nan)
            continue
        b0 = (es - hit["ws"]) // bin_size
        b1 = min((ee - hit["ws"] + bin_size - 1) // bin_size, hit["targets"].shape[0])
        if b0 >= hit["targets"].shape[0]:
            psi_vals.append(np.nan)
            continue
        t = hit["targets"][b0:b1, task_idx]
        m = hit["mask"][b0:b1, task_idx]
        psi_vals.append(float((t * m).sum() / m.sum()) if m.sum() > 0 else np.nan)
    return pd.Series(psi_vals, index=events_df.index)


def load_events_for_chrom(cell_type, chrom="chr11", drop_no_psi: bool = True):
    subdir = CELL_TYPE_MAP.get(cell_type)
    if not subdir:
        raise ValueError(f"Unknown cell type {cell_type}. Available: {list(CELL_TYPE_MAP.keys())}")
    
    base_path = os.path.join(RMATS_BASE, subdir)
    events_list = []
    
    def process_pr(pr_obj, label):
        if pr_obj is None or len(pr_obj) == 0:
            return
        df = pr_obj.df
        df = df[df['Chromosome'] == chrom].copy()
        
        if not df.empty:
            # Check for PSI columns or parse them
            if 'psi' not in df.columns:
                # Prefer read-weighted pooling (matches training label generation):
                #   psi_pooled = sum(IJC) / (sum(IJC) + sum(SJC)) across replicates
                # _load_rmats_common always emits ijc_N / sjc_N for all event types.
                ijc_cols = [c for c in df.columns if c.startswith('ijc_')]
                sjc_cols = [c for c in df.columns if c.startswith('sjc_')]
                if ijc_cols and sjc_cols:
                    total_ijc = df[ijc_cols].sum(axis=1).astype(float)
                    total_sjc = df[sjc_cols].sum(axis=1).astype(float)
                    denom = total_ijc + total_sjc
                    df['psi'] = np.where(denom > 0, total_ijc / denom, np.nan)
                elif 'IncLevel1' in df.columns:
                    # Fallback: unweighted mean from raw rMATS string
                    df['psi'] = df['IncLevel1'].apply(parse_rmats_psi)
                else:
                    logging.warning(f"No PSI source found for {label}. Columns: {df.columns.tolist()}")
                    df['psi'] = np.nan

            # Drop rows where we couldn't parse a PSI (skipped in npz mode)
            if drop_no_psi:
                df = df.dropna(subset=['psi'])
            
            df['event_type'] = label
            events_list.append(df)

    logging.info(f"Loading SE events for {cell_type}...")
    process_pr(load_rmats_se(os.path.join(base_path, "SE.MATS.JC.txt")), "SE")
    
    logging.info(f"Loading RI events for {cell_type}...")
    process_pr(load_rmats_ri(os.path.join(base_path, "RI.MATS.JC.txt")), "RI")
    
    logging.info(f"Loading A3SS events for {cell_type}...")
    process_pr(load_rmats_alt_site(os.path.join(base_path, "A3SS.MATS.JC.txt")), "A3SS")

    logging.info(f"Loading A5SS events for {cell_type}...")
    process_pr(load_rmats_alt_site(os.path.join(base_path, "A5SS.MATS.JC.txt")), "A5SS")
    
    if not events_list:
        return pd.DataFrame()
        
    return pd.concat(events_list, ignore_index=True)

def load_constitutive_events(cell_type: str, chrom: str, bed_base_path: str = BED_BASE_PATH) -> pd.DataFrame:
    """
    Load confidence=0 (skeleton/annotation) events from the master BED for a cell type.
    These represent constitutively spliced regions used to train the background signal.
    Returns a DataFrame with Chromosome, Start, End, psi, event_type='CONST'.
    """
    bed_path = os.path.join(bed_base_path, f"{cell_type}_master.bed")
    if not os.path.isfile(bed_path):
        raise FileNotFoundError(f"Master BED not found: {bed_path}")
    df = pd.read_csv(bed_path, sep="\t", header=None,
                     names=["Chromosome", "Start", "End", "psi", "confidence"])
    df = df[(df["Chromosome"] == chrom) & (df["confidence"] == 0.0)].copy()
    df["event_type"] = "CONST"
    df = df.reset_index(drop=True)
    logging.info(f"Loaded {len(df)} constitutive (confidence=0) events on {chrom} for {cell_type}")
    return df


def _model_is_conditional(model) -> bool:
    base_model = _unwrap_borzoi_model(model)
    head = getattr(base_model, "head", None)
    return isinstance(head, ConditionalHead) or getattr(base_model, "conditional_head_ref", None) is not None

def predict_events(model, events_df, genome, task_idx, device="cuda"):
    model.eval()
    
    # Model Specs
    input_len = 524288
    bin_size = 32

    # Dummy pass for output shape
    dummy = torch.zeros(1, 4, input_len).float().to(device)
    cond_mode = _model_is_conditional(model)
    with torch.no_grad():
        if cond_mode:
            cell_ids = torch.tensor([task_idx], dtype=torch.long, device=device)
            dummy_out = model((dummy, cell_ids))
        else:
            dummy_out = model(dummy)
        n_bins_out = dummy_out.shape[-1]
    
    output_len = n_bins_out * bin_size
    crop_bp = (input_len - output_len) // 2
    
    preds = []
    
    # Inference Loop
    pbar = tqdm(events_df.iterrows(), total=len(events_df), desc="Predicting", mininterval=1.0)
    for idx, row in pbar:
        chrom = row['Chromosome']
        start = row['Start']
        end = row['End']
        
        # 1. Prepare Window
        center = (start + end) // 2
        w_start = int(center - (input_len // 2))
        w_end = int(w_start + input_len)
        
        # 2. Fetch Sequence
        try:
            seq = genome.fetch(chrom, w_start, w_end).upper()
        except Exception as e:
            # logging.warning(f"Fetch error {chrom}:{w_start}-{w_end}: {e}")
            preds.append(np.nan)
            continue
            
        if len(seq) == input_len + 1:
            seq = seq[:input_len]
            
        if len(seq) != input_len:
            # logging.warning(f"Length mismatch {chrom}:{w_start}-{w_end} (Got {len(seq)}, expected {input_len})")
            preds.append(np.nan)
            continue
            
        one_hot = strings_to_one_hot([seq])
        tens = torch.from_numpy(one_hot).float() if isinstance(one_hot, np.ndarray) else one_hot.float()
        tens = tens.to(device)
        
        # 3. Model Forward Pass
        with torch.no_grad():
            if cond_mode:
                cell_ids = torch.full((tens.shape[0],), task_idx, dtype=torch.long, device=device)
                out = model((tens, cell_ids))
                if out.ndim == 3:
                    track = out[0, 0, :].cpu().numpy()
                else:
                    track = out[0].cpu().numpy()
            else:
                out = model(tens)
                track = out[0, task_idx, :].cpu().numpy()

        # 4. Map Coordinates to Bins
        out_start_genome = w_start + crop_bp
        rel_start = start - out_start_genome
        rel_end = end - out_start_genome
        
        # Convert to bins
        b_start = max(0, int(rel_start // bin_size))
        # Use ceil for end to ensure we cover the partial bin if the region extends into it
        b_end = min(len(track), int(np.ceil(rel_end / bin_size)))
        
        # Handle sub-bin events (where start/end map to same bin index due to integer division)
        # If region is smaller than 32bp, ensure we grab at least one bin
        if b_start == b_end:
            b_end = b_start + 1

        if b_start >= len(track):
            preds.append(np.nan)
            continue

        # 5. Aggregation Strategy: Mean Pooling
        # We average over the entire event region (Exon for SE, Intron for RI, etc.)
        # This is robust to "blurry" 32bp resolution because it integrates the total signal.
        region_vals = track[b_start:b_end]
        
        if len(region_vals) == 0:
            preds.append(np.nan)
        else:
            val = np.mean(region_vals)
            preds.append(val)
        
    return preds

def resolve_checkpoint(ckpt_arg, checkpoint_type):
    """
    Resolve a checkpoint path from either a direct .ckpt file or an experiment directory.

    If ckpt_arg is a .ckpt file, it is returned as-is (ignores checkpoint_type).
    If ckpt_arg is a directory, looks for a 'checkpoints/' subdirectory and resolves
    'last' -> last.ckpt, or 'best' -> the sole non-last .ckpt file (most recently
    modified if multiple exist).
    """
    if os.path.isfile(ckpt_arg) and ckpt_arg.endswith(".ckpt"):
        return ckpt_arg

    # Resolve checkpoint directory
    ckpt_dir = ckpt_arg
    candidate_dir = os.path.join(ckpt_arg, "checkpoints")
    if os.path.isdir(candidate_dir):
        ckpt_dir = candidate_dir

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if checkpoint_type == "last":
        last_path = os.path.join(ckpt_dir, "last.ckpt")
        if not os.path.isfile(last_path):
            raise FileNotFoundError(f"last.ckpt not found in {ckpt_dir}")
        return last_path

    # Best: all .ckpt files except last.ckpt
    candidates = [f for f in glob.glob(os.path.join(ckpt_dir, "*.ckpt")) if not f.endswith("last.ckpt")]
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found in {ckpt_dir}")
    if len(candidates) > 1:
        logging.warning("Multiple best checkpoint candidates found; using most recently modified: %s", candidates)
        candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def compute_event_stats(events_df: pd.DataFrame) -> list:
    """Compute per-event-type and combined Pearson/Spearman statistics."""
    rows = []
    for et in sorted(events_df['event_type'].unique()):
        sub = events_df[events_df['event_type'] == et]
        if len(sub) < 2:
            continue
        r_p, _ = pearsonr(sub['psi'], sub['predicted_psi'])
        r_s, _ = spearmanr(sub['psi'], sub['predicted_psi'])
        rows.append({'event_type': et, 'pearson_r': r_p, 'spearman_r': r_s, 'n_samples': len(sub)})
    if len(events_df) >= 2:
        r_p, _ = pearsonr(events_df['psi'], events_df['predicted_psi'])
        r_s, _ = spearmanr(events_df['psi'], events_df['predicted_psi'])
        rows.append({'event_type': 'ALL', 'pearson_r': r_p, 'spearman_r': r_s, 'n_samples': len(events_df)})
    return rows


def plot_separated_events(events_df, psi_label, file_tag, chrom, psi_source, out_dir):
    """2x2 subplot grid with one scatter per event type."""
    EVENT_ORDER = ["SE", "RI", "A3SS", "A5SS"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, et in zip(axes.flat, EVENT_ORDER):
        sub = events_df[events_df['event_type'] == et]
        ax.scatter(sub['psi'], sub['predicted_psi'], alpha=0.5, s=15)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(psi_label, fontsize=9)
        ax.set_ylabel("Predicted PSI", fontsize=9)
        if len(sub) >= 2:
            r_p, _ = pearsonr(sub['psi'], sub['predicted_psi'])
            r_s, _ = spearmanr(sub['psi'], sub['predicted_psi'])
            ax.set_title(f"{et}  (N={len(sub)})", fontsize=10, fontweight='bold')
            ax.text(0.05, 0.92, f"Pearson={r_p:.3f}\nSpearman={r_s:.3f}",
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        else:
            ax.set_title(f"{et}  (N={len(sub)}, skipped)", fontsize=10)
    fig.suptitle(f"{file_tag} ({chrom}) [{psi_source}]", fontsize=11)
    fig.tight_layout()
    out = os.path.join(out_dir, f"scatter_{file_tag}_{chrom}_{psi_source}_by_event.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved per-event scatter to {out}")


def plot_comparison(model_stats_rows, cell_type, chrom, psi_source,
                    baseline_csv, file_tag, out_dir, pretrained_csv=None):
    """Grouped bar chart: baselines vs fine-tuned model, per event type.

    Bars (left to right per group):
      MaxEnt only | Full features | [Pretrained Borzoi] | Fine-tuned Borzoi

    pretrained_csv: optional path to pretrained_borzoi_{chrom}_summary.csv
    produced by eval_pretrained_borzoi.py. When absent the 3rd bar is omitted.
    """
    EVENT_ORDER = ["CONST", "SE", "RI", "A3SS", "A5SS", "ALL"]
    EVENT_DISPLAY = {
        "CONST": "Constitutive",
        "SE": "SE",
        "RI": "RI",
        "A3SS": "A3SS",
        "A5SS": "A5SS",
        "ALL": "All alt\nevents",
    }

    bdf = pd.read_csv(baseline_csv)
    bdf = bdf[(bdf['cell_type'] == cell_type) & (bdf['split'] == 'test')]

    # Pretrained lookup (may be empty)
    pretrained_lookup: dict = {}
    if pretrained_csv and os.path.isfile(pretrained_csv):
        pdf = pd.read_csv(pretrained_csv)
        pdf = pdf[(pdf['cell_type'] == cell_type) & (pdf['chrom'] == chrom)]
        pretrained_lookup = {r['event_type']: r['pearson_r']
                             for _, r in pdf.iterrows()}

    model_lookup = {r['event_type']: r['pearson_r'] for r in model_stats_rows}
    events = [e for e in EVENT_ORDER if e in bdf['event_type'].values or e in model_lookup]

    # Build bar spec: always include MaxEnt + full_features + model;
    # insert pretrained only when data is available for at least one event.
    bar_spec = [
        ("maxent_only",   "#aec7e8",  "MaxEnt only\n(donor+acceptor)"),
        ("full_features", "#1f77b4",  "Full features\n(+GC+length)"),
    ]
    if pretrained_lookup:
        bar_spec.append(("pretrained", "#f0a070", "Pretrained\nBorzoi"))
    bar_spec.append(("model", "salmon", "PSI-Adapted\nBorzoi"))

    n_bars = len(bar_spec)
    width  = 0.8 / n_bars                           # bars share 80 % of each slot
    x      = np.arange(len(events))
    fig, ax = plt.subplots(figsize=(1.8 * len(events) + 1.5, 4))

    for i, (bname, col, label) in enumerate(bar_spec):
        offset = (i - (n_bars - 1) / 2) * width
        if bname == 'model':
            vals = [model_lookup.get(e, np.nan) for e in events]
        elif bname == 'pretrained':
            vals = [pretrained_lookup.get(e, np.nan) for e in events]
        else:
            vals = []
            for e in events:
                row = bdf[(bdf['event_type'] == e) & (bdf['baseline'] == bname)]
                vals.append(float(row['pearson_r'].values[0]) if len(row) else np.nan)
        ax.bar(x + offset, vals, width, label=label,
               color=col, edgecolor='black', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_DISPLAY.get(e, e) for e in events], fontsize=10)
    ax.set_ylabel("Pearson r  (test, chr11)", fontsize=10)
    ct_display = cell_type.replace("_", " ")
    ax.set_title(ct_display, fontsize=11)
    ax.set_ylim(-0.2, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9, frameon=False)
    fig.tight_layout()

    out = os.path.join(out_dir, f"comparison_{file_tag}_{chrom}_{psi_source}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved comparison chart to {out}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to train_config.yaml")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to .ckpt file or experiment directory")
    parser.add_argument('--checkpoint', type=str, choices=['best', 'last'], default='best', help="Which checkpoint to use when --ckpt is a directory (default: best)")
    parser.add_argument('--cell-type', type=str, default="B_Cell", help="Cell type the model was trained on (controls task index)")
    parser.add_argument('--rmats-cell-type', type=str, default=None, help="Cell type to draw rMATS events from (defaults to --cell-type)")
    parser.add_argument('--chrom', type=str, default="chr11", help="Chromosome")
    parser.add_argument('--out-dir', type=str, default="./eval_plots", help="Output directory")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of events")
    parser.add_argument('--psi-source', choices=['npz', 'rmats'], default='npz',
        help="Ground-truth PSI source. 'npz': use NPZ target bins (default). "
             "'rmats': use raw rMATS IJC/SJC counts (original behavior).")
    parser.add_argument('--npz-dir', type=str, default=None,
        help="Path to NPZ split directory (e.g. .../test). "
             "Defaults to config.target_tsv_base_path/test when --psi-source npz.")
    parser.add_argument('--separated-event-type', action='store_true',
        help="Produce a 2x2 per-event-type scatter panel and a per-event stats CSV.")
    parser.add_argument('--baseline-csv', type=str, default=None,
        help="Path to MaxEnt baseline summary.csv. When provided with --separated-event-type, "
             "generates a Pearson comparison bar chart.")
    parser.add_argument('--constitutive-only', action='store_true',
        help="Skip rMATS loading. Evaluate only constitutive (confidence=0) events from the "
             "master BED, append a CONST row to the existing stats CSV in --out-dir, and replot.")
    parser.add_argument('--bed-base-path', type=str, default=None,
        help=f"Directory containing {{cell_type}}_master.bed files. "
             f"Defaults to {BED_BASE_PATH}")
    parser.add_argument('--const-limit', type=int, default=None,
        help="Randomly subsample constitutive events to this count before prediction (seed 42).")
    parser.add_argument('--pretrained-csv', type=str, default=None,
        help="Path to pretrained_borzoi_{chrom}_summary.csv produced by "
             "eval_pretrained_borzoi.py. When provided, a 4th bar ('Pretrained Borzoi') "
             "is added to the comparison chart.")
    args = parser.parse_args()

    # Load Config
    cfg_dict = load_config(args.config)
    config = SimpleNamespace(**cfg_dict)
    
    if args.cell_type not in config.cell_types:
        raise ValueError(f"Cell type {args.cell_type} not found.")
    task_idx = config.cell_types.index(args.cell_type)

    rmats_cell_type = args.rmats_cell_type or args.cell_type

    # Resolve NPZ directory (needed for both modes)
    npz_dir = args.npz_dir or os.path.join(
        getattr(config, 'target_tsv_base_path', ''), 'test'
    )

    if args.constitutive_only:
        # --- Constitutive-only mode ---
        # Skip rMATS loading. Evaluate skeleton (confidence=0) events, append CONST row
        # to the existing stats CSV, and replot.
        if not os.path.isdir(npz_dir):
            logging.error(
                f"NPZ test dir not found: {npz_dir}. "
                "Pass --npz-dir explicitly or check target_tsv_base_path in config."
            )
            return
        logging.info(f"Loading NPZ windows for {args.chrom} from {npz_dir} ...")
        windows = load_npz_windows(npz_dir, args.chrom)
        logging.info(f"  {len(windows)} windows loaded.")

        bed_base = args.bed_base_path or BED_BASE_PATH
        events_df = load_constitutive_events(rmats_cell_type, args.chrom, bed_base)
        if events_df.empty:
            logging.error("No constitutive events found on %s for %s.", args.chrom, rmats_cell_type)
            return

        events_df['psi'] = lookup_npz_psi(windows, events_df, task_idx)
        events_df = events_df.dropna(subset=['psi'])
        logging.info(f"Constitutive events with NPZ coverage: {len(events_df)}")

        if events_df.empty:
            logging.error("No constitutive events with NPZ coverage.")
            return

        if args.const_limit and len(events_df) > args.const_limit:
            events_df = events_df.sample(n=args.const_limit, random_state=42).reset_index(drop=True)
            logging.info(f"Subsampled to {len(events_df)} constitutive events.")

    else:
        # --- Standard mode: load rMATS alternative splicing events ---
        events_df = load_events_for_chrom(
            rmats_cell_type, args.chrom,
            drop_no_psi=(args.psi_source == 'rmats'),
        )
        if events_df.empty:
            logging.error("No events found.")
            return

        if args.psi_source == 'rmats':
            events_df = events_df.dropna(subset=['psi'])
        else:
            if not os.path.isdir(npz_dir):
                logging.error(
                    f"NPZ test dir not found: {npz_dir}. "
                    "Pass --npz-dir explicitly or check target_tsv_base_path in config."
                )
                return
            logging.info(f"Loading NPZ windows for {args.chrom} from {npz_dir} ...")
            windows = load_npz_windows(npz_dir, args.chrom)
            logging.info(f"  {len(windows)} windows loaded.")
            events_df['psi'] = lookup_npz_psi(windows, events_df, task_idx)
            events_df = events_df.dropna(subset=['psi'])
            logging.info(f"Events with NPZ coverage (mask > 0): {len(events_df)}")

        if events_df.empty:
            logging.error("No events with valid PSI after source filtering.")
            return

        if args.limit:
            if args.separated_event_type:
                pieces = [
                    grp.sample(n=min(args.limit, len(grp)), random_state=42)
                    for _, grp in events_df.groupby("event_type")
                ]
                events_df = pd.concat(pieces).reset_index(drop=True)
            else:
                events_df = events_df.sample(n=min(args.limit, len(events_df)), random_state=42).reset_index(drop=True)

    # Load Model
    model_params = dict(getattr(config, "model_params", {}))
    if "n_tasks" not in model_params or model_params.get("n_tasks") in (None, 0):
        model_params["n_tasks"] = len(getattr(config, "cell_types", []))
    
    train_params = dict(getattr(config, "train_params", {}))
    task_value = train_params.get("task")
    if isinstance(task_value, str) and task_value.lower() == "multilabel":
        train_params["task"] = "binary"
    else:
        train_params["task"] = task_value.lower() if isinstance(task_value, str) else task_value
        
    train_params["logger"] = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_checkpoint(args.ckpt, args.checkpoint)
    logging.info(f"Using checkpoint: {ckpt_path}")

    # Unified loading using lora_patch
    ft_model = load_finetuned_checkpoint(ckpt_path)
        
    ft_model.to(device)
    ft_model.eval()
    ft_model.freeze()

    genome = grelu.io.genome.get_genome(GENOME_FASTA)

    # Predict
    preds = predict_events(ft_model, events_df, genome, task_idx, device=device)
    events_df['predicted_psi'] = preds
    
    n_total = len(events_df)
    events_df = events_df.dropna(subset=['predicted_psi'])
    n_valid = len(events_df)
    logging.info(f"Valid predictions: {n_valid}/{n_total}")
    
    if n_valid == 0:
        logging.error("No valid predictions generated! Aborting plot.")
        return

    # Apply Sigmoid if linear output
    final_act = model_params.get("final_act_func", "linear")
    if final_act == "linear":
        logging.info("Applying Sigmoid to logits...")
        raw_vals = torch.tensor(events_df['predicted_psi'].values)
        events_df['predicted_psi'] = torch.sigmoid(raw_vals).numpy()

    # Shared output setup
    os.makedirs(args.out_dir, exist_ok=True)
    psi_label = "NPZ target PSI" if args.psi_source == 'npz' else "Measured PSI (rMATS)"
    file_tag = f"{args.cell_type}_on_{rmats_cell_type}" if rmats_cell_type != args.cell_type else args.cell_type

    if args.constitutive_only:
        # Compute CONST stats, merge into the existing stats CSV, and replot.
        const_stats = [r for r in compute_event_stats(events_df) if r['event_type'] == 'CONST']
        stats_csv = os.path.join(args.out_dir, f"stats_{file_tag}_{args.chrom}_{args.psi_source}.csv")
        if os.path.isfile(stats_csv):
            existing = pd.read_csv(stats_csv)
            existing = existing[existing['event_type'] != 'CONST']
        else:
            logging.warning("No existing stats CSV found at %s — writing CONST-only file.", stats_csv)
            existing = pd.DataFrame(columns=['cell_type', 'chrom', 'psi_source', 'event_type',
                                             'pearson_r', 'spearman_r', 'n_samples'])
        const_df_stats = pd.DataFrame(const_stats)
        const_df_stats.insert(0, 'psi_source', args.psi_source)
        const_df_stats.insert(0, 'chrom', args.chrom)
        const_df_stats.insert(0, 'cell_type', args.cell_type)
        updated = pd.concat([existing, const_df_stats], ignore_index=True)
        updated.to_csv(stats_csv, index=False)
        logging.info(f"Updated stats CSV (added CONST row): {stats_csv}")
        if args.baseline_csv and os.path.isfile(args.baseline_csv):
            all_rows = updated[updated['cell_type'] == args.cell_type].to_dict('records')
            plot_comparison(all_rows, args.cell_type, args.chrom, args.psi_source,
                            args.baseline_csv, file_tag, args.out_dir,
                            pretrained_csv=args.pretrained_csv)
        return

    # Per-event scatter, stats CSV, and optional comparison chart
    if args.separated_event_type:
        plot_separated_events(events_df, psi_label, file_tag, args.chrom, args.psi_source, args.out_dir)
        stats_rows = compute_event_stats(events_df)
        stats_df = pd.DataFrame(stats_rows)
        stats_df.insert(0, 'psi_source', args.psi_source)
        stats_df.insert(0, 'chrom', args.chrom)
        stats_df.insert(0, 'cell_type', args.cell_type)
        stats_csv = os.path.join(args.out_dir, f"stats_{file_tag}_{args.chrom}_{args.psi_source}.csv")
        stats_df.to_csv(stats_csv, index=False)
        logging.info(f"Saved per-event stats to {stats_csv}")
        if args.baseline_csv and os.path.isfile(args.baseline_csv):
            plot_comparison(stats_rows, args.cell_type, args.chrom, args.psi_source,
                            args.baseline_csv, file_tag, args.out_dir,
                            pretrained_csv=args.pretrained_csv)

    # Combined scatter (always produced)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=events_df,
        x='psi',
        y='predicted_psi',
        hue='event_type',
        alpha=0.6,
        s=20
    )

    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)

    if len(events_df) > 1:
        r_p, _ = pearsonr(events_df['psi'], events_df['predicted_psi'])
        r_s, _ = spearmanr(events_df['psi'], events_df['predicted_psi'])
        label = f"model={args.cell_type}, events={rmats_cell_type}" if rmats_cell_type != args.cell_type else args.cell_type
        plt.title(
            f"{label} ({args.chrom}) [{args.psi_source}]: "
            f"Pearson={r_p:.3f}, Spearman={r_s:.3f}, N={len(events_df)}"
        )

    plt.xlabel(psi_label)
    plt.ylabel("Predicted PSI")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    out_file = os.path.join(args.out_dir, f"scatter_{file_tag}_{args.chrom}_{args.psi_source}.png")
    plt.savefig(out_file)
    logging.info(f"Saved plot to {out_file}")

    # Save TSV for inspection
    out_tsv = os.path.join(args.out_dir, f"preds_{file_tag}_{args.chrom}_{args.psi_source}.tsv")
    events_df.to_csv(out_tsv, sep="\t", index=False)
    logging.info(f"Saved results TSV to {out_tsv}")

if __name__ == "__main__":
    main()