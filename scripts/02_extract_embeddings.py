"""
03_extract_embeddings.py

Extract per-event embeddings from Borzoi at the end of each major model
component and save them to disk for downstream analysis (UMAP, probes, etc.).

This is the GPU-heavy step. Run once; 04_umap.py and 05_probe.py both load
from the saved output files without re-running GPU inference.

Outputs written to --out-dir:
  embeddings.npz    — one key per layer, float32 arrays (N, 2*channels)
                      NaN rows correspond to events that failed inference
  input_dedup.csv   — deduplicated label table aligned row-by-row with the
                      embedding arrays (same deduplication as this script)

Component hooks (paths relative to BorzoiModel):
  conv_tower      embedding.conv_tower.blocks.6
  transformer     embedding.transformer_tower.blocks.7
  unet_dec_1      embedding.unet_tower.blocks.1.sconv
  unet_dec_0      embedding.unet_tower.blocks.0.sconv
  final_conv      embedding.pointwise_conv
  head_shared     head.shared  (fine-tuned SplitHead models only)

Usage:
  python 03_extract_embeddings.py [--in input_table_chars.csv]
                                  [--out-dir results]
                                  [--ckpt path/to/checkpoint.ckpt]
                                  [--task-idx 0]
                                  [--genome path/to/genome.fa]
                                  [--overlap-slop 100]
                                  [--cuda-device 0]
"""

import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
_utils_dir  = os.path.join(_repo_root, "utils")
for _p in [_repo_root, _utils_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from losses import (
    FocalLoss, WeightedPSILoss, PSILoss, BhattacharyyaLoss,
    CompoundPSILoss, TaskDiversityLoss, CellTypeBhattacharyyaLoss, StrongPSILoss,
)
from lora_patch import load_finetuned_checkpoint, _unwrap_borzoi_model, ConditionalHead, SplitHead

import grelu.resources
import grelu.io.genome
from grelu.sequence.format import strings_to_one_hot


# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_LEN = 524288

COMPONENT_HOOKS = [
    ("embedding.conv_tower.blocks.6",        "conv_tower",  "last"),
    ("embedding.transformer_tower.blocks.7", "transformer", "first"),
    ("embedding.unet_tower.blocks.1.sconv",  "unet_dec_1",  "last"),
    ("embedding.unet_tower.blocks.0.sconv",  "unet_dec_0",  "last"),
    ("embedding.pointwise_conv",             "final_conv",  "last"),
    # head.shared added dynamically for SplitHead models
]


# ── Model helpers ─────────────────────────────────────────────────────────────

def _model_is_conditional(model) -> bool:
    base_model = _unwrap_borzoi_model(model)
    head = getattr(base_model, "head", None)
    if isinstance(head, ConditionalHead):
        return True
    return getattr(base_model, "conditional_head_ref", None) is not None


def _get_borzoi_model(model):
    """Return the BorzoiModel so both embedding.* and head.* paths are reachable."""
    m = _unwrap_borzoi_model(model)
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


# ── Hook utilities ────────────────────────────────────────────────────────────

class CaptureOutput:
    def __init__(self, module):
        self.module  = module
        self.output  = None
        self._handle = None

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self._handle:
            self._handle.remove()

    def _hook(self, module, input, output):
        self.output = output[0] if isinstance(output, tuple) else output


def get_layer(model, path: str):
    """Traverse a dot-separated path. Numeric parts index into ModuleLists."""
    curr = model
    for part in path.split("."):
        curr = curr[int(part)] if part.isdigit() else getattr(curr, part)
    return curr


# ── Overlap deduplication ─────────────────────────────────────────────────────

def deduplicate_shared(df: pd.DataFrame, slop_bp: int = 100) -> pd.DataFrame:
    """For shared events (|), keep the highest read_count per locus."""
    has_shared_col = "shared_cell_types" in df.columns
    shared_mask = (
        df["shared_cell_types"].str.contains(r"\|", na=False)
        if has_shared_col
        else pd.Series(False, index=df.index)
    )

    unique = df[~shared_mask].copy()
    shared = df[shared_mask].copy()

    if shared.empty:
        logging.info("No shared events to deduplicate.")
        return df.reset_index(drop=True)

    shared = shared.sort_values(["Chromosome", "center"]).reset_index(drop=True)
    group_id = 0
    groups   = []
    prev_chrom, prev_center = None, None

    for _, row in shared.iterrows():
        c, ctr = row["Chromosome"], row["center"]
        if prev_chrom != c or (ctr - prev_center) > slop_bp:
            group_id += 1
        groups.append(group_id)
        prev_chrom, prev_center = c, ctr

    shared["_group"] = groups
    kept = shared.loc[shared.groupby("_group")["read_count"].idxmax()].drop(columns=["_group"])

    logging.info(
        f"Deduplication: {len(shared)} shared -> {len(kept)} kept "
        f"(highest read_count per locus). {len(unique)} unique events unchanged."
    )
    return pd.concat([unique, kept], ignore_index=True)


# ── Sequence utilities ────────────────────────────────────────────────────────

def fetch_one_hot(genome, chrom: str, w_start: int, w_end: int, device):
    """Fetch a genomic window and return a one-hot tensor (1, 4, INPUT_LEN)."""
    try:
        seq = genome.fetch(chrom, w_start, w_end).upper()
    except Exception:
        return None

    if len(seq) == INPUT_LEN + 1:
        seq = seq[:INPUT_LEN]
    if len(seq) != INPUT_LEN:
        return None

    oh = strings_to_one_hot([seq])
    t  = torch.from_numpy(oh).float() if isinstance(oh, np.ndarray) else oh.float()
    return t.to(device)


# ── Embedding extraction ──────────────────────────────────────────────────────

def _pool_event_region(layer_out, seq_dim: str,
                        event_start: int, event_end: int,
                        w_start: int):
    """Mean+max pool the event region bins from a hook output."""
    emb = layer_out.squeeze(0)
    if seq_dim == "last":
        emb = emb.transpose(0, 1)   # (C, L) -> (L, C)

    seq_len = emb.shape[0]
    if seq_len == 0:
        return None

    bp_per_bin = INPUT_LEN / seq_len
    b_start = max(0, int((event_start - w_start) / bp_per_bin))
    b_end   = min(seq_len, max(b_start + 1,
                               int(np.ceil((event_end - w_start) / bp_per_bin))))

    region = emb[b_start:b_end, :]
    if region.shape[0] == 0:
        region = emb[seq_len // 2: seq_len // 2 + 1, :]

    mean_v = region.mean(dim=0)
    max_v  = region.max(dim=0)[0]
    return torch.cat([mean_v, max_v]).cpu()


def extract_embeddings(
    inference_model,
    borzoi_model,
    df: pd.DataFrame,
    genome,
    layer_path: str,
    seq_dim: str,
    task_idx: int,
    cond_mode: bool,
    device: str,
) -> np.ndarray | None:
    """Forward-pass every event in df; return (N, 2*C) float32 array."""
    try:
        target_layer = get_layer(borzoi_model, layer_path)
    except (AttributeError, IndexError) as e:
        logging.error(f"Layer '{layer_path}' not found: {e}")
        return None

    raw_embeddings = []

    with torch.no_grad(), CaptureOutput(target_layer) as hook:
        pbar = tqdm(df.iterrows(), total=len(df),
                    desc=f"  [{layer_path}]", mininterval=2.0)

        for _, row in pbar:
            chrom  = row["Chromosome"]
            start  = int(row["Start"])
            end    = int(row["End"])
            center = int(row["center"])

            w_start = center - INPUT_LEN // 2
            w_end   = w_start + INPUT_LEN

            tens = fetch_one_hot(genome, chrom, w_start, w_end, device)
            if tens is None:
                raw_embeddings.append(None)
                hook.output = None
                continue

            if cond_mode:
                cell_ids = torch.full((1,), task_idx, dtype=torch.long, device=device)
                _ = inference_model((tens, cell_ids))
            else:
                _ = inference_model(tens)

            if hook.output is None:
                raw_embeddings.append(None)
                continue

            emb = _pool_event_region(hook.output, seq_dim, start, end, w_start)
            raw_embeddings.append(emb)
            hook.output = None

    valid = [e for e in raw_embeddings if e is not None]
    if not valid:
        logging.warning(f"No valid embeddings for layer '{layer_path}'.")
        return None

    dim = valid[0].shape[0]
    mat = np.full((len(raw_embeddings), dim), np.nan, dtype=np.float32)
    for i, e in enumerate(raw_embeddings):
        if e is not None:
            mat[i] = e.numpy()

    return mat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract Borzoi splice-event embeddings and save to disk"
    )
    parser.add_argument(
        "--in", dest="input", type=str,
        default=os.path.join(_script_dir, "input_table_chars.csv"),
        help="Annotated input table (output of 02_add_characteristics.py)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default=os.path.join(_script_dir, "results"),
        help="Output directory for embeddings.npz and input_dedup.csv",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Fine-tuned checkpoint (.ckpt). Omit for pretrained borzoi-replicate-0.",
    )
    parser.add_argument(
        "--task-idx", type=int, default=0,
        help="Cell-type task index for conditional models (default: 0)",
    )
    parser.add_argument(
        "--genome", type=str, required=True,
        help="Path to genome FASTA (must have .fai index), e.g. GRCm38.primary_assembly.genome.fa",
    )
    parser.add_argument(
        "--overlap-slop", type=int, default=100,
        help="bp radius for shared-event deduplication (default: 100)",
    )
    parser.add_argument(
        "--cuda-device", type=int, default=None,
        help="CUDA device index (e.g. 0 or 1). Defaults to cuda:0 if a GPU is available.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Load existing embeddings.npz and only extract layers not already present.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error(f"Input table not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    logging.info(f"Loaded {len(df)} events")

    df = deduplicate_shared(df, slop_bp=args.overlap_slop)
    logging.info(f"Events after overlap deduplication: {len(df)}")

    if len(df) == 0:
        logging.error("No events remaining — aborting.")
        return

    # Device
    if torch.cuda.is_available():
        idx = args.cuda_device if args.cuda_device is not None else 0
        device = f"cuda:{idx}"
    else:
        device = "cpu"
    logging.info(f"Device: {device}")

    # Load model
    if args.ckpt:
        logging.info(f"Loading fine-tuned checkpoint: {args.ckpt}")
        inference_model = load_finetuned_checkpoint(args.ckpt)
        cond_mode       = _model_is_conditional(inference_model)
        logging.info(f"Conditional model: {cond_mode}  task_idx: {args.task_idx}")
    else:
        os.environ.setdefault("WANDB_MODE", "disabled")
        logging.info("Loading pretrained Borzoi (mouse_rep0)...")
        inference_model = grelu.resources.load_model(project="borzoi", model_name="mouse_rep0")
        inference_model.eval()
        cond_mode = False

    borzoi_model = _get_borzoi_model(inference_model)

    active_hooks = list(COMPONENT_HOOKS)
    head = getattr(borzoi_model, "head", None)
    # Unwrap PEFT ModulesToSaveWrapper (present when model was LoRA-wrapped with modules_to_save=["head"])
    _actual_head = head
    _head_shared_path = "head.shared"
    if head is not None and hasattr(head, "modules_to_save") and "default" in head.modules_to_save:
        _actual_head = head.modules_to_save["default"]
        _head_shared_path = "head.modules_to_save.default.shared"
    if isinstance(_actual_head, SplitHead):
        active_hooks.append((_head_shared_path, "head_shared", "last"))
        logging.info(f"SplitHead detected: adding head_shared hook ({_head_shared_path})")

    inference_model = inference_model.to(device).eval()
    for p in inference_model.parameters():
        p.requires_grad_(False)

    genome = grelu.io.genome.get_genome(args.genome)
    os.makedirs(args.out_dir, exist_ok=True)

    # Resume: pre-load existing layers so we only extract what's missing
    layer_mats: dict[str, np.ndarray] = {}
    emb_path = os.path.join(args.out_dir, "embeddings.npz")
    if args.resume and os.path.exists(emb_path):
        existing = np.load(emb_path, allow_pickle=False)
        layer_mats = {k: existing[k] for k in existing.files}
        logging.info(f"Resume mode: loaded {len(layer_mats)} existing layers: {list(layer_mats)}")

    for layer_path, component_label, seq_dim in active_hooks:
        if component_label in layer_mats:
            logging.info(f"Skipping {component_label} — already in embeddings.npz")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Component: {component_label}  layer: {layer_path}")

        mat = extract_embeddings(
            inference_model=inference_model,
            borzoi_model=borzoi_model,
            df=df,
            genome=genome,
            layer_path=layer_path,
            seq_dim=seq_dim,
            task_idx=args.task_idx,
            cond_mode=cond_mode,
            device=device,
        )

        if mat is not None:
            layer_mats[component_label] = mat
            n_ok = (~np.isnan(mat).any(axis=1)).sum()
            logging.info(f"  Valid embeddings: {n_ok}/{len(df)}  dim: {mat.shape[1]}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not layer_mats:
        logging.error("No embeddings extracted for any layer — aborting.")
        return

    # Save embeddings (emb_path defined above)
    np.savez(emb_path, **layer_mats)
    logging.info(f"\nSaved embeddings ({len(layer_mats)} layers) → {emb_path}")

    # Save deduplicated label table so downstream scripts are row-aligned
    # (skip if resuming and the file already exists — it hasn't changed)
    dedup_path = os.path.join(args.out_dir, "input_dedup.csv")
    if not (args.resume and os.path.exists(dedup_path)):
        df.to_csv(dedup_path, index=False)
        logging.info(f"Saved label table ({len(df)} rows) → {dedup_path}")

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
