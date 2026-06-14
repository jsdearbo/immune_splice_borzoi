import os
import sys
import logging
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
import glob

import grelu.resources
import grelu.visualize
import grelu.io.genome
from grelu.sequence.format import strings_to_one_hot

# Path setup: add repo utils/ to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
_utils_dir = os.path.join(_repo_root, "utils")
for _p in [_repo_root, _utils_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep custom loss symbols importable during checkpoint unpickling.
from losses import FocalLoss, WeightedPSILoss, PSILoss, BhattacharyyaLoss, CompoundPSILoss, TaskDiversityLoss, CellTypeBhattacharyyaLoss, StrongPSILoss

from lora_patch import load_finetuned_checkpoint, ConditionalHead, _unwrap_borzoi_model

GENES_OF_INTEREST = ["Ptprc", "Apoe", "Cfb", "Cklf", "Mtmr7", "Fcgr2b"]

CELL_TYPE_COLORS = {
    "B_Cell": "#1f77b4",    # blue
    "T_Cell": "salmon",
    "Macrophage": "#2ca02c", # green
}

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_gene_coordinates(gene_name, gtf_path):
    """Simple parser to find gene coordinates from GTF."""
    # Grep is faster than parsing the whole GTF in python for one gene
    import subprocess
    cmd = f"grep 'gene_name \"{gene_name}\"' {gtf_path} | grep '\tgene\t' | head -n 1"
    try:
        line = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        parts = line.split('\t')
        chrom = parts[0]
        start = int(parts[3])
        end = int(parts[4])
        strand = parts[6]
        return chrom, start, end, strand
    except subprocess.CalledProcessError:
        raise ValueError(f"Gene {gene_name} not found in {gtf_path}")


def get_exon_range_coordinates(gene_name, gtf_path, first_exon, last_exon, flank_bp=200):
    """
    Return the genomic span covering exons first_exon..last_exon for gene_name.

    For negative-strand genes, exon numbers increase as genomic position decreases
    (exon 1 is the most 3'-genomic, i.e. highest coordinate). We collect all exons
    in the requested range and return the min/max genomic extent plus a flank.

    Returns (chrom, span_start, span_end, strand) or None if not found.
    """
    import subprocess
    cmd = (
        f"grep 'gene_name \"{gene_name}\"' {gtf_path} "
        f"| grep '\texon\t'"
    )
    try:
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return None

    import re
    exon_coords = []
    for line in output.splitlines():
        parts = line.split('\t')
        if len(parts) < 9:
            continue
        attrs = parts[8]
        m = re.search(r'exon_number\s+(\d+)', attrs)
        if not m:
            continue
        exon_num = int(m.group(1))
        if first_exon <= exon_num <= last_exon:
            exon_coords.append((int(parts[3]), int(parts[4])))

    if not exon_coords:
        return None

    chrom = output.splitlines()[0].split('\t')[0]
    strand = output.splitlines()[0].split('\t')[6]
    span_start = min(c[0] for c in exon_coords) - flank_bp
    span_end   = max(c[1] for c in exon_coords) + flank_bp
    return chrom, span_start, span_end, strand


def _model_is_conditional(model) -> bool:
    base_model = _unwrap_borzoi_model(model)
    head = getattr(base_model, "head", None)
    if isinstance(head, ConditionalHead):
        return True
    return getattr(base_model, "conditional_head_ref", None) is not None


_label_cache = {}


def _align_label_track(label_vals, label_start, label_end, label_bin, output_start, output_end, x_axis, track_name):
    if (
        label_vals is None
        or label_start is None
        or label_end is None
        or label_bin is None
        or label_bin <= 0
    ):
        return None

    label_vals = np.asarray(label_vals, dtype=float).squeeze()
    if label_vals.ndim == 0 or label_vals.size <= 1:
        logging.warning(
            "Skipping label overlay for %s due to insufficient length (%s)",
            track_name,
            label_vals.shape,
        )
        return None

    label_range_start = label_start
    computed_end = label_start + label_bin * label_vals.size
    if label_end is not None and abs(computed_end - label_end) > max(label_bin, 1.0):
        logging.debug(
            "Label end mismatch for %s: computed=%d stored=%d",
            track_name,
            int(computed_end),
            int(label_end),
        )
    label_range_end = computed_end

    if output_end <= label_range_start or output_start >= label_range_end:
        logging.warning(
            "Label range for %s (%d-%d) does not overlap output window (%d-%d)",
            track_name,
            int(label_range_start),
            int(label_range_end),
            int(output_start),
            int(output_end),
        )
        return None

    positions = label_range_start + (np.arange(label_vals.size, dtype=float) + 0.5) * label_bin
    label_aligned = np.full_like(x_axis, np.nan, dtype=float)
    within = (x_axis >= positions[0]) & (x_axis <= positions[-1])
    if not within.any():
        logging.warning("No overlapping bins after alignment for %s", track_name)
        return None

    label_aligned[within] = np.interp(x_axis[within], positions, label_vals)
    return label_aligned


def load_label_tracks(data_root, chrom, window_start, window_end, target_bins):
    key = (chrom, int(window_start), int(window_end))
    _ = target_bins  # retained for caller compatibility; alignment occurs downstream
    if key in _label_cache:
        return _label_cache[key]

    splits = ("val", "train", "test")
    target_center = window_start + (window_end - window_start) // 2
    best = None
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue

        for root, _, files in os.walk(split_dir):
            shard_files = sorted(f for f in files if f.endswith(".npz"))
            for fname in shard_files:
                shard_path = os.path.join(root, fname)
                try:
                    with np.load(shard_path) as shard:
                        chroms = shard["chrom"]
                        starts = shard["window_start"]
                        ends = shard.get("window_end")
                        if ends is None:
                            ends = starts + 524288

                        mask_exact = (chroms == chrom) & (starts == window_start) & (ends == window_end)
                        matches = np.where(mask_exact)[0]
                        candidate_indices = matches.tolist()
                        if not candidate_indices:
                            mask_overlap = (chroms == chrom) & (starts <= target_center) & (ends > target_center)
                            overlap_indices = np.where(mask_overlap)[0]
                            candidate_indices.extend(int(idx) for idx in overlap_indices)

                        for idx in candidate_indices:
                            offset = abs(int(starts[idx]) - window_start)
                            if best is None or offset < best[0]:
                                targets = shard["targets"][idx].T
                                best = (
                                    offset,
                                    targets,
                                    shard_path,
                                    int(starts[idx]),
                                    int(ends[idx]),
                                    int(idx),
                                )
                except Exception as exc:
                    logging.warning("Failed to read %s: %s", shard_path, exc)
                    continue

        if best is not None:
            offset, targets, source_path, label_start, label_end, example_index = best
            total_bins = targets.shape[1]
            logging.info(
                "Using labeled window from %s (example=%d, start=%d, end=%d, offset=%d, bins=%d)",
                source_path,
                example_index,
                label_start,
                label_end,
                offset,
                total_bins,
            )
            _label_cache[key] = (targets, label_start, label_end)
            return _label_cache[key]

    logging.warning("No labeled data found for %s:%d", chrom, window_start)
    _label_cache[key] = None
    return None

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


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to train_config.yaml")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to .ckpt file or experiment directory")
    parser.add_argument('--checkpoint', type=str, choices=['best', 'last'], default='best', help="Which checkpoint to use when --ckpt is a directory (default: best)")
    parser.add_argument('--out-dir', type=str, default="./eval_plots", help="Output directory for plots")
    parser.add_argument('--plot-region', type=str, choices=['gene', 'full'], default='gene', help="Region to plot: 'gene' for gene bounds only, 'full' for entire output window")
    parser.add_argument('--genes', type=str, default=None,
        help="Comma-separated gene names, or path to a text file (one per line). "
             "Overrides the hardcoded GENES_OF_INTEREST list.")
    parser.add_argument('--genome', type=str, required=True,
        help="Path to genome FASTA (must have .fai index), e.g. GRCm38.primary_assembly.genome.fa")
    parser.add_argument('--gtf', type=str, required=True,
        help="Path to genome GTF annotation (e.g. gencode.vM23.annotation.gtf)")
    args = parser.parse_args()

    # Load Config
    cfg_dict = load_config(args.config)
    config = SimpleNamespace(**cfg_dict)
    tasks = list(getattr(config, "cell_types", []))

    os.makedirs(args.out_dir, exist_ok=True)

    if args.genes:
        if os.path.isfile(args.genes):
            with open(args.genes) as f:
                genes_to_run = [l.strip() for l in f if l.strip()]
        else:
            genes_to_run = [g.strip() for g in args.genes.split(',') if g.strip()]
    else:
        genes_to_run = GENES_OF_INTEREST

    dataset_root = getattr(config, "target_tsv_base_path", None)
    if dataset_root and not os.path.isdir(dataset_root):
        logging.warning("target_tsv_base_path %s not found; labeled traces disabled.", dataset_root)
        dataset_root = None

    ## wandb problems =============================================================================
    # Disable W&B reattachment when instantiating Lightning modules from checkpoints
    os.environ.setdefault("WANDB_MODE", "disabled")

    # Recreate Lightning hyperparameters but drop logger objects that trigger wandb attach
    model_params = dict(getattr(config, "model_params", {}))
    if "n_tasks" not in model_params or model_params.get("n_tasks") in (None, 0):
        model_params["n_tasks"] = len(getattr(config, "cell_types", []))
    train_params = dict(getattr(config, "train_params", {}))
    task_value = train_params.get("task")
    if isinstance(task_value, str):
        task_lower = task_value.lower()
        if task_lower == "multilabel":
            train_params["task"] = "binary"
        else:
            train_params["task"] = task_lower
    train_params["logger"] = None
    ## end wandb problems =============================================================================
    
    # -------------------------------------------------------
    # 1. Load Fine-Tuned Model
    # -------------------------------------------------------
    ckpt_path = resolve_checkpoint(args.ckpt, args.checkpoint)
    logging.info(f"Using checkpoint: {ckpt_path}")

    logging.info("Loading Fine-Tuned Model...")
    # We use load_finetuned_checkpoint for ALL cases (LoRA or Full)
    # because it now correctly handles the custom SplitHead reconstruction.
    ft_model = load_finetuned_checkpoint(ckpt_path)
    ft_model.eval()
    ft_model.freeze()

    # BatchNorm layers may have accumulated pathological running stats (very small
    # running_var) if the model was undertrained or if the locon adapters changed the
    # activation distribution during training. Force BN to use batch statistics during
    # inference (train mode for BN only) so each window normalises itself.
    _bn_layers_set_train = 0
    for m in ft_model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.train()
            _bn_layers_set_train += 1
    if _bn_layers_set_train:
        logging.info(f"Set {_bn_layers_set_train} BatchNorm1d layer(s) to train mode for inference "
                     "(using batch statistics instead of running stats).")

    cond_mode = _model_is_conditional(ft_model)
    
    # Log head type for debugging
    base_model = _unwrap_borzoi_model(ft_model)
    head = getattr(base_model, "head", None)
    head_type = type(head).__name__ if head is not None else "None"
    logging.info(f"Loaded head type: {head_type}")
    
    if cond_mode:
        if not tasks:
            raise ValueError("Conditional head detected but config.cell_types is empty.")
        logging.info("Detected conditional head; conditioning on %d cell types.", len(tasks))
    else:
        logging.info("Not using conditional head; outputting %d tasks simultaneously.", len(tasks))
    
    # -------------------------------------------------------
    # 2. Load Base Model (Comparison)
    # -------------------------------------------------------
    logging.info("Loading Base Pretrained Model...")
    pt_model = grelu.resources.load_model(project="borzoi", model_name="mouse_rep0")
    pt_model.eval()
    pt_model.freeze()

    if torch.cuda.is_available():
        ft_model = ft_model.cuda()
        pt_model = pt_model.cuda()

    genome = grelu.io.genome.get_genome(args.genome)
    
    for gene_name in genes_to_run:
        try:
            # -------------------------------------------------------
            # 2. Prepare Input Sequence
            # -------------------------------------------------------
            logging.info(f"Fetching sequence for {gene_name}...")
            chrom, gene_start, gene_end, strand = get_gene_coordinates(gene_name, args.gtf)
            gene_center = (gene_start + gene_end) // 2
            logging.info(f"Gene coords: {chrom}:{gene_start}-{gene_end} ({strand})")
        
            # Borzoi Input: 524,288 bp
            input_len = 524288
            window_start = gene_center - (input_len // 2)
            window_end = window_start + input_len
            
            seq = genome.fetch(chrom, window_start, window_end).upper()       
            one_hot = strings_to_one_hot([seq]) # shape: (1, 4, L)

            if isinstance(one_hot, np.ndarray):
                input_tensor = torch.from_numpy(one_hot).float()
            else:
                input_tensor = one_hot.float()
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
        
            # Inference
            with torch.no_grad():
                if cond_mode:
                    preds_per_task = []
                    for idx, _ in enumerate(tasks):
                        cell_ids = torch.full((input_tensor.shape[0],), idx, dtype=torch.long, device=input_tensor.device)
                        out = torch.sigmoid(ft_model((input_tensor, cell_ids)))
                        # out shape: (B=1, C=1, L)
                        out = out.squeeze(0).squeeze(0)  # (L,)
                        track = out.detach().cpu().numpy()
                        preds_per_task.append(track)
                    ft_preds = np.stack(preds_per_task, axis=0)  # (n_tasks, L)
                else:
                    raw_output = ft_model(input_tensor)  # (B, n_tasks, L)
                    ft_preds_tensor = torch.sigmoid(raw_output)  # Apply sigmoid to get probabilities
                    ft_preds = ft_preds_tensor.cpu().numpy()[0]  # Remove batch dim -> (n_tasks, L)
                    logging.info(f"ft_preds shape: {ft_preds.shape}, expected ({len(tasks)}, L)")
                pt_preds = torch.sigmoid(pt_model(input_tensor)).cpu().numpy()[0]

            label_tracks = None
            label_start = None
            label_end = None
            label_bin_size = None
            if dataset_root:
                result = load_label_tracks(dataset_root, chrom, window_start, window_end, ft_preds.shape[-1])
                if result is not None:
                    label_tracks, label_start, label_end = result
                    if label_tracks is not None and label_start is not None and label_end is not None:
                        span = label_end - label_start
                        bins = label_tracks.shape[1] if label_tracks.size > 0 else 0
                        if bins > 0 and span > 0:
                            label_bin_size = span / float(bins)
                if label_tracks is not None and label_tracks.shape[0] != len(tasks):
                    logging.warning(
                        "Label task count mismatch for %s: expected %d, got %d. Skipping labels.",
                        gene_name,
                        len(tasks),
                        label_tracks.shape[0],
                    )
                    label_tracks = None
                    label_start = None
                    label_end = None
                    label_bin_size = None
        
            # Plotting
            logging.info(f"Generating Plots for {gene_name}...")
            
            # Cropping parameters (these should match what was used during training, borzoi output is 196608 bp)
            output_len_bp = 196608 # Standard Borzoi output width
            bin_size = 32
            output_bins = output_len_bp // bin_size
            
            # Calculate where the output window starts relative to the input start
            # Borzoi crops (524288 - 196608) / 2 = 163840 bp from each side
            crop_bp = 163840
            output_start_genome = window_start + crop_bp
            output_end_genome = output_start_genome + output_len_bp
            
            # Prepare tracks for plotting
            # We will plot:
            # 1. FT Predictions (B_Cell, T_Cell, Macrophage)
            # 2. Selected Base Tracks (optional, e.g. CAGE if relevant, otherwise skip)            
            if not tasks:
                raise ValueError("No cell types provided in config; cannot label plots.")
            
            # Create a dictionary of tracks for grelu.visualize
            tracks = []

            for i, task in enumerate(tasks):
                # ft_preds is (n_tasks, L), so ft_preds[i] is (L,)
                pred_vals = ft_preds[i]
                # Ensure pred_vals is 1D
                if pred_vals.ndim > 1:
                    pred_vals = pred_vals.squeeze()
                logging.info(f"Task {task}: pred_vals shape = {pred_vals.shape}")
                label_vals = None
                if label_tracks is not None and i < label_tracks.shape[0]:
                    label_vals = np.array(label_tracks[i])
                tracks.append({
                    "name": f"FT {task} (PSI)",
                    "values": pred_vals,
                    "label_values": label_vals,
                    "color": CELL_TYPE_COLORS.get(task, f"C{i}"),
                    "task": task,
                    "label_start": label_start,
                    "label_end": label_end,
                    "label_bin_size": label_bin_size,
                })
            
            # Select specific region to zoom in on (The gene body)
            # Calculate bin indices for the gene
            gene_rel_start = gene_start - output_start_genome
            gene_rel_end = gene_end - output_start_genome
            
            start_bin = max(0, gene_rel_start // bin_size)
            end_bin = min(output_bins, gene_rel_end // bin_size)
                        
            # Create one axis per track (predictions + labels on same axis)
            total_axes = len(tracks)
            fig, axes = plt.subplots(total_axes, 1, figsize=(15, 3*total_axes), sharex=True)
            if total_axes == 1:
                axes = [axes]

            for ax_idx, track in enumerate(tracks):
                ax = axes[ax_idx]

                # track["values"] is already (L,), no squeeze needed
                y = track["values"]
                if y.ndim == 0 or y.size <= 1:
                    logging.warning("Skipping track %s due to insufficient length (%s)", track["name"], y.shape)
                    ax.set_visible(False)
                    continue
                
                # Verify y is 1D before plotting
                if y.ndim != 1:
                    logging.error(f"Track {track['name']} has unexpected shape {y.shape}, expected 1D")
                    ax.set_visible(False)
                    continue

                x_axis = output_start_genome + (np.arange(y.size) + 0.5) * bin_size
                logging.debug(f"Plotting {track['name']}: x_axis shape={x_axis.shape}, y shape={y.shape}")
                
                # Plot predictions
                ax.plot(x_axis, y, color=track["color"], label=track["name"], linewidth=1.5)
                ax.set_ylabel("PSI")

                # Plot labels on same axis if available
                label_aligned = _align_label_track(
                    track.get("label_values"),
                    track.get("label_start"),
                    track.get("label_end"),
                    track.get("label_bin_size"),
                    output_start_genome,
                    output_end_genome,
                    x_axis,
                    track["name"],
                )

                if label_aligned is not None and np.isfinite(label_aligned).any():
                    ax.plot(
                        x_axis,
                        label_aligned,
                        linestyle=":",
                        color="black",
                        linewidth=1.5,
                        label=f"Label {track['task']}",
                    )

                # Set y-axis limits based on prediction values
                finite_vals = y[np.isfinite(y)]
                if finite_vals.size > 0:
                    y_min = finite_vals.min()
                    y_max = finite_vals.max()
                    if np.isclose(y_min, y_max):
                        pad = max(abs(y_min) * 0.05, 0.05)
                        ax.set_ylim(y_min - pad, y_max + pad)
                    else:
                        pad = (y_max - y_min) * 0.05
                        ax.set_ylim(y_min - pad, y_max + pad)
                else:
                    ax.set_ylim(0, 1)
                
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)
                ax.axvspan(gene_start, gene_end, color='gray', alpha=0.1, label="Gene Body")
        
                if args.plot_region == 'gene':
                    gene_width = gene_end - gene_start
                    buffer = int(gene_width * 0.05)
                    ax.set_xlim(gene_start - buffer, gene_end + buffer)

            plt.xlabel(f"Genomic Coordinates ({chrom})")
            plt.suptitle(f"Splicing Predictions for {gene_name} ({chrom}:{gene_start}-{gene_end})")
            
            out_file = os.path.join(args.out_dir, f"{gene_name}_predictions.png")
            plt.tight_layout()
            plt.savefig(out_file)
            logging.info(f"Saved plot to {out_file}")
            plt.close(fig)

            # -------------------------------------------------------
            # Extra zoomed plot for Ptprc: exons 3-7
            # -------------------------------------------------------
            if gene_name == "Ptprc":
                # Hardcoded exon 3-7 coordinates for Ptprc (chr1, - strand):
                #   exon 3 starts at 138175305 (higher coord on - strand)
                #   exon 7 ends   at 138062859 (lower coord on - strand)
                _ptprc_flank = 200
                ex_span_start = 138115619 - _ptprc_flank
                ex_span_end   = 138126326 + _ptprc_flank
                logging.info(f"Ptprc exons 3-7 zoom (hardcoded): {chrom}:{ex_span_start}-{ex_span_end}")

                fig2, axes2 = plt.subplots(total_axes, 1, figsize=(15, 3*total_axes), sharex=True)
                if total_axes == 1:
                    axes2 = [axes2]

                for ax_idx, track in enumerate(tracks):
                    ax2 = axes2[ax_idx]
                    y = track["values"]
                    if y.ndim == 0 or y.size <= 1 or y.ndim != 1:
                        ax2.set_visible(False)
                        continue

                    x_axis = output_start_genome + (np.arange(y.size) + 0.5) * bin_size
                    ax2.plot(x_axis, y, color=track["color"], label=track["name"], linewidth=1.5)
                    ax2.set_ylabel("PSI")

                    label_aligned = _align_label_track(
                        track.get("label_values"),
                        track.get("label_start"),
                        track.get("label_end"),
                        track.get("label_bin_size"),
                        output_start_genome,
                        output_end_genome,
                        x_axis,
                        track["name"],
                    )
                    if label_aligned is not None and np.isfinite(label_aligned).any():
                        ax2.plot(
                            x_axis,
                            label_aligned,
                            linestyle=":",
                            color="black",
                            linewidth=1.5,
                            label=f"Label {track['task']}",
                        )

                    finite_vals = y[np.isfinite(y)]
                    if finite_vals.size > 0:
                        y_min, y_max = finite_vals.min(), finite_vals.max()
                        if np.isclose(y_min, y_max):
                            pad = max(abs(y_min) * 0.05, 0.05)
                        else:
                            pad = (y_max - y_min) * 0.05
                        ax2.set_ylim(y_min - pad, y_max + pad)
                    else:
                        ax2.set_ylim(0, 1)

                    ax2.set_xlim(ex_span_start, ex_span_end)
                    ax2.legend(loc="upper right")
                    ax2.grid(True, alpha=0.3)

                plt.xlabel(f"Genomic Coordinates ({chrom})")
                plt.suptitle(
                    f"Splicing Predictions for {gene_name} — Exons 3-7 "
                    f"({chrom}:{ex_span_start}-{ex_span_end})"
                )
                zoom_file = os.path.join(args.out_dir, f"{gene_name}_exons3-7_zoom.png")
                plt.tight_layout()
                plt.savefig(zoom_file)
                logging.info(f"Saved zoomed Ptprc exons 3-7 plot to {zoom_file}")
                plt.close(fig2)

        except Exception as e:
            logging.error(f"Error processing {gene_name}: {e}")
            continue
        
if __name__ == "__main__":
    main()