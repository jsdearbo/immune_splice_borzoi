# ===========================================================================
# Central path configuration for the manuscript figure-generation code.
#
# These are the data-run locations the figure scripts read from. They are
# currently the original on-disk run outputs on the development server. For a
# public release, repoint them (via the environment variables below) at the
# Zenodo-archived artifacts so the figures reproduce from the deposited data.
#
# NOTE (finishing the port): most scripts in this directory still define these
# paths as their own local constants (ported verbatim from the scratch repos).
# The remaining work is to have each script import from here instead. Until
# then, this file is the single source of truth / checklist of what to move.
# ===========================================================================
import os
from pathlib import Path

def _p(env, default):
    return Path(os.environ.get(env, default))

# This repo
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
OUT_DIR = _p("FIG_OUT_DIR", str(FIGURES_DIR / "out"))

# --- Model run outputs ------------------------------------------------------
# Fine-tuning experiments: checkpoints + per-cell-type / multitask event-level
# prediction TSVs (preds_<CellType>_chr11_npz.tsv). Consumed by
# multitask_collapse.py and consolidated_comparison_table.py.
EXPERIMENTS_ROOT = _p(
    "SB_EXPERIMENTS_ROOT",
    "/mnt/data_sda/jdearborn/scratch/T_and_B_cells/experiments",
)

# Embedding / UMAP / probe run (per-model {ALL,SE,RI,...}_results with
# probe_{knn,linear}/*.csv, embeddings.npz, and reports/). Consumed by
# probe_comparison.py and locon_vs_frozen_probe.py.
UMAP_RUN_ROOT = _p(
    "SB_UMAP_RUN_ROOT",
    "/mnt/data_sda/jdearborn/scratch/JD_build/interpretation/umapping/20260511_run_1",
)
PRETRAINED_UMAP_ROOT = _p(
    "SB_PRETRAINED_UMAP_ROOT",
    "/mnt/data_sda/jdearborn/scratch/JD_build/interpretation/umapping/pretrained_run_1/pretrained",
)

# Pretrained-Borzoi zero-shot evaluation: per-event prediction TSVs
# (preds_pretrained_<CellType>_chr11.tsv) produced by eval_pretrained_borzoi.py.
# Consumed by consolidated_comparison_table.py (Appendix Table A2).
PRETRAINED_EVAL_ROOT = _p(
    "SB_PRETRAINED_EVAL_ROOT",
    "/mnt/data_sda/jdearborn/scratch/JD_build/pretrained_borzoi_eval",
)

# --- Upstream data (companion repo Imm_cell_rMATS) --------------------------
# rMATS event tables + NPZ label shards used when (re)running eval_scatter.py /
# eval_pretrained_borzoi.py inference. Produced by the Imm_cell_rMATS pipeline.
RMATS_RESULTS = _p(
    "SB_RMATS_RESULTS",
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/results",
)
# Master BED files of PSI-labelled constitutive events (eval_scatter.py).
RMATS_LABELLED_BED = _p(
    "SB_RMATS_LABELLED_BED",
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/data_labeling/results/"
    "PSI_labelled_bed/master_beds",
)
LABELS_NPZ_ROOT = _p(
    "SB_LABELS_NPZ_ROOT",
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/data_labeling/results/labelled_data/"
    "single_cell_type_datasets/multitask_balanced",
)
GENOME_FASTA = _p(
    "SB_GENOME_FASTA",
    "/mnt/data_sda/jdearborn/genomes/mouse/mm10/fasta/GRCm38.primary_assembly.genome.fa",
)
# Borzoi mouse track metadata (targets_mouse.txt) for the pretrained baseline.
BORZOI_TARGETS_MOUSE = _p(
    "SB_BORZOI_TARGETS_MOUSE",
    "/mnt/data_sda/jdearborn/git_repos/edit_baskerville/borzoi/examples/targets_mouse.txt",
)

CELL_TYPES = ["B_cell", "Macrophage", "T_cell"]

if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, Path):
            print(f"{k:24s} {'OK ' if v.exists() else 'MISSING'} {v}")
