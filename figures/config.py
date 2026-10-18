# ===========================================================================
# Central path configuration for the manuscript figure-generation code.
#
# Every figure/analysis script in this directory resolves its data locations
# through the constants defined here (no hardcoded paths in the scripts).
#
# Public data & models are archived at Zenodo (see ZENODO_DOI below). To
# reproduce the figures from the deposit:
#   1. download + unpack the Zenodo archive, then
#   2. set SB_DATA_ROOT to the unpacked directory (each root below is then
#      resolved under it using _LAYOUT), or
#   3. set any individual SB_<NAME> env var (or edit the dev defaults) to
#      point a single root anywhere.
# Resolution order per root: SB_<NAME> env var  >  SB_DATA_ROOT/_LAYOUT  >
# development-server default.
# ===========================================================================
import os
from pathlib import Path

# Public archive of data + model checkpoints + figure-input run outputs.
ZENODO_DOI = "10.5281/zenodo.23442182"
ZENODO_URL = "https://doi.org/10.5281/zenodo.23442182"

# Layout of each root WITHIN the Zenodo deposit, relative to SB_DATA_ROOT.
# Adjust if the deposit uses different folder names.
_LAYOUT = {
    "EXPERIMENTS_ROOT":     "experiments",
    "UMAP_RUN_ROOT":        "umap_run",
    "PRETRAINED_UMAP_ROOT": "umap_run/pretrained",
    "PRETRAINED_EVAL_ROOT": "pretrained_eval",
    "RMATS_RESULTS":        "rmats/results",
    "RMATS_LABELLED_BED":   "rmats/master_beds",
    "LABELS_NPZ_ROOT":      "labels_npz",
    "GENOME_FASTA":         "genome/GRCm38.primary_assembly.genome.fa",
    "BORZOI_TARGETS_MOUSE": "borzoi/targets_mouse.txt",
}

_DATA_ROOT = os.environ.get("SB_DATA_ROOT")

def _root(name, dev_default):
    env = os.environ.get("SB_" + name)
    if env:
        return Path(env)
    if _DATA_ROOT:
        return Path(_DATA_ROOT) / _LAYOUT[name]
    return Path(dev_default)

# This repo
REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
OUT_DIR = Path(os.environ.get("FIG_OUT_DIR", str(FIGURES_DIR / "out")))

# --- Model run outputs ------------------------------------------------------
# Fine-tuning experiments: checkpoints + per-cell-type / multitask event-level
# prediction TSVs (preds_<CellType>_chr11_npz.tsv). Consumed by
# multitask_collapse.py and consolidated_comparison_table.py.
EXPERIMENTS_ROOT = _root(
    "EXPERIMENTS_ROOT",
    "/mnt/data_sda/jdearborn/scratch/T_and_B_cells/experiments",
)

# Embedding / UMAP / probe run (per-model {ALL,SE,RI,...}_results with
# probe_{knn,linear}/*.csv, embeddings.npz, and reports/). Consumed by
# probe_comparison.py and locon_vs_frozen_probe.py.
UMAP_RUN_ROOT = _root(
    "UMAP_RUN_ROOT",
    "/mnt/data_sda/jdearborn/scratch/JD_build/interpretation/umapping/20260511_run_1",
)
PRETRAINED_UMAP_ROOT = _root(
    "PRETRAINED_UMAP_ROOT",
    "/mnt/data_sda/jdearborn/scratch/JD_build/interpretation/umapping/pretrained_run_1/pretrained",
)

# Pretrained-Borzoi zero-shot evaluation: per-event prediction TSVs
# (preds_pretrained_<CellType>_chr11.tsv) produced by eval_pretrained_borzoi.py.
# Consumed by consolidated_comparison_table.py (Appendix Table A2).
PRETRAINED_EVAL_ROOT = _root(
    "PRETRAINED_EVAL_ROOT",
    "/mnt/data_sda/jdearborn/scratch/JD_build/pretrained_borzoi_eval",
)

# --- Upstream data (companion repo Imm_cell_rMATS) --------------------------
# rMATS event tables + NPZ label shards used when (re)running eval_scatter.py /
# eval_pretrained_borzoi.py inference. Produced by the Imm_cell_rMATS pipeline.
RMATS_RESULTS = _root(
    "RMATS_RESULTS",
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/results",
)
# Master BED files of PSI-labelled constitutive events (eval_scatter.py).
RMATS_LABELLED_BED = _root(
    "RMATS_LABELLED_BED",
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/data_labeling/results/"
    "PSI_labelled_bed/master_beds",
)
LABELS_NPZ_ROOT = _root(
    "LABELS_NPZ_ROOT",
    "/mnt/data_sda/jdearborn/scratch/Imm_cell_rMATS/data_labeling/results/labelled_data/"
    "single_cell_type_datasets/multitask_balanced",
)
GENOME_FASTA = _root(
    "GENOME_FASTA",
    "/mnt/data_sda/jdearborn/genomes/mouse/mm10/fasta/GRCm38.primary_assembly.genome.fa",
)
# Borzoi mouse track metadata (targets_mouse.txt) for the pretrained baseline.
BORZOI_TARGETS_MOUSE = _root(
    "BORZOI_TARGETS_MOUSE",
    "/mnt/data_sda/jdearborn/git_repos/edit_baskerville/borzoi/examples/targets_mouse.txt",
)

CELL_TYPES = ["B_cell", "Macrophage", "T_cell"]

if __name__ == "__main__":
    print(f"ZENODO_DOI = {ZENODO_DOI}")
    print(f"SB_DATA_ROOT = {_DATA_ROOT or '(unset -> dev defaults)'}\n")
    for k, v in sorted(globals().items()):
        if k.isupper() and isinstance(v, Path):
            print(f"{k:24s} {'OK ' if v.exists() else 'MISSING'} {v}")
