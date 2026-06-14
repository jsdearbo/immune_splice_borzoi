# immune_splice_borzoi

Code accompanying the manuscript:

> **Learning and probing sequence representations of immune cell alternative splicing**  
> Jacob S. Dearborn, Damas W. Limoge, Tylar Kirch, Zachary D. Miller, William Dowell, Sylvester Languon, Robert C. Cockrell, Devdoot Majumdar

This repository covers fine-tuning of the [Borzoi](https://www.nature.com/articles/s41592-024-02523-z) sequence model on immune cell PSI-informed labels, followed by layerwise embedding analysis (UMAP + probing) and attribution analysis. The upstream step of generating PSI-labelled training data (NPZ shards from RNA-seq) is covered by the companion repo [Imm_cell_rMATS](https://github.com/jsdearbo/Imm_cell_rMATS).

## Overview

The pipeline is organized as a sequence of numbered scripts:

| Step | Script | Description |
|------|--------|-------------|
| 00 | `scripts/00_fine_tune.py` | Fine-tune Borzoi on NPZ-sharded PSI labels |
| 01 | `scripts/01_eval_finetuned.py` | Evaluate fine-tuned model: scatter plots, gene-track visualization |
| 02 | `scripts/02_extract_embeddings.py` | GPU: extract per-event embeddings from each Borzoi layer |
| 03 | `scripts/03_umap.py` | CPU: UMAP scatter plots from saved embeddings |
| 04 | `scripts/04_probe.py` | Linear/KNN probe analysis to decode splicing attributes by layer |
| 05 | `scripts/05_attribution_prep.py` | Prepare event CSV for attribution pipeline |
| 06 | `scripts/06_attribution.sh` | Attribution + MoDISco + FIMO enrichment (calls eLife companion repo) |

## Requirements

### System

- Conda / Mamba
- NVIDIA GPU (A100 or similar recommended for steps 00 and 02)
- MEME Suite ≥ 5.5 (for step 06 FIMO enrichment)
- Reference genome: mm10 FASTA + `.fai` index; Gencode vM23 GTF

### Companion repos

This repo depends on two companion repos:

1. **[Imm_cell_rMATS](https://github.com/jsdearbo/Imm_cell_rMATS)** — generates NPZ training shards from RNA-seq BAMs. Run this first to produce the `target_tsv_base_path` used by step 00.

2. **[elife_delay_splice](https://github.com/jsdearbo/elife_delay_splice)** — provides attribution, MoDISco, and FIMO enrichment scripts used by step 06. Clone it alongside this repo and pass its path via `--elife-repo`.

## Installation

```bash
git clone https://github.com/jsdearbo/immune_splice_borzoi
cd immune_splice_borzoi
conda env create -f environment.yml
conda activate immune-splice-borzoi
```

Also clone the eLife companion repo if you plan to run the attribution analysis:

```bash
git clone https://github.com/jsdearbo/elife_delay_splice
```

## Usage

### 1. Fine-tune Borzoi

Create a YAML config (see `config/example_config.yaml` for all options and documentation):

```bash
python scripts/00_fine_tune.py --config config/example_config.yaml
```

Key config fields to set:
- `target_tsv_base_path`: directory of NPZ shard manifests from Imm_cell_rMATS
- `train_params.devices`: GPU indices (e.g. `[0]` or `[0, 1]`)
- `train_params.save_dir`: local checkpoint output directory

### 2. Evaluate predictions

```bash
python scripts/01_eval_finetuned.py \
    --config config/example_config.yaml \
    --ckpt /path/to/checkpoint.ckpt \
    --genome /path/to/GRCm38.primary_assembly.genome.fa \
    --gtf /path/to/gencode.vM23.annotation.gtf \
    --out-dir ./eval_plots
```

### 3. Extract embeddings (GPU)

```bash
python scripts/02_extract_embeddings.py \
    --in /path/to/input_dedup.csv \
    --ckpt /path/to/checkpoint.ckpt \
    --genome /path/to/GRCm38.primary_assembly.genome.fa \
    --out-dir ./SE_results \
    --cuda-device 0
```

### 4. UMAP visualization

```bash
python scripts/03_umap.py \
    --embeddings ./SE_results/embeddings.npz \
    --out-dir ./SE_results
```

### 5. Probe analysis

```bash
python scripts/04_probe.py \
    --embeddings ./SE_results/embeddings.npz \
    --out-dir ./SE_results \
    --method linear  # or "knn"
```

### 6. Attribution analysis

Requires the eLife companion repo. First prepare the event CSV:

```bash
python scripts/05_attribution_prep.py \
    --input-csv ./SE_results/input_dedup.csv \
    --cell-type B_cell \
    --event-type SE \
    --out-csv ./attribution_runs/B_cell_SE/events.csv \
    --psi-thresholds 0.5 0.95
```

Then run the full attribution pipeline:

```bash
bash scripts/06_attribution.sh \
    --run-name B_cell_SE \
    --config manuscript_runs/configs/locon_B_cell_SE.yaml \
    --input-csv ./SE_results/input_dedup.csv \
    --elife-repo /path/to/elife_delay_splice \
    --out-dir ./attribution_runs \
    --cuda-device 0
```

## Reproducing manuscript results

Exact configs used for paper figures are in `manuscript_runs/configs/`. The event CSVs used for attribution are in `manuscript_runs/event_tables/`.

> **Note:** Paths in these configs are absolute and will need to be updated to match your local environment:
> - `target_tsv_base_path` → your local NPZ shard directory
> - `checkpoint_path` → downloaded checkpoint (see Zenodo link in paper)
> - `genome_name`, `bigwig_dir`, `gtf_file` → your local reference genome files
> - `experiment_dir`, `coord_file_path` → your local output directories

Training configs used for the manuscript:

| Config file | Description |
|------------|-------------|
| `multitask_locon_v1.yaml` | Multitask LoCon model (B cell + Macrophage + T cell) |
| `B_cell_locon_v1.yaml` | Single-cell-type B cell LoCon model |
| `Macrophage_locon_v1.yaml` | Single-cell-type Macrophage LoCon model |
| `T_cell_locon_v1.yaml` | Single-cell-type T cell LoCon model |

Attribution configs (one per cell type × event type combination):
`locon_{B_cell,Macrophage,T_cell}_{SE,RI}.yaml`

## Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.6.0 | CUDA 12.6 |
| pytorch-lightning | 2.4.0 | |
| peft | 0.17.1 | LoRA implementation |
| gReLU | dev | Custom fork: `github.com/jsdearbo/edit_gReLU` |
| umap-learn | 0.5.11 | For step 03 |
| tangermeme | 0.4.2 | Sequence utilities |
| MEME Suite | ≥5.5 | System install; for step 06 FIMO |

## Citation

If you use this code, please cite:

```
Dearborn et al. (2026). Learning and probing sequence representations of
immune cell alternative splicing. [journal]
```

Please also cite Borzoi:

```
Linder et al. (2023). Predicting RNA-seq coverage from DNA sequence as a
unifying model of gene regulation. Nature Methods.
```
