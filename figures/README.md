# Manuscript figures

Figure- and table-generation code for *"Learning and probing sequence
representations of immune cell alternative splicing"* (MLCB 2026 / PMLR).

These scripts are the **assembly layer**: they consume the outputs of the model
pipeline in [`../scripts/`](../scripts) (fine-tune → eval → extract embeddings →
UMAP → probe → attribution) and the upstream labeling pipeline in the companion
repo [`Imm_cell_rMATS`](https://github.com/jsdearbo/Imm_cell_rMATS), and turn
them into the manuscript's figures and tables.

All run-output locations are centralized in [`config.py`](config.py) (overridable
via environment variables). See **Reproducing from the archive** below for the
one remaining step (pointing `config.py` at the public deposit).

## Figure / table → code map

| Manuscript item | Script here | Consumes | Notes |
|---|---|---|---|
| **Fig 1** (labeling framework; A–D) | — | — | Lives in companion **`Imm_cell_rMATS/analysis/figure1/`** (`panel_*` + `compose_figure1.tex`). Not duplicated here. |
| **Fig 2A–C** (single-cell-type prediction) | `prediction_bars.py` | model stats + ridge baselines (`BASELINE_SUMMARY`) + pretrained summary | CPU-only replot from saved metrics. `eval_scatter.py` is the inference source that produced those metrics (and Appendix Fig A3). |
| **Fig 2D–E** (multitask compression) | `multitask_collapse.py` | per-event pred TSVs (`EXPERIMENTS_ROOT`) | Correlation matrices + observed-vs-predicted PSI range; emits PNG + PDF. |
| **Fig 2** final composite | `compose_figure2.tex` | panel PDFs above | `pdflatex compose_figure2.tex`. |
| **Fig 3A–B** (UMAP grids: pretrained / adapted) | `umap_grid.py` | embeddings (`UMAP_RUN_ROOT`) | Compact grid per model (shared colorbar/row, no ticks); seeded, CPU. `../scripts/03_umap.py` still emits the per-panel UMAPs for Appendix A4–A5. |
| **Fig 3C–D** (KNN probe bars) | `probe_comparison.py` | probe CSVs (`UMAP_RUN_ROOT`, `PRETRAINED_UMAP_ROOT`) | Also emits Appendix Figs A6–A7. |
| **Fig 3E** (frozen-trunk UMAP grid) | `umap_grid.py --layers head_shared` | frozen-model embeddings | Same script, `frozen_*` run dir. |
| **Fig 3F** (LoCon vs frozen, by probe) | `locon_vs_frozen_probe.py` | probe CSVs (`UMAP_RUN_ROOT`) | New for the revision. |
| **Fig 3** final composite | `compose_figure3.tex` | panel PDFs/PNGs above | `pdflatex compose_figure3.tex`. |
| **Appendix A1** (model schematic) | — | — | Illustrative schematic, assembled manually (not analysis code). |
| **Appendix A2** (labeling workflow) | — | — | Companion `Imm_cell_rMATS` (flowchart). |
| **Appendix A3** (event diagnostics) | `eval_scatter.py` → `plot_separated_events` | as Fig 2A–C | |
| **Appendix A4–A5** (UMAP attr maps) | `../scripts/03_umap.py` | embeddings | |
| **Appendix A6–A7** (probe bars / linear) | `probe_comparison.py` | probe CSVs | |
| **Appendix A8** (Ptprc collapse) | `../scripts/01_eval_finetuned.py` | gene-track preds | Gene-of-interest track plot. |
| **Appendix A9–A11** (motifs) | `../scripts/05–06` + companion repo | attributions | Attribution → TF-MoDISco → FIMO. |
| **Appendix Table A2** (4-config comparison) | `consolidated_comparison_table.py` | per-event pred TSVs (`EXPERIMENTS_ROOT`, `PRETRAINED_EVAL_ROOT`) | Per-config bootstrap 95% CIs (main table) + **paired** Δr effect sizes on common events (LoCon/Frozen/Pretrained, each pair joined on event_key). |
| **Fig 1C event-level companion** (Results/Discussion stat) | `event_level_psi_fraction.py` | solo rMATS tables (`RMATS_RESULTS`) | Event-level PSI distribution (~18% intermediate under the ≥20-read filter) vs the bin-level Fig 1C (~1–2%). CPU-only. |

Pretrained-Borzoi zero-shot per-event predictions (needed by Table A2) are
produced by `eval_pretrained_borzoi.py` → `PRETRAINED_EVAL_ROOT`.

## Running

```bash
conda activate immune-splice-borzoi      # same env as the model pipeline
python figures/config.py                 # sanity-check that data paths resolve
python figures/multitask_collapse.py --ft-type locon
python figures/consolidated_comparison_table.py --n-boot 2000
python figures/event_level_psi_fraction.py
python figures/locon_vs_frozen_probe.py
python figures/probe_comparison.py --group both --method both
pdflatex figures/compose_figure3.tex
```

Outputs land in `figures/out/` (git-ignored). `eval_scatter.py` /
`eval_pretrained_borzoi.py` require model checkpoints + a GPU (they run
inference); the rest only read saved run outputs and are CPU-only.

## Reproducing from the archive

**Path routing is complete.** Every assembly script in this directory resolves
its data locations through [`config.py`](config.py) (no hardcoded data paths
remain — the former `_EXPT_BASE` / `_RUN_ROOT` / `PRETRAINED_DIR` / `RMATS_BASE`
constants are now thin aliases over `config.*`), and the model-pipeline scripts
in [`../scripts/`](../scripts) take every path as an explicit CLI argument
(`--config`, `--ckpt`, `--embeddings`, `--genome`, …). `python figures/config.py`
prints each resolved root and whether it currently exists.

Two steps remain to make the release fully self-contained from public data:

1. **Point `config.py` at the deposit.** `config.py` resolves each root in the
   order `SB_<NAME>` env var → `SB_DATA_ROOT`/`_LAYOUT` → dev-server default.
   Once the run outputs are deposited, set `SB_DATA_ROOT` to the unpacked archive
   (or individual `SB_<NAME>` vars) and the scripts reproduce the figures from
   public data rather than the dev-server scratch dirs. Adjust `_LAYOUT` in
   `config.py` if the deposit uses different folder names.
2. **Pin versions** — the figures in the paper were built at a specific commit;
   tag it and record the tag/DOI in the manuscript's Code Availability statement
   (Zenodo deposit [`10.5281/zenodo.23442182`](https://doi.org/10.5281/zenodo.23442182)).

## Honesty notes (reproducibility scope)

- **All three main figures are now composited programmatically** via
  `compose_figure{1,2,3}.tex` (`pdflatex`), with no Adobe Illustrator step.
  Figures 1 and 2 were previously hand-assembled in Illustrator; the camera-ready
  revision replaced that with the LaTeX compositors here (Fig 1 lives in the
  companion `Imm_cell_rMATS/analysis/figure1/`). Font/legend styling is shared
  across panels via `figstyle.py`.
- `multitask_collapse.py` keeps its original internal name (`fig5_*`) in the
  module docstring; the manuscript figure numbering changed during revision
  (old Fig 5 → Fig 2D–E). The file header banner states the current mapping.
