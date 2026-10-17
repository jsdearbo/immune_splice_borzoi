# Manuscript figures

Figure- and table-generation code for *"Learning and probing sequence
representations of immune cell alternative splicing"* (MLCB 2026 / PMLR).

These scripts are the **assembly layer**: they consume the outputs of the model
pipeline in [`../scripts/`](../scripts) (fine-tune → eval → extract embeddings →
UMAP → probe → attribution) and the upstream labeling pipeline in the companion
repo [`Imm_cell_rMATS`](https://github.com/jsdearbo/Imm_cell_rMATS), and turn
them into the manuscript's figures and tables.

All run-output locations are centralized in [`config.py`](config.py) (overridable
via environment variables). See **Finishing the port** below for the known gap.

## Figure / table → code map

| Manuscript item | Script here | Consumes | Notes |
|---|---|---|---|
| **Fig 1** (labeling framework; A–D) | — | — | Lives in companion **`Imm_cell_rMATS/analysis/figure1/`** (`panel_*` + `compose_figure1.tex`). Not duplicated here. |
| **Fig 2A–C** (single-cell-type prediction) | `eval_scatter.py` → `plot_comparison` | rMATS events + model checkpoints (inference) | Also emits Appendix Fig A3 panels (`plot_separated_events`). |
| **Fig 2D–E** (multitask compression) | `multitask_collapse.py` | per-event pred TSVs (`EXPERIMENTS_ROOT`) | Correlation matrices + observed-vs-predicted PSI range. |
| **Fig 2** final composite | *(manual)* | panels above | Assembled in Adobe Illustrator — no compositor script (see **Honesty notes**). |
| **Fig 3A–B** (UMAP: pretrained / adapted) | `../scripts/03_umap.py` | embeddings (`UMAP_RUN_ROOT`) | Produced by the clean pipeline, run per model. |
| **Fig 3C–D** (KNN probe bars) | `probe_comparison.py` | probe CSVs (`UMAP_RUN_ROOT`, `PRETRAINED_UMAP_ROOT`) | Also emits Appendix Figs A6–A7. |
| **Fig 3E** (frozen-trunk UMAP) | `../scripts/03_umap.py` | frozen-model embeddings | Same script, `frozen_*` run dirs. |
| **Fig 3F** (LoCon vs frozen, by probe) | `locon_vs_frozen_probe.py` | probe CSVs (`UMAP_RUN_ROOT`) | New for the revision. |
| **Fig 3** final composite | `compose_figure3.tex` | panel PDFs/PNGs above | `pdflatex compose_figure3.tex`. |
| **Appendix A1** (model schematic) | `model_taxonomy_schematic.py` | none (illustrative) | Self-contained. |
| **Appendix A2** (labeling workflow) | — | — | Companion `Imm_cell_rMATS` (flowchart). |
| **Appendix A3** (event diagnostics) | `eval_scatter.py` → `plot_separated_events` | as Fig 2A–C | |
| **Appendix A4–A5** (UMAP attr maps) | `../scripts/03_umap.py` | embeddings | |
| **Appendix A6–A7** (probe bars / linear) | `probe_comparison.py` | probe CSVs | |
| **Appendix A8** (Ptprc collapse) | `../scripts/01_eval_finetuned.py` | gene-track preds | Gene-of-interest track plot. |
| **Appendix A9–A11** (motifs) | `../scripts/05–06` + eLife companion | attributions | Attribution → TF-MoDISco → FIMO. |
| **Appendix Table A2** (4-config comparison) | `consolidated_comparison_table.py` | per-event pred TSVs (`EXPERIMENTS_ROOT`, `PRETRAINED_EVAL_ROOT`) | Bootstrap 95% CIs + effect sizes. |

Pretrained-Borzoi zero-shot per-event predictions (needed by Table A2) are
produced by `eval_pretrained_borzoi.py` → `PRETRAINED_EVAL_ROOT`.

## Running

```bash
conda activate immune-splice-borzoi      # same env as the model pipeline
python figures/config.py                 # sanity-check that data paths resolve
python figures/multitask_collapse.py --ft-type locon
python figures/consolidated_comparison_table.py --n-boot 2000
python figures/locon_vs_frozen_probe.py
python figures/probe_comparison.py --group both --method both
python figures/model_taxonomy_schematic.py
pdflatex figures/compose_figure3.tex
```

Outputs land in `figures/out/` (git-ignored). `eval_scatter.py` /
`eval_pretrained_borzoi.py` require model checkpoints + a GPU (they run
inference); the rest only read saved run outputs and are CPU-only.

## Finishing the port (known gap)

This is a **scaffold**. Each script was copied verbatim from the scratch repos
(provenance in each file's header banner) and still defines its data paths as
local constants. To make the release self-contained:

1. **Route paths through `config.py`** — replace the hardcoded `_EXPT_BASE` /
   `_RUN_ROOT` / `PRETRAINED_DIR` / `RMATS_BASE` / etc. constants in each script
   with imports from `config.py`. `config.py` already enumerates every root.
2. **Archive the run outputs to Zenodo** and set the `config.py` defaults (or
   documented env vars) to the deposited layout, so the scripts reproduce the
   figures from public data rather than the dev-server scratch dirs.
3. **Pin versions** — the figures in the paper were built at a specific commit;
   tag it and record the tag/DOI in the manuscript's Code Availability statement.

## Honesty notes (reproducibility scope)

- **Figures 1 and 2 are assembled manually** (Adobe Illustrator; see the
  software table, Appendix Table A1). The scripts here reproduce the *panels and
  the underlying numbers*; final panel layout/lettering is manual. Figure 3 is
  the exception — `compose_figure3.tex` assembles it programmatically.
- `multitask_collapse.py` keeps its original internal name (`fig5_*`) in the
  module docstring; the manuscript figure numbering changed during revision
  (old Fig 5 → Fig 2D–E). The file header banner states the current mapping.
