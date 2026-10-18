#!/usr/bin/env bash
# make_figures.sh -- regenerate manuscript figure panels + tables.
#
# Scaffold orchestrator: documents the order and surfaces the manual/companion
# steps. Scripts read run outputs via figures/config.py (check with
# `python figures/config.py` first). CPU-only unless noted.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "== Sanity-check data paths =="
python config.py

echo "== Appendix Table A2 (bootstrap comparison) =="
python consolidated_comparison_table.py --n-boot 2000
# requires pretrained per-event preds first (GPU inference, run once):
#   python eval_pretrained_borzoi.py --cell-type B_Cell --out-dir "$SB_PRETRAINED_EVAL_ROOT"
#   (repeat for Macrophage, T_Cell)

echo "== Figure 2A-C (prediction bars, CPU replot) =="
python prediction_bars.py

echo "== Figure 2D-E (multitask compression) =="
python multitask_collapse.py --ft-type locon

echo "== Figure 2 composite =="
pdflatex -interaction=nonstopmode compose_figure2.tex

echo "== Figure 3C-D / Appendix A6-A7 (probe comparison) =="
python probe_comparison.py --group both --method both

echo "== Figure 3F (LoCon vs frozen) =="
python locon_vs_frozen_probe.py

echo "== Figure 3 composite =="
pdflatex -interaction=nonstopmode compose_figure3.tex

echo
echo "== Manual / companion-repo steps (not run here) =="
echo "  Fig 2A-C regeneration : python eval_scatter.py ... (GPU inference) only if"
echo "                          recomputing predictions; prediction_bars.py above"
echo "                          replots A-C from saved metrics, CPU-only."
echo "  Appendix A3 scatter   : python eval_scatter.py --separated-event-type (GPU)"
echo "  Fig 3A-B, A4-A5, A8   : ../scripts/03_umap.py, ../scripts/01_eval_finetuned.py"
echo "  Fig 1, Appendix A2    : companion repo Imm_cell_rMATS/analysis/figure1/"
echo "  Appendix A1 schematic : hand-made illustrative figure (not analysis; no script in the repo)"
echo "  Appendix A9-A11       : ../scripts/05-06 + eLife companion (attribution/MoDISco/FIMO)"
echo "  Fig 1 / Fig 2 / Fig 3 : composited programmatically (compose_figure{1,2,3}.tex) --"
echo "                          no Adobe Illustrator step."
