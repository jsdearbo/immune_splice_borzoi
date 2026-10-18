# ===========================================================================
# Figure 2A-C (per-cell-type prediction accuracy by event class).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# CPU-only replot of the single-cell-type prediction bars. The original panels
# were produced by eval_scatter.py, which runs Borzoi inference; this script
# instead reads the saved per-event-class metrics (model stats, ridge baselines,
# pretrained summary) so the figure reproduces from the Zenodo deposit without a
# GPU. Numbers are identical to the inference run (same held-out chr11 events).
# ===========================================================================
"""prediction_bars.py

Manuscript Figure 2A-C: grouped bar charts of held-out (chr11) Pearson r by
event class (Constitutive, SE, RI, A3SS, A5SS, All alt events) for each cell
type, comparing four models:

  MaxEnt only | Full features | Pretrained Borzoi | PSI-Adapted Borzoi

Inputs (all via figures/config.py):
  model stats    EXPERIMENTS_ROOT/<locon run>/<ct>/evaluation_test/scatterplots/
                 stats_<CellType>_chr11_npz.csv   (event_type, pearson_r)
  ridge baselines BASELINE_SUMMARY                (cell_type, event_type, baseline, split, pearson_r)
  pretrained      PRETRAINED_EVAL_ROOT/pretrained_borzoi_chr11_summary.csv

Outputs (under FIG_OUT_DIR): fig2/comparison_<CellType>.pdf  (one per cell type)

Usage:
  python prediction_bars.py
  python prediction_bars.py --chrom chr11
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import figstyle


# Single-cell-type LoCon run that supplies the PSI-Adapted model stats.
_LOCON_RUN = "20260511_single_cell_types_locon4"
# directory key -> filename tag
CELL_TYPES = {"B_cell": "B_Cell", "Macrophage": "Macrophage", "T_cell": "T_Cell"}

EVENT_ORDER = ["CONST", "SE", "RI", "A3SS", "A5SS", "ALL"]
EVENT_DISPLAY = {
    "CONST": "Constitutive", "SE": "SE", "RI": "RI",
    "A3SS": "A3SS", "A5SS": "A5SS", "ALL": "All alt\nevents",
}

# (baseline key, color, legend label)
BAR_SPEC = [
    ("maxent_only",   "#aec7e8", "MaxEnt only\n(donor+acceptor)"),
    ("full_features", "#1f77b4", "Full features\n(+GC+length)"),
    ("pretrained",    "#f0a070", "Pretrained\nBorzoi"),
    ("model",         "salmon",  "PSI-Adapted\nBorzoi"),
]


def model_stats_path(ct_dir: str, ct_tag: str, chrom: str) -> str:
    return os.path.join(
        str(config.EXPERIMENTS_ROOT), _LOCON_RUN, ct_dir,
        "evaluation_test", "scatterplots", f"stats_{ct_tag}_chr11_npz.csv",
    )


def make_legend(out_dir):
    """Standalone horizontal legend shared by panels A-C (drawn once in the
    composite rather than repeated on every panel -- Reviewer 2)."""
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=c, edgecolor="black", linewidth=0.5, label=l)
               for _, c, l in BAR_SPEC]
    fig = plt.figure(figsize=(9, 0.5))
    fig.legend(handles=handles, loc="center", ncol=len(BAR_SPEC),
               frameon=False, handlelength=1.3, columnspacing=1.8)
    out = os.path.join(out_dir, "comparison_legend.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved -> {out}")
    return out


def plot_cell_type(ct_tag, chrom, baseline_df, pretrained_df, out_dir):
    """One grouped bar chart (Pearson r by event class) for a cell type."""
    model_df = pd.read_csv(model_stats_path(_CT_DIR[ct_tag], ct_tag, chrom))
    model_lookup = dict(zip(model_df["event_type"], model_df["pearson_r"]))

    bdf = baseline_df[(baseline_df["cell_type"] == ct_tag) & (baseline_df["split"] == "test")]
    pdf = pretrained_df[(pretrained_df["cell_type"] == ct_tag) & (pretrained_df["chrom"] == chrom)]
    pretrained_lookup = dict(zip(pdf["event_type"], pdf["pearson_r"]))

    events = [e for e in EVENT_ORDER
              if e in set(bdf["event_type"]) or e in model_lookup]
    n_bars = len(BAR_SPEC)
    width = 0.8 / n_bars
    x = np.arange(len(events))

    fig, ax = plt.subplots(figsize=(1.5 * len(events) + 1.3, 1.7))
    for i, (bname, col, label) in enumerate(BAR_SPEC):
        offset = (i - (n_bars - 1) / 2) * width
        if bname == "model":
            vals = [model_lookup.get(e, np.nan) for e in events]
        elif bname == "pretrained":
            vals = [pretrained_lookup.get(e, np.nan) for e in events]
        else:
            vals = []
            for e in events:
                row = bdf[(bdf["event_type"] == e) & (bdf["baseline"] == bname)]
                vals.append(float(row["pearson_r"].values[0]) if len(row) else np.nan)
        ax.bar(x + offset, vals, width, label=label,
               color=col, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_DISPLAY.get(e, e) for e in events])
    ax.set_ylabel("Pearson r  (test, chr11)")
    ax.set_title(ct_tag.replace("_", " "))
    ax.set_ylim(-0.2, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    # No per-panel legend: A-C share one standalone legend in the composite
    # (see make_legend), so repeating it on each panel is redundant (Reviewer 2).

    out = os.path.join(out_dir, f"comparison_{ct_tag}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved -> {out}")
    return out


_CT_DIR = {tag: d for d, tag in CELL_TYPES.items()}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    figstyle.apply()

    parser = argparse.ArgumentParser(description="Figure 2A-C prediction bars (CPU replot)")
    parser.add_argument("--chrom", default="chr11")
    parser.add_argument("--out-dir", default=os.path.join(str(config.OUT_DIR), "fig2"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    baseline_df = pd.read_csv(str(config.BASELINE_SUMMARY))
    pretrained_df = pd.read_csv(
        os.path.join(str(config.PRETRAINED_EVAL_ROOT),
                     f"pretrained_borzoi_{args.chrom}_summary.csv"))

    for ct_tag in CELL_TYPES.values():
        logging.info(f"{ct_tag}")
        plot_cell_type(ct_tag, args.chrom, baseline_df, pretrained_df, args.out_dir)
    make_legend(args.out_dir)
    logging.info("Done.")


if __name__ == "__main__":
    main()
