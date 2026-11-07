# ===========================================================================
# Figure 3F (LoCon vs frozen-trunk PSI accessibility by probe type).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/interpretation/umapping/08_locon_vs_frozen_psi_probe.py
# Data inputs: all run-output locations resolve through figures/config.py
#   (see figures/README.md "Reproducing from the archive").
# ===========================================================================
"""
locon_vs_frozen_probe.py

Manuscript Figure 3F: direct LoCon-vs-frozen-trunk PSI-accessibility comparison
under KNN and linear probes, across trunk/head layers, averaged over the three
single-cell-type models (B Cell, Macrophage, T Cell).

Visualizes the claim in Results ("LoCon-adapted models showed a modest
improvement over frozen-trunk models ... under k=5 KNN probes, whereas
linear-probe performance was broadly comparable") directly, rather than
leaving it as text pointing at two separate appendix figures.

Data: <UMAP_RUN_ROOT>/{locon,frozen}_{B_cell,Macrophage,T_cell}/ALL_results/probe_{knn,linear}/psi.csv (see figures/config.py)
Output: <FIG_OUT_DIR>/fig3/locon_vs_frozen_psi_probe.pdf
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).parent
import config
import figstyle

_RUN_ROOT = config.UMAP_RUN_ROOT
_OUT_DIR = config.OUT_DIR / "fig3"

CELL_TYPES = ["B_cell", "Macrophage", "T_cell"]
KEEP_LAYERS = ["conv_tower", "transformer", "final_conv", "head_shared"]
LAYER_LABELS = {"conv_tower": "Conv Tower", "transformer": "Transformer",
                 "final_conv": "Final Conv", "head_shared": "New Decoder"}
MODEL_COLORS = {"frozen": "#aec7e8", "locon": "#1f77b4"}
MODEL_LABELS = {"frozen": "Frozen-trunk", "locon": "LoCon-adapted"}


def load_pearson(model_key: str, cell_type: str, method: str) -> pd.Series:
    path = _RUN_ROOT / f"{model_key}_{cell_type}" / "ALL_results" / f"probe_{method}" / "psi.csv"
    df = pd.read_csv(path).set_index("layer")
    return df.loc[KEEP_LAYERS, "pearson_r"]


def main():
    figstyle.apply()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(KEEP_LAYERS))
    width = 0.32

    for ax, method in zip(axes, ["knn", "linear"]):
        for mi, model_key in enumerate(["frozen", "locon"]):
            vals = np.array([load_pearson(model_key, ct, method).values for ct in CELL_TYPES])
            mean = vals.mean(axis=0)
            sem = vals.std(axis=0, ddof=1) / np.sqrt(len(CELL_TYPES))
            xpos = x + (mi - 0.5) * width
            ax.bar(xpos, mean, width, yerr=sem, capsize=3,
                   color=MODEL_COLORS[model_key], edgecolor="black", linewidth=0.6,
                   label=MODEL_LABELS[model_key])
            for xi, m_ in zip(xpos, mean):
                ax.text(xi, m_ + 0.02, f"{m_:.2f}", ha="center", va="bottom",
                        fontsize=figstyle.ANNOT_SM)

        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS[l] for l in KEEP_LAYERS])
        ax.set_title(f"{'$k$=5 KNN' if method == 'knn' else 'Linear'} probe")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.8)

    axes[0].set_ylabel("PSI Pearson r\n(mean $\\pm$ SEM across B/Mac/T)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("LoCon vs. frozen-trunk: PSI accessibility by probe type")
    plt.tight_layout()

    out_path = _OUT_DIR / "locon_vs_frozen_psi_probe.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
