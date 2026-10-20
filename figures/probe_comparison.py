# ===========================================================================
# Figure 3C-D and Appendix Figs A6-A7 (grouped KNN/linear probe bars across pretrained + 3 cell-type models).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/interpretation/umapping/07_20260511_probe_comparison.py
# Data inputs: run outputs under the paths in figures/config.py (currently
#   hardcoded in this script -- see figures/README.md 'Finishing the port').
# ===========================================================================
"""
probe_comparison.py

Manuscript Figure 3C-D and Appendix Figs A6-A7: grouped bar charts comparing
linear/KNN probe performance across 4 models (pretrained + 3 cell types) for
the locon4 and frozen-trunk runs (paths in figures/config.py).

Two model groups:
  locon:  Pretrained | Locon B Cell | Locon Macrophage | Locon T Cell
  frozen: Pretrained | Frozen B Cell | Frozen Macrophage | Frozen T Cell

Layers shown (unet_dec_0 / unet_dec_1 omitted):
  conv_tower  -> Conv Tower
  transformer -> Transformer
  final_conv  -> Final Conv
  head_shared -> New Decoder   (fine-tuned models only; absent in pretrained)

Outputs (under FIG_OUT_DIR, see figures/config.py): probe_comparison/
  locon_probe_comparison_linear.pdf
  locon_probe_comparison_knn.pdf
  frozen_probe_comparison_linear.pdf
  frozen_probe_comparison_knn.pdf

Usage:
  python probe_comparison.py
  python probe_comparison.py --group locon --method knn
  python probe_comparison.py --event-types SE RI --attributes psi donor_score
"""

import os
import argparse
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ── Constants ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
import config
import figstyle

_PRETRAINED_ROOT = str(config.PRETRAINED_UMAP_ROOT)
_RUN_ROOT        = str(config.UMAP_RUN_ROOT)

KEEP_LAYERS = ["conv_tower", "transformer", "final_conv", "head_shared"]
LAYER_LABELS = {
    "conv_tower":  "Conv Tower",
    "transformer": "Transformer",
    "final_conv":  "Final Conv",
    "head_shared": "New Decoder",
}

MODEL_STYLE = {
    "pretrained":   ("Pretrained",           "#aec7e8"),
    "locon_bcell":  ("Locon B Cell",         "#1f77b4"),
    "locon_mac":    ("Locon Macrophage",     "#ff7f0e"),
    "locon_tcell":  ("Locon T Cell",         "#2ca02c"),
    "frozen_bcell": ("Frozen B Cell",        "#1f77b4"),
    "frozen_mac":   ("Frozen Macrophage",    "#ff7f0e"),
    "frozen_tcell": ("Frozen T Cell",        "#2ca02c"),
}

LOCON_ROOTS = OrderedDict([
    ("pretrained",  _PRETRAINED_ROOT),
    ("locon_bcell", os.path.join(_RUN_ROOT, "locon_B_cell")),
    ("locon_mac",   os.path.join(_RUN_ROOT, "locon_Macrophage")),
    ("locon_tcell", os.path.join(_RUN_ROOT, "locon_T_cell")),
])
FROZEN_ROOTS = OrderedDict([
    ("pretrained",   _PRETRAINED_ROOT),
    ("frozen_bcell", os.path.join(_RUN_ROOT, "frozen_B_cell")),
    ("frozen_mac",   os.path.join(_RUN_ROOT, "frozen_Macrophage")),
    ("frozen_tcell", os.path.join(_RUN_ROOT, "frozen_T_cell")),
])

GROUPS = {"locon": LOCON_ROOTS, "frozen": FROZEN_ROOTS}

# (primary_col, secondary_col, y_label) per attribute
ATTR_CFG = {
    "psi":            ("pearson_r", "r2",  "Pearson r"),
    "donor_score":    ("pearson_r", "r2",  "Pearson r"),
    "acceptor_score": ("pearson_r", "r2",  "Pearson r"),
    "gc_content":     ("pearson_r", "r2",  "Pearson r"),
    "element_length": ("pearson_r", "r2",  "Pearson r"),
    "cell_type":      ("accuracy",  "auc", "Accuracy"),
}

DEFAULT_ATTRIBUTES  = list(ATTR_CFG.keys())
DEFAULT_EVENT_TYPES = ["SE", "RI", "A3SS", "A5SS", "ALL"]
BAR_WIDTH = 0.20


# ── Helpers ───────────────────────────────────────────────────────────────────

def probe_dir(model_root: str, event_type: str, method: str) -> str:
    return os.path.join(model_root, f"{event_type}_results", f"probe_{method}")


def load_probe_csv(path: str, primary_col: str) -> "pd.DataFrame | None":
    """Load a per-attribute probe CSV; filter to KEEP_LAYERS and sort."""
    if not os.path.isfile(path):
        logging.debug(f"CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    if primary_col not in df.columns:
        logging.warning(f"Column '{primary_col}' missing in {path} (columns: {list(df.columns)})")
        return None
    df = df[df["layer"].isin(KEEP_LAYERS)].copy()
    df["_order"] = df["layer"].map({l: i for i, l in enumerate(KEEP_LAYERS)})
    df = df.sort_values("_order").reset_index(drop=True)
    return df


def _get_vals(df: "pd.DataFrame | None", layer: str,
              primary_col: str, secondary_col: str) -> "tuple[float, float]":
    if df is None:
        return np.nan, np.nan
    row = df[df["layer"] == layer]
    if not len(row):
        return np.nan, np.nan
    pval = float(row[primary_col].values[0])
    sval_raw = row[secondary_col].values[0] if secondary_col in row.columns else np.nan
    sval = float(sval_raw) if pd.notna(sval_raw) else np.nan
    return pval, sval


def _annotate_bar(ax, x_center, bar_val, annot_val, annot_col):
    if not (np.isfinite(bar_val) and np.isfinite(annot_val)):
        return
    prefix = {"r2": "r²", "pearson_r": "r", "auc": "auc", "accuracy": "acc"}.get(annot_col, annot_col)
    # Rotate vertical: 3-4 grouped bars per layer are too narrow to fit the
    # annotations horizontally (they otherwise overlap into unreadable runs).
    offset = 0.02 if bar_val >= 0 else -0.02
    va = "bottom" if bar_val >= 0 else "top"
    ax.text(x_center, bar_val + offset, f"{prefix}={annot_val:.2f}",
            ha="center", va=va, rotation=90, fontsize=figstyle.ANNOT_SM, color="#333333")


def plot_comparison(attr: str, model_dfs: dict, event_type: str, method: str,
                    primary_col: str, secondary_col: str, ylabel: str,
                    out_dir: str, fname_prefix: str = "",
                    ymax_fixed: "float | None" = None, fig_height: float = 5.0) -> str:
    n_models = len(model_dfs)
    x = np.arange(len(KEEP_LAYERS))
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * BAR_WIDTH

    fig, ax = plt.subplots(figsize=(max(7, len(KEEP_LAYERS) * 2.2), fig_height))
    all_bar_vals = []
    first_bar = {k: True for k in model_dfs}

    for mi, (model_key, df) in enumerate(model_dfs.items()):
        label, color = MODEL_STYLE[model_key]
        for i, layer in enumerate(KEEP_LAYERS):
            bar_val, annot_val = _get_vals(df, layer, primary_col, secondary_col)
            if not np.isfinite(bar_val):
                continue
            all_bar_vals.append(bar_val)
            xpos = x[i] + offsets[mi]
            ax.bar(xpos, bar_val, BAR_WIDTH, color=color, edgecolor="black", linewidth=0.6,
                   label=label if first_bar[model_key] else "_nolegend_")
            first_bar[model_key] = False
            _annotate_bar(ax, xpos, bar_val, annot_val, secondary_col)

    ymin = min(0.0, (min(all_bar_vals) - 0.08)) if all_bar_vals else -0.1
    # Fixed top (e.g. 1.0) gives the legend clear room above the vertical r²/r
    # annotations on tall bars; otherwise auto headroom above the bars.
    if ymax_fixed is not None:
        ymax = ymax_fixed
    else:
        ymax = (max(all_bar_vals) + 0.26) if all_bar_vals else 1.0
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([LAYER_LABELS[l] for l in KEEP_LAYERS])
    ax.set_ylabel(ylabel)
    attr_disp = figstyle.ATTR_LABELS.get(attr, attr)
    ax.set_title(f"{attr_disp}  —  {method.upper()} probe  |  {event_type} events")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{fname_prefix}{attr}_{event_type}_{method}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved -> {out_path}")
    return out_path


def make_pdf(png_paths: list, pdf_path: str):
    with PdfPages(pdf_path) as pdf:
        for p in png_paths:
            if not os.path.isfile(p):
                continue
            img = plt.imread(p)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.axis("off")
            plt.tight_layout(pad=0)
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)
    logging.info(f"  Saved PDF -> {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    figstyle.apply()

    parser = argparse.ArgumentParser(description="Probe comparison plots (KNN/linear across models)")
    parser.add_argument("--group",       choices=["locon", "frozen", "both"], default="both")
    parser.add_argument("--method",      choices=["linear", "knn", "both"],   default="both")
    parser.add_argument("--event-types", nargs="+", default=DEFAULT_EVENT_TYPES)
    parser.add_argument("--attributes",  nargs="+", default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--out-dir",     default=os.path.join(str(config.OUT_DIR), "probe_comparison"))
    parser.add_argument("--ymax",        type=float, default=None,
                        help="Fixed y-axis top (e.g. 1.0) so the legend clears the bar annotations.")
    parser.add_argument("--fig-height",  type=float, default=5.0,
                        help="Figure height in inches (taller gives the legend more room).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    groups  = ["locon", "frozen"] if args.group  == "both" else [args.group]
    methods = ["linear", "knn"]   if args.method == "both" else [args.method]

    for group in groups:
        roots = GROUPS[group]
        for method in methods:
            logging.info(f"\n{'='*60}")
            logging.info(f"Group: {group.upper()}  |  Method: {method.upper()}")
            png_paths = []

            for event_type in args.event_types:
                for attr in args.attributes:
                    if attr not in ATTR_CFG:
                        logging.warning(f"Unknown attribute '{attr}' — skipping.")
                        continue
                    primary_col, secondary_col, ylabel = ATTR_CFG[attr]
                    logging.info(f"  {event_type} / {attr}")

                    model_dfs = {}
                    for key, root in roots.items():
                        csv_path = os.path.join(probe_dir(root, event_type, method), f"{attr}.csv")
                        model_dfs[key] = load_probe_csv(csv_path, primary_col)

                    if all(v is None for v in model_dfs.values()):
                        logging.warning(f"  No data for {event_type}/{attr} ({method}) — skipping.")
                        continue

                    png = plot_comparison(attr, model_dfs, event_type, method,
                                          primary_col, secondary_col, ylabel,
                                          args.out_dir, fname_prefix=f"{group}_",
                                          ymax_fixed=args.ymax, fig_height=args.fig_height)
                    png_paths.append(png)

            if png_paths:
                pdf_path = os.path.join(args.out_dir, f"{group}_probe_comparison_{method}.pdf")
                make_pdf(png_paths, pdf_path)

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
