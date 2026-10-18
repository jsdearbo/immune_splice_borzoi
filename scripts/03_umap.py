"""
04_umap.py

Load saved embeddings (from 03_extract_embeddings.py) and produce
colour-coded UMAP scatter plots per layer. No GPU required.

Reads:
  --embeddings   embeddings.npz (from 03_extract_embeddings.py)
  --in           input_dedup.csv (default: input_dedup.csv next to embeddings)

Writes per-layer PNGs to --out-dir using the same subfolder structure as
03_umap_embedding.py so make_umap_reports.py can compile them as before:
  <out-dir>/cell_type/umap_{layer}_cell_type.png
  <out-dir>/event_type/umap_{layer}_event_type.png
  <out-dir>/gc_content/umap_{layer}_gc_content.png
  <out-dir>/maxent/umap_{layer}_donor_score.png
  <out-dir>/maxent/umap_{layer}_acceptor_score.png
  <out-dir>/psi/umap_{layer}_psi.png
  <out-dir>/element_length/umap_{layer}_element_length.png

Usage:
  python 04_umap.py --embeddings SE_results/embeddings.npz
                    [--in SE_results/input_dedup.csv]
                    [--out-dir SE_results]
                    [--model-tag pretrained_SE]
"""

import os
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap as umap_lib

# Shared figure styling lives in ../figures/figstyle.py (one font scale across
# all manuscript panels); add it to the path since this script is in scripts/.
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
import figstyle


# ── Constants ─────────────────────────────────────────────────────────────────

# Layers in network forward-pass order (subset present in NPZ may be smaller)
LAYER_ORDER = [
    "conv_tower",
    "transformer",
    "unet_dec_0",
    "unet_dec_1",
    "final_conv",
    "head_shared",
]

# (subfolder, column_name, kind)
COLOR_SCHEMES = [
    ("cell_type",     "cell_type",      "categorical"),
    ("event_type",    "event_type",     "categorical"),
    ("gc_content",    "gc_content",     "continuous"),
    ("maxent",        "donor_score",    "continuous"),
    ("maxent",        "acceptor_score", "continuous"),
    ("psi",           "psi",            "continuous"),
    ("element_length","element_length", "continuous"),
]

CATEGORICAL_PALETTES = {
    "cell_type":  ["#e6194B", "#4363d8", "#3cb44b"],
    "event_type": ["#f58231", "#911eb4", "#42d4f4", "#bfef45"],
}

CONTINUOUS_CMAPS = {
    "gc_content":     "viridis",
    "donor_score":    "plasma",
    "acceptor_score": "plasma",
    "psi":            "RdYlBu_r",
    "element_length": "cividis",
}


# ── UMAP ─────────────────────────────────────────────────────────────────────

def run_umap(mat: np.ndarray) -> np.ndarray | None:
    """Run UMAP on valid (non-NaN) rows. Returns (N, 2) with NaN for bad rows."""
    valid_mask = ~np.isnan(mat).any(axis=1)
    if valid_mask.sum() < 5:
        logging.warning("Fewer than 5 valid embeddings — skipping UMAP.")
        return None

    reducer = umap_lib.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    )
    coords_valid = reducer.fit_transform(mat[valid_mask])

    coords = np.full((len(mat), 2), np.nan)
    coords[valid_mask] = coords_valid
    return coords


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_categorical(ax, coords, labels, palette, title, col_name):
    categories = sorted(labels.dropna().unique())
    color_map  = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    for cat in categories:
        mask = (labels == cat) & ~np.isnan(coords[:, 0])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color_map[cat], label=str(cat),
            s=18, alpha=0.75, linewidths=0,
        )

    ax.legend(title=col_name, markerscale=1.5, loc="best", framealpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)


def _plot_continuous(ax, coords, values, cmap_name, title, col_name):
    valid    = ~np.isnan(coords[:, 0])
    nan_vals = np.isnan(values)

    grey_mask = valid & nan_vals
    if grey_mask.any():
        ax.scatter(coords[grey_mask, 0], coords[grey_mask, 1],
                   c="#cccccc", s=12, alpha=0.4, linewidths=0, label="N/A")

    plot_mask = valid & ~nan_vals
    if plot_mask.any():
        sc = ax.scatter(
            coords[plot_mask, 0], coords[plot_mask, 1],
            c=values[plot_mask], cmap=cmap_name,
            s=18, alpha=0.8, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=col_name, shrink=0.8)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)


def save_umap_plots(
    coords: np.ndarray,
    df: pd.DataFrame,
    component_label: str,
    out_dir: str,
    model_tag: str,
):
    """Save one PNG per colour scheme for a single layer's UMAP."""
    n_valid = (~np.isnan(coords[:, 0])).sum()

    for subfolder, col_name, kind in COLOR_SCHEMES:
        if col_name not in df.columns:
            logging.debug(f"Skipping '{col_name}' — column not in table.")
            continue

        subdir = os.path.join(out_dir, subfolder)
        os.makedirs(subdir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        # Concise title: layer | attribute. Model tag and N (={n_valid}) are
        # conveyed by the panel grouping and caption, so they are omitted here
        # to keep the small composited panels legible.
        title = f"{component_label} | {col_name}"

        if kind == "categorical":
            palette = CATEGORICAL_PALETTES.get(col_name, list(mcolors.TABLEAU_COLORS.values()))
            _plot_categorical(ax, coords, df[col_name], palette, title, col_name)
        else:
            vals = pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=float)
            _plot_continuous(ax, coords, vals,
                             CONTINUOUS_CMAPS.get(col_name, "viridis"), title, col_name)

        plt.tight_layout()
        out_path = os.path.join(subdir, f"umap_{component_label}_{col_name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logging.info(f"  -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    figstyle.apply()

    parser = argparse.ArgumentParser(
        description="Run UMAP on saved embeddings and save colour-coded scatter plots"
    )
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to embeddings.npz (output of 03_extract_embeddings.py)",
    )
    parser.add_argument(
        "--in", dest="input", type=str, default=None,
        help="Deduplicated label table. Default: input_dedup.csv in the embeddings directory.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for UMAP PNGs. Default: directory of --embeddings.",
    )
    parser.add_argument(
        "--model-tag", type=str, default="",
        help="Label shown in plot titles.",
    )
    args = parser.parse_args()

    emb_dir = os.path.dirname(os.path.abspath(args.embeddings))

    if args.input is None:
        args.input = os.path.join(emb_dir, "input_dedup.csv")
    if args.out_dir is None:
        args.out_dir = emb_dir

    for path, label in [(args.embeddings, "embeddings"), (args.input, "label table")]:
        if not os.path.exists(path):
            logging.error(f"Missing {label}: {path}")
            return

    logging.info(f"Loading embeddings from {args.embeddings}")
    data = np.load(args.embeddings, allow_pickle=False)
    layers_present = [l for l in LAYER_ORDER if l in data]
    logging.info(f"Layers found: {layers_present}")

    logging.info(f"Loading label table from {args.input}")
    df = pd.read_csv(args.input).reset_index(drop=True)
    logging.info(f"Events: {len(df)}")

    os.makedirs(args.out_dir, exist_ok=True)

    for layer in layers_present:
        mat = data[layer]

        if len(mat) != len(df):
            logging.error(
                f"Row count mismatch for layer '{layer}': "
                f"embeddings={len(mat)}, table={len(df)}. Skipping."
            )
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Layer: {layer}  shape: {mat.shape}")

        coords = run_umap(mat)
        if coords is None:
            continue

        save_umap_plots(
            coords=coords,
            df=df,
            component_label=layer,
            out_dir=args.out_dir,
            model_tag=args.model_tag,
        )

    logging.info("\nAll layers done.")


if __name__ == "__main__":
    main()
