# ===========================================================================
# Figure 3 panels A/B/E -- compact UMAP grids.
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
# ===========================================================================
"""umap_grid.py

One compact UMAP grid per model for Figure 3 (panels A pretrained, B adapted,
E frozen-trunk). Rows = colour attributes (MaxEnt donor score, PSI), columns =
network layers. A single shared colorbar per row, no axis ticks (UMAP
coordinates are arbitrary), bold column titles (layer) and row labels
(attribute). This maximises the scatter area and removes the per-cell clutter
(ticks, repeated UMAP-1/2 labels, per-panel colorbars) of the old one-PNG-per-
panel layout.

CPU only; UMAP is seeded (random_state=42) so layouts match the published
figure. Coordinates are cached next to the embeddings (umap_coords.npz) so
re-rendering (e.g. font tuning) does not recompute UMAP.

Usage:
  python umap_grid.py --embeddings <run>/RI_results/embeddings.npz --out A.pdf
  python umap_grid.py --embeddings <frozen>/RI_results/embeddings.npz \
                      --layers head_shared --out E.pdf
"""
import os
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap as umap_lib

import figstyle

KEEP_LAYERS = ["conv_tower", "transformer", "final_conv", "head_shared"]
LAYER_LABELS = {
    "conv_tower": "Conv Tower", "transformer": "Transformer",
    "final_conv": "Final Conv", "head_shared": "New Decoder",
}
# (column, row-label, colormap) -- matches CONTINUOUS_CMAPS in scripts/03_umap.py
ATTRS = [
    ("donor_score", "MaxEnt donor score", "plasma"),
    ("psi",         "PSI",                "RdYlBu_r"),
]


def run_umap(mat: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(mat).any(axis=1)
    coords = np.full((len(mat), 2), np.nan)
    if valid.sum() >= 5:
        reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                                metric="cosine", random_state=42)
        coords[valid] = reducer.fit_transform(mat[valid])
    return coords


def layer_coords(npz, layers, cache_path):
    """UMAP coords per layer, cached next to the embeddings."""
    cache = {}
    if os.path.isfile(cache_path):
        z = np.load(cache_path)
        cache = {k: z[k] for k in z.files}
    out, dirty = {}, False
    for layer in layers:
        if layer in cache and cache[layer].shape[0] == npz[layer].shape[0]:
            out[layer] = cache[layer]
        else:
            logging.info(f"  UMAP {layer} ...")
            out[layer] = run_umap(npz[layer])
            cache[layer] = out[layer]
            dirty = True
    if dirty:
        np.savez(cache_path, **cache)
        logging.info(f"  cached coords -> {cache_path}")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ap = argparse.ArgumentParser(description="Compact UMAP grid for a model (Fig 3 A/B/E)")
    ap.add_argument("--embeddings", required=True, help="embeddings.npz")
    ap.add_argument("--in", dest="input", default=None, help="label table (default input_dedup.csv beside embeddings)")
    ap.add_argument("--layers", nargs="+", default=None, help="layers/columns (default: KEEP_LAYERS present)")
    ap.add_argument("--cell", type=float, default=2.0, help="inches per grid cell (native)")
    ap.add_argument("--poster", action="store_true", help="larger fonts for poster viewing distance")
    ap.add_argument("--dpi", type=int, default=200, help="output raster DPI (use >=300 for poster PNGs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    figstyle.apply(poster=args.poster)

    npz = np.load(args.embeddings)
    label_csv = args.input or os.path.join(os.path.dirname(args.embeddings), "input_dedup.csv")
    df = pd.read_csv(label_csv)

    layers = args.layers or [l for l in KEEP_LAYERS if l in npz.files]
    coords = layer_coords(npz, layers, os.path.join(os.path.dirname(args.embeddings), "umap_coords.npz"))

    nrows, ncols = len(ATTRS), len(layers)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(args.cell * ncols + 0.6, args.cell * nrows))

    for i, (col, row_label, cmap) in enumerate(ATTRS):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
        sc = None
        for j, layer in enumerate(layers):
            ax = axes[i][j]
            c = coords[layer]
            m = np.isfinite(c[:, 0]) & np.isfinite(vals)
            sc = ax.scatter(c[m, 0], c[m, 1], c=vals[m], cmap=cmap,
                            vmin=vmin, vmax=vmax, s=8, alpha=0.8,
                            linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.margins(0.02)
            ax.set_box_aspect(1)   # square panels
            if i == 0:
                ax.set_title(LAYER_LABELS.get(layer, layer))
            if j == 0:
                ax.set_ylabel(row_label)
        cb = fig.colorbar(sc, ax=axes[i].tolist(), fraction=0.035, pad=0.012)
        cb.outline.set_linewidth(0.5)

    fig.savefig(args.out, bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)
    logging.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
