"""
model_taxonomy_schematic.py

Illustrative schematic (no data dependency) comparing the four model
configurations used throughout the manuscript: what's frozen vs. trained in
the trunk and head, whether the configuration is cell-type-specific, and what
question each configuration is designed to answer.

Cell-type colors match Figure 1's PSI-distribution panel (B Cell #2166AC,
Macrophage #4DAC26, T Cell #D6604D) for visual consistency across the paper.

Output: out/model_taxonomy_schematic.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

CELL_COLORS = {"B": "#2166AC", "Mac": "#4DAC26", "T": "#D6604D"}
FROZEN_COLOR = "#B0B0B0"
TRAINED_COLOR = "#4878D0"

ROWS = [
    {
        "name": "Pretrained Borzoi",
        "trunk_state": "frozen",
        "head_state": "frozen",
        "shared_trunk": True,
        "n_heads": 1,
        "cell_specific": "No — one shared model,\nno PSI training at all",
        "tests": "What splicing-relevant information\nalready exists with zero\nPSI-directed training",
    },
    {
        "name": "Frozen-trunk\nhead-only",
        "trunk_state": "frozen",
        "head_state": "trained",
        "shared_trunk": False,
        "n_heads": 3,
        "cell_specific": "Yes — separate head\nper cell type",
        "tests": "What PSI-relevant information is\nalready decodable from pretrained\nrepresentations via a lightweight,\nper-cell-type head alone",
    },
    {
        "name": "Single-cell-type\nLoCon-adapted",
        "trunk_state": "trained",
        "head_state": "trained",
        "shared_trunk": False,
        "n_heads": 3,
        "cell_specific": "Yes — fully independent\nmodel per cell type",
        "tests": "Whether lightweight, per-cell-type\ntrunk adaptation improves performance\nor shifts representation geometry vs.\nthe frozen-trunk baseline",
    },
    {
        "name": "Shared-trunk\nmultitask LoCon",
        "trunk_state": "trained",
        "head_state": "trained",
        "shared_trunk": True,
        "n_heads": 3,
        "cell_specific": "Partially — heads are\nper-cell-type, trunk\nadaptation is shared",
        "tests": "Whether a jointly-adapted shared trunk\npreserves cell-type-specific PSI\ndifferences or compresses them\ntoward a shared profile",
    },
]

CELL_LABELS = ["B", "Mac", "T"]


def box(ax, x, y, w, h, state, label, fontsize=8):
    color = FROZEN_COLOR if state == "frozen" else TRAINED_COLOR
    edge = "#666666" if state == "frozen" else "#1c3f7a"
    hatch = "///" if state == "frozen" else None
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                        linewidth=1.2, edgecolor=edge, facecolor=color,
                        hatch=hatch, alpha=0.9 if state == "frozen" else 1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white" if state == "trained" else "#333333")
    return x + w / 2, y + h / 2


def draw_row(ax, y0, row_h, row):
    """Shared-trunk layout: one trunk box fanning out into n head boxes,
    all heads sized/spaced to fit strictly within [y0, y0+row_h]."""
    trunk_w, trunk_h = 1.3, min(row_h * 0.7, row_h - 0.1)
    trunk_x = 2.6
    trunk_y = y0 + (row_h - trunk_h) / 2

    tcx, tcy = box(ax, trunk_x, trunk_y, trunk_w, trunk_h, row["trunk_state"], "Trunk")

    head_x0 = trunk_x + trunk_w + 0.9
    n = row["n_heads"]
    head_w = 0.85
    gap = 0.12
    # Solve head_h so that n heads + (n-1) gaps fit exactly within row_h
    head_h = (row_h - (n - 1) * gap) / n
    head_y_start = y0

    for i in range(n):
        # i=0 (B) at top, i=n-1 (T) at bottom -> reverse for y (axis grows upward)
        hy = head_y_start + (n - 1 - i) * (head_h + gap)
        label = "Head" if n == 1 else f"Head\n({CELL_LABELS[i]})"
        hcx, hcy = box(ax, head_x0, hy, head_w, head_h, row["head_state"], label, fontsize=6.5)
        src = (tcx + trunk_w / 2, tcy)
        arrow = FancyArrowPatch(src, (hcx - head_w / 2, hcy),
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, color="#555555", shrinkA=2, shrinkB=2)
        ax.add_patch(arrow)
        if n > 1:
            ax.add_patch(plt.Rectangle((hcx - head_w / 2 - 0.08, hy), 0.05, head_h,
                                        color=list(CELL_COLORS.values())[i], lw=0))


def draw_independent_units(ax, y0, row_h, row):
    """For non-shared-trunk, multi-head rows: draw n independent trunk->head
    units stacked top-to-bottom in B / Mac / T order."""
    n = row["n_heads"]
    unit_h = row_h / n
    trunk_w, head_w = 1.05, 0.75
    trunk_h = unit_h * 0.68
    head_h = unit_h * 0.68
    trunk_x = 2.6
    head_x = trunk_x + trunk_w + 0.7

    for i in range(n):
        # i=0 (B) at top, i=n-1 (T) at bottom -> reverse for y (axis grows upward)
        uy0 = y0 + (n - 1 - i) * unit_h
        ty = uy0 + (unit_h - trunk_h) / 2
        hy = uy0 + (unit_h - head_h) / 2
        tcx, tcy = box(ax, trunk_x, ty, trunk_w, trunk_h, row["trunk_state"], "Trunk", fontsize=7)
        hcx, hcy = box(ax, head_x, hy, head_w, head_h, row["head_state"], f"Head\n({CELL_LABELS[i]})", fontsize=6.5)
        arrow = FancyArrowPatch((tcx + trunk_w / 2, tcy), (hcx - head_w / 2, hcy),
                                 arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color="#555555", shrinkA=2, shrinkB=2)
        ax.add_patch(arrow)
        ax.add_patch(plt.Rectangle((trunk_x - 0.12, uy0 + 0.03), 0.06, unit_h - 0.06,
                                    color=list(CELL_COLORS.values())[i], lw=0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_rows = len(ROWS)
    row_h = 1.7
    fig_h = n_rows * row_h + 0.6
    fig, ax = plt.subplots(figsize=(11, fig_h))

    for ri, row in enumerate(ROWS):
        y0 = (n_rows - 1 - ri) * row_h

        # Row label
        ax.text(0.05, y0 + row_h / 2, row["name"], ha="left", va="center",
                fontsize=10.5, fontweight="bold")

        if row["shared_trunk"] or row["n_heads"] == 1:
            draw_row(ax, y0 + 0.15, row_h - 0.3, row)
        else:
            draw_independent_units(ax, y0 + 0.15, row_h - 0.3, row)

        # Cell-type-specific column
        ax.text(7.3, y0 + row_h / 2, row["cell_specific"], ha="left", va="center", fontsize=8)

        # Tests column
        ax.text(9.3, y0 + row_h / 2, row["tests"], ha="left", va="center", fontsize=8)

        if ri < n_rows - 1:
            ax.axhline(y0, xmin=0.01, xmax=0.99, color="#DDDDDD", linewidth=0.8)

    # Column headers
    header_y = n_rows * row_h + 0.15
    for xpos, label in [(0.05, "Configuration"), (2.6, "Architecture"),
                         (7.3, "Cell-type-specific?"), (9.3, "Tests")]:
        ax.text(xpos, header_y, label, ha="left", va="bottom", fontsize=10, fontweight="bold")

    # Legend
    legend_y = -0.55
    ax.add_patch(FancyBboxPatch((0.05, legend_y), 0.35, 0.28, boxstyle="round,pad=0.02",
                                 facecolor=FROZEN_COLOR, edgecolor="#666666", hatch="///"))
    ax.text(0.5, legend_y + 0.14, "Frozen (not trained)", va="center", fontsize=8.5)
    ax.add_patch(FancyBboxPatch((3.1, legend_y), 0.35, 0.28, boxstyle="round,pad=0.02",
                                 facecolor=TRAINED_COLOR, edgecolor="#1c3f7a"))
    ax.text(3.55, legend_y + 0.14, "Trained", va="center", fontsize=8.5)
    for i, (ct, c) in enumerate(CELL_COLORS.items()):
        ax.add_patch(plt.Rectangle((5.6 + i * 1.1, legend_y), 0.12, 0.28, color=c, lw=0))
        ax.text(5.75 + i * 1.1, legend_y + 0.14, ct, va="center", fontsize=8.5)

    ax.set_xlim(-0.1, 12.2)
    ax.set_ylim(legend_y - 0.3, header_y + 0.35)
    ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "model_taxonomy_schematic.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
