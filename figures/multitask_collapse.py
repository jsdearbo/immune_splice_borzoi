# ===========================================================================
# Figure 2D-E (multitask compression: inter-cell-type PSI correlation matrices and observed-vs-predicted range).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/interpretation/umapping/fig5_multitask_collapse.py
# Data inputs: run outputs under the paths in figures/config.py (currently
#   hardcoded in this script -- see figures/README.md 'Finishing the port').
# ===========================================================================
"""
multitask_collapse.py

Manuscript Figure 2D-E: the shared-trunk multitask model compresses cell-type
PSI differences relative to single-task models.

Panel A — Observed vs predicted inter-cell-type PSI range (scatter, single vs multitask)
Panel B — Pairwise ΔPSI preservation (slope/r of pred_delta ~ obs_delta for B–T, B–Mac, T–Mac)
Panel C — Correlation matrices (3×3 Pearson r: observed | single-task | multitask)

Data: held-out chr11 event-level predictions from the fine-tuning experiment
runs (paths in figures/config.py).
  Events shared across all 3 cell types → Panels A and C.
  Pairwise-shared events → Panel B.

Usage:
  python multitask_collapse.py
  python multitask_collapse.py --ft-type locon
  python multitask_collapse.py --both          # frozen_trunk + locon, separate PDFs
  python multitask_collapse.py --out-dir PATH
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ── Constants ─────────────────────────────────────────────────────────────────

import config
import figstyle

_EXPT_BASE = config.EXPERIMENTS_ROOT
_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_OUT = config.OUT_DIR / "fig2_multitask"

FT_CONFIGS = {
    "frozen_trunk": {
        "multi_base":  _EXPT_BASE / "20260511_multitask_frozen_trunk/cosine_decay_fix/evaluation_test/scatterplots",
        "single_base": _EXPT_BASE / "20260511_single_cell_types_frozen_trunk",
        "label": "Frozen Trunk",
    },
    "locon": {
        "multi_base":  _EXPT_BASE / "20260511_multitask_locon/cosine_decay_fix/evaluation_test/scatterplots",
        "single_base": _EXPT_BASE / "20260511_single_cell_types_locon4",
        "label": "LoCon",
    },
}

CELL_TYPES = ["B_cell", "Macrophage", "T_cell"]
_CELL_SUBDIR = {"B_cell": "b_cell", "Macrophage": "macrophage", "T_cell": "t_cell"}
_CELL_FNAME  = {"B_cell": "B_Cell", "Macrophage": "Macrophage", "T_cell": "T_Cell"}
CELL_LABELS  = {"B_cell": "B Cell", "Macrophage": "Mac", "T_cell": "T Cell"}

EVENT_COLORS = {"SE": "#1f77b4", "RI": "#ff7f0e", "A3SS": "#2ca02c", "A5SS": "#d62728"}
MODEL_COLORS = {"single": "#4878d0", "multi": "#ee854a"}
MODEL_LABELS = {"single": "Single-task", "multi": "Multitask"}

PAIRS = [
    ("B_cell", "T_cell",     "B–T"),
    ("B_cell", "Macrophage", "B–Mac"),
    ("T_cell", "Macrophage", "T–Mac"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["event_key"] = (df["Chromosome"].astype(str) + ":" +
                       df["Strand"].astype(str) + ":" +
                       df["Start"].astype(str) + ":" +
                       df["End"].astype(str) + ":" +
                       df["event_type"].astype(str))
    # rMATS can report the same exon with multiple upstream/downstream partners;
    # keep one row per unique (position, event_type) for clean joins.
    df = df.drop_duplicates("event_key", keep="first")
    return df[["event_key", "event_type", "psi", "predicted_psi"]].copy()


def load_all_preds(ft_type: str) -> dict:
    """Returns {cell_type: {"single": df, "multi": df}} where each df has
    columns: event_key, event_type, psi, predicted_psi."""
    cfg = FT_CONFIGS[ft_type]
    data = {}
    for ct in CELL_TYPES:
        multi_path  = cfg["multi_base"] / _CELL_SUBDIR[ct] / f"preds_{_CELL_FNAME[ct]}_chr11_npz.tsv"
        single_path = cfg["single_base"] / ct / "evaluation_test" / "scatterplots" / f"preds_{_CELL_FNAME[ct]}_chr11_npz.tsv"
        for label, path in [("multi", multi_path), ("single", single_path)]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing: {path}")
        data[ct] = {"multi": _load_tsv(multi_path), "single": _load_tsv(single_path)}
        n = len(data[ct]["single"])
        logging.info(f"  {ct}: {n} events")
    return data


def build_three_way(data: dict) -> pd.DataFrame:
    """Inner join events across all 3 cell types (for both model types).

    Returns df with columns:
      event_key, event_type,
      psi_B, psi_Mac, psi_T,
      pred_single_B, pred_single_Mac, pred_single_T,
      pred_multi_B,  pred_multi_Mac,  pred_multi_T
    """
    b_s = (data["B_cell"]["single"]
           .rename(columns={"psi": "psi_B", "predicted_psi": "pred_single_B"}))
    mac_s = (data["Macrophage"]["single"][["event_key", "psi", "predicted_psi"]]
             .rename(columns={"psi": "psi_Mac", "predicted_psi": "pred_single_Mac"}))
    tc_s = (data["T_cell"]["single"][["event_key", "psi", "predicted_psi"]]
            .rename(columns={"psi": "psi_T", "predicted_psi": "pred_single_T"}))

    b_m   = data["B_cell"]["multi"][["event_key", "predicted_psi"]].rename(columns={"predicted_psi": "pred_multi_B"})
    mac_m = data["Macrophage"]["multi"][["event_key", "predicted_psi"]].rename(columns={"predicted_psi": "pred_multi_Mac"})
    tc_m  = data["T_cell"]["multi"][["event_key", "predicted_psi"]].rename(columns={"predicted_psi": "pred_multi_T"})

    df = (b_s
          .merge(mac_s, on="event_key", how="inner")
          .merge(tc_s,  on="event_key", how="inner")
          .merge(b_m,   on="event_key", how="inner")
          .merge(mac_m, on="event_key", how="inner")
          .merge(tc_m,  on="event_key", how="inner"))
    logging.info(f"  3-way shared events: {len(df)}")
    return df


def build_pairwise(data: dict, ct1: str, ct2: str) -> pd.DataFrame:
    """Inner join events between two cell types."""
    d1_s = (data[ct1]["single"]
            .rename(columns={"psi": f"psi_{ct1}", "predicted_psi": f"pred_single_{ct1}"}))
    d2_s = (data[ct2]["single"][["event_key", "psi", "predicted_psi"]]
            .rename(columns={"psi": f"psi_{ct2}", "predicted_psi": f"pred_single_{ct2}"}))
    d1_m = data[ct1]["multi"][["event_key", "predicted_psi"]].rename(columns={"predicted_psi": f"pred_multi_{ct1}"})
    d2_m = data[ct2]["multi"][["event_key", "predicted_psi"]].rename(columns={"predicted_psi": f"pred_multi_{ct2}"})

    return (d1_s
            .merge(d2_s, on="event_key", how="inner")
            .merge(d1_m, on="event_key", how="inner")
            .merge(d2_m, on="event_key", how="inner"))


# ── Panel A: Observed vs Predicted PSI Range ──────────────────────────────────

def panel_a(df3: pd.DataFrame, out_dir: Path, ft_label: str) -> Path:
    import matplotlib.transforms as mtransforms

    df = df3.copy()
    obs_cols    = ["psi_B", "psi_Mac", "psi_T"]
    single_cols = ["pred_single_B", "pred_single_Mac", "pred_single_T"]
    multi_cols  = ["pred_multi_B",  "pred_multi_Mac",  "pred_multi_T"]

    df["obs_range"]    = df[obs_cols].max(axis=1)    - df[obs_cols].min(axis=1)
    df["range_single"] = df[single_cols].max(axis=1) - df[single_cols].min(axis=1)
    df["range_multi"]  = df[multi_cols].max(axis=1)  - df[multi_cols].min(axis=1)

    mean_obs = df["obs_range"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, model_key in zip(axes, ["single", "multi"]):
        pred_col = f"range_{model_key}"
        for etype, color in EVENT_COLORS.items():
            sub = df[df["event_type"] == etype]
            if len(sub) == 0:
                continue
            ax.scatter(sub["obs_range"], sub[pred_col], c=color, alpha=0.45, s=14,
                       label=etype, linewidths=0, rasterized=True)

        # y=x reference (perfect prediction)
        ax.plot([0, 1], [0, 1], color="#999999", linewidth=1.2, linestyle="--")

        # Vertical dotted line at mean observed range
        ax.axvline(mean_obs, color="#666666", linewidth=1.0, linestyle=":")

        # Horizontal line at mean predicted range
        mean_pred = df[pred_col].mean()
        ax.axhline(mean_pred, color="black", linewidth=1.5, linestyle="-")

        # Annotate mean predicted on the y-axis side using a blended transform
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.02, mean_pred + 0.02, f"mean pred = {mean_pred:.3f}",
                transform=trans, ha="left", va="bottom", fontsize=figstyle.ANNOT, color="black")

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Observed PSI range")
        ax.set_title(MODEL_LABELS[model_key])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.97, 0.04, f"n = {len(df)}", transform=ax.transAxes,
                fontsize=figstyle.ANNOT, ha="right", va="bottom", color="#555555")

    axes[0].set_ylabel("Predicted PSI range")

    # Mean observed annotation on the x-axis of the left panel
    trans0 = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
    axes[0].text(mean_obs + 0.01, 0.97, f"mean obs = {mean_obs:.3f}",
                 transform=trans0, ha="left", va="top", fontsize=figstyle.ANNOT, color="#666666")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=8, label=e)
               for e, c in EVENT_COLORS.items()]
    axes[0].legend(handles=handles, frameon=False, loc="upper left")

    fig.suptitle(f"Inter-cell-type PSI range: observed vs predicted  [{ft_label}]",
                 y=1.01)
    plt.tight_layout()

    out_path = out_dir / f"fig5a_range_{ft_label.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")  # vector, for compose_figure2
    plt.close(fig)
    logging.info(f"  Saved -> {out_path}")
    return out_path


# ── Panel B: Pairwise ΔPSI Preservation ──────────────────────────────────────

def panel_b(data: dict, out_dir: Path, ft_label: str,
            min_delta: float = 0.0) -> Path:
    records = []
    for ct1, ct2, pair_name in PAIRS:
        df = build_pairwise(data, ct1, ct2)
        obs_delta = df[f"psi_{ct1}"] - df[f"psi_{ct2}"]
        if min_delta > 0:
            keep = obs_delta.abs() >= min_delta
            df = df[keep].copy()
            obs_delta = obs_delta[keep]
        for model_key in ["single", "multi"]:
            pred_delta = df[f"pred_{model_key}_{ct1}"] - df[f"pred_{model_key}_{ct2}"]
            mask = obs_delta.notna() & pred_delta.notna()
            slope, _, r, _, _ = stats.linregress(obs_delta[mask], pred_delta[mask])
            records.append({"pair": pair_name, "model": model_key,
                            "slope": slope, "pearson_r": r, "n": mask.sum()})
    res = pd.DataFrame(records)

    pair_names = [p[2] for p in PAIRS]
    x = np.arange(len(pair_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    for mi, model_key in enumerate(["single", "multi"]):
        sub = res[res["model"] == model_key].set_index("pair").reindex(pair_names)
        xpos = x + (mi - 0.5) * w
        bars = ax.bar(xpos, sub["slope"], w,
                      color=MODEL_COLORS[model_key], edgecolor="black", linewidth=0.6,
                      label=MODEL_LABELS[model_key])
        for xi, (_, row) in zip(xpos, sub.iterrows()):
            # annotate above bar if positive, below if negative
            offset = 0.008 if row["slope"] >= 0 else -0.008
            va = "bottom" if row["slope"] >= 0 else "top"
            ax.text(xi, row["slope"] + offset, f"{row['slope']:.2f}",
                    ha="center", va=va, fontsize=figstyle.ANNOT_SM)

    ymin = min(res["slope"].min() * 1.4, -0.05)
    ymax = max(res["slope"].max() * 1.4,  0.10)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names)
    ax.set_ylabel("Slope  (pred ΔPSI ~ obs ΔPSI)")
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Pairwise ΔPSI preservation  [{ft_label}]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    # Annotate n (from records, already filtered)
    for xi, pair_name in zip(x, pair_names):
        n = res[(res["pair"] == pair_name) & (res["model"] == "single")]["n"].values[0]
        ax.text(xi, ymin * 0.85, f"n={n}", ha="center", va="top",
                fontsize=figstyle.ANNOT_SM, color="#555555")

    plt.tight_layout()

    out_path = out_dir / f"fig5b_deltapsi_{ft_label.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"  Saved -> {out_path}")
    return out_path


# ── Panel C: Correlation Matrices ─────────────────────────────────────────────

def _corr_matrix(df3: pd.DataFrame, cols: list) -> np.ndarray:
    """3×3 Pearson r matrix from three column lists."""
    mat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            x = df3[cols[i]].values
            y = df3[cols[j]].values
            mask = np.isfinite(x) & np.isfinite(y)
            mat[i, j] = stats.pearsonr(x[mask], y[mask])[0]
    return mat


def panel_c(df3: pd.DataFrame, out_dir: Path, ft_label: str) -> Path:
    tick_labels = ["B Cell", "Mac", "T Cell"]

    obs_cols    = ["psi_B",        "psi_Mac",       "psi_T"]
    single_cols = ["pred_single_B","pred_single_Mac","pred_single_T"]
    multi_cols  = ["pred_multi_B", "pred_multi_Mac", "pred_multi_T"]

    panels = [
        ("Observed",    _corr_matrix(df3, obs_cols)),
        ("Single-task", _corr_matrix(df3, single_cols)),
        ("Multitask",   _corr_matrix(df3, multi_cols)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))

    for ax, (title, mat) in zip(axes, panels):
        im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(tick_labels)
        ax.set_yticks(range(3)); ax.set_yticklabels(tick_labels)
        for i in range(3):
            for j in range(3):
                val = mat[i, j]
                txt_color = "white" if val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_color)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Inter-cell-type PSI correlations  [{ft_label}]  n = {len(df3)}")
    plt.tight_layout()

    out_path = out_dir / f"fig5c_corrmat_{ft_label.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")  # vector, for compose_figure2
    plt.close(fig)
    logging.info(f"  Saved -> {out_path}")
    return out_path


# ── PDF assembly ──────────────────────────────────────────────────────────────

def make_pdf(png_paths: list, pdf_path: Path):
    with PdfPages(pdf_path) as pdf:
        for p in png_paths:
            img = plt.imread(p)
            fig, ax = plt.subplots(figsize=(12, 6))
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

    parser = argparse.ArgumentParser(description="Figure 2D-E multitask compression panels")
    parser.add_argument("--ft-type", choices=list(FT_CONFIGS.keys()), default="frozen_trunk",
                        help="Fine-tuning strategy (default: frozen_trunk)")
    parser.add_argument("--both", action="store_true",
                        help="Generate panels for both frozen_trunk and locon")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--min-obs-range", type=float, default=0.15,
                        help="Min observed PSI range (max-min across 3 cell types) for panels A/C (default: 0.15)")
    parser.add_argument("--min-obs-delta", type=float, default=0.10,
                        help="Min |observed ΔPSI| for pairwise events in panel B (default: 0.10)")
    args = parser.parse_args()

    ft_types = list(FT_CONFIGS.keys()) if args.both else [args.ft_type]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ft_type in ft_types:
        ft_label = FT_CONFIGS[ft_type]["label"]
        logging.info(f"\n{'='*60}")
        logging.info(f"Fine-tuning type: {ft_label}")

        logging.info("Loading prediction TSVs...")
        data = load_all_preds(ft_type)

        logging.info("Building 3-way event intersection...")
        df3 = build_three_way(data)

        # Option 1: filter to events with meaningful observed inter-cell-type variation
        obs_cols = ["psi_B", "psi_Mac", "psi_T"]
        df3["obs_range"] = df3[obs_cols].max(axis=1) - df3[obs_cols].min(axis=1)
        n_before = len(df3)
        df3 = df3[df3["obs_range"] >= args.min_obs_range].copy()
        logging.info(f"  Panels A/C: {n_before} → {len(df3)} events (obs_range ≥ {args.min_obs_range})")

        logging.info("Panel A: observed vs predicted PSI range...")
        pa = panel_a(df3, args.out_dir, ft_label)

        logging.info("Panel B: pairwise ΔPSI preservation...")
        pb = panel_b(data, args.out_dir, ft_label, min_delta=args.min_obs_delta)

        logging.info("Panel C: correlation matrices...")
        pc = panel_c(df3, args.out_dir, ft_label)

        pdf_path = args.out_dir / f"fig5_{ft_type}.pdf"
        make_pdf([pa, pb, pc], pdf_path)

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
