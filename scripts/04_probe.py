"""
05_probe.py

Linear or KNN probe analysis on saved Borzoi splice-event embeddings.

For each layer × attribute combination, fit a probe via cross-validation
and report how well that attribute is encoded in the embedding space. This
reveals which network layers capture each biological/sequence property.

Classification targets (cell_type, event_type): accuracy + macro-AUC
Regression targets (gc_content, psi, donor_score, acceptor_score,
                    element_length):  R² + Pearson r

Reads:
  --embeddings   embeddings.npz (from 03_extract_embeddings.py)
  --in           input_dedup.csv (default: input_dedup.csv next to embeddings)

Writes to --out-dir:
  probe_metrics_{method}.csv   — per layer × attribute × metric
  probe_summary_{method}.pdf   — bar charts of primary metric by layer

Usage:
  python 05_probe.py --embeddings SE_results/embeddings.npz
                     [--in SE_results/input_dedup.csv]
                     [--out-dir SE_results]
                     [--method linear]
                     [--cv-folds 5]
                     [--targets cell_type psi ...]
                     [--layers conv_tower transformer ...]
"""

import os
import logging
import argparse
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ── Constants ─────────────────────────────────────────────────────────────────

LAYER_ORDER = [
    "conv_tower",
    "transformer",
    "unet_dec_0",
    "unet_dec_1",
    "final_conv",
    "head_shared",
]

LAYER_LABELS = {
    "conv_tower":  "Conv Tower",
    "transformer": "Transformer",
    "unet_dec_0":  "UNet Dec 0",
    "unet_dec_1":  "UNet Dec 1",
    "final_conv":  "Final Conv",
    "head_shared": "Shared Head",
}

# (attribute_name, task_type)
ALL_TARGETS: list[tuple[str, Literal["classification", "regression"]]] = [
    ("cell_type",      "classification"),
    ("event_type",     "classification"),
    ("gc_content",     "regression"),
    ("psi",            "regression"),
    ("donor_score",    "regression"),
    ("acceptor_score", "regression"),
    ("element_length", "regression"),
]

PRIMARY_METRIC = {
    "classification": "accuracy",
    "regression":     "r2",
}


# ── Probe evaluation ──────────────────────────────────────────────────────────

def _make_estimator(method: str, task: str):
    if task == "classification":
        if method == "linear":
            return LogisticRegression(max_iter=1000, class_weight="balanced",
                                      solver="lbfgs")
        return KNeighborsClassifier(n_neighbors=5)
    else:
        if method == "linear":
            return Ridge()
        return KNeighborsRegressor(n_neighbors=5)


def fit_eval(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    task: str,
    cv_folds: int,
) -> dict[str, float]:
    """
    Cross-validate a probe and return a dict of metric → value.
    Uses cross_val_predict so we can compute all metrics in one pass.
    Returns NaN dict on failure.
    """
    nan_result = {"accuracy": np.nan, "auc": np.nan, "r2": np.nan, "pearson_r": np.nan}

    if len(X) < cv_folds * 2:
        logging.warning(f"Too few samples ({len(X)}) for {cv_folds}-fold CV — skipping.")
        return nan_result

    estimator = _make_estimator(method, task)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if task == "classification":
                le     = LabelEncoder()
                y_enc  = le.fit_transform(y)
                if len(le.classes_) < 2:
                    logging.info(
                        f"Skipping — only 1 class present ({le.classes_[0]!r}); "
                        "not meaningful to probe."
                    )
                    return nan_result
                cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                y_pred = cross_val_predict(estimator, X, y_enc, cv=cv, method="predict")

                acc = accuracy_score(y_enc, y_pred)

                if hasattr(estimator, "predict_proba"):
                    y_prob = cross_val_predict(estimator, X, y_enc, cv=cv, method="predict_proba")
                    n_classes = len(le.classes_)
                    if n_classes == 2:
                        auc = roc_auc_score(y_enc, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(y_enc, y_prob, multi_class="ovr", average="macro")
                else:
                    auc = np.nan

                return {"accuracy": acc, "auc": auc, "r2": np.nan, "pearson_r": np.nan}

            else:  # regression
                cv     = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                y_pred = cross_val_predict(estimator, X, y, cv=cv)

                r2 = r2_score(y, y_pred)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pr, _ = pearsonr(y, y_pred)
                if not np.isfinite(pr):
                    pr = np.nan

                return {"accuracy": np.nan, "auc": np.nan, "r2": r2, "pearson_r": pr}

    except Exception as e:
        logging.warning(f"Probe failed: {e}")
        return nan_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Linear/KNN probe analysis on saved Borzoi embeddings"
    )
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to embeddings.npz (output of 03_extract_embeddings.py)",
    )
    parser.add_argument(
        "--in", dest="input", type=str, default=None,
        help="Deduplicated label table. Default: input_dedup.csv next to embeddings.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory. Default: directory of --embeddings.",
    )
    parser.add_argument(
        "--method", type=str, default="linear", choices=["linear", "knn"],
        help="Probe method: linear (default) or knn",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        metavar="ATTR",
        help="Attributes to probe (default: all). "
             "Choices: cell_type event_type gc_content psi donor_score acceptor_score element_length",
    )
    parser.add_argument(
        "--layers", nargs="+", default=None,
        metavar="LAYER",
        help="Layers to probe (default: all present in NPZ). "
             "Choices: conv_tower transformer unet_dec_0 unet_dec_1 final_conv head_shared",
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

    logging.info(f"Loading label table from {args.input}")
    df = pd.read_csv(args.input).reset_index(drop=True)
    logging.info(f"Events: {len(df)}")

    # Determine layers and targets
    layers_in_npz = [l for l in LAYER_ORDER if l in data]
    if args.layers is not None:
        layers = [l for l in args.layers if l in layers_in_npz]
        unknown = [l for l in args.layers if l not in layers_in_npz]
        if unknown:
            logging.warning(f"Layers not found in NPZ, skipping: {unknown}")
    else:
        layers = layers_in_npz
    logging.info(f"Layers to probe: {layers}")

    target_info = ALL_TARGETS
    if args.targets is not None:
        target_names = set(args.targets)
        target_info = [(a, t) for a, t in ALL_TARGETS if a in target_names]
        unknown = target_names - {a for a, _ in target_info}
        if unknown:
            logging.warning(f"Unknown targets, skipping: {unknown}")
    logging.info(f"Targets: {[a for a, _ in target_info]}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Run probes
    records = []

    for layer in layers:
        mat = data[layer]

        if len(mat) != len(df):
            logging.error(
                f"Row count mismatch for '{layer}': "
                f"embeddings={len(mat)}, table={len(df)}. Skipping layer."
            )
            continue

        # Drop rows with NaN embeddings (failed inference)
        valid_emb = ~np.isnan(mat).any(axis=1)
        X_all  = StandardScaler().fit_transform(mat[valid_emb])
        df_sub = df[valid_emb].reset_index(drop=True)

        logging.info(f"\n{'='*60}")
        logging.info(f"Layer: {layer}  valid rows: {valid_emb.sum()}/{len(df)}")

        for attr, task in target_info:
            if attr not in df_sub.columns:
                logging.debug(f"  {attr}: column not in table — skipping")
                continue

            y_raw    = df_sub[attr].values
            valid_y  = ~pd.isna(y_raw)
            n_valid  = valid_y.sum()

            if n_valid < args.cv_folds * 2:
                logging.info(f"  {attr}: only {n_valid} non-NaN samples — skipping")
                continue

            X = X_all[valid_y]
            y = y_raw[valid_y]
            if task == "regression":
                y = y.astype(float)

            metrics = fit_eval(X, y, method=args.method, task=task, cv_folds=args.cv_folds)

            logging.info(
                f"  {attr} ({task}): "
                + ", ".join(
                    f"{k}={v:.3f}" for k, v in metrics.items() if np.isfinite(v)
                )
            )

            for metric_name, value in metrics.items():
                records.append({
                    "layer":     layer,
                    "attribute": attr,
                    "task":      task,
                    "metric":    metric_name,
                    "value":     value,
                    "n_samples": int(n_valid),
                    "method":    args.method,
                })

    if not records:
        logging.warning("No probe results collected.")
        return

    results_df = pd.DataFrame(records)

    # Save CSV
    csv_path = os.path.join(args.out_dir, f"probe_metrics_{args.method}.csv")
    results_df.to_csv(csv_path, index=False)
    logging.info(f"\nSaved metrics → {csv_path}")

    # Save per-attribute PNGs, then assemble into PDF
    png_dir = _save_probe_pngs(results_df, layers, target_info, args.method, args.out_dir)
    logging.info(f"Saved probe PNGs → {png_dir}")
    pdf_path = os.path.join(args.out_dir, f"probe_summary_{args.method}.pdf")
    _make_probe_pdf(png_dir, args.method, target_info, pdf_path)
    logging.info(f"Saved summary → {pdf_path}")
    logging.info("\nDone.")


def _save_probe_pngs(
    results_df: pd.DataFrame,
    layers: list[str],
    target_info: list[tuple],
    method: str,
    out_dir: str,
) -> str:
    """Save one PNG per attribute to <out_dir>/probe_<method>/.

    Returns the PNG directory path.
    Skips attributes where every layer returned NaN (e.g. single-class event_type).
    Handles negative R² correctly by extending the y-axis below zero.
    """
    png_dir = os.path.join(out_dir, f"probe_{method}")
    os.makedirs(png_dir, exist_ok=True)

    layer_labels = [LAYER_LABELS.get(l, l) for l in layers]

    for attr, task in target_info:
        primary   = PRIMARY_METRIC[task]
        sec_metric = "auc" if task == "classification" else "pearson_r"

        sub = results_df[
            (results_df["attribute"] == attr) & (results_df["metric"] == primary)
        ]
        if sub.empty:
            continue

        layer_vals = []
        for l in layers:
            row = sub[sub["layer"] == l]
            layer_vals.append(float(row["value"].values[0]) if len(row) else np.nan)

        finite_vals = [v for v in layer_vals if np.isfinite(v)]
        if not finite_vals:
            logging.info(f"  {attr}: all layers NaN — skipping plot")
            continue

        sub2 = results_df[
            (results_df["attribute"] == attr) & (results_df["metric"] == sec_metric)
        ]
        sec_vals = []
        for l in layers:
            row = sub2[sub2["layer"] == l]
            sec_vals.append(float(row["value"].values[0]) if len(row) else np.nan)

        # Save plot data as CSV alongside the PNG
        n_samples_vals = []
        for l in layers:
            row = results_df[
                (results_df["attribute"] == attr) &
                (results_df["layer"] == l)
            ]
            n_samples_vals.append(int(row["n_samples"].values[0]) if len(row) else 0)

        csv_data = pd.DataFrame({
            "layer":       layers,
            "layer_label": layer_labels,
            primary:       layer_vals,
            sec_metric:    sec_vals,
            "n_samples":   n_samples_vals,
        })
        csv_path = os.path.join(png_dir, f"{attr}.csv")
        csv_data.to_csv(csv_path, index=False)
        logging.info(f"  -> {csv_path}")

        fig, ax = plt.subplots(figsize=(max(6, len(layers) * 1.4), 5))
        x = np.arange(len(layers))

        colors     = ["#4363d8" if np.isfinite(v) else "#cccccc" for v in layer_vals]
        bar_heights = [v if np.isfinite(v) else 0.0 for v in layer_vals]
        bars = ax.bar(x, bar_heights, color=colors, width=0.6, alpha=0.85)

        # Annotate with secondary metric only when primary is also valid
        for bar, pv, sv in zip(bars, layer_vals, sec_vals):
            if np.isfinite(pv) and np.isfinite(sv):
                offset = 0.02 if pv >= 0 else -0.04
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    pv + offset,
                    f"{sv:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#555555",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(primary, fontsize=10)
        ax.set_title(
            f"{attr}  |  {task}  |  method: {method}\n"
            f"primary: {primary}   (annotated: {sec_metric})",
            fontsize=11, fontweight="bold",
        )

        ymin = min(0.0, min(finite_vals) - 0.10)
        ymax = max(0.1, max(finite_vals) + 0.20)
        ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        out_path = os.path.join(png_dir, f"{attr}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        logging.info(f"  -> {out_path}")

    return png_dir


def _make_probe_pdf(
    png_dir: str,
    method: str,
    target_info: list[tuple],
    pdf_path: str,
):
    """Assemble saved probe PNGs into a single summary PDF."""
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.60, "Probe Analysis Summary",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=26, fontweight="bold")
        ax.text(0.5, 0.48, f"Method: {method}",
                ha="center", va="center", transform=ax.transAxes, fontsize=16)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        for attr, _ in target_info:
            img_path = os.path.join(png_dir, f"{attr}.png")
            if not os.path.exists(img_path):
                continue
            img = plt.imread(img_path)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.axis("off")
            plt.tight_layout(pad=0)
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
