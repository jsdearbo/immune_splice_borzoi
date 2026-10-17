# ===========================================================================
# Appendix Table A2 (consolidated 4-configuration comparison with bootstrap 95% CIs).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/interpretation/umapping/09_consolidated_comparison_table.py
# Data inputs: run outputs under the paths in figures/config.py (currently
#   hardcoded in this script -- see figures/README.md 'Finishing the port').
# ===========================================================================
"""
consolidated_comparison_table.py

Manuscript Appendix Table A2: consolidated comparison of event-level PSI
prediction performance for pretrained Borzoi (zero-shot), frozen-trunk
head-only, single-cell-type LoCon, and shared-trunk multitask LoCon models,
each with a bootstrap 95% CI, per cell type, on the same held-out chr11
alternative-splicing events (SE+RI+A3SS+A5SS, i.e. the "All alt events"
condition shown in Fig 2A-C / Appendix A3).

Pretrained Borzoi's per-event predictions are produced by a separate inference
pass (eval_pretrained_borzoi.py -> preds_pretrained_{cell_type}_chr11.tsv); the
other three configurations reuse the preds_*.tsv files written during their
original evaluation runs.

Usage:
  python consolidated_comparison_table.py
  python consolidated_comparison_table.py --n-boot 5000 --out out.md
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import multitask_collapse as fmc

import config

PRETRAINED_DIR = config.PRETRAINED_EVAL_ROOT

CONFIGS = {
    "pretrained": {
        "label": "Pretrained Borzoi (zero-shot)",
        "base": PRETRAINED_DIR,
        "pattern": "pretrained",
    },
    "frozen_trunk": {
        "label": "Frozen-trunk head-only",
        "base": config.EXPERIMENTS_ROOT / "20260511_single_cell_types_frozen_trunk",
        "pattern": "single",
    },
    "locon": {
        "label": "Single-cell-type LoCon",
        "base": config.EXPERIMENTS_ROOT / "20260511_single_cell_types_locon4",
        "pattern": "single",
    },
    "multitask": {
        "label": "Shared-trunk multitask LoCon",
        "base": config.EXPERIMENTS_ROOT / "20260511_multitask_locon" / "cosine_decay_fix" / "evaluation_test" / "scatterplots",
        "pattern": "multitask",
    },
}


def load_preds(config_key: str, ct: str) -> pd.DataFrame:
    cfg = CONFIGS[config_key]
    fname = fmc._CELL_FNAME[ct]
    if cfg["pattern"] == "single":
        path = cfg["base"] / ct / "evaluation_test" / "scatterplots" / f"preds_{fname}_chr11_npz.tsv"
        return fmc._load_tsv(path)
    elif cfg["pattern"] == "pretrained":
        path = cfg["base"] / f"preds_pretrained_{fname}_chr11.tsv"
        return pd.read_csv(path, sep="\t")[["event_type", "psi", "predicted_psi"]]
    else:
        path = cfg["base"] / fmc._CELL_SUBDIR[ct] / f"preds_{fname}_chr11_npz.tsv"
        return fmc._load_tsv(path)


def bootstrap_pearson(psi: np.ndarray, pred: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = len(psi)
    point = stats.pearsonr(psi, pred)[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stats.pearsonr(psi[idx], pred[idx])[0]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi, n


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=config.OUT_DIR / "tableA2_comparison.md")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for config_key, cfg in CONFIGS.items():
        for ct in fmc.CELL_TYPES:
            df = load_preds(config_key, ct)
            valid = df.dropna(subset=["psi", "predicted_psi"])
            point, lo, hi, n = bootstrap_pearson(
                valid["psi"].values, valid["predicted_psi"].values, args.n_boot, rng
            )
            logging.info(f"{cfg['label']:30s} {ct:12s} r={point:.3f} [{lo:.3f}, {hi:.3f}] n={n}")
            rows.append({
                "configuration": cfg["label"], "cell_type": fmc.CELL_LABELS[ct],
                "pearson_r": point, "ci_lo": lo, "ci_hi": hi, "n": n,
            })

    res = pd.DataFrame(rows)

    # Effect sizes, bootstrapped jointly per pair (resampling each config's own
    # event indices independently; event sets differ per model's held-out
    # predictions, so this is a difference-of-independent-bootstrap-distributions
    # CI, which is conservative but valid).
    def bootstrap_diff(key_a: str, key_b: str, ct: str):
        """key_a - key_b, e.g. locon - frozen_trunk."""
        d_a = load_preds(key_a, ct).dropna(subset=["psi", "predicted_psi"])
        d_b = load_preds(key_b, ct).dropna(subset=["psi", "predicted_psi"])
        n_a, n_b = len(d_a), len(d_b)
        boots_a = np.empty(args.n_boot)
        boots_b = np.empty(args.n_boot)
        for b in range(args.n_boot):
            idx_a = rng.integers(0, n_a, n_a)
            idx_b = rng.integers(0, n_b, n_b)
            boots_a[b] = stats.pearsonr(d_a["psi"].values[idx_a], d_a["predicted_psi"].values[idx_a])[0]
            boots_b[b] = stats.pearsonr(d_b["psi"].values[idx_b], d_b["predicted_psi"].values[idx_b])[0]
        diff = boots_a - boots_b
        lo, hi = np.percentile(diff, [2.5, 97.5])
        return diff.mean(), lo, hi

    diff_rows = []
    for ct in fmc.CELL_TYPES:
        d, lo, hi = bootstrap_diff("locon", "frozen_trunk", ct)
        diff_rows.append({"cell_type": fmc.CELL_LABELS[ct], "comparison": "LoCon − Frozen-trunk",
                           "diff": d, "ci_lo": lo, "ci_hi": hi})
        d, lo, hi = bootstrap_diff("locon", "pretrained", ct)
        diff_rows.append({"cell_type": fmc.CELL_LABELS[ct], "comparison": "LoCon − Pretrained",
                           "diff": d, "ci_lo": lo, "ci_hi": hi})
    diff_res = pd.DataFrame(diff_rows)

    # ── Write markdown table ──────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Consolidated comparison: event-level PSI prediction performance\n\n")
        f.write("All alt events (SE+RI+A3SS+A5SS), held-out chr11, Pearson r with 95% bootstrap CI ")
        f.write(f"(n={args.n_boot} resamples).\n\n")
        f.write("| Configuration | Cell type | Pearson r | 95% CI | n events |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in res.iterrows():
            f.write(f"| {r['configuration']} | {r['cell_type']} | {r['pearson_r']:.3f} | "
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['n']} |\n")
        f.write("\n## Effect sizes (Pearson r difference, 95% CI)\n\n")
        f.write("| Cell type | Comparison | Difference | 95% CI |\n")
        f.write("|---|---|---|---|\n")
        for _, r in diff_res.iterrows():
            f.write(f"| {r['cell_type']} | {r['comparison']} | {r['diff']:+.3f} | "
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] |\n")

    logging.info(f"\nSaved -> {args.out}")
    print(res.to_string(index=False))
    print()
    print(diff_res.to_string(index=False))


if __name__ == "__main__":
    main()
