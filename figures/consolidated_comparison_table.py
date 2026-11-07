# ===========================================================================
# Appendix Table A2 (consolidated 4-configuration comparison with bootstrap 95% CIs).
#
# Manuscript figure-generation code for "Learning and probing sequence
# representations of immune cell alternative splicing" (MLCB 2026 / PMLR).
#
# Ported from: JD_build/interpretation/umapping/09_consolidated_comparison_table.py
# Data inputs: all run-output locations resolve through figures/config.py
#   (see figures/README.md "Reproducing from the archive").
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

The main table reports each configuration's Pearson r with its own per-config
bootstrap CI (event sets differ per model, so n differs by row). The effect-size
section reports PAIRED Δr: each pair of configurations is compared on the events
they share (joined on event_key), resampling the same events for both models.
This is the analysis cited in the Appendix Table A2 note (LoCon − Pretrained,
Frozen-trunk − Pretrained, LoCon − Frozen-trunk, per cell type).

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
        df = pd.read_csv(path, sep="\t")
        # Build the same event_key as fmc._load_tsv so the pretrained predictions
        # can be joined to the adapted configurations for the paired effect sizes.
        # NOTE: we do NOT deduplicate here -- the main-table pretrained row is
        # reported over all rMATS-supported chr11 events (n~3200), matching the
        # published Table A2. Deduplication for the paired join happens in
        # bootstrap_diff_paired(), where a unique index is required.
        df["event_key"] = (df["Chromosome"].astype(str) + ":" +
                           df["Strand"].astype(str) + ":" +
                           df["Start"].astype(str) + ":" +
                           df["End"].astype(str) + ":" +
                           df["event_type"].astype(str))
        return df[["event_key", "event_type", "psi", "predicted_psi"]].copy()
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

    # ── Effect sizes: PAIRED differences in Pearson r on COMMON events ─────
    # Each pair of configurations is scored on the events they SHARE (joined on
    # event_key), and the SAME resampled events are used for both models in every
    # bootstrap replicate. This removes the event-set mismatch between pretrained
    # Borzoi (scored on all rMATS-supported chr11 events, n~3200) and the adapted
    # configurations (scored on their held-out prediction files, n~1000), and is
    # the analysis reported in the Appendix Table A2 note.
    #
    # REPRODUCIBILITY: the paired effect-size CIs printed in the manuscript were
    # generated with the legacy global NumPy RNG (np.random.seed + randint),
    # seeded once and drawn in cell-type x comparison order, with r via
    # np.corrcoef. We reuse that exact generator here so the published intervals
    # reproduce bit-for-bit. (The per-configuration CIs in the main table above
    # use np.random.default_rng via `rng`; the two halves of Table A2 were
    # generated by different RNGs and both are reproduced faithfully.)
    np.random.seed(args.seed)

    def bootstrap_diff_paired(key_a: str, key_b: str, ct: str):
        """Paired Δr = r(key_a) − r(key_b) on their common events, 95% CI.

        The join needs a unique event_key index, so we deduplicate (keep="first")
        and then drop rows with missing psi/predicted_psi -- the dedupe-then-dropna
        order used to generate the manuscript numbers. (fmc._load_tsv already
        deduplicates the adapted configs, so this only affects pretrained.)"""
        def prep(key):
            return (load_preds(key, ct)
                    .drop_duplicates("event_key", keep="first")
                    .dropna(subset=["psi", "predicted_psi"])
                    .set_index("event_key"))
        a, b = prep(key_a), prep(key_b)
        common = a.index.intersection(b.index)
        obs = a.loc[common, "psi"].values               # observed PSI (shared labels)
        pa = a.loc[common, "predicted_psi"].values
        pb = b.loc[common, "predicted_psi"].values
        n = len(common)
        r_a = np.corrcoef(obs, pa)[0, 1]
        r_b = np.corrcoef(obs, pb)[0, 1]
        boots = np.empty(args.n_boot)
        for bi in range(args.n_boot):
            idx = np.random.randint(0, n, n)             # SAME events for both models
            oo = obs[idx]
            boots[bi] = np.corrcoef(oo, pa[idx])[0, 1] - np.corrcoef(oo, pb[idx])[0, 1]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return r_a - r_b, lo, hi, r_a, r_b, n

    diff_rows = []
    for ct in fmc.CELL_TYPES:
        for comp_label, key_a, key_b in [
            ("LoCon − Pretrained", "locon", "pretrained"),
            ("Frozen-trunk − Pretrained", "frozen_trunk", "pretrained"),
            ("LoCon − Frozen-trunk", "locon", "frozen_trunk"),
        ]:
            d, lo, hi, r_a, r_b, n = bootstrap_diff_paired(key_a, key_b, ct)
            logging.info(f"{fmc.CELL_LABELS[ct]:8s} {comp_label:26s} "
                         f"Δr={d:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                         f"(r_a={r_a:.3f} r_b={r_b:.3f}, common n={n})")
            diff_rows.append({"cell_type": fmc.CELL_LABELS[ct], "comparison": comp_label,
                              "diff": d, "ci_lo": lo, "ci_hi": hi, "n_common": n})
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
        f.write("\n## Effect sizes (paired Δ Pearson r on common events, 95% CI)\n\n")
        f.write("Paired bootstrap: each pair is scored on the events it shares (joined on ")
        f.write("event_key), resampling the same events for both models.\n\n")
        f.write("| Cell type | Comparison | Paired Δr | 95% CI | n common |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in diff_res.iterrows():
            f.write(f"| {r['cell_type']} | {r['comparison']} | {r['diff']:+.3f} | "
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['n_common']} |\n")

    logging.info(f"\nSaved -> {args.out}")
    print(res.to_string(index=False))
    print()
    print(diff_res.to_string(index=False))


if __name__ == "__main__":
    main()
