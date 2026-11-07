# ===========================================================================
# Event-level PSI distribution (companion to the bin-level distribution in
# Fig 1C) for "Learning and probing sequence representations of immune cell
# alternative splicing" (MLCB 2026 / PMLR).
#
# Fig 1C and the per-bin scarcity statistics in Results/Discussion are computed
# over 32 bp *output bins* (the resolution the model is trained at), where the
# PSI label distribution is highly imbalanced (~90-91% < 0.1, ~7-8% > 0.9,
# ~1-2% intermediate). This script reports the complementary *event-level*
# distribution: the fraction of individual alternative-splicing events whose
# aggregated PSI falls in the skipped (<0.1), intermediate (0.1-0.9), and
# included (>0.9) ranges. The event-level intermediate fraction (~18% under the
# paper's >=20-read filter) is roughly an order of magnitude larger than the
# bin-level one, which is why the manuscript frames the scarcity argument at the
# training-bin level rather than the event level.
#
# Data inputs: solo per-cell-type rMATS runs (one row per genome-wide AS event)
# under config.RMATS_RESULTS / {B_only, Macrophage_only, T_only}. Each event's
# PSI is the mean of its per-replicate IncLevel1 values; the >=20-read filter
# matches the aggregate junction-read threshold used for event definition in the
# Methods (IJC_SAMPLE_1 + SJC_SAMPLE_1 summed across replicates).
# ===========================================================================
"""
event_level_psi_fraction.py

Usage:
  python event_level_psi_fraction.py
  python event_level_psi_fraction.py --min-reads 20 --out out.md
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Solo per-cell-type rMATS result subdirectories under config.RMATS_RESULTS.
SOLO_DIRS = {"B Cell": "B_only", "Macrophage": "Macrophage_only", "T Cell": "T_only"}
# Event classes used throughout the paper (MXE is excluded, as in the main text).
CLASSES = ["SE", "RI", "A3SS", "A5SS"]
LO, HI = 0.1, 0.9  # intermediate = [LO, HI]


def _repsum(field) -> float:
    """Sum comma-separated per-replicate counts, ignoring NA/blank entries."""
    return sum(float(x) for x in str(field).split(",") if x not in ("NA", "nan", ""))


def event_psis(solo_dir: Path, min_reads: int) -> np.ndarray:
    """Mean-across-replicate PSI for every event passing the read filter."""
    vals = []
    for cls in CLASSES:
        path = solo_dir / f"{cls}.MATS.JC.txt"
        df = pd.read_csv(path, sep="\t",
                         usecols=["IncLevel1", "IJC_SAMPLE_1", "SJC_SAMPLE_1"])
        for inc, ijc, sjc in zip(df["IncLevel1"], df["IJC_SAMPLE_1"], df["SJC_SAMPLE_1"]):
            reps = [float(x) for x in str(inc).split(",") if x not in ("NA", "nan", "")]
            if not reps:
                continue
            if _repsum(ijc) + _repsum(sjc) >= min_reads:
                vals.append(np.mean(reps))
    return np.array(vals)


def fractions(v: np.ndarray) -> tuple:
    """(% < LO, % intermediate, % > HI) for a PSI array."""
    lo = (v < LO).mean() * 100
    mid = ((v >= LO) & (v <= HI)).mean() * 100
    hi = (v > HI).mean() * 100
    return lo, mid, hi


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-reads", type=int, default=20,
                    help="Aggregate junction-read threshold (paper filter = 20). "
                         "The report also always includes the unfiltered row.")
    ap.add_argument("--out", type=Path, default=config.OUT_DIR / "event_level_psi_fraction.md")
    args = ap.parse_args()

    results_root = Path(config.RMATS_RESULTS)
    rows = []
    for thr in sorted({0, args.min_reads}):
        tag = "unfiltered (>=1 valid replicate)" if thr == 0 \
            else f">={thr} aggregate junction reads (paper filter)"
        logging.info(f"--- event-level PSI, {tag} ---")
        pooled = []
        for ct, sub in SOLO_DIRS.items():
            v = event_psis(results_root / sub, thr)
            pooled.append(v)
            lo, mid, hi = fractions(v)
            logging.info(f"  {ct:12s} n={len(v):6d}  <0.1={lo:5.1f}%  "
                         f"0.1-0.9={mid:5.1f}%  >0.9={hi:5.1f}%")
            rows.append({"filter": tag, "cell_type": ct, "n": len(v),
                         "pct_lo": lo, "pct_mid": mid, "pct_hi": hi})
        allv = np.concatenate(pooled)
        lo, mid, hi = fractions(allv)
        logging.info(f"  {'POOLED':12s} n={len(allv):6d}  <0.1={lo:5.1f}%  "
                     f"0.1-0.9={mid:5.1f}%  >0.9={hi:5.1f}%")
        rows.append({"filter": tag, "cell_type": "POOLED", "n": len(allv),
                     "pct_lo": lo, "pct_mid": mid, "pct_hi": hi})

    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Event-level PSI distribution (companion to bin-level Fig 1C)\n\n")
        f.write("Fraction of alternative-splicing events by aggregated PSI range, from solo "
                "per-cell-type rMATS runs (SE+RI+A3SS+A5SS). Compare with the bin-level "
                "distribution in Fig 1C (~90-91% <0.1, ~7-8% >0.9, ~1-2% intermediate).\n\n")
        for tag in res["filter"].unique():
            sub = res[res["filter"] == tag]
            f.write(f"## {tag}\n\n")
            f.write("| Cell type | n events | PSI<0.1 | PSI 0.1-0.9 | PSI>0.9 |\n")
            f.write("|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                f.write(f"| {r['cell_type']} | {r['n']} | {r['pct_lo']:.1f}% | "
                        f"{r['pct_mid']:.1f}% | {r['pct_hi']:.1f}% |\n")
            f.write("\n")

    logging.info(f"Saved -> {args.out}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
