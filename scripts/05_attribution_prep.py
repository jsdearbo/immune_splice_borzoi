"""
Prepare an attribution-ready CSV from input_dedup.csv.

Filters to a specific cell type and event type, renames columns to match the
attribution pipeline's expected format, synthesizes a unique_ID field,
and adds a psi_bin column for use as the groupby_column in 02_run_modisco.py
and 03_run_enrichment.py.

Two binning modes:
  Quantile (default): Balanced tertile bins — works well for RI events, which
      have a spread PSI distribution. Bin labels: low_psi, intermediate_psi,
      high_psi.
  Fixed thresholds (--psi-thresholds L H): Bins events as PSI < L (low_psi),
      L ≤ PSI ≤ H (intermediate_psi), PSI > H (high_psi). Recommended for SE
      events whose PSI is heavily skewed toward 1.0, e.g.:
          --psi-thresholds 0.5 0.95

Usage:
    python 00_prepare_events.py \
        --input-csv  /path/to/{SE,RI}_results/input_dedup.csv \
        --cell-type  B_cell \
        --event-type SE \
        --out-csv    /path/to/run_dir/events.csv \
        [--psi-thresholds 0.5 0.95]

Output columns include all original columns plus:
    chrom, start, end, strand  (renamed from Chromosome, Start, End, Strand)
    unique_ID                  (synthesized identifier)
    psi_bin                    (low_psi | intermediate_psi | high_psi)
"""

import argparse
import sys
import pandas as pd
import numpy as np


BIN_LABELS = {
    2: ["low_psi", "high_psi"],
    3: ["low_psi", "intermediate_psi", "high_psi"],
    4: ["low_psi", "low_mid_psi", "high_mid_psi", "high_psi"],
}

COLUMN_RENAMES = {
    "Chromosome": "chrom",
    "Start":      "start",
    "End":        "end",
    "Strand":     "strand",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare attribution-ready CSV from input_dedup.csv"
    )
    p.add_argument("--input-csv",  required=True, help="Path to input_dedup.csv")
    p.add_argument("--cell-type",  required=True, help="Cell type to filter (e.g. B_cell)")
    p.add_argument("--event-type", required=True, help="Event type to filter (e.g. SE, RI)")
    p.add_argument("--out-csv",    required=True, help="Output CSV path")
    p.add_argument("--psi-bins",   type=int, default=3,
                   help="Number of quantile bins (default: 3 → low/intermediate/high)")
    p.add_argument("--psi-thresholds", type=float, nargs=2, metavar=("LOW", "HIGH"),
                   help="Fixed PSI thresholds: PSI<LOW → low_psi, LOW≤PSI≤HIGH → intermediate_psi, "
                        "PSI>HIGH → high_psi.  Overrides --psi-bins. "
                        "Recommended for SE events (e.g. 0.5 0.95).")
    p.add_argument("--events-per-bin", type=int, default=100,
                   help="Max events to keep per PSI bin after specificity ranking "
                        "(default: 100; 0 = no limit). Events exclusive to the "
                        "target cell type are selected first.")
    p.add_argument("--psi-bin-filter", default=None, metavar="BIN",
                   help="If set, keep only events from this PSI bin after binning "
                        "and sampling (e.g. high_psi, low_psi, intermediate_psi). "
                        "For use in per-bin pipeline runs.")
    p.add_argument("--exclusive-only", action="store_true",
                   help="Hard-filter to events exclusive to the target cell type "
                        "(n_shared == 1 in shared_cell_types column). Applied before "
                        "PSI binning and sampling.")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)

    # Filter — case-insensitive to handle B_Cell vs B_cell etc.
    mask = (
        df["cell_type"].str.lower() == args.cell_type.lower()
    ) & (
        df["event_type"].str.upper() == args.event_type.upper()
    )
    if not mask.any():
        print(f"ERROR: no rows for cell_type={args.cell_type}, event_type={args.event_type}",
              file=sys.stderr)
        print(f"  Available cell_types : {sorted(df['cell_type'].unique().tolist())}",
              file=sys.stderr)
        print(f"  Available event_types: {sorted(df['event_type'].unique().tolist())}",
              file=sys.stderr)
        sys.exit(1)
    df = df[mask].copy()

    print(f"  Filtered to {len(df)} events "
          f"(cell_type={args.cell_type}, event_type={args.event_type})")

    # Rename genomic coordinate columns
    df = df.rename(columns=COLUMN_RENAMES)

    # Compute n_shared early so it's available for both exclusive filter and sampling
    if "shared_cell_types" in df.columns:
        df["_n_shared"] = df["shared_cell_types"].str.split("|").str.len()

    # Hard-filter to exclusive events if requested
    if args.exclusive_only:
        if "_n_shared" not in df.columns:
            print("ERROR: --exclusive-only requires shared_cell_types column in input CSV",
                  file=sys.stderr)
            sys.exit(1)
        before = len(df)
        df = df[df["_n_shared"] == 1].copy()
        print(f"  Exclusive-only filter: {before} → {len(df)} events (n_shared == 1)")
        if df.empty:
            print(f"ERROR: No exclusive events for cell_type={args.cell_type}, "
                  f"event_type={args.event_type}", file=sys.stderr)
            sys.exit(1)

    # Synthesize unique_ID
    df["unique_ID"] = (
        df["cell_type"] + "__" +
        df["event_type"] + "__" +
        df["gene_symbol"] + "__" +
        df["chrom"].astype(str) + "_" +
        df["start"].astype(str) + "_" +
        df["end"].astype(str)
    )

    # PSI binning
    labels = ["low_psi", "intermediate_psi", "high_psi"]

    if args.psi_thresholds is not None:
        # Fixed-threshold binning — recommended for SE events
        lo, hi = args.psi_thresholds
        if not (0.0 <= lo < hi <= 1.0):
            print(f"ERROR: --psi-thresholds must satisfy 0 <= LOW < HIGH <= 1; got {lo} {hi}",
                  file=sys.stderr)
            sys.exit(1)
        conditions = [df["psi"] < lo, df["psi"] <= hi]
        choices    = ["low_psi", "intermediate_psi"]
        df["psi_bin"] = np.select(conditions, choices, default="high_psi")
        print(f"  PSI binning: fixed thresholds LOW={lo}, HIGH={hi}")
    else:
        # Quantile-based binning — works well for RI events with spread PSI
        if args.psi_bins not in BIN_LABELS:
            print(f"ERROR: --psi-bins must be one of {list(BIN_LABELS)}; got {args.psi_bins}",
                  file=sys.stderr)
            sys.exit(1)
        # Use labels=False then remap: avoids ValueError when duplicates="drop"
        # collapses bin edges and produces fewer bins than requested.
        bin_ints = pd.qcut(df["psi"], q=args.psi_bins, labels=False, duplicates="drop")
        n_actual = int(bin_ints.max()) + 1
        if n_actual < args.psi_bins:
            print(f"  Warning: PSI distribution collapsed to {n_actual} quantile bins "
                  f"(requested {args.psi_bins}); using {n_actual}-bin labels.")
        labels = BIN_LABELS.get(n_actual, [f"bin_{i}" for i in range(n_actual)])
        df["psi_bin"] = bin_ints.map({i: labels[i] for i in range(n_actual)})
        print(f"  PSI binning: quantile ({n_actual} bins)")

    # Report bin sizes and warn if any bin has a very narrow PSI range
    counts = df["psi_bin"].value_counts()
    print("  PSI bin counts:")
    for label in labels:
        n = counts.get(label, 0)
        if n == 0:
            print(f"    {label}: n=0  (no events in this bin)")
            continue
        psi_vals = df.loc[df["psi_bin"] == label, "psi"]
        spread = psi_vals.max() - psi_vals.min()
        flag = "  ← narrow range, bins may not be biologically distinct" if spread < 0.01 else ""
        print(f"    {label}: n={n}  PSI [{psi_vals.min():.3f}, {psi_vals.max():.3f}]{flag}")

    # ── Specificity-ranked sampling per bin ─────────────────────────────────
    # Sort: exclusive events (n_shared=1) first, then by |PSI - bin median|
    # descending so the most extreme values within each tier are preferred.
    if args.events_per_bin > 0:
        if "shared_cell_types" in df.columns:
            bin_medians = df.groupby("psi_bin")["psi"].transform("median")
            if "_n_shared" not in df.columns:  # may already be set by exclusive filter
                df["_n_shared"] = df["shared_cell_types"].str.split("|").str.len()
            df["_psi_dev"] = (df["psi"] - bin_medians).abs()
            df = df.sort_values(
                ["psi_bin", "_n_shared", "_psi_dev"],
                ascending=[True, True, False],
            )
            df = df.drop(columns=["_n_shared", "_psi_dev"])

        # Explicit per-bin head() — avoids pandas groupby/apply column-drop issues
        parts = []
        for label in labels:
            sub = df[df["psi_bin"] == label]
            parts.append(sub.head(args.events_per_bin))
        df = pd.concat(parts, ignore_index=True)

        print(f"  After specificity sampling (max {args.events_per_bin}/bin):")
        for label in labels:
            sub = df[df["psi_bin"] == label]
            if len(sub) == 0:
                print(f"    {label}: n=0")
                continue
            if "shared_cell_types" in sub.columns:
                n_excl = (sub["shared_cell_types"].str.split("|").str.len() == 1).sum()
                excl_str = f"  exclusive={n_excl}"
            else:
                excl_str = ""
            print(f"    {label}: n={len(sub)}{excl_str}  "
                  f"PSI [{sub['psi'].min():.3f}, {sub['psi'].max():.3f}]")

    # Verify unique_ID is actually unique (warn if not — can happen with duplicate coordinates)
    n_dup = df["unique_ID"].duplicated().sum()
    if n_dup > 0:
        print(f"  WARNING: {n_dup} duplicate unique_IDs detected; appending row index to deduplicate")
        dup_mask = df["unique_ID"].duplicated(keep=False)
        df.loc[dup_mask, "unique_ID"] = (
            df.loc[dup_mask, "unique_ID"] + "__" +
            df.loc[dup_mask].reset_index().index.astype(str)
        )

    # Optional: restrict output to a single PSI bin (for per-bin pipeline runs)
    if args.psi_bin_filter is not None:
        before = len(df)
        df = df[df["psi_bin"] == args.psi_bin_filter].copy()
        print(f"  PSI bin filter '{args.psi_bin_filter}': {before} → {len(df)} events")
        if df.empty:
            print(f"ERROR: No events remain after filtering to psi_bin='{args.psi_bin_filter}'",
                  file=sys.stderr)
            sys.exit(1)

    # Drop internal helper columns before writing
    df = df.drop(columns=[c for c in ["_n_shared", "_psi_dev"] if c in df.columns])

    df.to_csv(args.out_csv, index=False)
    print(f"  Wrote {len(df)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
