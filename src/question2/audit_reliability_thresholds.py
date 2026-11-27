
"""
Question 2 - Reliability Threshold Audit

Compare the original reliability flag with alternative
correlation and peak-contrast screening rules.

No original data or experiment results are modified.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "results/question2/ten_heat_reconstruction_delay/"
    "all_window_delay_estimates.csv"
)

OUTPUT = (
    ROOT / "results/question2/reliability_threshold_audit"
    / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

OUTPUT.mkdir(parents=True, exist_ok=False)


def as_bool(series):
    return (
        series.astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )


def spearman(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 3:
        return np.nan
    return pair["x"].corr(pair["y"], method="spearman")


def summarize(group, mask, rule):
    d = group.loc[mask].copy()

    if d.empty:
        return {
            "rule": rule,
            "windows": 0,
            "retention_pct": 0.0,
            "extreme_windows": 0,
            "median_delay_s": np.nan,
            "mean_delay_s": np.nan,
            "flow_delay_spearman": np.nan,
        }

    extreme = (
        (d["delay_s"] < 10) | (d["delay_s"] > 55)
    )

    return {
        "rule": rule,
        "windows": len(d),
        "retention_pct": round(
            100 * len(d) / len(group), 2
        ),
        "extreme_windows": int(extreme.sum()),
        "median_delay_s": d["delay_s"].median(),
        "mean_delay_s": d["delay_s"].mean(),
        "flow_delay_spearman": spearman(
            d["mean_lance_flow"],
            d["delay_s"],
        ),
    }


print("=" * 65)
print("RELIABILITY THRESHOLD AUDIT")
print("=" * 65)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

numeric_cols = [
    "attachment",
    "window_center_s",
    "mean_lance_flow",
    "lance_flow_std",
    "delay_s",
    "correlation",
    "peak_contrast",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["original_reliable"] = as_bool(df["reliable"])
df["boundary_bool"] = as_bool(df["boundary_peak"])

df["extreme_delay"] = (
    (df["delay_s"] < 10) | (df["delay_s"] > 55)
)

# --------------------------------------------------
# 1. Define exploratory screening rules
# --------------------------------------------------

rules = {
    "original_reliable": df["original_reliable"],

    "quality_050_015": (
        (df["correlation"] >= 0.50)
        & (df["peak_contrast"] >= 0.15)
        & (~df["boundary_bool"])
    ),

    "quality_060_020": (
        (df["correlation"] >= 0.60)
        & (df["peak_contrast"] >= 0.20)
        & (~df["boundary_bool"])
    ),

    "quality_070_030": (
        (df["correlation"] >= 0.70)
        & (df["peak_contrast"] >= 0.30)
        & (~df["boundary_bool"])
    ),

    "original_plus_050_015": (
        df["original_reliable"]
        & (df["correlation"] >= 0.50)
        & (df["peak_contrast"] >= 0.15)
        & (~df["boundary_bool"])
    ),

    "original_plus_060_020": (
        df["original_reliable"]
        & (df["correlation"] >= 0.60)
        & (df["peak_contrast"] >= 0.20)
        & (~df["boundary_bool"])
    ),
}

# --------------------------------------------------
# 2. Overall and signal-level comparisons
# --------------------------------------------------

overall = []
by_signal = []

for name, mask in rules.items():

    overall.append(
        summarize(df, mask, name)
    )

    for signal, group in df.groupby("signal"):

        by_signal.append({
            "signal": signal,
            **summarize(
                group,
                mask.loc[group.index],
                name,
            ),
        })

overall_df = pd.DataFrame(overall)
signal_df = pd.DataFrame(by_signal)

overall_df.to_csv(
    OUTPUT / "threshold_comparison_overall.csv",
    index=False,
    encoding="utf-8-sig",
)

signal_df.to_csv(
    OUTPUT / "threshold_comparison_by_signal.csv",
    index=False,
    encoding="utf-8-sig",
)

# --------------------------------------------------
# 3. Heat-level comparisons
# --------------------------------------------------

heat_results = []

for (signal, heat), group in df.groupby(
    ["signal", "heat"]
):

    for name, mask in rules.items():

        result = summarize(
            group,
            mask.loc[group.index],
            name,
        )

        heat_results.append({
            "signal": signal,
            "heat": heat,
            **result,
        })

heat_df = pd.DataFrame(heat_results)

heat_df.to_csv(
    OUTPUT / "threshold_comparison_by_heat.csv",
    index=False,
    encoding="utf-8-sig",
)

# --------------------------------------------------
# 4. Mark individual windows
# --------------------------------------------------

flags = df.copy()

for name, mask in rules.items():
    flags[name] = mask

flags["delay_jump_from_previous_s"] = (
    flags.sort_values(
        ["signal", "heat", "window_center_s"]
    )
    .groupby(["signal", "heat"])["delay_s"]
    .diff()
    .abs()
)

flags.to_csv(
    OUTPUT / "all_window_quality_flags.csv",
    index=False,
    encoding="utf-8-sig",
)

# --------------------------------------------------
# 5. Extreme window retention
# --------------------------------------------------

extreme = flags[flags["extreme_delay"]].copy()

extreme_results = []

for name in rules:
    extreme_results.append({
        "rule": name,
        "total_extreme_windows": len(extreme),
        "retained_extreme_windows": int(
            extreme[name].sum()
        ),
        "rejected_extreme_windows": int(
            (~extreme[name]).sum()
        ),
    })

extreme_df = pd.DataFrame(extreme_results)

extreme_df.to_csv(
    OUTPUT / "extreme_window_retention.csv",
    index=False,
    encoding="utf-8-sig",
)

# --------------------------------------------------
# 6. Print results
# --------------------------------------------------

print("\n1. Overall threshold comparison")
print(overall_df.to_string(index=False))

print("\n2. Comparison by gas")
print(signal_df.to_string(index=False))

print("\n3. Extreme-window retention")
print(extreme_df.to_string(index=False))

print("\n4. Heat-level comparison")
print(heat_df.to_string(index=False))

print("\n" + "=" * 65)
print("AUDIT COMPLETED")
print("=" * 65)

print(f"\nOutput directory:\n{OUTPUT}")

print("\nGenerated files:")
for path in sorted(OUTPUT.glob("*.csv")):
    print(" ", path.name)

print("\nOriginal experiment files were not modified.")
print("All screening thresholds are exploratory, not final.")
