
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results/question2/ten_heat_reconstruction_delay/"
    / "all_window_delay_estimates.csv"
)

OUTPUT = (
    ROOT / "results/question2/within_between_heat_audit"
    / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

OUTPUT.mkdir(parents=True, exist_ok=False)

df = pd.read_csv(INPUT)

for col in [
    "mean_lance_flow",
    "delay_s",
    "correlation",
    "peak_contrast",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["reliable"] = (
    df["reliable"].astype(str).str.lower() == "true"
)

df["boundary_peak"] = (
    df["boundary_peak"].astype(str).str.lower() == "true"
)

# Primary and sensitivity screening rules
df["primary"] = (
    df["reliable"]
    & (df["correlation"] >= 0.50)
    & (df["peak_contrast"] >= 0.15)
    & (~df["boundary_peak"])
)

df["sensitivity"] = (
    df["reliable"]
    & (df["correlation"] >= 0.60)
    & (df["peak_contrast"] >= 0.20)
    & (~df["boundary_peak"])
)

rules = {
    "original": "reliable",
    "primary_050_015": "primary",
    "sensitivity_060_020": "sensitivity",
}


def calculate(group, rule, signal):

    d = group.loc[group[rule]].copy()

    d = d.dropna(
        subset=["mean_lance_flow", "delay_s", "heat"]
    )

    if len(d) < 3:
        return None

    x = d["mean_lance_flow"]
    y = d["delay_s"]

    # Overall relationship
    raw_pearson = x.corr(y)
    raw_spearman = x.corr(y, method="spearman")

    # Within-heat centering
    d["flow_deviation"] = (
        d["mean_lance_flow"]
        - d.groupby("heat")["mean_lance_flow"].transform("mean")
    )

    d["delay_deviation"] = (
        d["delay_s"]
        - d.groupby("heat")["delay_s"].transform("mean")
    )

    dx = d["flow_deviation"]
    dy = d["delay_deviation"]

    denominator = np.sum(dx ** 2)

    within_slope = (
        np.sum(dx * dy) / denominator
        if denominator > 1e-12
        else np.nan
    )

    within_corr = dx.corr(dy)

    # Between-heat relationship
    heat_means = (
        d.groupby("heat")
        .agg(
            mean_flow=("mean_lance_flow", "mean"),
            mean_delay=("delay_s", "mean"),
            windows=("delay_s", "size"),
        )
        .reset_index()
    )

    between_pearson = (
        heat_means["mean_flow"].corr(
            heat_means["mean_delay"]
        )
    )

    between_spearman = (
        heat_means["mean_flow"].corr(
            heat_means["mean_delay"],
            method="spearman",
        )
    )

    return {
        "signal": signal,
        "rule": rule,
        "windows": len(d),
        "heats": d["heat"].nunique(),
        "raw_pearson": raw_pearson,
        "raw_spearman": raw_spearman,
        "within_heat_pearson": within_corr,
        "within_heat_slope": within_slope,
        "between_heat_pearson": between_pearson,
        "between_heat_spearman": between_spearman,
    }


# 1. Overall decomposition
overall = []

for signal, group in df.groupby("signal"):

    for rule, column in rules.items():

        result = calculate(group, column, signal)

        if result is not None:
            overall.append(result)

overall_df = pd.DataFrame(overall)

overall_df.to_csv(
    OUTPUT / "within_between_overall.csv",
    index=False,
    encoding="utf-8-sig",
)

# 2. Individual heat relationships
heat_results = []

for signal, group in df.groupby("signal"):

    for rule, column in rules.items():

        selected = group.loc[group[column]]

        for heat, h in selected.groupby("heat"):

            h = h.dropna(
                subset=["mean_lance_flow", "delay_s"]
            )

            x = h["mean_lance_flow"]
            y = h["delay_s"]

            if len(h) >= 3 and x.nunique() >= 2:

                pearson = x.corr(y)
                spearman = x.corr(y, method="spearman")

                slope = (
                    np.sum(
                        (x - x.mean()) * (y - y.mean())
                    )
                    / np.sum((x - x.mean()) ** 2)
                )

            else:
                pearson = np.nan
                spearman = np.nan
                slope = np.nan

            heat_results.append({
                "signal": signal,
                "rule": rule,
                "heat": heat,
                "windows": len(h),
                "flow_mean": x.mean(),
                "delay_mean": y.mean(),
                "delay_median": y.median(),
                "pearson": pearson,
                "spearman": spearman,
                "slope": slope,
            })

heat_df = pd.DataFrame(heat_results)

heat_df.to_csv(
    OUTPUT / "within_heat_results.csv",
    index=False,
    encoding="utf-8-sig",
)

# 3. Print results
print("=" * 65)
print("WITHIN-HEAT / BETWEEN-HEAT ANALYSIS")
print("=" * 65)

print("\n1. Overall decomposition")

print(
    overall_df.round(4).to_string(index=False)
)

print("\n2. Individual heat results")

print(
    heat_df.round(4).to_string(index=False)
)

print("\n3. Output directory:")
print(OUTPUT)

print("\nGenerated files:")
print("  within_between_overall.csv")
print("  within_heat_results.csv")

print("\nOriginal data preserved.")
