from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "results/question2/ten_heat_reconstruction_delay"
    / "all_window_delay_estimates.csv"
)

OUTPUT = (
    ROOT / "results/question2/within_between_robustness"
    / f"run_{datetime.now():%Y%m%d_%H%M%S}"
)
OUTPUT.mkdir(parents=True, exist_ok=False)

df = pd.read_csv(INPUT)

for c in ["mean_lance_flow", "delay_s",
          "correlation", "peak_contrast"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["reliable"] = df["reliable"].astype(str).str.lower().eq("true")
df["boundary_peak"] = df["boundary_peak"].astype(str).str.lower().eq("true")

df["primary"] = (
    df["reliable"]
    & (df["correlation"] >= 0.50)
    & (df["peak_contrast"] >= 0.15)
    & (~df["boundary_peak"])
)

df["strict"] = (
    df["reliable"]
    & (df["correlation"] >= 0.60)
    & (df["peak_contrast"] >= 0.20)
    & (~df["boundary_peak"])
)

rules = ["reliable", "primary", "strict"]

scenarios = [
    "all",
    "exclude_6C02057",
    "minimum_3_windows",
    "exclude_6C02057_minimum_3",
]


def analyze(d):
    if len(d) < 3:
        return {}

    x = d["mean_lance_flow"]
    y = d["delay_s"]

    d = d.copy()

    d["dx"] = x - d.groupby("heat")["mean_lance_flow"].transform("mean")
    d["dy"] = y - d.groupby("heat")["delay_s"].transform("mean")

    dx, dy = d["dx"], d["dy"]

    denom = np.sum(dx ** 2)

    within_slope = (
        np.sum(dx * dy) / denom
        if denom > 1e-12 else np.nan
    )

    within_corr = dx.corr(dy)

    means = d.groupby("heat").agg(
        flow=("mean_lance_flow", "mean"),
        delay=("delay_s", "mean")
    )

    return {
        "windows": len(d),
        "heats": d["heat"].nunique(),
        "within_corr": within_corr,
        "within_slope": within_slope,
        "between_pearson": means["flow"].corr(means["delay"]),
        "between_spearman": means["flow"].corr(
            means["delay"], method="spearman"
        ),
    }


results = []
loo_results = []

for signal in ["co", "co2"]:
    for rule in rules:

        base = df[
            (df["signal"] == signal) & df[rule]
        ].dropna(
            subset=["mean_lance_flow", "delay_s"]
        )

        for scenario in scenarios:

            d = base.copy()

            if scenario in [
                "exclude_6C02057",
                "exclude_6C02057_minimum_3"
            ]:
                d = d[d["heat"] != "6C02057"]

            if scenario in [
                "minimum_3_windows",
                "exclude_6C02057_minimum_3"
            ]:
                counts = d.groupby("heat").size()
                valid_heats = counts[counts >= 3].index
                d = d[d["heat"].isin(valid_heats)]

            result = analyze(d)

            if result:
                results.append({
                    "signal": signal,
                    "rule": rule,
                    "scenario": scenario,
                    **result
                })

            # Leave-one-heat-out robustness
            for heat in d["heat"].unique():

                remaining = d[d["heat"] != heat]
                loo = analyze(remaining)

                if loo:
                    loo_results.append({
                        "signal": signal,
                        "rule": rule,
                        "scenario": scenario,
                        "excluded_heat": heat,
                        **loo
                    })

summary = pd.DataFrame(results)
loo = pd.DataFrame(loo_results)

summary.to_csv(
    OUTPUT / "robustness_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

loo.to_csv(
    OUTPUT / "leave_one_heat_out.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 65)
print("WITHIN / BETWEEN ROBUSTNESS AUDIT")
print("=" * 65)

print("\n1. Scenario comparison")
print(summary.round(4).to_string(index=False))

print("\n2. Leave-one-heat-out results")

for (signal, rule, scenario), g in loo.groupby(
    ["signal", "rule", "scenario"]
):
    print(
        f"\n{signal.upper()} | {rule} | {scenario}"
    )
    print(
        "Within correlation range:",
        round(g["within_corr"].min(), 4),
        "to",
        round(g["within_corr"].max(), 4)
    )
    print(
        "Negative proportion:",
        round((g["within_corr"] < 0).mean(), 3)
    )

print("\nOutput:", OUTPUT)
print("Original data preserved.")