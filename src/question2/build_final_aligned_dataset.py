
from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "results" / "question2" /
    "dynamic_delay_validation_corrected" /
    "run_20261127_152936"
)

OUTPUT_BASE = (
    ROOT / "results" / "question2" /
    "final_aligned_dataset"
)

OUTPUT_DIR = OUTPUT_BASE / (
    "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


# ============================================================
# 2. Configuration
# ============================================================

CO_FIXED_DELAY = 32.0
CO2_FIXED_DELAY = 34.0

REQUIRED_COLUMNS = [
    "time_s",
    "oxygen_lance_flow_m3_s",
    "oxygen_cumulative_m3",
    "co_dynamic_valid",
    "co_flow_fixed_aligned_m3_s",
    "co_flow_dynamic_aligned_m3_s",
    "co2_flow_fixed_aligned_m3_s",
    "co_cumulative_fixed_aligned_m3",
    "co_cumulative_dynamic_aligned_m3",
    "co2_cumulative_fixed_aligned_m3",
]

NUMERIC_COLUMNS = [
    "time_s",
    "oxygen_lance_flow_m3_s",
    "oxygen_cumulative_m3",
    "co_dynamic_delay_s",
    "co_delay_at_interval_midpoint_s",
    "co2_delay_s",
    "co2_delay_at_interval_midpoint_s",
    "co_flow_fixed_aligned_m3_s",
    "co_flow_dynamic_aligned_m3_s",
    "co2_flow_fixed_aligned_m3_s",
    "co_cumulative_fixed_aligned_m3",
    "co_cumulative_dynamic_aligned_m3",
    "co2_cumulative_fixed_aligned_m3",
]


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def last_valid(series):
    values = series.dropna()
    return float(values.iloc[-1]) if len(values) else np.nan


def correlation(x, y, method="pearson"):
    temp = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(temp) < 3:
        return np.nan

    if temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return np.nan

    return temp["x"].corr(temp["y"], method=method)


# ============================================================
# 3. Load and align all ten heats
# ============================================================

files = sorted(
    INPUT_DIR.glob("heat_*_corrected_aligned.csv")
)

if len(files) != 10:
    raise RuntimeError(
        f"Expected 10 corrected files, found {len(files)}"
    )

all_heats = []
summary_rows = []
quality_rows = []

print("=" * 65)
print("FINAL TEN-HEAT DATASET CONSTRUCTION")
print("=" * 65)

for file in files:

    match = re.search(r"heat_(\d+)", file.name)
    heat_number = int(match.group(1))

    df = pd.read_csv(file)

    missing_columns = [
        c for c in REQUIRED_COLUMNS if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file.name}: missing columns {missing_columns}"
        )

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.sort_values("time_s").reset_index(drop=True)

    if df["time_s"].duplicated().any():
        raise ValueError(f"{file.name}: duplicated timestamps")

    if not df["time_s"].is_monotonic_increasing:
        raise ValueError(f"{file.name}: invalid time order")

    # --------------------------------------------------------
    # CO: dynamic candidate when valid, fixed fallback otherwise
    # --------------------------------------------------------

    co_valid = parse_bool(df["co_dynamic_valid"])

    co_dynamic = df["co_flow_dynamic_aligned_m3_s"]
    co_fixed = df["co_flow_fixed_aligned_m3_s"]

    df["co_flow_final_aligned_m3_s"] = (
        co_dynamic.where(
            co_valid & co_dynamic.notna(),
            co_fixed
        )
    )

    co_cum_dynamic = df["co_cumulative_dynamic_aligned_m3"]
    co_cum_fixed = df["co_cumulative_fixed_aligned_m3"]

    df["co_cumulative_final_aligned_m3"] = (
        co_cum_dynamic.where(
            co_valid & co_cum_dynamic.notna(),
            co_cum_fixed
        )
    )

    df["co_delay_final_s"] = (
        df["co_delay_at_interval_midpoint_s"]
        .where(
            co_valid &
            df["co_delay_at_interval_midpoint_s"].notna(),
            CO_FIXED_DELAY
        )
    )

    df["co_model_used"] = np.where(
        co_valid & co_dynamic.notna(),
        "dynamic_candidate",
        "fixed_fallback"
    )

    # --------------------------------------------------------
    # CO2: fixed 34-second baseline
    # --------------------------------------------------------

    df["co2_flow_final_aligned_m3_s"] = (
        df["co2_flow_fixed_aligned_m3_s"]
    )

    df["co2_cumulative_final_aligned_m3"] = (
        df["co2_cumulative_fixed_aligned_m3"]
    )

    df["co2_delay_final_s"] = CO2_FIXED_DELAY

    df["co2_model_used"] = "fixed_baseline"

    # --------------------------------------------------------
    # Identification
    # --------------------------------------------------------

    df["heat_number"] = heat_number
    df["heat_label"] = f"heat_{heat_number:02d}"

    df = df.sort_values("time_s").reset_index(drop=True)

    # --------------------------------------------------------
    # Data quality checks
    # --------------------------------------------------------

    co_cum = df["co_cumulative_final_aligned_m3"]
    co2_cum = df["co2_cumulative_final_aligned_m3"]

    co_drops = int((co_cum.diff() < -1e-8).sum())
    co2_drops = int((co2_cum.diff() < -1e-8).sum())

    if co_drops or co2_drops:
        print(
            f"WARNING: Heat {heat_number} "
            f"cumulative monotonicity issue: "
            f"CO={co_drops}, CO2={co2_drops}"
        )

    quality_rows.append({
        "heat_number": heat_number,
        "rows": len(df),
        "co_missing": int(
            df["co_flow_final_aligned_m3_s"].isna().sum()
        ),
        "co2_missing": int(
            df["co2_flow_final_aligned_m3_s"].isna().sum()
        ),
        "co_negative": int(
            (df["co_flow_final_aligned_m3_s"] < 0).sum()
        ),
        "co2_negative": int(
            (df["co2_flow_final_aligned_m3_s"] < 0).sum()
        ),
        "co_cumulative_drops": co_drops,
        "co2_cumulative_drops": co2_drops,
    })

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary_rows.append({
        "heat_number": heat_number,
        "rows": len(df),

        "duration_s": (
            df["time_s"].max() - df["time_s"].min()
        ),

        "oxygen_total_m3": last_valid(
            df["oxygen_cumulative_m3"]
        ),

        "co_valid_points": int(
            df["co_flow_final_aligned_m3_s"].notna().sum()
        ),

        "co2_valid_points": int(
            df["co2_flow_final_aligned_m3_s"].notna().sum()
        ),

        "co_dynamic_points": int(
            (df["co_model_used"] == "dynamic_candidate").sum()
        ),

        "co_mean_flow": df[
            "co_flow_final_aligned_m3_s"
        ].mean(),

        "co2_mean_flow": df[
            "co2_flow_final_aligned_m3_s"
        ].mean(),

        "co_total_m3": last_valid(co_cum),

        "co2_total_m3": last_valid(co2_cum),

        "oxygen_co_pearson": correlation(
            df["oxygen_lance_flow_m3_s"],
            df["co_flow_final_aligned_m3_s"]
        ),

        "oxygen_co2_pearson": correlation(
            df["oxygen_lance_flow_m3_s"],
            df["co2_flow_final_aligned_m3_s"]
        ),

        "oxygen_co_spearman": correlation(
            df["oxygen_lance_flow_m3_s"],
            df["co_flow_final_aligned_m3_s"],
            "spearman"
        ),

        "oxygen_co2_spearman": correlation(
            df["oxygen_lance_flow_m3_s"],
            df["co2_flow_final_aligned_m3_s"],
            "spearman"
        ),
    })

    # --------------------------------------------------------
    # Save individual heat dataset
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"heat_{heat_number:02d}_final_aligned.csv"
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    all_heats.append(df)

    print(
        f"Heat {heat_number:02d}: "
        f"{len(df)} rows | "
        f"CO dynamic points: "
        f"{summary_rows[-1]['co_dynamic_points']} | "
        f"CO total: {summary_rows[-1]['co_total_m3']:.3f} | "
        f"CO2 total: {summary_rows[-1]['co2_total_m3']:.3f}"
    )


# ============================================================
# 4. Combine ten heats
# ============================================================

combined = pd.concat(
    all_heats,
    ignore_index=True
)

summary = pd.DataFrame(summary_rows)
quality = pd.DataFrame(quality_rows)

combined.to_csv(
    OUTPUT_DIR / "all_heats_final_aligned.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    OUTPUT_DIR / "heat_process_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

quality.to_csv(
    OUTPUT_DIR / "data_quality_audit.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. Generate per-heat relationship plots
# ============================================================

for df in all_heats:

    heat_number = int(df["heat_number"].iloc[0])

    fig, axes = plt.subplots(
        2, 1,
        figsize=(13, 9),
        sharex=True
    )

    # Flow relationships
    axes[0].plot(
        df["time_s"],
        df["oxygen_lance_flow_m3_s"],
        label="Oxygen lance flow"
    )

    axes[0].plot(
        df["time_s"],
        df["co_flow_final_aligned_m3_s"],
        label="CO flow"
    )

    axes[0].plot(
        df["time_s"],
        df["co2_flow_final_aligned_m3_s"],
        label="CO2 flow"
    )

    axes[0].set_ylabel("Flow rate")
    axes[0].set_title("Aligned gas flow and oxygen lance flow")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative relationships
    axes[1].plot(
        df["time_s"],
        df["oxygen_cumulative_m3"],
        label="Oxygen cumulative"
    )

    axes[1].plot(
        df["time_s"],
        df["co_cumulative_final_aligned_m3"],
        label="CO cumulative"
    )

    axes[1].plot(
        df["time_s"],
        df["co2_cumulative_final_aligned_m3"],
        label="CO2 cumulative"
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Cumulative volume")
    axes[1].set_title("Cumulative gas production")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Heat {heat_number:02d} - Process Relationships"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        f"heat_{heat_number:02d}_process_relationships.png",
        dpi=180
    )

    plt.close(fig)


# ============================================================
# 6. Ten-heat comparison chart
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(summary))
width = 0.36

ax.bar(
    x - width / 2,
    summary["co_total_m3"],
    width,
    label="CO"
)

ax.bar(
    x + width / 2,
    summary["co2_total_m3"],
    width,
    label="CO2"
)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"Heat {i:02d}" for i in summary["heat_number"]]
)

ax.set_ylabel("Cumulative gas volume")
ax.set_title("Final Cumulative Gas Volume Across Ten Heats")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "ten_heat_gas_comparison.png",
    dpi=180
)

plt.close(fig)


# ============================================================
# 7. README
# ============================================================

readme = """
QUESTION 2 - FINAL ALIGNED DATASET

Input:
Ten corrected heat datasets from dynamic_delay_validation_corrected.

CO:
Use dynamic candidate when co_dynamic_valid is True.
Otherwise use fixed 32-second fallback.

CO2:
Use fixed 34-second baseline.

Time convention:
Positive effective delay means analyzer observation occurs
after the corresponding oxygen-lance event.

Important:
Effective delay is not independently identified physical
transportation time.

Cumulative:
Use the cumulative aligned outputs generated by the
corrected validation program.

No endpoint extrapolation is introduced here.

Unit warning:
Flow units remain subject to verification against the
original data source.

Outputs:
- Ten per-heat final aligned CSV files
- all_heats_final_aligned.csv
- heat_process_summary.csv
- data_quality_audit.csv
- Ten per-heat relationship plots
- ten_heat_gas_comparison.png
"""

(OUTPUT_DIR / "README.txt").write_text(
    readme,
    encoding="utf-8"
)


# ============================================================
# 8. Final report
# ============================================================

print("\n" + "=" * 65)
print("FINAL DATASET COMPLETED")
print("=" * 65)

print("\nHeat summary:")
print(
    summary[
        [
            "heat_number",
            "rows",
            "co_dynamic_points",
            "co_total_m3",
            "co2_total_m3",
            "oxygen_co_pearson",
            "oxygen_co2_pearson",
        ]
    ].round(4).to_string(index=False)
)

print("\nData quality:")
print(quality.to_string(index=False))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nAll original experimental data preserved.")
