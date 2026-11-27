"""
Question 2 - Extreme Delay Correlation Peak Audit

Purpose:
1. Identify the 21 extreme reliable delay windows.
2. Recover their original correlation-versus-lag curves.
3. Compare estimated delays with actual correlation maxima.
4. Identify competing local peaks and peak separation.
5. Generate audit tables and figures.

Original data are never modified.
"""

from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

ESTIMATE_FILE = (
    ROOT / "results/question2/ten_heat_reconstruction_delay/"
    "all_window_delay_estimates.csv"
)

CURVE_FILE = (
    ROOT / "results/question2/ten_heat_reconstruction_delay/"
    "all_window_correlation_curves.csv"
)

OUTPUT_DIR = (
    ROOT / "results/question2/extreme_delay_peak_audit"
    / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

EXTREME_LOW = 10.0
EXTREME_HIGH = 55.0

# Minimum separation between primary and competing peaks.
MIN_PEAK_SEPARATION_S = 5.0


# --------------------------------------------------
# 2. Utility functions
# --------------------------------------------------

def make_key(row):
    return (
        str(int(float(row["attachment"]))),
        str(row["heat"]).strip(),
        str(row["signal"]).strip().lower(),
        round(float(row["window_center_s"]), 4),
    )


def find_local_peaks(lags, correlations):
    """Find interior local maxima."""
    peaks = []

    for i in range(1, len(correlations) - 1):
        if (
            correlations[i] >= correlations[i - 1]
            and correlations[i] >= correlations[i + 1]
            and (
                correlations[i] > correlations[i - 1]
                or correlations[i] > correlations[i + 1]
            )
        ):
            peaks.append(i)

    return peaks


def analyze_curve(curve, estimated_delay):
    curve = curve.sort_values("lag_s").drop_duplicates("lag_s")

    lags = curve["lag_s"].to_numpy(dtype=float)
    corr = curve["correlation"].to_numpy(dtype=float)

    valid = np.isfinite(lags) & np.isfinite(corr)
    lags = lags[valid]
    corr = corr[valid]

    if len(lags) < 3:
        raise ValueError("Insufficient correlation points.")

    # Primary maximum
    primary_idx = int(np.argmax(corr))

    primary_lag = float(lags[primary_idx])
    primary_corr = float(corr[primary_idx])

    # Local maxima
    peak_indices = find_local_peaks(lags, corr)

    if primary_idx not in peak_indices:
        peak_indices.append(primary_idx)

    # Competing local maxima separated from the primary peak
    secondary = [
        i for i in peak_indices
        if abs(lags[i] - primary_lag)
        >= MIN_PEAK_SEPARATION_S
    ]

    secondary_idx = None

    if secondary:
        secondary_idx = max(
            secondary,
            key=lambda i: corr[i]
        )

        secondary_lag = float(lags[secondary_idx])
        secondary_corr = float(corr[secondary_idx])
        peak_gap = primary_corr - secondary_corr

    else:
        secondary_lag = np.nan
        secondary_corr = np.nan
        peak_gap = np.nan

    # Correlation at originally estimated delay
    nearest_idx = int(np.argmin(abs(lags - estimated_delay)))
    corr_at_estimated = float(corr[nearest_idx])

    return {
        "curve_points": len(lags),
        "curve_lag_min_s": float(lags.min()),
        "curve_lag_max_s": float(lags.max()),
        "curve_primary_peak_lag_s": primary_lag,
        "curve_primary_peak_corr": primary_corr,
        "curve_secondary_peak_lag_s": secondary_lag,
        "curve_secondary_peak_corr": secondary_corr,
        "recomputed_peak_gap": peak_gap,
        "local_peak_count": len(peak_indices),
        "corr_at_estimated_delay": corr_at_estimated,
        "estimated_delay_lag_difference_s": (
            primary_lag - estimated_delay
        ),
        "primary_peak_near_boundary": (
            primary_lag <= lags.min() + 0.5
            or primary_lag >= lags.max() - 0.5
        ),
        "lags": lags,
        "correlations": corr,
        "primary_idx": primary_idx,
        "secondary_idx": secondary_idx,
    }


# --------------------------------------------------
# 3. Load original data
# --------------------------------------------------

print("=" * 65)
print("EXTREME DELAY CORRELATION PEAK AUDIT")
print("=" * 65)

if not ESTIMATE_FILE.exists():
    raise FileNotFoundError(ESTIMATE_FILE)

if not CURVE_FILE.exists():
    raise FileNotFoundError(CURVE_FILE)

estimates = pd.read_csv(ESTIMATE_FILE)
curves = pd.read_csv(CURVE_FILE)

for col in [
    "attachment",
    "window_center_s",
    "delay_s",
]:
    estimates[col] = pd.to_numeric(
        estimates[col], errors="coerce"
    )

for col in ["lag_s", "correlation", "window_center_s"]:
    curves[col] = pd.to_numeric(
        curves[col], errors="coerce"
    )

estimates["reliable_bool"] = (
    estimates["reliable"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

extreme = estimates[
    estimates["reliable_bool"]
    & (
        (estimates["delay_s"] < EXTREME_LOW)
        | (estimates["delay_s"] > EXTREME_HIGH)
    )
].copy()

print(f"\nTotal original estimates: {len(estimates)}")
print(f"Extreme reliable windows: {len(extreme)}")
print(f"Correlation curve records: {len(curves)}")


# --------------------------------------------------
# 4. Index correlation curves
# --------------------------------------------------

curve_groups = {}

for key, group in curves.groupby(
    ["attachment", "heat", "signal", "window_center_s"]
):
    normalized_key = (
        str(int(float(key[0]))),
        str(key[1]).strip(),
        str(key[2]).strip().lower(),
        round(float(key[3]), 4),
    )

    curve_groups[normalized_key] = group.copy()


# --------------------------------------------------
# 5. Analyze every extreme window
# --------------------------------------------------

records = []
plot_items = []

for _, row in extreme.iterrows():

    key = make_key(row)

    print(
        f"\nProcessing {row['signal']} | "
        f"{row['heat']} | "
        f"{row['window_center_s']} s"
    )

    base = {
        "attachment": row["attachment"],
        "heat": row["heat"],
        "signal": row["signal"],
        "window_center_s": row["window_center_s"],
        "estimated_delay_s": row["delay_s"],
        "reported_correlation": row["correlation"],
        "reported_peak_contrast": row["peak_contrast"],
        "boundary_peak_original": row["boundary_peak"],
        "reliable_original": row["reliable"],
    }

    if key not in curve_groups:
        print("  WARNING: Correlation curve not found.")

        records.append({
            **base,
            "curve_found": False,
        })
        continue

    result = analyze_curve(
        curve_groups[key],
        float(row["delay_s"])
    )

    records.append({
        **base,
        "curve_found": True,
        **{
            k: v for k, v in result.items()
            if k not in [
                "lags",
                "correlations",
                "primary_idx",
                "secondary_idx",
            ]
        },
    })

    plot_items.append({
        "row": row,
        "result": result,
    })

    print(
        f"  Estimated delay: {row['delay_s']:.2f} s"
    )
    print(
        f"  Curve maximum: {result['curve_primary_peak_lag_s']:.2f} s"
    )
    print(
        f"  Peak correlation: {result['curve_primary_peak_corr']:.4f}"
    )
    print(
        f"  Peak gap: {result['recomputed_peak_gap']:.4f}"
    )


# --------------------------------------------------
# 6. Save audit table
# --------------------------------------------------

audit = pd.DataFrame(records)

audit.to_csv(
    OUTPUT_DIR / "extreme_window_peak_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nAudit table saved.")


# --------------------------------------------------
# 7. Generate correlation curve figures
# --------------------------------------------------

if plot_items:

    n = len(plot_items)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18, 4.0 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for i, item in enumerate(plot_items):

        ax = axes[i]

        row = item["row"]
        res = item["result"]

        lags = res["lags"]
        corr = res["correlations"]

        ax.plot(
            lags,
            corr,
            linewidth=1.4,
            label="Correlation curve",
        )

        ax.axvline(
            float(row["delay_s"]),
            linestyle="--",
            linewidth=1.2,
            label="Original estimated delay",
        )

        p = res["primary_idx"]

        ax.scatter(
            lags[p],
            corr[p],
            marker="*",
            s=100,
            label="Global maximum",
            zorder=5,
        )

        s = res["secondary_idx"]

        if s is not None:
            ax.scatter(
                lags[s],
                corr[s],
                marker="x",
                s=65,
                label="Competing local peak",
                zorder=5,
            )

        ax.set_title(
            f"{row['signal'].upper()} | {row['heat']}\n"
            f"Window center: {row['window_center_s']} s",
            fontsize=10,
        )

        ax.set_xlabel("Candidate delay (s)")
        ax.set_ylabel("Correlation")

        ax.grid(alpha=0.25)

    for j in range(len(plot_items), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
    )

    fig.suptitle(
        "Extreme Delay Windows - Correlation Peak Audit",
        fontsize=16,
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(
        OUTPUT_DIR / "extreme_window_correlation_audit.png",
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_DIR / "extreme_window_correlation_audit.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# --------------------------------------------------
# 8. Special analysis: consecutive CO windows
#    in heat 6B03206
# --------------------------------------------------

target_heat = "6B03206"

sequence = [
    item for item in plot_items
    if (
        str(item["row"]["heat"]) == target_heat
        and str(item["row"]["signal"]).lower() == "co"
    )
]

if sequence:

    fig, ax = plt.subplots(figsize=(11, 6))

    for item in sequence:

        row = item["row"]
        res = item["result"]

        ax.plot(
            res["lags"],
            res["correlations"],
            linewidth=1.5,
            label=(
                f"Center {row['window_center_s']} s "
                f"(delay={row['delay_s']} s)"
            ),
        )

    ax.set_title(
        "CO Correlation Curves - Heat 6B03206"
    )

    ax.set_xlabel("Candidate delay (s)")
    ax.set_ylabel("Correlation")

    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "heat_6B03206_CO_sequence.png",
        dpi=180,
    )

    fig.savefig(
        OUTPUT_DIR / "heat_6B03206_CO_sequence.pdf",
    )

    plt.close(fig)


# --------------------------------------------------
# 9. Final summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("AUDIT COMPLETED")
print("=" * 65)

print(f"\nOutput directory:\n{OUTPUT_DIR}")

if not audit.empty:

    display_cols = [
        "heat",
        "signal",
        "window_center_s",
        "estimated_delay_s",
        "curve_primary_peak_lag_s",
        "curve_primary_peak_corr",
        "recomputed_peak_gap",
        "local_peak_count",
        "estimated_delay_lag_difference_s",
    ]

    available = [
        c for c in display_cols if c in audit.columns
    ]

    print("\nPeak audit summary:")

    print(
        audit[available].to_string(
            index=False
        )
    )

    print("\nCurve matching:")
    print(
        f"  Matched: {audit['curve_found'].sum()}"
    )
    print(
        f"  Missing: {(~audit['curve_found']).sum()}"
    )

print("\nOriginal experiment files were not modified.")