
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "deliverables" / "question2" / "01_process_signals"
OUT_DIR = DATA_DIR / "draft_figures"

HEAT_NUMBER = 10
HEAT_ID = "6C02108"

DATA_FILE = (
    DATA_DIR /
    f"heat_{HEAT_NUMBER:02d}_{HEAT_ID}_six_signals.csv"
)

OUT_PNG = OUT_DIR / "heat_10_continuous_process_signals_v1.png"
OUT_PDF = OUT_DIR / "heat_10_continuous_process_signals_v1.pdf"


# ==========================================
# 2. DATA QUALITY DIAGNOSTIC
# ==========================================

def diagnose_data(df, required_columns):

    print("\nMISSING DATA DIAGNOSTIC")
    print("-" * 60)

    print("Original rows:", len(df))

    print("\nMissing values by column:")
    print(df[required_columns].isna().sum())

    bad_rows = df.loc[
        df[required_columns].isna().any(axis=1),
        required_columns
    ]

    print("\nRows containing missing values:")

    if bad_rows.empty:
        print("No missing rows.")
    else:
        print(bad_rows.to_string(index=False))


# ==========================================
# 3. CUMULATIVE CONSISTENCY CHECK
# ==========================================

def check_cumulative_consistency(
    df,
    flow_col,
    cumulative_col
):
    """
    Compare integrated instantaneous flow with
    source cumulative volume.

    Missing observations are not filled.
    Integration starts from the first valid
    observation shared by flow and cumulative data.
    """

    valid = df[
        ["time_s", flow_col, cumulative_col]
    ].copy()

    valid = valid.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = (
        valid.dropna()
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    if len(valid) < 2:
        raise ValueError(
            f"Insufficient valid data: {flow_col}"
        )

    time = valid["time_s"].to_numpy(dtype=float)

    flow = valid[flow_col].to_numpy(dtype=float)

    cumulative = valid[
        cumulative_col
    ].to_numpy(dtype=float)

    # Check time intervals
    dt = np.diff(time)

    if np.any(dt < 0):
        raise ValueError(
            f"Time sequence is not increasing: {flow_col}"
        )

    # Trapezoidal integration
    integrated = np.zeros(len(time))

    integrated[1:] = np.cumsum(
        (flow[:-1] + flow[1:]) / 2 * dt
    )

    # Compare cumulative increments from
    # the same first valid observation
    source_increment = (
        cumulative - cumulative[0]
    )

    difference = (
        source_increment - integrated
    )

    source_total = float(source_increment[-1])

    integrated_total = float(integrated[-1])

    max_difference = float(
        np.max(np.abs(difference))
    )

    final_difference = (
        source_total - integrated_total
    )

    relative_error = (
        abs(final_difference)
        / abs(source_total) * 100
        if abs(source_total) > 1e-12
        else np.nan
    )

    return {
        "source_increment": source_total,
        "integrated_flow": integrated_total,
        "max_difference": max_difference,
        "final_difference": final_difference,
        "relative_error": relative_error,
        "start_time": float(time[0]),
        "end_time": float(time[-1]),
        "valid_rows": len(valid),
    }


def validate_all_cumulative_signals(df):

    pairs = [
        (
            "Oxygen",
            "oxygen_lance_flow_m3_s",
            "oxygen_cumulative_m3"
        ),
        (
            "CO",
            "co_flow_m3_s",
            "co_cumulative_m3"
        ),
        (
            "CO2",
            "co2_flow_m3_s",
            "co2_cumulative_m3"
        )
    ]

    print("\nCUMULATIVE CONSISTENCY CHECK")
    print("-" * 60)

    results = {}

    for name, flow_col, cumulative_col in pairs:

        result = check_cumulative_consistency(
            df,
            flow_col,
            cumulative_col
        )

        results[name] = result

        print(f"\n{name}:")

        print(
            f"Valid observations:           "
            f"{result['valid_rows']}"
        )

        print(
            f"Integration time range:       "
            f"{result['start_time']:.1f} - "
            f"{result['end_time']:.1f} s"
        )

        print(
            f"Source cumulative increment:  "
            f"{result['source_increment']:.3f}"
        )

        print(
            f"Integrated flow:              "
            f"{result['integrated_flow']:.3f}"
        )

        print(
            f"Maximum absolute difference:  "
            f"{result['max_difference']:.3f}"
        )

        print(
            f"Final difference:             "
            f"{result['final_difference']:.3f}"
        )

        print(
            f"Final relative difference:    "
            f"{result['relative_error']:.4f}%"
        )

    return results


# ==========================================
# 4. MAIN PLOTTING FUNCTION
# ==========================================

def main():

    # --------------------------------------
    # Load source data
    # --------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "time_s",
        "oxygen_lance_flow_m3_s",
        "oxygen_cumulative_m3",
        "co_flow_m3_s",
        "co2_flow_m3_s",
        "co_cumulative_m3",
        "co2_cumulative_m3",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Convert columns to numeric
    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------
    # Data diagnostic
    # --------------------------------------

    diagnose_data(
        df,
        required_columns
    )

    # Preserve all valid timestamp records.
    # Do not delete rows because individual
    # process signals contain missing values.

    df = (
        df.dropna(subset=["time_s"])
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    # --------------------------------------
    # Validate cumulative signals
    # --------------------------------------

    validation_results = (
        validate_all_cumulative_signals(df)
    )

    # --------------------------------------
    # Time information
    # --------------------------------------

    time = df["time_s"]

    # ======================================
    # 5. FIGURE LAYOUT
    # ======================================

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 11),
        sharex=True
    )

    fig.suptitle(
        f"Heat {HEAT_NUMBER:02d} - Continuous Process Signals",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    fig.text(
        0.5, 0.945,
        f"Heat ID: {HEAT_ID} | "
        f"Observations: {len(df)} | "
        f"Duration: "
        f"{time.max() - time.min():.1f} s",
        ha="center",
        fontsize=10
    )

    # ======================================
    # (a) OXYGEN LANCE PROCESS
    # ======================================

    ax1 = axes[0]

    line1, = ax1.plot(
        time,
        df["oxygen_lance_flow_m3_s"],
        linewidth=1.1,
        linestyle="-",
        label="Oxygen lance flow"
    )

    ax1.set_ylabel(
        "Oxygen flow (m³/s)"
    )

    ax1.set_title(
        "(a) Oxygen Lance Process",
        loc="left",
        fontsize=12
    )

    ax1.grid(alpha=0.25)

    # Secondary axis
    ax2 = ax1.twinx()

    line2, = ax2.plot(
        time,
        df["oxygen_cumulative_m3"],
        color="C1",
        linewidth=1.5,
        linestyle="--",
        label="Cumulative oxygen consumption"
    )

    ax2.set_ylabel(
        "Cumulative oxygen (m³)"
    )

    ax1.legend(
        [line1, line2],
        [
            line1.get_label(),
            line2.get_label()
        ],
        loc="upper left",
        fontsize=9
    )

    # ======================================
    # (b) OFF-GAS FLOW
    # ======================================

    axes[1].plot(
        time,
        df["co_flow_m3_s"],
        linewidth=1.2,
        linestyle="-",
        label="CO flow"
    )

    axes[1].plot(
        time,
        df["co2_flow_m3_s"],
        linewidth=1.2,
        linestyle="-",
        label="CO₂ flow"
    )

    axes[1].set_ylabel(
        "Gas flow (m³/s)"
    )

    axes[1].set_title(
        "(b) Off-gas Flow",
        loc="left",
        fontsize=12
    )

    axes[1].legend(
        loc="upper right",
        fontsize=9
    )

    axes[1].grid(alpha=0.25)

    # ======================================
    # (c) CUMULATIVE OFF-GAS VOLUME
    # ======================================

    axes[2].plot(
        time,
        df["co_cumulative_m3"],
        linewidth=1.4,
        linestyle="--",
        label="Cumulative CO"
    )

    axes[2].plot(
        time,
        df["co2_cumulative_m3"],
        linewidth=1.4,
        linestyle="--",
        label="Cumulative CO₂"
    )

    axes[2].set_ylabel(
        "Cumulative gas volume (m³)"
    )

    axes[2].set_xlabel(
        "Process time (s)"
    )

    axes[2].set_title(
        "(c) Cumulative Off-gas Volume",
        loc="left",
        fontsize=12
    )

    axes[2].legend(
        loc="upper left",
        fontsize=9
    )

    axes[2].grid(alpha=0.25)

    # ======================================
    # 6. FOOTER
    # ======================================

    fig.text(
        0.5, 0.015,
        "Solid lines: instantaneous flow. "
        "Dashed lines: cumulative volume. "
        "Units follow source field definitions.",
        ha="center",
        fontsize=8
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        hspace=0.38
    )

    # ======================================
    # 7. SAVE FIGURES
    # ======================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # 8. OUTPUT SUMMARY
    # ======================================

    print("\n" + "=" * 60)

    print(f"HEAT {HEAT_NUMBER:02d} PROCESS SIGNAL FIGURE GENERATED")

    print("=" * 60)

    print("Rows:", len(df))

    print(
        "Time range:",
        time.min(),
        "-",
        time.max()
    )

    print("\nMissing values retained:")

    for col in required_columns:
        print(
            f"{col}: {df[col].isna().sum()}"
        )

    print("\nFinal cumulative values:")

    for col in [
        "oxygen_cumulative_m3",
        "co_cumulative_m3",
        "co2_cumulative_m3"
    ]:

        print(
            f"{col}: "
            f"{df[col].iloc[-1]:.3f}"
        )

    print("\nPNG:", OUT_PNG)

    print("PDF:", OUT_PDF)

    print("\nAll cumulative validation checks completed.")


# ==========================================
# 9. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
