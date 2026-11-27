
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. PATH CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "deliverables" / "question2" / "04_alignment"
OUT_DIR = DATA_DIR

# Only generate Heat 03 in this run
HEAT_NUMBERS = range(8, 9)

HEAT_ID_MAP = {
    1: "6B03205",
    2: "6B03206",
    3: "6B03270",
    4: "6B03271",
    5: "6C02057",
    6: "6C02058",
    7: "6C02059",
    8: "6C02060",
    9: "6C02107",
    10: "6C02108",
}


# =====================================================
# 2. MODEL USAGE SUMMARY
# =====================================================

def model_summary(series):

    labels = {
        "dynamic_candidate": "Dynamic",
        "fixed_fallback": "Fixed fallback",
        "fixed_baseline": "Fixed baseline",
    }

    counts = series.value_counts(dropna=True)

    return "; ".join(
        f"{labels.get(name, name)}: {count}"
        for name, count in counts.items()
    )


# =====================================================
# 3. REPRESENTATIVE DELAY SELECTION
# =====================================================

def representative_delay(df, delay_column):

    valid = df.dropna(
        subset=[
            "oxygen_interval_midpoint_s",
            delay_column
        ]
    ).copy()

    if valid.empty:
        raise ValueError(
            f"No valid observations for {delay_column}"
        )

    median_delay = valid[delay_column].median()

    # Find observations closest to median delay
    deviation = (
        valid[delay_column] - median_delay
    ).abs()

    min_deviation = deviation.min()

    candidates = valid.loc[
        deviation <= min_deviation + 1e-9
    ].copy()

    # Select the candidate nearest to the
    # central event-time position
    center_time = (
        valid["oxygen_interval_midpoint_s"].min()
        + valid["oxygen_interval_midpoint_s"].max()
    ) / 2

    idx = (
        candidates["oxygen_interval_midpoint_s"]
        - center_time
    ).abs().idxmin()

    row = valid.loc[idx]

    event_time = float(
        row["oxygen_interval_midpoint_s"]
    )

    delay = float(row[delay_column])

    analyzer_time = event_time + delay

    return {
        "event_time": event_time,
        "analyzer_time": analyzer_time,
        "delay": delay,
    }


# =====================================================
# 4. DELAY ARROW
# =====================================================

def add_delay_arrow(
    ax,
    result,
    gas_name,
    label_prefix=None
):

    event_time = result["event_time"]
    analyzer_time = result["analyzer_time"]
    delay = result["delay"]

    y_bottom, y_top = ax.get_ylim()

    span = y_top - y_bottom

    arrow_y = y_top + span * 0.08
    guide_bottom = y_top - span * 0.12

    # Dashed vertical guides
    ax.vlines(
        [event_time, analyzer_time],
        guide_bottom,
        arrow_y,
        colors="black",
        linestyles="dashed",
        linewidth=0.9,
        alpha=0.8
    )

    # Double-headed arrow
    ax.annotate(
        "",
        xy=(analyzer_time, arrow_y),
        xytext=(event_time, arrow_y),
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            linewidth=1.6
        )
    )

    # Label
    if label_prefix is not None:
        label = (
            f"{label_prefix}: {delay:.1f} s"
        )
    else:
        label = (
            f"{gas_name} effective delay: "
            f"{delay:.1f} s"
        )

    ax.text(
        (event_time + analyzer_time) / 2,
        arrow_y + span * 0.015,
        label,
        ha="center",
        va="bottom",
        fontsize=9
    )

    ax.set_ylim(
        y_bottom,
        arrow_y + span * 0.17
    )


# =====================================================
# 5. DATA LOADING
# =====================================================

def load_and_prepare(data_file):

    df = pd.read_csv(data_file)

    numeric_columns = [
        "time_s",
        "oxygen_interval_midpoint_s",
        "oxygen_lance_flow_m3_s",
        "co_flow_m3_s",
        "co2_flow_m3_s",
        "co_flow_final_aligned_m3_s",
        "co2_flow_final_aligned_m3_s",
        "co_delay_final_s",
        "co2_delay_final_s",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


# =====================================================
# 6. PLOT ONE HEAT
# =====================================================

def plot_one_heat(heat_number):

    heat_label = f"heat_{heat_number:02d}"

    heat_id = HEAT_ID_MAP[heat_number]

    data_file = (
        DATA_DIR
        / f"{heat_label}_final_aligned.csv"
    )

    out_png = (
        OUT_DIR
        / f"{heat_label}_alignment_example_v5_2.png"
    )

    out_pdf = (
        OUT_DIR
        / f"{heat_label}_alignment_example_v5_2.pdf"
    )

    if not data_file.exists():

        print(f"[SKIP] File not found: {data_file}")

        return

    df = load_and_prepare(data_file)

    # -------------------------------------------------
    # Time references
    # -------------------------------------------------

    analyzer_time = df["time_s"]

    event_time = df[
        "oxygen_interval_midpoint_s"
    ]

    # -------------------------------------------------
    # Process signals
    # -------------------------------------------------

    oxygen_flow = df[
        "oxygen_lance_flow_m3_s"
    ]

    co_raw = df["co_flow_m3_s"]

    co_aligned = df[
        "co_flow_final_aligned_m3_s"
    ]

    co2_raw = df["co2_flow_m3_s"]

    co2_aligned = df[
        "co2_flow_final_aligned_m3_s"
    ]

    # -------------------------------------------------
    # Model usage
    # -------------------------------------------------

    co_models = model_summary(
        df["co_model_used"]
    )

    co2_models = model_summary(
        df["co2_model_used"]
    )

    # -------------------------------------------------
    # Median delays
    # -------------------------------------------------

    co_median = df[
        "co_delay_final_s"
    ].median()

    co2_median = df[
        "co2_delay_final_s"
    ].median()

    # -------------------------------------------------
    # Representative observations
    # -------------------------------------------------

    co_example = representative_delay(
        df,
        "co_delay_final_s"
    )

    co2_example = representative_delay(
        df,
        "co2_delay_final_s"
    )

    # -------------------------------------------------
    # AUDIT: ensure arrow and footer use same records
    # -------------------------------------------------

    print("\n" + "=" * 60)

    print(f"HEAT {heat_number:02d} REPRESENTATIVE DELAY AUDIT")

    print("CO representative:", co_example)

    print("CO2 representative:", co2_example)

    print("=" * 60 + "\n")

    # -------------------------------------------------
    # Figure configuration
    # -------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True
    )

    fig.suptitle(
        f"Heat {heat_number:02d} - Process Signal Alignment",
        fontsize=18,
        fontweight="bold",
        y=0.975
    )

    fig.text(
        0.5,
        0.938,
        f"Heat ID: {heat_id}    |    "
        f"Median effective delay: "
        f"CO {co_median:.1f} s    |    "
        f"CO2 {co2_median:.1f} s",
        ha="center",
        fontsize=11
    )

    # =================================================
    # (a) OXYGEN LANCE
    # =================================================

    axes[0].plot(
        event_time,
        oxygen_flow,
        linewidth=1.3,
        label="Oxygen lance flow (event time)"
    )

    axes[0].set_title(
        "(a) Oxygen Lance Process Signal",
        loc="left",
        fontsize=12
    )

    axes[0].set_ylabel(
        "Oxygen flow\n(reported unit)"
    )

    axes[0].legend(
        loc="upper left"
    )

    axes[0].grid(
        alpha=0.25
    )

    # =================================================
    # (b) CO ALIGNMENT
    # =================================================

    axes[1].plot(
        analyzer_time,
        co_raw,
        linewidth=1.1,
        label="Raw CO (analyzer observation time)"
    )

    axes[1].plot(
        event_time,
        co_aligned,
        linewidth=1.3,
        label="Aligned CO (oxygen-lance event time)"
    )

    axes[1].set_title(
        "(b) CO Signal Alignment",
        loc="left",
        fontsize=12
    )

    axes[1].set_ylabel(
        "CO flow\n(reported unit)"
    )

    axes[1].grid(
        alpha=0.25
    )

    add_delay_arrow(
        axes[1],
        co_example,
        "CO"
    )

    axes[1].legend(
        loc="upper right",
        fontsize=9
    )

    # =================================================
    # (c) CO2 ALIGNMENT
    # =================================================

    axes[2].plot(
        analyzer_time,
        co2_raw,
        linewidth=1.1,
        label="Raw CO2 (analyzer observation time)"
    )

    axes[2].plot(
        event_time,
        co2_aligned,
        linewidth=1.3,
        label="Aligned CO2 (oxygen-lance event time)"
    )

    axes[2].set_title(
        "(c) CO2 Signal Alignment",
        loc="left",
        fontsize=12
    )

    axes[2].set_ylabel(
        "CO2 flow\n(reported unit)"
    )

    axes[2].set_xlabel(
        "Time / s (blue: analyzer time; orange: event time)"
    )

    axes[2].grid(
        alpha=0.25
    )

    add_delay_arrow(
        axes[2],
        co2_example,
        "CO2",
        label_prefix="CO2 fixed baseline"
    )

    axes[2].legend(
        loc="upper right",
        fontsize=9
    )

    # =================================================
    # 7. FOOTER
    # =================================================

    footer = (
        f"Model usage - CO: {co_models} | "
        f"CO2: {co2_models}\n"

        r"Delay definition: "
        r"$\tau=t_{analyzer}-t_{event}$"

        "\n"

        f"CO example: "
        f"event={co_example['event_time']:.1f} s, "
        f"analyzer={co_example['analyzer_time']:.1f} s, "
        f"tau={co_example['delay']:.1f} s"

        "\n"

        f"CO2 example: "
        f"event={co2_example['event_time']:.1f} s, "
        f"analyzer={co2_example['analyzer_time']:.1f} s, "
        f"tau={co2_example['delay']:.1f} s"

        "\n"

        "Event time = oxygen-lance interval midpoint. "
        "Raw signals use analyzer time; "
        "aligned signals use event time."

        "\n"

        "Flow units: source-unit verification pending."
    )

    fig.text(
        0.5,
        0.015,
        footer,
        ha="center",
        va="bottom",
        fontsize=7.8
    )

    # =================================================
    # 8. LAYOUT
    # =================================================

    fig.subplots_adjust(
        top=0.90,
        bottom=0.22,
        hspace=0.35
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =================================================
    # 9. SAVE
    # =================================================

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("=" * 60)

    print(f"Heat {heat_number:02d} completed")

    print("PNG:", out_png)

    print("PDF:", out_pdf)

    print("CO median delay:", co_median)

    print("CO2 median delay:", co2_median)

    print("CO models:", co_models)

    print("CO2 models:", co2_models)

    print("=" * 60)


# =====================================================
# 10. MAIN
# =====================================================

def main():

    print("=" * 60)

    print("PROCESS SIGNAL ALIGNMENT - V5.2")

    print("=" * 60)

    for heat_number in HEAT_NUMBERS:

        try:

            plot_one_heat(heat_number)

        except Exception as e:

            print(
                f"[ERROR] Heat {heat_number:02d}: {e}"
            )

            raise

    print("\nBatch generation finished.")


if __name__ == "__main__":

    main()
