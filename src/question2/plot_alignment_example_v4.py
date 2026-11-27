
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "deliverables" / "question2" / "04_alignment"
    / "heat_01_final_aligned.csv"
)

OUT_DIR = ROOT / "deliverables" / "question2" / "04_alignment"

OUT_PNG = OUT_DIR / "heat_01_alignment_example_v4.png"
OUT_PDF = OUT_DIR / "heat_01_alignment_example_v4.pdf"


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


def representative_delay(df, delay_column):
    valid = df.dropna(
        subset=["oxygen_interval_midpoint_s", delay_column]
    ).copy()

    if valid.empty:
        raise ValueError(f"No valid observations for {delay_column}")

    median_delay = valid[delay_column].median()

    idx = (valid[delay_column] - median_delay).abs().idxmin()
    row = valid.loc[idx]

    event_time = float(row["oxygen_interval_midpoint_s"])
    delay = float(row[delay_column])
    analyzer_time = event_time + delay

    return {
        "event_time": event_time,
        "analyzer_time": analyzer_time,
        "delay": delay,
    }


def add_delay_arrow(ax, result, gas_name):
    event_time = result["event_time"]
    analyzer_time = result["analyzer_time"]
    delay = result["delay"]

    y_bottom, y_top = ax.get_ylim()
    span = y_top - y_bottom

    arrow_y = y_top + span * 0.08
    guide_bottom = y_top - span * 0.12

    # Short dashed guides at both time coordinates
    ax.vlines(
        [event_time, analyzer_time],
        guide_bottom,
        arrow_y,
        colors="black",
        linestyles="dashed",
        linewidth=0.9,
        alpha=0.8
    )

    # Double-headed delay arrow
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

    # Delay label
    ax.text(
        (event_time + analyzer_time) / 2,
        arrow_y + span * 0.015,
        f"{gas_name} effective delay: {delay:.1f} s",
        ha="center",
        va="bottom",
        fontsize=9
    )

    ax.set_ylim(
        y_bottom,
        arrow_y + span * 0.17
    )


def main():

    df = pd.read_csv(DATA_FILE)

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Time references
    analyzer_time = df["time_s"]
    event_time = df["oxygen_interval_midpoint_s"]

    # Process signals
    oxygen_flow = df["oxygen_lance_flow_m3_s"]

    co_raw = df["co_flow_m3_s"]
    co_aligned = df["co_flow_final_aligned_m3_s"]

    co2_raw = df["co2_flow_m3_s"]
    co2_aligned = df["co2_flow_final_aligned_m3_s"]

    # Model usage
    co_models = model_summary(df["co_model_used"])
    co2_models = model_summary(df["co2_model_used"])

    # Median effective delays
    co_median = df["co_delay_final_s"].median()
    co2_median = df["co2_delay_final_s"].median()

    # Representative delay observations
    co_example = representative_delay(df, "co_delay_final_s")
    co2_example = representative_delay(df, "co2_delay_final_s")

    # Figure
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 11),
        sharex=True
    )

    fig.suptitle(
        "Heat 01 - Process Signal Alignment",
        fontsize=18,
        fontweight="bold",
        y=0.975
    )

    fig.text(
        0.5, 0.938,
        f"Heat ID: 6B03205    |    "
        f"Median effective delay: "
        f"CO {co_median:.1f} s    |    "
        f"CO2 {co2_median:.1f} s",
        ha="center",
        fontsize=11
    )

    # -------------------------------------------------
    # (a) Oxygen lance
    # -------------------------------------------------

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

    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    # -------------------------------------------------
    # (b) CO alignment
    # -------------------------------------------------

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

    axes[1].grid(alpha=0.25)

    add_delay_arrow(
        axes[1],
        co_example,
        "CO"
    )

    axes[1].legend(
        loc="upper right",
        fontsize=9
    )

    # -------------------------------------------------
    # (c) CO2 alignment
    # -------------------------------------------------

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

    axes[2].grid(alpha=0.25)

    add_delay_arrow(
        axes[2],
        co2_example,
        "CO2"
    )

    axes[2].legend(
        loc="upper right",
        fontsize=9
    )

    # -------------------------------------------------
    # Footer
    # -------------------------------------------------

    footer = (
        f"Model usage — CO: {co_models} | CO2: {co2_models}\n"
        r"Delay definition: $\tau=t_{analyzer}-t_{event}$"
        "\n"
        f"CO example: event={co_example['event_time']:.1f} s, "
        f"analyzer={co_example['analyzer_time']:.1f} s, "
        f"tau={co_example['delay']:.1f} s"
        "\n"
        f"CO2 example: event={co2_example['event_time']:.1f} s, "
        f"analyzer={co2_example['analyzer_time']:.1f} s, "
        f"tau={co2_example['delay']:.1f} s"
        "\n"
        "Event time = oxygen-lance interval midpoint. "
        "Raw signals use analyzer time; aligned signals use event time."
        "\n"
        "Flow units: source-unit verification pending."
    )

    fig.text(
        0.5, 0.015,
        footer,
        ha="center",
        va="bottom",
        fontsize=7.8
    )

    fig.subplots_adjust(
        top=0.90,
        bottom=0.22,
        hspace=0.35
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    print("=" * 55)
    print("V4 FINAL FIGURE GENERATED")
    print("=" * 55)

    print("CO median delay:", co_median)
    print("CO2 median delay:", co2_median)

    print("CO representative:", co_example)
    print("CO2 representative:", co2_example)

    print("CO models:", co_models)
    print("CO2 models:", co2_models)

    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()
