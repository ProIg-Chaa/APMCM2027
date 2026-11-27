
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "deliverables" / "question2" / "04_alignment"
OUT_DIR = DATA_DIR

# 每次只修改这里：例如 Heat 09 就改成 range(9, 10)
HEAT_NUMBERS = range(8, 9)

# heat_number -> Heat ID
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


def get_delay_label(series, gas_name):
    """
    Generate an arrow label according to actual model usage.
    """
    models = series.dropna().unique()

    if len(models) == 1:
        model = models[0]

        if model == "fixed_fallback":
            return f"{gas_name} fixed fallback"

        if model == "fixed_baseline":
            return f"{gas_name} fixed baseline"

        if model == "dynamic_candidate":
            return f"{gas_name} dynamic candidate"

    return f"{gas_name} effective delay"



def representative_delay(df, delay_column):

    # 有效延迟数据
    valid_delay = df.dropna(
        subset=[delay_column]
    ).copy()

    # 有效事件时间数据
    valid_time = df.dropna(
        subset=["oxygen_interval_midpoint_s"]
    ).copy()

    if valid_delay.empty or valid_time.empty:
        raise ValueError(
            f"No valid observations for {delay_column}"
        )

    # 全部有效观测的中位数延迟
    median_delay = float(
        valid_delay[delay_column].median()
    )

    # 整炉时间范围的中点
    min_time = valid_time[
        "oxygen_interval_midpoint_s"
    ].min()

    max_time = valid_time[
        "oxygen_interval_midpoint_s"
    ].max()

    center_time = (min_time + max_time) / 2

    # 选择最接近整炉时间中点的事件记录
    idx = (
        valid_time["oxygen_interval_midpoint_s"]
        - center_time
    ).abs().idxmin()

    event_time = float(
        valid_time.loc[
            idx, "oxygen_interval_midpoint_s"
        ]
    )

    # 根据中位数延迟计算分析仪时间
    analyzer_time = event_time + median_delay

    return {
        "event_time": event_time,
        "analyzer_time": analyzer_time,
        "delay": median_delay,
    }


def add_delay_arrow(ax, result, label_prefix):
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

    # Arrow label
    ax.text(
        (event_time + analyzer_time) / 2,
        arrow_y + span * 0.015,
        f"{label_prefix}: {delay:.1f} s",
        ha="center",
        va="bottom",
        fontsize=9
    )

    ax.set_ylim(
        y_bottom,
        arrow_y + span * 0.17
    )


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


def plot_one_heat(heat_number):

    heat_label = f"heat_{heat_number:02d}"
    heat_id = HEAT_ID_MAP[heat_number]

    data_file = DATA_DIR / f"{heat_label}_final_aligned.csv"

    out_png = OUT_DIR / (
        f"{heat_label}_alignment_example_v5.png"
    )

    out_pdf = OUT_DIR / (
        f"{heat_label}_alignment_example_v5.pdf"
    )

    if not data_file.exists():
        print(f"[跳过] 文件不存在: {data_file}")
        return

    df = load_and_prepare(data_file)

    # Time variables
    analyzer_time = df["time_s"]
    event_time = df["oxygen_interval_midpoint_s"]

    # Oxygen signal
    oxygen_flow = df["oxygen_lance_flow_m3_s"]

    # CO signals
    co_raw = df["co_flow_m3_s"]
    co_aligned = df["co_flow_final_aligned_m3_s"]

    # CO2 signals
    co2_raw = df["co2_flow_m3_s"]
    co2_aligned = df["co2_flow_final_aligned_m3_s"]

    # Model usage
    co_models = model_summary(df["co_model_used"])
    co2_models = model_summary(df["co2_model_used"])

    # Median delays
    co_median = df["co_delay_final_s"].median()
    co2_median = df["co2_delay_final_s"].median()

    # Representative observations
    co_example = representative_delay(
        df,
        "co_delay_final_s"
    )

    co2_example = representative_delay(
        df,
        "co2_delay_final_s"
    )

    # Labels based on model usage
    co_label = get_delay_label(
        df["co_model_used"],
        "CO"
    )

    co2_label = get_delay_label(
        df["co2_model_used"],
        "CO2"
    )

    # Create figure
    fig, axes = plt.subplots(
        3, 1,
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

    # =====================================
    # (a) Oxygen lance process signal
    # =====================================

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

    # =====================================
    # (b) CO signal alignment
    # =====================================

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
        co_label
    )

    axes[1].legend(
        loc="upper right",
        fontsize=9
    )

    # =====================================
    # (c) CO2 signal alignment
    # =====================================

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
        co2_label
    )

    axes[2].legend(
        loc="upper right",
        fontsize=9
    )

    # =====================================
    # Footer
    # =====================================

    footer = (
        f"Model usage - CO: {co_models} | CO2: {co2_models}\n"
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
        0.5,
        0.015,
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

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save PNG
    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight"
    )

    # Save PDF
    fig.savefig(
        out_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("=" * 60)
    print(f"Heat {heat_number:02d} completed")
    print("PNG:", out_png)
    print("PDF:", out_pdf)
    print("CO models:", co_models)
    print("CO2 models:", co2_models)
    print("CO representative:", co_example)
    print("CO2 representative:", co2_example)


def main():

    print("=" * 60)
    print("BATCH GENERATION: ALIGNMENT FIGURES V5")
    print("=" * 60)

    for heat_number in HEAT_NUMBERS:

        try:
            plot_one_heat(heat_number)

        except Exception as e:
            print(
                f"[错误] Heat {heat_number:02d} "
                f"生成失败: {e}"
            )

    print("=" * 60)
    print("Batch generation finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
