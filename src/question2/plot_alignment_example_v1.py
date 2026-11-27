from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

FILE = ROOT / "deliverables" / "question2" / "04_alignment" / "heat_01_final_aligned.csv"
OUT_PNG = ROOT / "deliverables" / "question2" / "04_alignment" / "heat_01_alignment_example_v1.png"
OUT_PDF = ROOT / "deliverables" / "question2" / "04_alignment" / "heat_01_alignment_example_v1.pdf"


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def main():
    df = pd.read_csv(FILE)

    # 数值化
    for col in [
        "time_s",
        "oxygen_interval_midpoint_s",
        "oxygen_lance_flow_m3_s",
        "co_flow_m3_s",
        "co2_flow_m3_s",
        "co_flow_final_aligned_m3_s",
        "co2_flow_final_aligned_m3_s",
        "co_delay_final_s",
        "co2_delay_final_s",
    ]:
        df[col] = to_numeric(df[col])

    # 时间轴
    analyzer_time = df["time_s"]
    event_time = df["oxygen_interval_midpoint_s"]

    # 信号
    oxygen_flow = df["oxygen_lance_flow_m3_s"]
    co_raw = df["co_flow_m3_s"]
    co_aligned = df["co_flow_final_aligned_m3_s"]
    co2_raw = df["co2_flow_m3_s"]
    co2_aligned = df["co2_flow_final_aligned_m3_s"]

    # 模型说明（取第一个非空值）
    co_delay = df["co_delay_final_s"].dropna().iloc[0]
    co2_delay = df["co2_delay_final_s"].dropna().iloc[0]
    co_model = df["co_model_used"].dropna().iloc[0]
    co2_model = df["co2_model_used"].dropna().iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    # 1) 氧枪流量
    axes[0].plot(event_time, oxygen_flow, linewidth=1.2)
    axes[0].set_title("Heat 01 Alignment Example (V1)")
    axes[0].set_ylabel("Oxygen lance flow")
    axes[0].set_xlabel("Event time / s")
    axes[0].grid(True, alpha=0.3)

    note = (
        f"CO delay = {co_delay:.1f} s ({co_model})\n"
        f"CO2 delay = {co2_delay:.1f} s ({co2_model})\n"
        "Aligned gas signals are mapped to oxygen-lance event time.\n"
        "Raw gas signals remain on analyzer observation time."
    )
    axes[0].text(
        0.99, 0.98, note,
        transform=axes[0].transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    # 2) CO
    axes[1].plot(analyzer_time, co_raw, label="CO raw (analyzer time)", linewidth=1.0)
    axes[1].plot(event_time, co_aligned, label="CO aligned (event time)", linewidth=1.2)
    axes[1].set_ylabel("CO flow")
    axes[1].set_xlabel("Time / s")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) CO2
    axes[2].plot(analyzer_time, co2_raw, label="CO2 raw (analyzer time)", linewidth=1.0)
    axes[2].plot(event_time, co2_aligned, label="CO2 aligned (event time)", linewidth=1.2)
    axes[2].set_ylabel("CO2 flow")
    axes[2].set_xlabel("Time / s")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()