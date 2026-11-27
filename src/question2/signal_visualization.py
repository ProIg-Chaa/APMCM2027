
"""
APMCM2026 Problem B - Question 2
原始过程信号可视化

读取十炉 XLSB 数据，绘制：
1. 累积耗氧量
2. CO2 瞬时流量
3. CO 瞬时流量

不修改原始数据，不进行异常值处理。
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 路径与参数
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "question2"
    / "visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.5


# =========================
# 2. 中文绘图设置
# =========================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial"
]

plt.rcParams["axes.unicode_minus"] = False


# =========================
# 3. 文件排序
# =========================

def natural_sort_key(path):
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[0]) if numbers else 999


# =========================
# 4. 数据读取
# =========================

def load_heat(file_path):

    df = pd.read_excel(
        file_path,
        engine="calamine"
    )

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 4:
        raise ValueError(
            f"{file_path.name} 的数据列数不足。"
        )

    # 根据官方数据结构确定字段位置
    sequence_col = df.columns[0]
    oxygen_col = df.columns[-3]
    co2_col = df.columns[-2]
    co_col = df.columns[-1]

    data = pd.DataFrame({
        "sequence": pd.to_numeric(
            df[sequence_col], errors="coerce"
        ),
        "oxygen": pd.to_numeric(
            df[oxygen_col], errors="coerce"
        ),
        "co2": pd.to_numeric(
            df[co2_col], errors="coerce"
        ),
        "co": pd.to_numeric(
            df[co_col], errors="coerce"
        )
    })

    data = data.dropna(subset=["sequence"])

    # 时间从该炉第一个有效采样点开始
    data["time"] = (
        data["sequence"] - data["sequence"].iloc[0]
    ) * DT

    return data


# =========================
# 5. 绘制单炉三联图
# =========================

def plot_one_heat(file_path, data, heat_number):

    fig, axes = plt.subplots(
        3, 1,
        figsize=(13, 10),
        sharex=True
    )

    time = data["time"]

    # 图1：累积耗氧量
    axes[0].plot(
        time,
        data["oxygen"],
        linewidth=1
    )

    axes[0].set_title("Cumulative Oxygen Consumption")
    axes[0].set_ylabel("Oxygen (m³)")
    axes[0].grid(alpha=0.3)

    # 图2：CO2
    axes[1].plot(
        time,
        data["co2"],
        linewidth=1
    )

    axes[1].set_title("CO₂ Instantaneous Volume Flow")
    axes[1].set_ylabel("CO₂ (m³/s)")
    axes[1].grid(alpha=0.3)

    # 图3：CO
    axes[2].plot(
        time,
        data["co"],
        linewidth=1
    )

    axes[2].set_title("CO Instantaneous Volume Flow")
    axes[2].set_ylabel("CO (m³/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"Heat {heat_number} - {file_path.stem}",
        fontsize=14
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR /
        f"heat_{heat_number:02d}_raw_signals.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"已保存：{output_file.name}")


# =========================
# 6. 第一炉与第五炉对比
# =========================

def plot_comparison(data_1, data_5):

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9)
    )

    datasets = [
        (data_1, "Heat 1"),
        (data_5, "Heat 5")
    ]

    signals = [
        ("oxygen", "Cumulative Oxygen", "m³"),
        ("co2", "CO₂ Flow", "m³/s"),
        ("co", "CO Flow", "m³/s")
    ]

    for row, (data, heat_name) in enumerate(datasets):

        for col, (key, title, unit) in enumerate(signals):

            ax = axes[row, col]

            ax.plot(
                data["time"],
                data[key],
                linewidth=0.9
            )

            ax.set_title(f"{heat_name}: {title}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Comparison of Raw Process Signals: Heat 1 vs Heat 5",
        fontsize=15
    )

    fig.tight_layout()

    output_file = OUTPUT_DIR / "heat_01_vs_heat_05.png"

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"已保存：{output_file.name}")


# =========================
# 7. 主程序
# =========================

def main():

    files = sorted(
        RAW_DIR.glob("Attachment*.xlsb"),
        key=natural_sort_key
    )

    if len(files) != 10:
        raise ValueError(
            f"预期十炉数据，实际找到 {len(files)} 个文件。"
        )

    print("=" * 55)
    print("开始绘制十炉原始过程信号")
    print("=" * 55)

    all_data = {}

    for index, file_path in enumerate(files, start=1):

        print(f"\n读取第 {index} 炉：{file_path.name}")

        data = load_heat(file_path)

        all_data[index] = data

        print(
            f"采样点：{len(data)}，"
            f"时间范围：{data['time'].min():.1f} "
            f"- {data['time'].max():.1f} s"
        )

        plot_one_heat(
            file_path,
            data,
            index
        )

    # 第一炉和第五炉对比
    plot_comparison(
        all_data[1],
        all_data[5]
    )

    print("\n" + "=" * 55)
    print("全部绘图完成！")
    print(f"图片保存目录：{OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
