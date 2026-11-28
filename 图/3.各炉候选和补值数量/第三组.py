# -*- coding: utf-8 -*-
"""第三组图

数据：../../01_heat_comparison.csv（q1_plot_data/ 下）
运行：python 第三组.py；可用 Q1_OUT 覆盖输出目录
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "01_heat_comparison.csv")
OUT = os.environ.get("Q1_OUT", HERE)
DPI = 200
TITLE = "第三组 · 各炉候选与补值数量"  # 图中不显示素材/配色来源文字

DEEPBLUE = "#2E75B6"   # chart4 深蓝：主结果
PURPLE = "#7C6CC0"     # chart3 紫：本地 RPCA
GREEN = "#548235"      # chart4 绿：两者共同
ORANGE = "#FAAC3C"     # chart10 橙
RED = "#FF0000"        # chart6 红：对照
SOFTRED = "#D8636F"    # chart9 柔红（仅用于结论框文字，保证可读）
PINK5 = "#F5D7EE"      # chart2 饼图顶部 5% 扇区粉红（彤彤指定）
BLUE_L = "#DEEBF7"     # chart1 浅蓝
BLUE_M = "#86AEDE"     # chart3 中蓝
PURPLE_L = "#E0DCF0"   # chart1 浅紫
GREEN_L = "#E2F0D9"    # chart1 浅绿
YELLOW_L = "#FFF2CC"   # chart1 浅黄
GRID = "#D9D9D9"       # PPT 网格色
BLACK = "#000000"
# chart1.xml 精确参数：gapWidth=100（柱宽=间隙）、边框 16478 EMU≈1.30pt、
# 数据标签 dLblPos=outEnd + showVal=1
BAR_W = 0.50           # 堆叠柱宽（gapWidth=100 → 0.5）
GRP_W = 0.25           # 分组柱单柱宽（每组总宽 0.5，两柱各 0.25）
EDGE_LW = 1.3          # 黑色描边宽度


def style_axes(ax, xlabel, ylabel):
    for sp in ax.spines.values():
        sp.set_color(BLACK); sp.set_linewidth(1.0); sp.set_visible(True)
    ax.tick_params(direction="in", top=True, right=True,
                   length=4, width=0.8, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6, ls="-", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)


def panel(ax, tag, title):
    head = "(%s) %s" % (tag, title) if tag else title
    ax.set_title(head, loc="left",
                 fontsize=10, fontweight="bold", pad=6)


def legend(ax, loc="upper left", fs=7.5):
    ax.legend(loc=loc, fontsize=fs, frameon=False,
              borderpad=0.3, handlelength=1.6)


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect, h_pad=1.5, w_pad=1.2) if rect else fig.tight_layout(h_pad=1.5, w_pad=1.2)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("已出图:", name, flush=True)


def main():
    df = pd.read_csv(SRC)
    x = np.arange(len(df))
    labs = [str(v) for v in df["attachment"]]
    w = GRP_W

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))

    # (a) 平滑版标记数构成：chart1 浅色填充 + 黑色描边
    ax = axes[0, 0]
    parts = [("teammate_hampel_only", "Hampel only", BLUE_L),
             ("teammate_rpca_only", "RPCA only", PURPLE_L),
             ("teammate_both", "Both", GREEN_L),
             ("teammate_segment", "Segment anomaly", YELLOW_L)]
    bottom = np.zeros(len(df))
    for col, name, color in parts:
        v = df[col].to_numpy(float)
        ax.bar(x, v, BAR_W, bottom=bottom, label=name, color=color,
               edgecolor=BLACK, lw=EDGE_LW)
        bottom += v
    ax.bar_label(ax.containers[-1], labels=["%d" % int(t) for t in bottom],
                 fontsize=6.5, padding=2)
    panel(ax, "a", "Smoothed-version flags (total 108,539)")
    style_axes(ax, "Furnace ID", "Flag count")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    legend(ax, loc="upper left", fs=7.5)

    # (b) 本地 RPCA 与整合版补值：chart3 紫 + chart4 深蓝
    ax = axes[0, 1]
    b1 = ax.bar(x - w / 2, df["local_rpca_spike"], w,
                label="Local RPCA only", color=PURPLE,
                edgecolor=BLACK, lw=EDGE_LW)
    b2 = ax.bar(x + w / 2, df["fused_spikes"], w,
                label="Fused final imputation", color=DEEPBLUE,
                edgecolor=BLACK, lw=EDGE_LW)
    ax.bar_label(b1, fmt="%d", fontsize=6.5, padding=1)
    ax.bar_label(b2, fmt="%d", fontsize=6.5, padding=1)
    panel(ax, "b", "Local RPCA vs fused imputations (separate scale)")
    style_axes(ax, "Furnace ID", "Count")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    legend(ax, loc="upper right")   # 避开第 1 炉 385 的高柱

    # (c) 限幅补值构成：深蓝 + chart3 中蓝
    ax = axes[1, 0]
    a = df["clipped_reconstructed"].to_numpy(float)
    b = df["clipped_lower_bound"].to_numpy(float)
    ax.bar(x, a, BAR_W, label="Hermite reconstruction (562)", color=DEEPBLUE,
           edgecolor=BLACK, lw=EDGE_LW)
    ax.bar(x, b, BAR_W, bottom=a, label="Observed lower bound (14,243)", color=BLUE_M,
           edgecolor=BLACK, lw=EDGE_LW)
    ax.bar_label(ax.containers[-1], labels=["%d" % int(t) for t in a + b],
                 fontsize=6.5, padding=2)
    panel(ax, "c", "Clipping imputation: Hermite reconstruction vs lower bound")
    style_axes(ax, "Furnace ID", "Saturation count")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    legend(ax, loc="upper left")

    # (d) 峰保真：chart6 红 + chart4 深蓝
    ax = axes[1, 1]
    t = df["teammate_peaks_with_at_least_20pct_loss"].to_numpy(float)
    f = df["fused_peaks_with_at_least_20pct_loss"].to_numpy(float)
    b1 = ax.bar(x - w / 2, t, w, label="Smoothed: >=20% loss", color=PINK5,
                edgecolor=BLACK, lw=EDGE_LW)
    ax.bar(x + w / 2, f, w, label="Fused: >=20% loss", color=DEEPBLUE,
           edgecolor=BLACK, lw=EDGE_LW)
    ax.bar_label(b1, fmt="%d", fontsize=6.5, padding=1)
    for xi in x:
        ax.text(xi + w / 2, t.max() * 0.015, "0", ha="center", va="bottom",
                fontsize=7, color=DEEPBLUE, fontweight="bold")
    ax.text(0.985, 0.93, "8,460 audited peaks: 1,648 vs 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=SOFTRED, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=BLACK, lw=0.7))
    panel(ax, "d", "Peaks with >=20% height loss")
    style_axes(ax, "Furnace ID", "Peak count")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    legend(ax, loc="upper left")

    save(fig, "G3_Candidate_and_Imputation_Counts.png", rect=[0, 0, 1, 1.0])

    # 单图：峰保真
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, t, w, label="Teammate smoothed", color=PINK5,
                edgecolor=BLACK, lw=EDGE_LW)
    ax.bar(x + w / 2, f, w, label="Fused (original sampled intensity preserved)", color=DEEPBLUE,
           edgecolor=BLACK, lw=EDGE_LW)
    ax.bar_label(b1, fmt="%d", fontsize=8, padding=2)
    for xi in x:
        ax.text(xi + w / 2, t.max() * 0.015, "0", ha="center", va="bottom",
                fontsize=9, color=DEEPBLUE, fontweight="bold")
    panel(ax, "", "Audited peaks with >=20% height loss")
    style_axes(ax, "Furnace ID (1-10)", "Peak count")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.text(0.985, 0.93, "Total 1,648 vs 0", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold", color=SOFTRED,
            bbox=dict(boxstyle="square,pad=0.4", fc="white", ec=BLACK, lw=0.8))
    legend(ax, loc="upper left", fs=9)
    save(fig, "G3_Peak_Fidelity_Height_Loss.png")


if __name__ == "__main__":
    main()
