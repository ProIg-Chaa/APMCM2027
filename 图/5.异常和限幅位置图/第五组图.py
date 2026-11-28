# -*- coding: utf-8 -*-
"""第五组图


  · 波长计数分布.png：四个统计量的 10 炉叠加折线（均值用饼图浅色，单炉用绿色）
  · 异常位置_十炉.png：2×5 散点，按候选类别着色
  · 编码分面.png：复核点三个编码字段的分面散点

运行：python 第五组图.py
输出：本文件夹下 3 张 PNG；可用环境变量 Q1_OUT 覆盖输出目录。
"""
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("Q1_OUT", HERE)
DPI = 200

# ---------- 基础色 ----------
BLUE = "#1E77B4"
RED = "#D62728"
ORANGE = "#FF7F0E"
PURPLE = "#9467BD"
GREEN = "#2CA02C"
C_GRID = "#D9D9D9"

# ---------- 饼图色板（第四/五组均值线） ----------
PIE_BLUE = "#D8EAF4"
PIE_PURPLE = "#E0DCF0"
PIE_PEACH = "#F2CDB4"
PIE_PINK = "#F5D7EE"

# ---------- 单炉曲线（与浅色均值线区分） ----------
SINGLE_CURVE = "#548235"


def style_axes(ax, xlabel, ylabel, fs=9):
    """四边黑框 + 内向刻度 + 浅灰实线网格。"""
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.0)
        sp.set_visible(True)
    ax.tick_params(direction="in", top=True, right=True,
                   length=4, width=0.8, labelsize=8)
    ax.grid(True, color=C_GRID, lw=0.6, ls="-", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)


def panel(ax, tag, title, fs=10):
    head = "(%s) %s" % (tag, title) if tag else title
    ax.set_title(head, loc="left", fontsize=fs, fontweight="bold", pad=6)


def legend(ax, loc="upper right", ncol=1, fs=7.5):
    ax.legend(loc=loc, ncol=ncol, fontsize=fs, frameon=True,
              edgecolor="black", fancybox=False, borderpad=0.5,
              handlelength=1.6, columnspacing=1.0)


def save(fig, name, rect=None):
    if rect:
        fig.tight_layout(rect=rect, h_pad=1.5, w_pad=1.2)
    else:
        fig.tight_layout(h_pad=1.5, w_pad=1.2)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  已出图:", name, flush=True)


def csv(name):
    return pd.read_csv(os.path.join(DATA, name))


# =====================================================================
# 第五组：异常与限幅位置图
# =====================================================================
def g5():
    w = csv("05_wavelength_counts.csv")
    files = sorted(glob.glob(os.path.join(DATA, "06_review_points", "*.csv")))

    # ---------- 图 1：波长方向计数分布 2×2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))
    cols = [("teammate_flags", "Smoothed flags", PIE_PINK, "a"),
            ("local_rpca_spikes", "Local RPCA spikes", PIE_PURPLE, "b"),
            ("clipped", "Clipping saturation points", PIE_PEACH, "c"),
            ("clipped_reconstructed", "Hermite reconstructions", PIE_BLUE, "d")]
    NOTE = {
        "teammate_flags": "Spike at short wavelengths (~380 nm)",
        "local_rpca_spikes": "Concentrated in 550-700 nm",
        "clipped": "Clipping concentrated at 700-850 nm",
        "clipped_reconstructed": "Hermite reconstructions near ~770 nm",
    }
    for ax, (col, name, color, tag) in zip(axes.flat, cols):
        for h, g in w.groupby("attachment"):
            ax.plot(g["wavelength_nm"], g[col], color=SINGLE_CURVE, lw=0.8, alpha=0.42,
                    zorder=1)
        mean = w.groupby("wavelength_nm")[col].mean()
        ax.plot(mean.index.to_numpy(), mean.to_numpy(), color=color, lw=2.4,
                label="10-furnace mean", zorder=3)
        ax.plot([], [], color=SINGLE_CURVE, lw=0.8, alpha=0.8,
                label="Single furnace", zorder=1)
        panel(ax, tag, "Wavelength-wise counts: %s" % name)
        style_axes(ax, "Wavelength / nm", "Count")
        legend(ax, loc="lower right", fs=8)
        ax.text(0.985, 0.90, NOTE[col],
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#555555",
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="none", alpha=0.85))
    save(fig, "G5_Wavelength_Count_Distribution.png", rect=[0, 0, 1, 1.0])

    # ---------- 图 2：10 炉异常位置 2×5（按候选类别着色） ----------
    CLASS_COLOR = {0: "#BFBFBF", 1: BLUE, 2: ORANGE, 3: RED, 4: GREEN}
    CLASS_LABEL = {0: "Unflagged", 1: "Hampel only", 2: "RPCA only",
                   3: "Both", 4: "Segment anomaly"}
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.4))
    for i, (ax, f) in enumerate(zip(axes.flat, files)):
        d = pd.read_csv(f)
        for k in [0, 1, 2, 3, 4]:
            g = d[d.teammate_candidate_class == k]
            if len(g) == 0:
                continue
            ax.scatter(g["time_s"], g["wavelength_nm"], s=1.6,
                       color=CLASS_COLOR[k], alpha=0.55, lw=0,
                       label=CLASS_LABEL[k], rasterized=True, zorder=2)
        heat = os.path.basename(f).replace(".csv", "")
        panel(ax, chr(97 + i), "%s (%d points)" % (heat.split("_", 1)[-1], len(d)), fs=9)
        style_axes(ax, "Time / s", "Wavelength / nm", fs=8)
        ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], marker="o", lw=0, color=CLASS_COLOR[k],
                      markersize=6, label=CLASS_LABEL[k]) for k in [0, 1, 2, 3, 4]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, fontsize=9.5, frameon=True, edgecolor="black",
               fancybox=False, borderpad=0.5, columnspacing=1.6)
    save(fig, "G5_Anomaly_Locations_Ten_Furnaces.png", rect=[0, 0, 1, 0.94])

    # ---------- 图 3：三个编码字段分面（选含限幅的代表炉） ----------
    pick = None
    for f in files:
        d = pd.read_csv(f)
        if (d.clipping_status > 0).any():
            pick = (f, d)
            break
    if pick is None:
        pick = (files[0], pd.read_csv(files[0]))
    f, d = pick
    heat = os.path.basename(f).replace(".csv", "").split("_", 1)[-1]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    specs = [
        ("teammate_candidate_class", {0: "#BFBFBF", 1: BLUE, 2: ORANGE, 3: RED, 4: GREEN},
         ["Unflagged", "Hampel only", "RPCA only", "Both", "Segment"], "Candidate source"),
        ("spike_source", {0: "#BFBFBF", 1: PURPLE, 2: ORANGE},
         ["No imputation", "Local method", "Teammate candidate recheck"], "Imputation source"),
        ("clipping_status", {0: "#BFBFBF", 1: ORANGE, 2: BLUE},
         ["No clipping", "Observed lower bound", "Hermite estimate"], "Clipping handling"),
    ]
    for j, (ax, (col, cmap, names, title)) in enumerate(zip(axes, specs)):
        for k, c in cmap.items():
            g = d[d[col] == k]
            if len(g) == 0:
                continue
            ax.scatter(g["time_s"], g["wavelength_nm"], s=2.0, color=c,
                       alpha=0.6, lw=0, label=names[k], rasterized=True, zorder=2)
        panel(ax, chr(97 + j), "%s (Furnace %s)" % (title, heat))
        style_axes(ax, "Time / s", "Wavelength / nm", fs=9)
        legend(ax, loc="upper right", fs=8)
    save(fig, "G5_Encoding_Facets.png", rect=[0, 0, 1, 1.0])


if __name__ == "__main__":
    print("=== 第五组 ===", flush=True)
    g5()
    print("全部完成，输出目录:", OUT, flush=True)
