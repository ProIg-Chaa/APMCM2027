# -*- coding: utf-8 -*-
"""第四组图



运行：python 第四组图.py
输出：本文件夹下 4 张 PNG；可用环境变量 Q1_OUT 覆盖输出目录。
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
DATA = os.path.dirname(os.path.dirname(HERE))
OUT = os.environ.get("Q1_OUT", HERE)
DPI = 200

# ---------- 基础色 ----------
GREY = "#7F7F7F"
BLUE = "#1E77B4"      # 整合版参考竖线
C_GRID = "#D9D9D9"

# ---------- 截图1 饼图色板（第四组柱状图统一用色） ----------
PIE_BLUE = "#D8EAF4"    # 44%
PIE_YELLOW = "#FFF2CC"  # 28%
PIE_PURPLE = "#E0DCF0"  # 11%
PIE_GREEN = "#E2F0D9"   # 7%
PIE_PINK = "#F5D7EE"    # 5%
PIE_PEACH = "#F2CDB4"   # 5%

# ---------- 截图2 折线图系列色 ----------
LINE_RED = "#D8636F"
LINE_PURPLE = "#6B23A4"
LINE_ORANGE = "#FAAC3C"
LINE_YELLOW = "#EDE532"

METHOD_ORDER = ["observed", "teammate_repair_only", "teammate_gaussian",
                "local_rpca", "fused_dual_confirmed", "fused_neighbor_agreement"]
METHOD_LABEL = {
    "observed": "Observed (raw)",
    "teammate_repair_only": "Teammate repair only",
    "teammate_gaussian": "Teammate full-map smoothing",
    "local_rpca": "Local robust PCA",
    "fused_dual_confirmed": "Fused (dual confirmed)",
    "fused_neighbor_agreement": "Fused (neighbor agreement)",
}
METHOD_COLOR = dict(zip(METHOD_ORDER,
                        [PIE_YELLOW, PIE_PEACH, PIE_PINK,
                         PIE_PURPLE, PIE_BLUE, PIE_GREEN]))
# 折线图按截图2的“纯色 + 实/虚线 + 不同 marker”方式区分六个方法
LINE_STYLE = [
    (LINE_RED, "^", "-"),     # observed
    (LINE_PURPLE, "o", "-"),  # teammate_repair_only
    (LINE_ORANGE, "s", "-"),  # teammate_gaussian
    (LINE_YELLOW, "D", "-"),  # local_rpca
    (LINE_RED, "^", "--"),    # fused_dual_confirmed
    (LINE_PURPLE, "o", "--"), # fused_neighbor_agreement
]
METHOD_SHORT = ["Observed", "Teammate repair only", "Teammate full-map smoothing",
                "Local robust PCA", "Fused (dual confirmed)", "Fused (neighbor agreement)"]


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


def style_line_axes(ax, xlabel, ylabel, fs=10):
    """截图2 折线图式样：四边细黑框、内向刻度、仅水平浅灰网格、加粗刻度字。"""
    style_axes(ax, xlabel, ylabel, fs=fs)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color=C_GRID, lw=0.8, ls="-", alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, width=1.0, length=4)
    ax.set_xlabel(xlabel, fontsize=fs, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fs, fontweight="bold")
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


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
# 第四组：定量对照
# =====================================================================
def g4():
    s = csv("02_spike_injection_summary.csv")
    byp = csv("02_spike_injection_by_patch.csv")
    clip = csv("02_clipping_holdout.csv")
    interp = csv("04_interpolation_holdout.csv")
    peak = csv("04_peak_preservation.csv")

    # ---------- 图 1：注入实验 2×2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    amps = ["0.15", "0.35", "0.65"]
    xa = np.arange(3)
    w6 = 0.13

    # (a) 注入幅度 × 召回率（comparison 集，6 方法分组柱）
    ax = axes[0, 0]
    comp = s[(s.split == "comparison") & (s.amplitude_fraction.isin(amps))]
    for k, m in enumerate(METHOD_ORDER):
        g = comp[comp.method == m].set_index("amplitude_fraction")
        v = [g.loc[a, "recall"] if a in g.index else 0 for a in amps]
        ax.bar(xa + (k - 2.5) * w6, v, w6, label=METHOD_LABEL[m],
               color=METHOD_COLOR[m], edgecolor="black", lw=0.8)
    panel(ax, "a", "Recall of injected spikes (comparison set)")
    style_axes(ax, "Injection amplitude α (value x(1+α))", "Recall")
    ax.set_xticks(xa); ax.set_xticklabels(amps)
    ax.set_ylim(0, 1.12)
    legend(ax, loc="upper left", ncol=2, fs=7)

    # (b) 逐窗口召回率箱线图
    ax = axes[0, 1]
    byp = byp.copy()
    byp["recall"] = byp["detected"] / byp["injected"]
    groups = [byp[byp.method == m]["recall"].to_numpy(float)
              for m in METHOD_ORDER]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="black", lw=0.9),
                    capprops=dict(color="black", lw=0.9),
                    flierprops=dict(marker="o", ms=2.5, mfc="white",
                                    mec=GREY, alpha=0.6))
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLOR[m])
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_alpha(1.0)
    panel(ax, "b", "Recall distribution across windows (540 patches)")
    style_axes(ax, "Method", "Recall")
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels(METHOD_SHORT, rotation=22, ha="right", fontsize=7.5)
    ax.set_ylim(-0.03, 1.08)

    # (c) 注入点修复误差 vs 控制点扰动（split=all, amp=all）
    ax = axes[1, 0]
    allrow = s[(s.split == "all") & (s.amplitude_fraction == "all")]
    xm = np.arange(6); w2 = 0.36
    inj = [float(allrow[allrow.method == m]["injected_mae"].iloc[0]) for m in METHOD_ORDER]
    ctl = [float(allrow[allrow.method == m]["control_mae"].iloc[0]) for m in METHOD_ORDER]
    b1 = ax.bar(xm - w2 / 2, inj, w2, label="Injected MAE", color=PIE_PEACH,
                edgecolor="black", lw=0.8)
    b2 = ax.bar(xm + w2 / 2, ctl, w2, label="Control MAE", color=PIE_BLUE,
                edgecolor="black", lw=0.8)
    ax.bar_label(b1, fmt="%d", fontsize=6.5, padding=1)
    ax.bar_label(b2, fmt="%d", fontsize=6.5, padding=1)
    panel(ax, "c", "Repair error: injected vs unflagged control points")
    style_axes(ax, "Method", "MAE / counts")
    ax.set_xticks(xm)
    ax.set_xticklabels(METHOD_SHORT, rotation=22, ha="right", fontsize=7.5)
    legend(ax, loc="upper right")

    # (d) 未标记控制点被标记比例（split=all, amp=all）
    ax = axes[1, 1]
    frac = [float(allrow[allrow.method == m]["control_flag_fraction"].iloc[0]) * 100
            for m in METHOD_ORDER]
    b = ax.bar(xm, frac, 0.55, color=[METHOD_COLOR[m] for m in METHOD_ORDER],
               edgecolor="black", lw=0.8)
    ax.bar_label(b, fmt="%.3f", fontsize=7, padding=2)
    panel(ax, "d", "Flagged fraction of unflagged control points")
    style_axes(ax, "Method", "Flagged fraction / %")
    ax.set_xticks(xm)
    ax.set_xticklabels(METHOD_SHORT, rotation=22, ha="right", fontsize=7.5)
    ax.text(0.985, 0.93, "Control points flagged\nFull smoothing 0.477% vs fused 0.003%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
            color="#B03A2E", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="black", lw=0.7))

    save(fig, "G4_Spike_Injection_and_Method_Comparison.png", rect=[0, 0, 1, 1.0])

    # 单图：召回率（放大版）—— 按截图2的折线图格式
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for k, m in enumerate(METHOD_ORDER):
        g = comp[comp.method == m].set_index("amplitude_fraction")
        v = [g.loc[a, "recall"] if a in g.index else 0 for a in amps]
        c, mk, ls = LINE_STYLE[k]
        ax.plot(xa, v, color=c, lw=1.8, ls=ls, marker=mk, ms=6.5,
                mec=c, mfc=c, mew=0.8, label=METHOD_LABEL[m])
    style_line_axes(ax, "Injection amplitude α", "Recall / %", fs=11)
    ax.set_xticks(xa); ax.set_xticklabels(amps)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%d" % int(t * 100) for t in ticks])
    ax.set_ylim(0, 1.08)
    # 截图2中的竖向虚线参考线（此处取中间幅度 α=0.35）
    ax.axvline(1.0, color="#A6A6A6", lw=1.1, ls="--", zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=8.5, frameon=True, edgecolor="black", fancybox=False,
              borderpad=0.5, handlelength=2.2, columnspacing=1.2)
    fig.text(0.5, 0.01, "Figure 4: Effect of injection amplitude on spike recall (comparison set)",
             ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 0.88], h_pad=1.2, w_pad=1.2)
    p = os.path.join(OUT, "G4_Injection_Amplitude_vs_Recall.png")
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  已出图:", "G4_Injection_Amplitude_vs_Recall.png", flush=True)

    # ---------- 图 2：限幅重建 1×2 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    wl = ["1–4", "5–8", "9–16", "17–32", "33–64", "65–128"]
    for ax, sp, tag in zip(axes, ["calibration", "holdout"], ["a", "b"]):
        d = clip[clip.split == sp].dropna(subset=["flat_mae", "hermite_mae"])
        xx = np.arange(len(d)); w2 = 0.36
        b1 = ax.bar(xx - w2 / 2, d["flat_mae"], w2, label="Flat extrapolation",
                    color=PIE_PEACH, edgecolor="black", lw=0.8)
        b2 = ax.bar(xx + w2 / 2, d["hermite_mae"], w2, label="Hermite arc reconstruction",
                    color=PIE_BLUE, edgecolor="black", lw=0.8)
        ax.bar_label(b1, fmt="%d", fontsize=6.5, padding=1)
        ax.bar_label(b2, fmt="%d", fontsize=6.5, padding=1)
        panel(ax, tag, "Clipping plateau reconstruction MAE (%s set)" % sp)
        style_axes(ax, "Plateau width / channels", "MAE / counts")
        ax.set_xticks(xx); ax.set_xticklabels([wl[i] for i in range(len(d))])
        legend(ax, loc="upper left")
    save(fig, "G4_Clipping_Plateau_Reconstruction.png", rect=[0, 0, 1, 1.0])

    # ---------- 图 3：插值与谱峰保真 2×2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    # (a)(b) 插值：时间轴 / 波长轴分开（量级差 ~16 倍）
    for ax, gap, tag, title in zip(
            axes[0], ["isolated_frames", "three_channel_blocks"], ["a", "b"],
            ["Temporal interpolation (isolated frames, MAE≈1900)", "Spectral interpolation (3-channel blocks, MAE≈120)"]):
        d = interp[interp.gap == gap].sort_values("attachment")
        xx = np.arange(len(d)); w2 = 0.36
        b1 = ax.bar(xx - w2 / 2, d["pchip_mae"], w2, label="PCHIP",
                    color=PIE_BLUE, edgecolor="black", lw=0.8)
        ax.bar(xx + w2 / 2, d["linear_mae"], w2, label="Linear interpolation",
               color=PIE_PEACH, edgecolor="black", lw=0.8)
        panel(ax, tag, title)
        style_axes(ax, "Furnace ID", "MAE / counts")
        ax.set_xticks(xx); ax.set_xticklabels(d["attachment"].astype(str))
        legend(ax, loc="upper left")

    # (c) 峰高保持率分布
    ax = axes[1, 0]
    rt = (peak["teammate_peak_intensity"] / peak["observed_peak_intensity"]).to_numpy(float)
    bins = np.linspace(0.4, 2.0, 65)
    ax.hist(rt, bins=bins, histtype="stepfilled", color=PIE_PINK, alpha=0.9,
            label="Teammate smoothed", edgecolor=LINE_RED, lw=0.8)
    # 整合版是 delta 分布（全部严格 =1.0），画直方会把纵轴撑到 8460 压扁对照，
    # 改用竖线标注，两条证据才都看得清
    ax.axvline(1.0, color=BLUE, lw=2.2,
               label="Fused (all 8,460 peaks = 1.0)")
    ax.text(0.985, 0.93, "Smoothed: 19.5% of peaks drop >=20%\nFused: 0%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#B03A2E", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="black", lw=0.7))
    panel(ax, "c", "Peak height retention distribution (8,460 peaks)")
    style_axes(ax, "Reconstructed / observed peak height", "Frequency")
    ax.set_xlim(0.4, 2.0)
    legend(ax, loc="upper left")

    # (d) 峰位偏移分布
    ax = axes[1, 1]
    dt = np.abs(peak["teammate_peak_nm"] - peak["observed_peak_nm"]).to_numpy(float)
    bins2 = np.linspace(0, 3, 61)
    ax.hist(dt, bins=bins2, histtype="stepfilled", color=PIE_PINK, alpha=0.9,
            label="Teammate smoothed", edgecolor=LINE_RED, lw=0.8)
    ax.axvline(0, color=BLUE, lw=2.2, label="Fused (shift always 0)")
    ax.text(0.985, 0.62, "Smoothed median shift 0.35 nm\n5% of peaks shift >1 nm",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#B03A2E", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.35", fc="white", ec="black", lw=0.7))
    panel(ax, "d", "Peak position shift distribution (8,460 peaks)")
    style_axes(ax, "|Reconstructed - observed peak position| / nm", "Frequency")
    ax.set_xlim(-0.06, 3)
    legend(ax, loc="upper right")

    save(fig, "G4_Interpolation_and_Peak_Fidelity.png", rect=[0, 0, 1, 1.0])


if __name__ == "__main__":
    print("=== 第四组 ===", flush=True)
    g4()
    print("全部完成，输出目录:", OUT, flush=True)
