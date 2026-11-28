# -*- coding: utf-8 -*-
"""


数据：../../03_spectral_sections.csv（q1_plot_data/ 下）
运行：python 截面图.py   （需 numpy / pandas / matplotlib）
输出：本文件夹下 5 张单图 + 1 张 2×3 拼版；可用 Q1_OUT 覆盖输出目录
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.linewidth"] = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "03_spectral_sections.csv")
OUT = os.environ.get("Q1_OUT", HERE)
SAT = 65535
SHOW_LOCAL = False
DPI = 200

C_FUSED = "#1E77B4"   # tab10 蓝：整合版（主结果）
C_TEAM = "#D62728"    # tab10 红：平滑版（对照）
C_OBS = "#3C3C3C"     # 深灰：原测散点（空心圆）
C_FIX = "#F9AC3B"     # 素材1 橙：被补值的尖刺
C_LOCAL = "#7B6CC0"   # 素材1 紫：本地版（可选）
C_SAT = "#808080"
C_GRID = "#D9D9D9"
C_HERM = "#EDF1F7"    # Hermite 段极淡底
C_LOWER = "#F7F3EA"   # 下界段极淡底

ORDER = ["isolated_spike", "coherent_peak", "teammate_supported_spike",
         "short_clipping", "long_clipping"]
LABEL = ["a", "b", "c", "d", "e"]
TITLE = {
    "isolated_spike": "Isolated spike imputation",
    "coherent_peak": "Coherent peak fidelity",
    "teammate_supported_spike": "Spike imputation supported by teammate mask",
    "short_clipping": "Short plateau reconstruction (Hermite arc)",
    "long_clipping": "Long plateau: preserved observed lower bound",
}


def style_axes(ax):
    """四边黑框 + 内向刻度 + 浅灰实线网格。"""
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(1.0)
        sp.set_visible(True)
    ax.tick_params(direction="in", top=True, right=True,
                   length=4, width=0.8, labelsize=8.5)
    ax.grid(True, color=C_GRID, lw=0.6, ls="-", alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("Wavelength / nm", fontsize=9.5)
    ax.set_ylabel("Intensity / counts", fontsize=9.5)


def draw(ax, g, ex, tag="", compact=False):
    w = g["wavelength_nm"].to_numpy(dtype=float)
    obs = g["observed"].to_numpy(dtype=float)
    fused = g["fused"].to_numpy(dtype=float)
    team = g["teammate_smoothed"].to_numpy(dtype=float)
    local = g["local_complete"].to_numpy(dtype=float)
    src = g["spike_source"].to_numpy()
    clip = g["clipping_status"].to_numpy()
    n = len(w)
    every = max(1, n // 14)          # 每条线约 14 个 marker，避免长平顶糊成一片

    herm = clip == 2
    if herm.any():
        ax.axvspan(w[herm].min(), w[herm].max(), color=C_HERM, zorder=0)
    lower = clip == 1
    if lower.any() and lower.mean() < 0.7:
        ax.axvspan(w[lower].min(), w[lower].max(), color=C_LOWER, zorder=0)


    ax.plot(w, fused, color=C_FUSED, lw=1.7, marker="s", ms=4.2,
            markevery=every, label="Fused", zorder=3)
    ax.plot(w, team, color=C_TEAM, lw=1.4, ls="--", marker="^", ms=4.4,
            markevery=every, label="Teammate smoothed", zorder=2)
    if SHOW_LOCAL:
        ax.plot(w, local, color=C_LOCAL, lw=1.0, ls=":", marker="D", ms=3.4,
                markevery=every, label="Local", zorder=2)


    ax.scatter(w, obs, s=16 if not compact else 11, facecolors="white",
               edgecolors=C_OBS, linewidths=0.8, label="Observed", zorder=4)

    # 补值点：橙色空心圈 + 竖线连接原值与补回值
    fixed = src > 0
    if fixed.any():
        ax.scatter(w[fixed], obs[fixed], s=80 if not compact else 48,
                   facecolors="none", edgecolors=C_FIX, linewidths=1.5,
                   label="Imputed spike", zorder=5)
        for i in np.where(fixed)[0]:
            ax.plot([w[i], w[i]], [obs[i], fused[i]], color=C_FIX,
                    lw=0.9, alpha=0.8, zorder=2)

    # 饱和上限：只画参考线，说明统一放图例，避免图内文字与曲线/标题重叠
    if (obs == SAT).any() or (clip > 0).any():
        ax.axhline(SAT, color=C_SAT, lw=1.0, ls="-.", zorder=1)

    # 子图标题放在画框外上方（避免长平顶数据贴顶时把标题盖住）
    heat = str(g["heat"].iloc[0]); t = float(g["time_s"].iloc[0])
    if tag:
        ax.set_title("(%s) %s" % (tag, TITLE.get(ex, ex)),
                     loc="left", fontsize=10.5 if not compact else 9.5,
                     fontweight="bold", pad=6)
    ax.text(0.985, 0.035, "Furnace %s | t = %.1f s" % (heat, t),
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
            color="#555555")
    style_axes(ax)

    # 长平顶：大片空白区加一句注记，说明为何不重建峰高
    if ex == "long_clipping":
        ax.text(0.5, 0.52,
                "640 / 645 observations at the 65,535 lower bound\nFused preserves observed values; true peak height not estimated",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, linespacing=1.5, color="#333333",
                bbox=dict(boxstyle="square,pad=0.5", fc="white",
                          ec="black", lw=0.8))

    # 谱峰保真结论框（核心证据）：白底黑框，贴近论文注记
    if ex == "coherent_peak":
        msg = ("Observed peak %.0f\nFused %.0f (100%% preserved)\nSmoothed %.0f (%.0f%% lower)"
               % (obs.max(), fused.max(), team.max(),
                  100 * (1 - team.max() / obs.max())))
        ax.text(0.02, 0.96, msg, transform=ax.transAxes, fontsize=8.2,
                va="top", linespacing=1.5,
                bbox=dict(boxstyle="square,pad=0.45", fc="white",
                          ec="black", lw=0.8, alpha=0.92))


def legend_handles():
    h = [
        Line2D([0], [0], marker="o", mfc="white", mec=C_OBS, lw=0,
               markersize=6, label="Observed"),
        Line2D([0], [0], color=C_FUSED, lw=1.7, marker="s", ms=4.2,
               label="Fused (primary result)"),
        Line2D([0], [0], color=C_TEAM, lw=1.4, ls="--", marker="^", ms=4.4,
               label="Teammate smoothed"),
        Line2D([0], [0], marker="o", mfc="none", mec=C_FIX, lw=0,
               markersize=7, label="Imputed spike (spike_source>0)"),
        Line2D([0], [0], color=C_SAT, lw=1.0, ls="-.", label="Saturation limit 65,535"),
    ]
    if SHOW_LOCAL:
        h.append(Line2D([0], [0], color=C_LOCAL, lw=1.0, ls=":", marker="D",
                        ms=3.4, label="Local"))
    return h


def main():
    df = pd.read_csv(SRC)
    print("共 %d 行，example: %s" % (len(df), df["example"].unique().tolist()), flush=True)

    # 1) 五张单图
    for i, ex in enumerate(ORDER):
        g = df[df["example"] == ex].sort_values("wavelength_nm")
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        draw(ax, g, ex, tag=LABEL[i])
        ax.legend(handles=legend_handles(), loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5,
                  frameon=True, edgecolor="black", fancybox=False,
                  borderpad=0.6, columnspacing=1.2)
        fig.subplots_adjust(bottom=0.24, top=0.93, left=0.10, right=0.97)
        p = os.path.join(OUT, "Section_%s.png" % ex)
        fig.savefig(p, dpi=DPI, facecolor="white")
        plt.close(fig)
        print("  已出图:", os.path.basename(p), "| %d 点" % len(g), flush=True)

    # 2) 2×3 拼版
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    for ax, ex, tag in zip(axes.flat[:5], ORDER, LABEL):
        g = df[df["example"] == ex].sort_values("wavelength_nm")
        draw(ax, g, ex, tag=tag, compact=True)

    last = axes.flat[5]
    last.axis("off")
    last.text(0.02, 0.92,
              "How to read\n\n"
              "- Open circles = raw instrument observations\n"
              "- Blue squares (solid) = fused; passes exactly through observed points where no imputation\n"
              "- Red triangles (dashed) = teammate full-map smoothing; peak heights systematically lowered\n"
              "- Orange circles = confirmed and imputed isolated spikes\n"
              "- Plateaus = 65,535 saturation readings (observed lower bound, not true peak height)\n"
              "- Light blue band = Hermite reconstruction; light beige band = preserved lower bound\n\n"
              "Source: 03_spectral_sections.csv\n"
              "Time t=(n-1)*0.5 s; first frame at 0 s\n"
              "Each furnace keeps its native wavelength axis (two calibrations not unified)",
              transform=last.transAxes, fontsize=9.5, va="top", linespacing=1.55,
              bbox=dict(boxstyle="square,pad=0.6", fc="white", ec="black", lw=0.9))

    fig.legend(handles=legend_handles(), loc="upper center",
               bbox_to_anchor=(0.5, 0.998), ncol=5, fontsize=9.5,
               frameon=True, edgecolor="black", fancybox=False,
               borderpad=0.55, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.875], h_pad=1.6, w_pad=1.2)
    p = os.path.join(OUT, "Local_Sections_Spike_Imputation_and_Peak_Fidelity.png")
    fig.savefig(p, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("拼版:", p, flush=True)


if __name__ == "__main__":
    main()
