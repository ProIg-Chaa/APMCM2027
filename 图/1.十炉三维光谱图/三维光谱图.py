# -*- coding: utf-8 -*-
"""
十炉连续三维光谱曲面

数据：07_surfaces/{附件}_{炉号}.csv
  首列 time_s = (原始序号-1)*0.5，第一帧 0 秒
  其余 2047 列列标题为波长(nm)，单元格为仪器强度计数
  时间网格不均匀（指定截面时刻已插入），波长各炉保留原生标定

运行：python 三维光谱图.py   （需 numpy / pandas / matplotlib）
输出：本文件夹下 10 张单炉曲面 + 1 张 2×5 拼版；可用 Q1_OUT 覆盖输出目录
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 可调参数 ====================
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "07_surfaces")
OUT = os.environ.get("Q1_OUT", HERE)
WAVE_STEP = 4        # 波长降采样步长（1=全 2047 通道，越大越快）
TIME_STEP = 1        # 时间降采样步长
ELEV, AZIM = 26, -58  # 视角：仰角、方位角
CMAP = "viridis"
DPI = 150
# ================================================


def load_surface(path):
    df = pd.read_csv(path)
    t = df["time_s"].to_numpy(dtype=float)
    lam = np.array([float(c) for c in df.columns[1:]], dtype=float)
    Z = df.iloc[:, 1:].to_numpy(dtype=float)
    return t, lam, Z


def draw_one(ax, t, lam, Z, title, colorbar=False, fig=None, show_title=False):
    tt = t[::TIME_STEP]
    ll = lam[::WAVE_STEP]
    zz = Z[::TIME_STEP, ::WAVE_STEP]
    X, Y = np.meshgrid(tt, ll, indexing="ij")   # X=时间(T,L), Y=波长(T,L)
    surf = ax.plot_surface(X, Y, zz, cmap=CMAP, linewidth=0,
                           antialiased=True, rcount=min(len(tt), 220),
                           ccount=min(len(ll), 220))
    ax.set_xlabel("Time / s", labelpad=8)
    ax.set_ylabel("Wavelength / nm", labelpad=8)
    ax.set_zlabel("Intensity / counts", labelpad=8)
    if show_title:
        ax.set_title(title, pad=10)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.tick_params(labelsize=8)
    if colorbar and fig is not None:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.12, label="Intensity")
    return surf


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.csv")))
    print("找到 %d 个曲面文件" % len(files), flush=True)

    # 1) 每炉一张单独高清图
    for f in files:
        base = os.path.basename(f).replace(".csv", "")
        t, lam, Z = load_surface(f)
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection="3d")
        draw_one(ax, t, lam, Z, "Furnace %s continuous 3D spectrum S(t, λ)" % base,
                 colorbar=True, fig=fig)
        fig.tight_layout()
        p = os.path.join(OUT, "Furnace_%s_3D_Spectrum.png" % base)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  已出图:", os.path.basename(p), "| t=%.1f–%.1f s, λ=%.3f–%.3f nm"
              % (t.min(), t.max(), lam.min(), lam.max()), flush=True)

    # 2) 十炉 2×5 拼版
    fig = plt.figure(figsize=(19, 13))
    for k, f in enumerate(files, start=1):
        base = os.path.basename(f).replace(".csv", "")
        t, lam, Z = load_surface(f)
        ax = fig.add_subplot(2, 5, k, projection="3d")
        draw_one(ax, t, lam, Z, "%s" % base, show_title=True)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    p = os.path.join(OUT, "Ten_Furnaces_3D_Spectral_Surfaces.png")
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("拼版:", p, flush=True)


if __name__ == "__main__":
    main()
