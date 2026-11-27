# -*- coding: utf-8 -*-
"""
问题1：局部 Hampel + 鲁棒 PCA 融合去异常、连续重构、三维光谱图

方法（对应团队计划"方向建议"第 2、3 条，用两类证据选定）：
  1) 局部 Hampel（沿波长 + 沿时间双向，滑动中位数 + MAD）给出一路候选；
  2) 鲁棒 PCA（稳定 ADMM 版，惩罚按二阶差分估计的噪声尺度自适应）分解出
     低秩主体 L 与残差；|原始 - L| 的稳健 z 分数给出另一路候选，
     它能发现局部 MAD 被大波动抬高而漏检的孤立异常点；
  3) 两路候选各自做"邻域孤立性"复核：成片出现的判为真实谱峰并放回；
  4) 修复：RPCA 判定的点用低秩估计 L 修复（全局结构信息），
     仅局部 Hampel 判定的点用同列线性插值（局部信息）；
  5) 高斯平滑得到连续信号 S(t, λ)，并输出 10 炉三维曲面。

运行：python Q1_融合去异常与重构.py [--heats 1 2 3 ...]
环境：系统 Python 3.13（numpy / pandas / scipy / pyxlsb 均已装，无需安装任何包）
"""
import os
import re
import glob
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.ndimage import median_filter, uniform_filter, gaussian_filter

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 可重现参数 ====================
SEED = 2027
T_STEP = 0.5
W_LAM = 7                    # 波长方向局部窗口
W_T = 5                      # 时间方向局部窗口
Z_LAM = 5.0                  # 波长方向 z 阈值
Z_T = 5.0                    # 时间方向 z 阈值
RPCA_Z = 5.0                 # RPCA 残差 z 阈值
RPCA_ITER = 30               # 鲁棒 PCA 迭代轮数
RPCA_TOL = 1e-3              # 收敛阈值
SPIKE_NEIGH_MAX = 0.34       # 邻域候选比例上限，高于此视为成片真峰并放回
SEG_MIN_LEN = 3              # 成段异常最短连续帧数
SEG_Z = 4.0                  # 成段异常行统计量 z 阈值
SIGMA_T, SIGMA_L = 2.0, 3.0  # 高斯平滑带宽（时间帧, 波长点）
DOWN_T, DOWN_L = 20, 30      # 三维图降采样步长
AUX_COLS = ["累计耗氧量", "CO2浓度", "CO浓度"]
# ==============================================


def find_heat_files(data_dir):
    """自动发现 10 个附件，中文命名（附件N...）优先。"""
    found = {}
    for p in glob.glob(os.path.join(data_dir, "*.xlsb")):
        name = os.path.basename(p)
        m = re.search(r"Attachment\s*(\d+)", name, re.I) or re.search(r"附件\s*(\d+)", name)
        if not m:
            continue
        k = int(m.group(1))
        if 1 <= k <= 10 and (k not in found or name.startswith("附件")):
            found[k] = p
    return {k: found[k] for k in sorted(found)}


def load_heat(path):
    xls = pd.ExcelFile(path, engine="pyxlsb")
    df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], engine="pyxlsb")
    arr = df.to_numpy()
    seq = np.asarray(arr[:, 0]).astype(int)
    lam = np.asarray([float(c) for c in df.columns[1:-3]])
    D = np.asarray(arr[:, 1:-3], dtype=float)
    aux = df.iloc[:, -3:].copy()
    aux.columns = AUX_COLS
    return df, seq, lam, D, aux


# ---------------- 鲁棒 PCA（稳定 ADMM，噪声自适应惩罚） ----------------
def robust_pca(D, max_iter=RPCA_ITER, tol=RPCA_TOL):
    """返回 (低秩 L, 稀疏 S, 秩, 相对残差, 噪声尺度)。"""
    baseline = np.median(D, axis=0)
    data = D - baseline
    dn = norm(data)
    if dn == 0:
        return D.copy(), np.zeros_like(D), 0, 0.0, 0.0
    # 二阶差分的稳健标准差 -> 噪声水平（局部线性谱 + 等方差噪声假设）
    curv = np.diff(D, n=2, axis=1)
    noise = max(1.0, float(np.median(np.abs(curv - np.median(curv)))) * 1.4826 / np.sqrt(6))
    sparse_pen = 4.0 * noise
    rank_pen = sparse_pen * np.sqrt(max(data.shape))
    rho = 0.25
    low_rank = np.zeros_like(data)
    sparse = np.zeros_like(data)
    low_dual = np.zeros_like(data)
    sparse_dual = np.zeros_like(data)
    rank = 0
    res = 1.0
    for _ in range(max_iter):
        lt = low_rank - low_dual
        st = sparse - sparse_dual
        corr = (data - lt - st) / (2 + rho)
        lp = lt + corr
        sp_ = st + corr
        # numpy 的 LAPACK 绑定比 scipy.linalg.svd 快约 10 倍
        U, s, Vt = np.linalg.svd(lp + low_dual, full_matrices=False)
        keep = s > rank_pen / rho
        next_low = (U[:, keep] * (s[keep] - rank_pen / rho)) @ Vt[keep]
        shifted = sp_ + sparse_dual
        next_sparse = np.sign(shifted) * np.maximum(np.abs(shifted) - sparse_pen / rho, 0.0)
        low_dual += lp - next_low
        sparse_dual += sp_ - next_sparse
        pr = np.hypot(norm(lp - next_low), norm(sp_ - next_sparse))
        dr = rho * np.hypot(norm(low_rank - next_low), norm(sparse - next_sparse))
        res = float(max(pr, dr) / max(1.0, dn))
        low_rank, sparse = next_low, next_sparse
        rank = int(keep.sum())
        if res <= tol:
            break
    return low_rank + baseline, sparse, rank, res, noise


def rpca_residual_candidates(D, L, z_thr=RPCA_Z):
    """残差 |D - L| 的稳健 z 分数 -> RPCA 候选。"""
    resid = D - L
    med = np.median(resid, axis=0)
    mad = np.median(np.abs(resid - med), axis=0)
    scale = np.maximum(1.4826 * mad, max(1.0, float(np.median(1.4826 * mad))))
    z = np.abs(resid - med) / scale
    return z, z > z_thr


# ---------------- 局部检测 ----------------
def local_sigma(A, size):
    med = median_filter(A, size=size, mode="nearest")
    mad = median_filter(np.abs(A - med), size=size, mode="nearest")
    return 1.4826 * mad, med


def hampel_candidates(D):
    """沿波长 + 沿时间双向局部中位数/MAD。"""
    sw, mw = local_sigma(D, (1, W_LAM))
    cand_w = np.abs(D - mw) > Z_LAM * sw
    st, mt = local_sigma(D, (W_T, 1))
    cand_t = np.abs(D - mt) > Z_T * st
    return cand_w & cand_t


def isolated_review(cand):
    """邻域孤立性复核：成片出现的判为真实谱峰并放回。"""
    neigh = uniform_filter(cand.astype(float), size=(1, W_LAM), mode="nearest")
    return cand & (neigh <= SPIKE_NEIGH_MAX)


def segment_rows(D):
    """行中位数沿时间 Hampel，连续多帧整体偏离 -> 成段异常。"""
    row = np.median(D, axis=1)
    sr, mr = local_sigma(row, (W_T,))
    bad = np.abs(row - mr) > SEG_Z * sr
    rows = np.zeros(len(row), dtype=bool)
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            if j - i + 1 >= SEG_MIN_LEN:
                rows[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    seg = np.zeros_like(D, dtype=bool)
    seg[rows, :] = True
    return seg


def fuse(D):
    """两路候选 + 各自孤立性复核 + 成段检测。"""
    L, S, rank, res, noise = robust_pca(D)
    _, cand_r = rpca_residual_candidates(D, L)
    cand_h = hampel_candidates(D)

    h_iso = isolated_review(cand_h)
    r_iso = isolated_review(cand_r)
    both = h_iso & r_iso
    h_only = h_iso & ~r_iso
    r_only = r_iso & ~h_iso
    seg = segment_rows(D)
    mask = h_iso | r_iso | seg
    return mask, h_only, r_only, both, seg, L, S, rank, res, noise


def repair(D, mask, L, rpca_flag):
    """RPCA 判定的点用低秩估计修复；其余用同列线性插值。
    向量化实现：按列做线性插值，避免逐列 Python 循环。"""
    R = D.copy().astype(float)
    T, Lw = D.shape
    idx = np.arange(T)
    any_bad = mask.any(axis=0)
    if not any_bad.any():
        return R
    cols = np.where(any_bad)[0]
    # 收集所有需要修复的行位置，逐列向量化插值
    bad_idx = [np.where(mask[:, j])[0] for j in cols]
    for j, bidx in zip(cols, bad_idx):
        good = ~mask[:, j]
        if good.sum() == 0:
            R[:, j] = L[:, j]
            continue
        R[bidx, j] = np.interp(idx[bidx], idx[good], D[good, j])
    # RPCA 判定的点改用低秩估计（全局结构更可靠）
    use_low = mask & rpca_flag
    if use_low.any():
        R[use_low] = L[use_low]
    return R


def load_all_from_csv(out_dir):
    data = []
    for h in range(1, 11):
        p = os.path.join(out_dir, "附件%d_重构后.csv" % h)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        seq = df.iloc[:, 0].to_numpy().astype(int)
        lam = np.array([float(c) for c in df.columns[1:-3]])
        S = df.iloc[:, 1:-3].to_numpy(dtype=float)
        data.append((h, seq, lam, S))
    return data


def make_plot(all_data, out_dir):
    n = len(all_data)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(9 * ncols, 6.5 * nrows))
    for k, (h, seq, lam, S) in enumerate(all_data, start=1):
        tt = seq[::DOWN_T]
        ll = lam[::DOWN_L]
        zz = S[::DOWN_T, ::DOWN_L]
        TT, LL = np.meshgrid(tt, ll, indexing="ij")
        ax = fig.add_subplot(nrows, ncols, k, projection="3d")
        ax.plot_surface(TT, LL, zz, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("波长 / nm")
        ax.set_zlabel("光强")
        ax.set_title("炉次 %d : S(t, λ)" % h)
    fig.suptitle("全部 %d 炉连续三维光谱图（局部Hampel+鲁棒PCA融合去异常，连续重构）" % n,
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, "连续三维光谱图.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heats", nargs="+", type=int, default=list(range(1, 11)))
    args = ap.parse_args()

    np.random.seed(SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "Q1_results")
    os.makedirs(out_dir, exist_ok=True)

    print("numpy", np.__version__, "| pandas", pd.__version__)
    print("参数: Z_LAM=%.1f Z_T=%.1f RPCA_Z=%.1f RPCA_ITER=%d SIGMA=(%.1f,%.1f)" %
          (Z_LAM, Z_T, RPCA_Z, RPCA_ITER, SIGMA_T, SIGMA_L))

    heat_files = find_heat_files(here)
    t0 = time.time()
    for h in args.heats:
        if h not in heat_files:
            print("未找到附件", h)
            continue
        print("\n[%d] %s" % (h, os.path.basename(heat_files[h])), flush=True)
        df, seq, lam, D, aux = load_heat(heat_files[h])
        print("  矩阵 %d x %d" % D.shape, flush=True)

        mask, h_only, r_only, both, seg, L, S, rank, res, noise = fuse(D)
        n = mask.sum()
        print("  RPCA: 秩=%d 残差=%.2e 噪声=%.1f" % (rank, res, noise), flush=True)
        print("  异常 %d (%.3f%%) | 仅Hampel %d | 仅RPCA %d | 两者 %d | 成段 %d" %
              (n, 100.0 * n / mask.size, h_only.sum(), r_only.sum(), both.sum(), seg.sum()),
              flush=True)

        rpca_flag = r_only | both | seg
        R = repair(D, mask, L, rpca_flag)
        S_smooth = gaussian_filter(R, sigma=(SIGMA_T, SIGMA_L), mode="nearest")

        wave_names = [str(round(float(x), 3)) for x in lam]
        seq_col = df.iloc[:, :1].copy()
        seq_col.columns = ["序列"]
        fixed = pd.concat([
            seq_col.reset_index(drop=True),
            pd.DataFrame(np.round(S_smooth, 2), columns=wave_names),
            aux.reset_index(drop=True),
        ], axis=1)
        p1 = os.path.join(out_dir, "附件%d_重构后.csv" % h)
        fixed.to_csv(p1, index=False, encoding="utf-8-sig", float_format="%.2f")

        mask_df = pd.DataFrame(mask.astype(np.int8), columns=wave_names)
        mask_df.insert(0, "序列", seq)
        p2 = os.path.join(out_dir, "附件%d_异常掩码.csv" % h)
        mask_df.to_csv(p2, index=False, encoding="utf-8-sig")

        # 分类掩码：1=仅Hampel 2=仅RPCA 3=两者 4=成段
        cls = np.zeros(mask.shape, dtype=np.int8)
        cls[h_only] = 1
        cls[r_only] = 2
        cls[both] = 3
        cls[seg] = 4
        cls_df = pd.DataFrame(cls, columns=wave_names)
        cls_df.insert(0, "序列", seq)
        p3 = os.path.join(out_dir, "附件%d_异常分类掩码.csv" % h)
        cls_df.to_csv(p3, index=False, encoding="utf-8-sig")
        print("  已导出 3 个表格", flush=True)

    plot_data = load_all_from_csv(out_dir)
    fig_path = make_plot(plot_data, out_dir)
    print("\n三维图:", fig_path)
    print("结果目录:", out_dir)
    print("总耗时 %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
