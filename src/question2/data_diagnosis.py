
"""
APMCM2026 Problem B - Question 2
十炉原始数据读取与质量诊断

功能：
1. 批量读取 data/raw 下的十个 XLSB 文件
2. 检查数据维度、列结构、缺失值和重复采样序号
3. 检查累积耗氧量的单调性
4. 对过程信号的一阶差分进行稳健异常候选检测
5. 输出诊断汇总 CSV，不修改原始数据
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


# =========================
# 1. 路径与参数
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "question2"
    / "data_diagnosis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_SPECTRAL_COLUMNS = 2047
SAMPLING_INTERVAL = 0.5

# 一阶差分稳健异常候选阈值
ROBUST_Z_THRESHOLD = 8.0


# =========================
# 2. 文件排序
# =========================

def natural_sort_key(path):
    """按照 Attachment 编号自然排序。"""
    numbers = re.findall(r"\d+", path.name)
    return int(numbers[0]) if numbers else 999


# =========================
# 3. 稳健异常候选检测
# =========================

def detect_difference_spikes(series):
    """
    基于一阶差分和 MAD 检测突变候选。
    仅用于诊断，不直接删除或修复数据。
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy()
    diff = np.diff(values)

    valid = np.isfinite(diff)

    if valid.sum() < 5:
        return 0

    valid_diff = diff[valid]

    median = np.median(valid_diff)
    mad = np.median(np.abs(valid_diff - median))

    if mad < 1e-12:
        return 0

    robust_z = (
        0.6745 * np.abs(valid_diff - median) / mad
    )

    return int(np.sum(robust_z > ROBUST_Z_THRESHOLD))


# =========================
# 4. 单炉数据诊断
# =========================

def diagnose_one_file(file_path):

    print(f"\n正在读取：{file_path.name}")

    try:
        df = pd.read_excel(
            file_path,
            engine="pyxlsb"
        )
    except Exception as exc:
        raise RuntimeError(
            f"读取失败：{file_path.name}\n原因：{exc}"
        ) from exc

    # 删除完全空白的行和列
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 4:
        raise ValueError(
            f"{file_path.name} 列数不足，无法识别数据结构。"
        )

    # 当前官方数据格式：
    # 第1列：Sequence
    # 中间2047列：光谱
    # 最后3列：累积耗氧量、CO2、CO

    sequence_col = df.columns[0]
    oxygen_col = df.columns[-3]
    co2_col = df.columns[-2]
    co_col = df.columns[-1]

    spectral_cols = list(df.columns[1:-3])

    # 数值化
    numeric_df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    sequence = numeric_df[sequence_col]
    oxygen = numeric_df[oxygen_col]
    co2 = numeric_df[co2_col]
    co = numeric_df[co_col]

    # 基本结构
    row_count = len(df)
    column_count = len(df.columns)
    spectral_count = len(spectral_cols)

    # 缺失值统计
    total_missing = int(numeric_df.isna().sum().sum())

    sequence_missing = int(sequence.isna().sum())
    oxygen_missing = int(oxygen.isna().sum())
    co2_missing = int(co2.isna().sum())
    co_missing = int(co.isna().sum())

    spectral_missing = int(
        numeric_df[spectral_cols].isna().sum().sum()
    )

    # 序号检查
    duplicate_sequence = int(
        sequence.duplicated().sum()
    )

    sequence_diff = sequence.diff().dropna()

    nonconsecutive_sequence = int(
        (sequence_diff != 1).sum()
    )

    # 累积耗氧量单调性检查
    oxygen_diff = oxygen.diff().dropna()

    oxygen_decreases = int(
        (oxygen_diff < 0).sum()
    )

    # 负值检查
    negative_oxygen = int((oxygen < 0).sum())
    negative_co2 = int((co2 < 0).sum())
    negative_co = int((co < 0).sum())

    # 过程信号突变候选数量
    oxygen_spikes = detect_difference_spikes(oxygen)
    co2_spikes = detect_difference_spikes(co2)
    co_spikes = detect_difference_spikes(co)

    # 光谱整体统计
    spectrum = numeric_df[spectral_cols].to_numpy(
        dtype=float
    )

    finite_spectrum = spectrum[np.isfinite(spectrum)]

    if finite_spectrum.size > 0:
        spectrum_min = float(np.min(finite_spectrum))
        spectrum_max = float(np.max(finite_spectrum))
        spectrum_mean = float(np.mean(finite_spectrum))
        spectrum_std = float(np.std(finite_spectrum))
    else:
        spectrum_min = np.nan
        spectrum_max = np.nan
        spectrum_mean = np.nan
        spectrum_std = np.nan

    # 时间长度
    duration_seconds = (
        max(row_count - 1, 0) * SAMPLING_INTERVAL
    )

    result = {
        "文件名": file_path.name,
        "采样行数": row_count,
        "总列数": column_count,
        "光谱列数": spectral_count,
        "是否符合2047个波长点": (
            spectral_count == EXPECTED_SPECTRAL_COLUMNS
        ),
        "采样时长_s": duration_seconds,

        "缺失值总数": total_missing,
        "序号缺失数": sequence_missing,
        "光谱缺失数": spectral_missing,
        "耗氧量缺失数": oxygen_missing,
        "CO2缺失数": co2_missing,
        "CO缺失数": co_missing,

        "重复序号数": duplicate_sequence,
        "序号不连续次数": nonconsecutive_sequence,

        "耗氧量下降次数": oxygen_decreases,
        "耗氧量负值数": negative_oxygen,
        "CO2负值数": negative_co2,
        "CO负值数": negative_co,

        "耗氧量突变候选数": oxygen_spikes,
        "CO2突变候选数": co2_spikes,
        "CO突变候选数": co_spikes,

        "光谱最小值": spectrum_min,
        "光谱最大值": spectrum_max,
        "光谱均值": spectrum_mean,
        "光谱标准差": spectrum_std,

        "耗氧量原始列名": str(oxygen_col),
        "CO2原始列名": str(co2_col),
        "CO原始列名": str(co_col),
    }

    print(
        f"完成：{row_count} 行，"
        f"{spectral_count} 个光谱列，"
        f"缺失值 {total_missing} 个"
    )

    return result


# =========================
# 5. 批量处理十炉数据
# =========================

def main():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"找不到原始数据目录：{RAW_DIR}"
        )

    files = sorted(
        RAW_DIR.glob("Attachment*.xlsb"),
        key=natural_sort_key
    )

    if not files:
        raise FileNotFoundError(
            "data/raw 中没有找到 Attachment*.xlsb 文件。"
        )

    print("=" * 60)
    print("APMCM2026 B题 - 第二问数据质量诊断")
    print("=" * 60)

    print(f"找到 {len(files)} 个 XLSB 文件。")

    if len(files) != 10:
        warnings.warn(
            f"预期有10炉数据，实际找到{len(files)}个文件。"
        )

    results = []

    for file_path in files:
        result = diagnose_one_file(file_path)
        results.append(result)

    summary = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "data_quality_summary.csv"

    summary.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("全部数据诊断完成！")
    print("=" * 60)

    display_columns = [
        "文件名",
        "采样行数",
        "光谱列数",
        "缺失值总数",
        "重复序号数",
        "耗氧量下降次数",
        "CO2突变候选数",
        "CO突变候选数",
    ]

    print(summary[display_columns].to_string(index=False))

    print(f"\n诊断报告已保存至：\n{output_file}")


if __name__ == "__main__":
    main()
