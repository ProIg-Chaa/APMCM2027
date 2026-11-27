"""Complete Problem B Question 2 with emissions integration and relationship analysis."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "question2" / "ten_heat_reconstruction_delay"
OUTPUT = ROOT / "results" / "question2" / "final_process_relationships"
DT = 0.5
DELAYS = {"co2": 34.0, "co": 32.0}
LABELS = {
    "oxygen_flow": ("Oxygen-lance flow", r"m$^3$/s", "#266b59"),
    "oxygen": ("Cumulative oxygen", r"m$^3$", "#266b59"),
    "co2": (r"CO$_2$ flow", r"m$^3$/s", "#235883"),
    "co": ("CO flow", r"m$^3$/s", "#ad522a"),
    "cumulative_co2": (r"Total CO$_2$ emission", r"m$^3$", "#235883"),
    "cumulative_co": ("Total CO emission", r"m$^3$", "#ad522a"),
}


def integrate_signals(frame):
    time = frame["time_s"].to_numpy()
    oxygen = frame["oxygen_monotone_linear"].to_numpy()
    co2 = frame["co2_robust_linear"].to_numpy()
    co = frame["co_robust_linear"].to_numpy()
    if not np.allclose(np.diff(time), DT):
        raise ValueError("Expected a uniform 0.5 s time grid.")
    result = pd.DataFrame({
        "time_s": time,
        # The first value is undefined because the reconstructed lance flow is
        # the interval slope of cumulative oxygen over the preceding 0.5 s.
        "oxygen_lance_flow_m3_s": np.r_[np.nan, np.diff(oxygen) / DT],
        "oxygen_cumulative_m3": oxygen,
        "co2_flow_m3_s": co2,
        "co_flow_m3_s": co,
        "co2_cumulative_m3": cumulative_trapezoid(co2, time, initial=0),
        "co_cumulative_m3": cumulative_trapezoid(co, time, initial=0),
    })
    midpoint = (time[:-1] + time[1:]) / 2
    interval = pd.DataFrame({
        "time_s": midpoint,
        "oxygen_lance_flow_m3_s": np.diff(oxygen) / DT,
        "co2_flow_m3_s": (co2[:-1] + co2[1:]) / 2,
        "co_flow_m3_s": (co[:-1] + co[1:]) / 2,
    })
    return result, interval


def correlation(function, x, y):
    value = function(np.asarray(x), np.asarray(y))
    return float(value.statistic), float(value.pvalue)


def r_squared(x, y):
    design = np.column_stack((np.ones(len(x)), x))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ coefficient
    denominator = np.sum((y - y.mean())**2)
    score = 1 - np.sum((y - prediction)**2) / denominator if denominator > 0 else np.nan
    return coefficient[0], coefficient[1], score


def analyze_heat(attachment, heat, cumulative, interval):
    oxygen = cumulative["oxygen_cumulative_m3"]
    rows = {"attachment": attachment, "heat": heat}
    for signal in ("co2", "co"):
        gas = cumulative[f"{signal}_flow_m3_s"]
        pearson, pearson_p = correlation(pearsonr, oxygen, gas)
        spearman, spearman_p = correlation(spearmanr, oxygen, gas)
        rows.update({
            f"oxygen_vs_{signal}_flow_pearson": pearson,
            f"oxygen_vs_{signal}_flow_pearson_p": pearson_p,
            f"oxygen_vs_{signal}_flow_spearman": spearman,
            f"oxygen_vs_{signal}_flow_spearman_p": spearman_p,
        })

        cumulative_gas = cumulative[f"{signal}_cumulative_m3"]
        intercept, slope, score = r_squared(oxygen, cumulative_gas)
        rows.update({
            f"oxygen_vs_cumulative_{signal}_intercept_m3": intercept,
            f"oxygen_vs_cumulative_{signal}_slope": slope,
            f"oxygen_vs_cumulative_{signal}_r2": score,
        })

        lag_samples = round(DELAYS[signal] / DT)
        source = interval["oxygen_lance_flow_m3_s"].to_numpy()[:-lag_samples]
        target = interval[f"{signal}_flow_m3_s"].to_numpy()[lag_samples:]
        aligned_pearson, aligned_pearson_p = correlation(pearsonr, source, target)
        aligned_spearman, aligned_spearman_p = correlation(spearmanr, source, target)
        rows.update({
            f"lance_flow_vs_aligned_{signal}_pearson": aligned_pearson,
            f"lance_flow_vs_aligned_{signal}_pearson_p": aligned_pearson_p,
            f"lance_flow_vs_aligned_{signal}_spearman": aligned_spearman,
            f"lance_flow_vs_aligned_{signal}_spearman_p": aligned_spearman_p,
        })

    rows.update({
        "duration_s": cumulative["time_s"].iloc[-1],
        "final_oxygen_m3": cumulative["oxygen_cumulative_m3"].iloc[-1],
        "total_co2_m3": cumulative["co2_cumulative_m3"].iloc[-1],
        "total_co_m3": cumulative["co_cumulative_m3"].iloc[-1],
        "co2_per_oxygen": (
            cumulative["co2_cumulative_m3"].iloc[-1]
            / cumulative["oxygen_cumulative_m3"].iloc[-1]
        ),
        "co_per_oxygen": (
            cumulative["co_cumulative_m3"].iloc[-1]
            / cumulative["oxygen_cumulative_m3"].iloc[-1]
        ),
    })
    return rows


def style(ax, title, ylabel):
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d8dce0", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, folder, stem):
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(folder / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_heat(folder, attachment, heat, cumulative, interval):
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 7.1), sharex=True, layout="constrained")
    series = (
        (interval["time_s"], interval["oxygen_lance_flow_m3_s"], "oxygen_flow"),
        (cumulative["time_s"], cumulative["oxygen_cumulative_m3"], "oxygen"),
        (cumulative["time_s"], cumulative["co2_flow_m3_s"], "co2"),
        (cumulative["time_s"], cumulative["co_flow_m3_s"], "co"),
        (cumulative["time_s"], cumulative["co2_cumulative_m3"], "cumulative_co2"),
        (cumulative["time_s"], cumulative["co_cumulative_m3"], "cumulative_co"),
    )
    for index, (ax, (time, values, key)) in enumerate(zip(axes.flat, series)):
        label, unit, color = LABELS[key]
        ax.plot(time, values, color=color)
        style(ax, f"({chr(97 + index)})  {label}", unit)
    axes[-1, 0].set_xlabel("Elapsed time (s)")
    axes[-1, 1].set_xlabel("Elapsed time (s)")
    fig.suptitle(f"Heat {heat} · reconstructed process signals", fontsize=12)
    save(fig, folder, f"heat_{attachment:02d}_{heat}_six_signals")


def plot_relationships(folder, cumulative_frames):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), layout="constrained")
    palette = plt.cm.tab10(np.linspace(0, 1, len(cumulative_frames)))
    for color, (attachment, heat, data) in zip(palette, cumulative_frames):
        oxygen_fraction = data["oxygen_cumulative_m3"] / data["oxygen_cumulative_m3"].iloc[-1]
        axes[0, 0].plot(
            oxygen_fraction, data["co2_flow_m3_s"],
            color=color, alpha=0.75, label=heat)
        axes[0, 1].plot(
            oxygen_fraction, data["co_flow_m3_s"],
            color=color, alpha=0.75)
        co2_fraction = data["co2_cumulative_m3"] / data["co2_cumulative_m3"].iloc[-1]
        co_fraction = data["co_cumulative_m3"] / data["co_cumulative_m3"].iloc[-1]
        axes[1, 0].plot(oxygen_fraction, co2_fraction, color=color, alpha=0.75)
        axes[1, 1].plot(oxygen_fraction, co_fraction, color=color, alpha=0.75)

    style(axes[0, 0], r"(a)  CO$_2$ flow over oxygen progress", r"CO$_2$ flow (m$^3$/s)")
    style(axes[0, 1], "(b)  CO flow over oxygen progress", r"CO flow (m$^3$/s)")
    style(axes[1, 0], r"(c)  Normalized total CO$_2$ emission", "Emission fraction")
    style(axes[1, 1], "(d)  Normalized total CO emission", "Emission fraction")
    for ax in axes.flat:
        ax.set_xlabel("Cumulative-oxygen fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=5, frameon=False)
    save(fig, folder, "figure_1_oxygen_emission_relationships")


def plot_cross_heat_totals(folder, totals):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), layout="constrained")
    for index, (signal, color) in enumerate((("co2", "#235883"), ("co", "#ad522a"))):
        ax = axes[index]
        x = totals["final_oxygen_m3"].to_numpy()
        y = totals[f"total_{signal}_m3"].to_numpy()
        intercept, slope, score = r_squared(x, y)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, y, color=color)
        for row in totals.itertuples(index=False):
            ax.annotate(row.heat, (row.final_oxygen_m3, getattr(row, f"total_{signal}_m3")),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.plot(grid, intercept + slope * grid, color="#555555", ls="--")
        style(ax, f"({chr(97 + index)})  Final oxygen vs total {SIGNALS_TITLE[signal]} · R²={score:.2f}",
              r"Total emission (m$^3$)")
        ax.set_xlabel(r"Final cumulative oxygen (m$^3$)")
    save(fig, folder, "figure_2_cross_heat_totals")


SIGNALS_TITLE = {"co2": r"CO$_2$", "co": "CO"}


def cross_heat_summary(totals):
    rows = []
    oxygen = totals["final_oxygen_m3"]
    for signal in ("co2", "co"):
        emission = totals[f"total_{signal}_m3"]
        pearson, pearson_p = correlation(pearsonr, oxygen, emission)
        spearman, spearman_p = correlation(spearmanr, oxygen, emission)
        intercept, slope, score = r_squared(oxygen, emission)
        rows.append({
            "signal": signal,
            "heats": len(totals),
            "pearson": pearson,
            "pearson_p": pearson_p,
            "spearman": spearman,
            "spearman_p": spearman_p,
            "ols_intercept_m3": intercept,
            "ols_slope": slope,
            "ols_r2": score,
        })
    return pd.DataFrame(rows)


def write_readme(output_folder, relationships, totals, across_heats):
    co2_cross = across_heats.loc[across_heats["signal"] == "co2"].iloc[0]
    co_cross = across_heats.loc[across_heats["signal"] == "co"].iloc[0]
    co2_r2 = relationships["oxygen_vs_cumulative_co2_r2"]
    co_r2 = relationships["oxygen_vs_cumulative_co_r2"]
    co2_aligned = relationships["lance_flow_vs_aligned_co2_pearson"]
    co_aligned = relationships["lance_flow_vs_aligned_co_pearson"]
    text = f"""# B题第二问：十炉过程曲线、累计排放与关系分析

## 计算口径

- 输入为上一阶段十炉重构结果，不读取或改写原始 XLSB 数据。
- 时间网格为 {DT:g} s。CO₂、CO 累计排放采用复合梯形积分：`E(t)=∫q(t)dt`。
- CO₂ 与 CO 的固定有效延迟分别采用 {DELAYS['co2']:g} s 和 {DELAYS['co']:g} s，来自十炉留一炉验证后选出的稳健基线。
- 每炉曲线表含氧枪流量、累积耗氧量、CO₂ 流量、CO 流量、CO₂ 累计排放和 CO 累计排放。氧枪流量为相邻累积耗氧量之差除以 {DT:g} s，首个时刻无前一区间，因此留空。

## 主要结果

- 十炉 CO₂ 总排放范围为 {totals['total_co2_m3'].min():.1f}–{totals['total_co2_m3'].max():.1f} m³，均值为 {totals['total_co2_m3'].mean():.1f} m³。
- 十炉 CO 总排放范围为 {totals['total_co_m3'].min():.1f}–{totals['total_co_m3'].max():.1f} m³，均值为 {totals['total_co_m3'].mean():.1f} m³。
- 单炉内累积耗氧量与累计 CO₂ 排放的 R² 为 {co2_r2.min():.3f}–{co2_r2.max():.3f}，与累计 CO 排放的 R² 为 {co_r2.min():.3f}–{co_r2.max():.3f}。
- 延迟对齐后，氧枪流量与 CO₂ 流量的 Pearson 相关系数为 {co2_aligned.min():.3f}–{co2_aligned.max():.3f}，与 CO 流量为 {co_aligned.min():.3f}–{co_aligned.max():.3f}，说明瞬时响应在炉次间有明显差异。
- 跨炉最终耗氧量对 CO₂ 总排放的 R² 为 {co2_cross['ols_r2']:.3f}，对 CO 总排放的 R² 为 {co_cross['ols_r2']:.3f}，均不足以支持稳定的跨炉线性关系。

## 解释边界

单炉“累积量—累积量”的高 R² 同时受到共同时间趋势影响，不能单独作为因果证据。论文中应把它表述为过程内同步累积特征，并结合延迟对齐后的瞬时流量关系及跨炉总量检验共同判断。当前证据支持固定延迟作为十炉稳健基线，不支持建立统一的动态延迟函数。

## 输出文件

- `per_heat_curves/`：十炉六条过程信号的可复用数据。
- `per_heat_relationships.csv`：逐炉瞬时、延迟对齐及累计关系指标。
- `per_heat_totals.csv`：逐炉最终耗氧量与 CO₂、CO 总排放。
- `cross_heat_relationships.csv`：跨炉总量相关与线性拟合。
- `figures/`：十张单炉六联图和两张十炉汇总图，均含 PNG 与 PDF。
- `experiment.json`：采样、延迟、积分方法和解释限制。
"""
    (output_folder / "README.md").write_text(text, encoding="utf-8")


def run(input_folder, output_folder):
    if output_folder.exists():
        raise FileExistsError(f"Use a new output directory: {output_folder}")
    heat_folders = sorted((input_folder / "per_heat").glob("heat_*"))
    if len(heat_folders) != 10:
        raise ValueError("Expected ten per-heat reconstruction folders.")
    output_folder.mkdir(parents=True)
    curves_folder = output_folder / "per_heat_curves"
    figures_folder = output_folder / "figures"
    curves_folder.mkdir()
    figures_folder.mkdir()

    relationship_rows = []
    cumulative_frames = []
    for heat_folder in heat_folders:
        match = heat_folder.name.split("_", 2)
        attachment = int(match[1])
        heat = match[2]
        reconstructed = pd.read_csv(heat_folder / "reconstructed_signals.csv")
        cumulative, interval = integrate_signals(reconstructed)
        cumulative.to_csv(
            curves_folder / f"heat_{attachment:02d}_{heat}_six_signals.csv",
            index=False, float_format="%.10g")
        relationship_rows.append(analyze_heat(attachment, heat, cumulative, interval))
        cumulative_frames.append((attachment, heat, cumulative))
        plot_heat(figures_folder, attachment, heat, cumulative, interval)

    relationships = pd.DataFrame(relationship_rows).sort_values("attachment")
    totals = relationships[[
        "attachment", "heat", "duration_s", "final_oxygen_m3",
        "total_co2_m3", "total_co_m3", "co2_per_oxygen", "co_per_oxygen",
    ]]
    across_heats = cross_heat_summary(totals)
    relationships.to_csv(
        output_folder / "per_heat_relationships.csv", index=False, float_format="%.10g")
    totals.to_csv(output_folder / "per_heat_totals.csv", index=False, float_format="%.10g")
    across_heats.to_csv(
        output_folder / "cross_heat_relationships.csv", index=False, float_format="%.10g")
    write_readme(output_folder, relationships, totals, across_heats)
    plot_relationships(figures_folder, cumulative_frames)
    plot_cross_heat_totals(figures_folder, totals)

    metadata = {
        "heats": 10,
        "sample_step_s": DT,
        "integration": "composite trapezoidal rule on robust reconstructed gas flow",
        "gas_units": "m^3/s",
        "cumulative_emission_units": "m^3",
        "fixed_effective_delay_s": DELAYS,
        "instantaneous_relationships": {
            "oxygen_cumulative_vs_observed_gas": "Pearson and Spearman",
            "oxygen_lance_flow_vs_future_gas": "Pearson and Spearman after fixed-delay alignment",
        },
        "cumulative_relationships": {
            "within_heat": "OLS cumulative gas = intercept + slope * cumulative oxygen",
            "cross_heat": "final totals across ten heats",
        },
        "caution": (
            "Cumulative-versus-cumulative association includes a shared time trend; "
            "aligned flow correlations and cross-heat totals are reported separately."
        ),
    }
    (output_folder / "experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "axes.linewidth": 0.65,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
    }):
        run(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"Question 2 final process analysis saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
