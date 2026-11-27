"""Apply the locked first-heat reconstruction and delay pipeline to all ten heats."""

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from scipy.stats import spearmanr

from heat_01_delay_experiment import (
    DT,
    estimate_delays,
    interval_signals,
)
from heat_01_reconstruction_experiment import fit


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "results" / "question2" / "ten_heat_reconstruction_delay"
MODEL_NAMES = ("fixed", "monotone_linear", "reciprocal", "isotonic")
SIGNALS = {"co2": r"CO$_2$", "co": "CO"}
COLORS = {
    "fixed": "#777777",
    "monotone_linear": "#266b59",
    "reciprocal": "#ad522a",
    "isotonic": "#235883",
}


def attachment_number(path):
    match = re.search(r"Attachment\s+(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot determine attachment number from {path.name}")
    return int(match.group(1))


def heat_number(path):
    match = re.search(r"Heat No\.\s+([^.]+)", path.name)
    if not match:
        raise ValueError(f"Cannot determine heat number from {path.name}")
    return match.group(1)


def load_heat(path):
    path = Path(path)

    df = pd.read_excel(
        path,
        sheet_name=0,
        engine="calamine"
    ).dropna(how="all")

    # 第一炉字段名称映射
    aliases = {
        "序列": "sequence",
        "Sequence": "sequence",
        "序列Sequence": "sequence",
        "累积耗氧量": "oxygen",
        "CO2浓度": "co2",
        "CO₂浓度": "co2",
        "CO浓度": "co",
    }

    # 兼容其他炉次原始字段名称
    aliases.update({
        "Cumulative oxygen consumption": "oxygen",
        "CO₂ concentration": "co2",
        "CO concentration": "co",
        "YQXH": "oxygen",
    })

    normalized = {
        col: aliases[col]
        for col in df.columns
        if col in aliases
    }

    required = {"oxygen", "co2", "co"}

    if not required.issubset(set(normalized.values())):
        raise ValueError(
            f"{path.name} 缺少必要字段，"
            f"实际字段：{df.columns[-6:].tolist()}"
        )

    sequence_col = next(
        (col for col, name in normalized.items()
         if name == "sequence"),
        None
    )

    if sequence_col is None:
        raise ValueError(f"{path.name} 缺少序列字段")

    data = pd.DataFrame({
        name: df[col]
        for col, name in normalized.items()
    })

    sequence = pd.to_numeric(
        data["sequence"], errors="coerce"
    ).to_numpy(dtype=float)

    oxygen = pd.to_numeric(
        data["oxygen"], errors="coerce"
    ).to_numpy(dtype=float)

    co2 = pd.to_numeric(
        data["co2"], errors="coerce"
    ).to_numpy(dtype=float)

    co = pd.to_numeric(
        data["co"], errors="coerce"
    ).to_numpy(dtype=float)

    if not (
        len(sequence) == len(oxygen)
        == len(co2) == len(co)
    ):
        raise ValueError("四组数据长度不一致")

    if (
        np.any(~np.isfinite(sequence))
        or not np.all(np.diff(sequence) == 1)
    ):
        raise ValueError("采样序号必须有限且连续")

    if np.any(~np.isfinite(oxygen)):
        raise ValueError("累积耗氧量存在非有限值")

    if np.any(~np.isfinite(co2)) or np.any(~np.isfinite(co)):
        raise ValueError("烟气数据存在非有限值")

    return (sequence - sequence[0]) * DT, {
        "oxygen": oxygen,
        "co2": co2,
        "co": co,
    }


def reconstruct(time, raw):
    return pd.DataFrame({
        "time_s": time,
        "oxygen_monotone_linear": fit("monotone_linear", time, raw["oxygen"])(time),
        "co2_robust_linear": fit("robust_linear", time, raw["co2"])(time),
        "co_robust_linear": fit("robust_linear", time, raw["co"])(time),
    })


def add_heat_columns(frame, attachment, heat):
    result = frame.copy()
    result.insert(0, "heat", heat)
    result.insert(0, "attachment", attachment)
    return result


def pava_decreasing(x, y):
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = np.asarray(x)[order], np.asarray(y)[order]
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    means = np.bincount(inverse, weights=sorted_y) / counts

    block_values = []
    block_weights = []
    block_starts = []
    block_ends = []
    for index, (value, weight) in enumerate(zip(-means, counts)):
        block_values.append(value)
        block_weights.append(weight)
        block_starts.append(index)
        block_ends.append(index)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / total_weight
            block_values[-2:] = [pooled]
            block_weights[-2:] = [total_weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()

    fitted = np.empty(len(unique_x))
    for value, start, end in zip(block_values, block_starts, block_ends):
        fitted[start:end + 1] = -value
    return unique_x, fitted


def fit_model(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if name == "fixed":
        level = float(np.median(y))
        return lambda target: np.full(np.asarray(target).shape, level)

    if name == "monotone_linear":
        design = np.column_stack((np.ones(len(x)), x))
        result = lsq_linear(
            design, y,
            bounds=([-np.inf, -np.inf], [np.inf, 0.0]),
            method="bvls",
        )
        if not result.success:
            raise RuntimeError(result.message)
        return lambda target: result.x[0] + result.x[1] * np.asarray(target)

    if name == "reciprocal":
        def residual(parameters):
            a, b, c = parameters
            return a / (x + b) + c - y

        initial = np.array([max(np.ptp(y), 1.0) * max(np.mean(x), 1.0), 1.0, max(y.min(), 0)])
        result = least_squares(
            residual,
            x0=initial,
            bounds=([0.0, 0.05, 0.0], [10000.0, 100.0, 60.0]),
            max_nfev=5000,
        )
        if not result.success:
            raise RuntimeError(result.message)
        a, b, c = result.x
        return lambda target: a / (np.asarray(target) + b) + c

    if name == "isotonic":
        knots, levels = pava_decreasing(x, y)
        return lambda target: np.interp(np.asarray(target), knots, levels)

    raise ValueError(name)


def cross_validate(reliable):
    prediction_rows = []
    for signal in SIGNALS:
        signal_data = reliable[reliable["signal"] == signal]
        for attachment in sorted(signal_data["attachment"].unique()):
            test = signal_data[signal_data["attachment"] == attachment]
            train = signal_data[signal_data["attachment"] != attachment]
            if len(test) == 0 or len(train) < 8:
                continue
            for model_name in MODEL_NAMES:
                model = fit_model(
                    model_name,
                    train["mean_lance_flow"].to_numpy(),
                    train["delay_s"].to_numpy(),
                )
                prediction = np.clip(model(test["mean_lance_flow"].to_numpy()), 0, 60)
                for row, predicted in zip(test.itertuples(index=False), prediction):
                    prediction_rows.append({
                        "signal": signal,
                        "model": model_name,
                        "test_attachment": attachment,
                        "test_heat": row.heat,
                        "window_center_s": row.window_center_s,
                        "mean_lance_flow": row.mean_lance_flow,
                        "observed_delay_s": row.delay_s,
                        "predicted_delay_s": predicted,
                        "error_s": predicted - row.delay_s,
                    })
    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    fold_rows = []
    if predictions.empty:
        return predictions, pd.DataFrame(), pd.DataFrame()
    for (signal, model), group in predictions.groupby(["signal", "model"]):
        heat_rmse = group.groupby("test_attachment")["error_s"].apply(
            lambda error: np.sqrt(np.mean(error**2)))
        for attachment, fold in group.groupby("test_attachment"):
            fold_rows.append({
                "signal": signal,
                "model": model,
                "test_attachment": attachment,
                "test_heat": fold["test_heat"].iloc[0],
                "test_windows": len(fold),
                "rmse_s": np.sqrt(np.mean(fold["error_s"]**2)),
                "mae_s": np.mean(np.abs(fold["error_s"])),
            })
        metric_rows.append({
            "signal": signal,
            "model": model,
            "test_windows": len(group),
            "test_heats": group["test_attachment"].nunique(),
            "window_rmse_s": np.sqrt(np.mean(group["error_s"]**2)),
            "window_mae_s": np.mean(np.abs(group["error_s"])),
            "mean_heat_rmse_s": heat_rmse.mean(),
        })
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(fold_rows)


def relationship_summary(reliable):
    rows = []
    for signal, data in reliable.groupby("signal"):
        correlation = spearmanr(data["mean_lance_flow"], data["delay_s"])
        rows.append({
            "signal": signal,
            "reliable_windows": len(data),
            "heats": data["attachment"].nunique(),
            "flow_min": data["mean_lance_flow"].min(),
            "flow_max": data["mean_lance_flow"].max(),
            "delay_min_s": data["delay_s"].min(),
            "delay_max_s": data["delay_s"].max(),
            "pooled_spearman_rho": correlation.statistic,
            "pooled_spearman_p_value": correlation.pvalue,
        })
    return pd.DataFrame(rows)


def full_models(reliable):
    curve_rows = []
    for signal in SIGNALS:
        data = reliable[reliable["signal"] == signal]
        if len(data) < 8:
            continue
        grid = np.linspace(data["mean_lance_flow"].min(), data["mean_lance_flow"].max(), 200)
        for model_name in MODEL_NAMES:
            model = fit_model(model_name, data["mean_lance_flow"], data["delay_s"])
            for flow, delay in zip(grid, np.clip(model(grid), 0, 60)):
                curve_rows.append({
                    "signal": signal,
                    "model": model_name,
                    "mean_lance_flow": flow,
                    "predicted_delay_s": delay,
                })
    return pd.DataFrame(curve_rows)


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


def plot_heat_summary(folder, summary):
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.5), sharex=True, layout="constrained")
    positions = np.arange(len(summary))
    width = 0.36
    for offset, signal, color in ((-0.5, "co2", "#235883"), (0.5, "co", "#ad522a")):
        axes[0].bar(
            positions + offset * width,
            summary[f"{signal}_reliable_windows"],
            width,
            color=color,
            label=SIGNALS[signal],
        )
    style(axes[0], "(a)  Reliable local-delay windows by heat", "Window count")
    axes[0].legend(frameon=False, ncols=2)
    for signal, color, marker in (("co2", "#235883", "o"), ("co", "#ad522a", "D")):
        valid = ~summary[f"{signal}_fixed_boundary"]
        axes[1].scatter(
            positions[valid],
            summary.loc[valid, f"{signal}_fixed_delay_s"],
            color=color,
            marker=marker,
            label=f"{SIGNALS[signal]} interior peak",
        )
        axes[1].scatter(
            positions[~valid],
            summary.loc[~valid, f"{signal}_fixed_delay_s"],
            facecolors="none",
            edgecolors=color,
            marker=marker,
            label=f"{SIGNALS[signal]} boundary peak",
        )
    style(axes[1], "(b)  Whole-heat fixed-delay candidates", "Candidate delay (s)")
    axes[1].set_xticks(positions, summary["heat"], rotation=35, ha="right")
    axes[1].set_xlabel("Heat")
    axes[1].legend(frameon=False, ncols=2, fontsize=8)
    save(fig, folder, "figure_1_heat_summary")


def plot_models(folder, reliable, curves, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), layout="constrained")
    heat_ids = sorted(reliable["attachment"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(heat_ids), 1)))
    for column, signal in enumerate(("co2", "co")):
        data = reliable[reliable["signal"] == signal]
        for color, attachment in zip(palette, heat_ids):
            heat = data[data["attachment"] == attachment]
            if len(heat):
                axes[0, column].scatter(
                    heat["mean_lance_flow"], heat["delay_s"],
                    s=22, color=color, alpha=0.75,
                    label=heat["heat"].iloc[0],
                )
        for model_name in MODEL_NAMES:
            curve = curves[(curves["signal"] == signal) & (curves["model"] == model_name)]
            if len(curve):
                axes[0, column].plot(
                    curve["mean_lance_flow"], curve["predicted_delay_s"],
                    color=COLORS[model_name], label=model_name.replace("_", " "),
                )
        style(axes[0, column], f"({chr(97 + column)})  {SIGNALS[signal]} pooled reliable windows",
              "Effective delay (s)")
        axes[0, column].set_xlabel(r"Mean oxygen-lance flow (m$^3$/s)")

        metric = metrics[metrics["signal"] == signal].set_index("model").reindex(MODEL_NAMES)
        bars = axes[1, column].bar(
            np.arange(len(metric)), metric["window_rmse_s"],
            color=[COLORS[name] for name in MODEL_NAMES],
        )
        axes[1, column].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        style(axes[1, column], f"({chr(99 + column)})  Leave-one-heat-out error",
              "Window RMSE (s)")
        axes[1, column].set_xticks(
            np.arange(len(metric)),
            ["Fixed", "Monotone\nlinear", "Reciprocal", "Isotonic"],
        )
        axes[1, column].set_ylim(0, np.nanmax(metric["window_rmse_s"]) * 1.20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    model_handles = handles[-len(MODEL_NAMES):]
    model_labels = labels[-len(MODEL_NAMES):]
    fig.legend(model_handles, model_labels, loc="outside upper center",
               ncols=len(MODEL_NAMES), frameon=False)
    save(fig, folder, "figure_2_delay_models")


def run(raw_folder, output_folder):
    if output_folder.exists():
        raise FileExistsError(f"Use a new output directory: {output_folder}")
    files = sorted(raw_folder.glob("Attachment*.xlsb"), key=attachment_number)
    if len(files) != 10 or [attachment_number(path) for path in files] != list(range(1, 11)):
        raise ValueError("Expected attachments 1 through 10.")

    output_folder.mkdir(parents=True)
    per_heat = output_folder / "per_heat"
    per_heat.mkdir()
    fixed_frames = []
    window_frames = []
    curve_frames = []
    summary_rows = []

    for path in files:
        attachment = attachment_number(path)
        heat = heat_number(path)
        time, raw = load_heat(path)
        reconstructed = reconstruct(time, raw)
        midpoint, flow, gases = interval_signals(reconstructed)
        fixed, windows, curves = estimate_delays(midpoint, flow, gases)
        fixed = add_heat_columns(fixed, attachment, heat)
        windows = add_heat_columns(windows, attachment, heat)
        curves = add_heat_columns(curves, attachment, heat)
        fixed_frames.append(fixed)
        window_frames.append(windows)
        curve_frames.append(curves)

        heat_folder = per_heat / f"heat_{attachment:02d}_{heat}"
        heat_folder.mkdir()
        reconstructed.to_csv(
            heat_folder / "reconstructed_signals.csv", index=False, float_format="%.10g")
        pd.DataFrame({
            "time_s": midpoint,
            "oxygen_lance_flow_m3_s": flow,
            "co2_flow_m3_s": gases["co2"],
            "co_flow_m3_s": gases["co"],
        }).to_csv(heat_folder / "aligned_interval_signals.csv",
                  index=False, float_format="%.10g")
        fixed.to_csv(heat_folder / "fixed_delay_candidates.csv",
                     index=False, float_format="%.10g")
        windows.to_csv(heat_folder / "window_delay_estimates.csv",
                       index=False, float_format="%.10g")

        fixed_index = fixed.set_index("signal")
        summary_rows.append({
            "attachment": attachment,
            "heat": heat,
            "observations": len(time),
            "duration_s": time[-1],
            "co2_reliable_windows": int(
                ((windows["signal"] == "co2") & windows["reliable"]).sum()),
            "co_reliable_windows": int(
                ((windows["signal"] == "co") & windows["reliable"]).sum()),
            "co2_fixed_delay_s": fixed_index.loc["co2", "delay_s"],
            "co2_fixed_correlation": fixed_index.loc["co2", "correlation"],
            "co2_fixed_boundary": fixed_index.loc["co2", "boundary_peak"],
            "co_fixed_delay_s": fixed_index.loc["co", "delay_s"],
            "co_fixed_correlation": fixed_index.loc["co", "correlation"],
            "co_fixed_boundary": fixed_index.loc["co", "boundary_peak"],
        })

    fixed_all = pd.concat(fixed_frames, ignore_index=True)
    windows_all = pd.concat(window_frames, ignore_index=True)
    curves_all = pd.concat(curve_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    reliable = windows_all[windows_all["reliable"]].copy()
    predictions, metrics, fold_metrics = cross_validate(reliable)
    relationships = relationship_summary(reliable)
    model_curves = full_models(reliable)

    summary.to_csv(output_folder / "heat_summary.csv", index=False, float_format="%.10g")
    fixed_all.to_csv(output_folder / "all_fixed_delay_candidates.csv",
                     index=False, float_format="%.10g")
    windows_all.to_csv(output_folder / "all_window_delay_estimates.csv",
                       index=False, float_format="%.10g")
    curves_all.to_csv(output_folder / "all_window_correlation_curves.csv",
                      index=False, float_format="%.10g")
    reliable.to_csv(output_folder / "reliable_windows.csv",
                    index=False, float_format="%.10g")
    predictions.to_csv(output_folder / "leave_one_heat_out_predictions.csv",
                       index=False, float_format="%.10g")
    metrics.to_csv(output_folder / "model_comparison.csv",
                   index=False, float_format="%.10g")
    fold_metrics.to_csv(output_folder / "per_heat_model_errors.csv",
                        index=False, float_format="%.10g")
    relationships.to_csv(output_folder / "pooled_relationship_summary.csv",
                         index=False, float_format="%.10g")
    model_curves.to_csv(output_folder / "fitted_model_curves.csv",
                        index=False, float_format="%.10g")

    selected = {}
    for signal in SIGNALS:
        subset = metrics[metrics["signal"] == signal]
        if len(subset):
            best = subset.loc[subset["mean_heat_rmse_s"].idxmin()]
            selected[signal] = {
                "model": best["model"],
                "selection_metric": "mean_heat_rmse_s",
                "leave_one_heat_out_mean_heat_rmse_s": best["mean_heat_rmse_s"],
                "leave_one_heat_out_window_rmse_s": best["window_rmse_s"],
                "leave_one_heat_out_window_mae_s": best["window_mae_s"],
                "test_heats": int(best["test_heats"]),
                "test_windows": int(best["test_windows"]),
            }
    metadata = {
        "attachments": list(range(1, 11)),
        "reconstruction": {
            "oxygen": "monotone piecewise linear, nominal 10 s knots",
            "co2": "Huber robust piecewise linear, nominal 5 s knots",
            "co": "Huber robust piecewise linear, nominal 5 s knots",
        },
        "delay_pipeline": {
            "window_s": 120.0,
            "step_s": 30.0,
            "lag_search_s": [0.0, 60.0],
            "local_preprocessing": "linear detrend",
            "minimum_correlation": 0.45,
            "minimum_peak_contrast": 0.08,
            "minimum_lance_flow_std": 0.10,
            "boundary_peaks_rejected": True,
        },
        "validation": {
            "scheme": "leave one heat out",
            "reported_metrics": ["window weighted", "equal-weight mean of per-heat RMSE"],
            "selection_metric": "equal-weight mean of per-heat RMSE",
        },
        "candidate_models": list(MODEL_NAMES),
        "selected_models": selected,
    }
    (output_folder / "experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with plt.rc_context({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "axes.linewidth": 0.65,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42,
    }):
        plot_heat_summary(output_folder, summary)
        plot_models(output_folder, reliable, model_curves, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.raw_dir.resolve(), args.output_dir.resolve())
    print(f"Ten-heat reconstruction and delay results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
