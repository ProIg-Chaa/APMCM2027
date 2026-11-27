"""Compare offline reconstructions of corrected APMCM Problem B heat 6B03205."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import lsq_linear, minimize
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[2]
HEAT = "6B03205"
INPUT = ROOT / "data" / "raw" / f"Attachment 1. Heat No. {HEAT}.xlsb"
INPUT_HASH = "8c73aaefadb9c8d8c3f83b193d9b4583dcfdfb239446c69b7eacd4120820783c"
OUTPUT = ROOT / "results" / "question2" / "heat_01_reconstruction"
DT = 0.5
GRID_STEP = 0.1
EPSILON = 1e-8
FOLDS = 5
GAPS = (6, 20)
PEAK_HALF_WIDTH = 1.5
PEAK_PROMINENCE_FRACTION = 0.10
PEAK_SEPARATION_S = 20.0
HUBER_SCALE_FLOOR = 0.05
SPACING = {"monotone_linear": 10.0, "robust_linear": 5.0}
METHODS = {
    "oxygen": ("linear", "monotone_linear", "pchip"),
    "co2": ("robust_linear", "pchip"),
    "co": ("robust_linear", "pchip"),
}
LABELS = {
    "oxygen": ("Cumulative oxygen", r"m$^3$"),
    "co2": (r"CO$_2$ flow", r"m$^3$/s"),
    "co": ("CO flow", r"m$^3$/s"),
}
STYLES = {
    "linear": ("Linear interpolation", "#696969", "--"),
    "monotone_linear": ("Monotone linear fit", "#266b59", "-"),
    "robust_linear": ("Robust linear fit", "#235883", "-"),
    "pchip": ("PCHIP", "#ad522a", "-."),
}


def load_heat(path):
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != INPUT_HASH:
        raise ValueError(f"Expected the corrected {HEAT} workbook. Got SHA-256 {digest}.")
    # Match the positional data contract of signal_visualization.load_heat.
    # Keep this entry point free of that module's plot setup and output creation.
    sheet = pd.read_excel(path, engine="calamine").dropna(how="all")
    sheet = sheet.dropna(axis=1, how="all")
    sequence, oxygen, co2, co = sheet.iloc[:, [0, -3, -2, -1]].to_numpy(dtype=float).T
    if not np.isfinite([sequence, oxygen, co2, co]).all() or not np.all(np.diff(sequence) == 1):
        raise ValueError("The corrected workbook requires finite values and consecutive sequence IDs.")
    return (sequence - sequence[0]) * DT, dict(zip(METHODS, (oxygen, co2, co)))


def fit(method, time, values):
    if method == "pchip":
        return PchipInterpolator(time, values, extrapolate=False)

    knots = time
    levels = values
    if method in SPACING:
        segment_count = int(np.ceil((time[-1] - time[0]) / SPACING[method]))
        knots = np.linspace(time[0], time[-1], segment_count + 1)
        if method == "monotone_linear":
            # Columns encode the initial level and each successive rise.
            rises = np.clip((time[:, None] - knots[:-1]) / np.diff(knots), 0, 1)
            design = np.column_stack((np.ones(len(time)), rises))
            result = lsq_linear(design, values, bounds=(0, np.inf), method="bvls")
            if not result.success:
                raise RuntimeError(f"Monotone fit failed: {result.message}")
            levels = np.cumsum(result.x)
        else:
            # Snap nominal knots to retained observations, then merge duplicates.
            # A knot inside a held-out gap would have no direct data support.
            right = np.clip(np.searchsorted(time, knots), 1, len(time) - 1)
            left = right - 1
            nearest = np.where(knots - time[left] <= time[right] - knots, left, right)
            knots = np.unique(time[nearest])
            levels = huber_levels(time, values, knots)

    def interpolate(target):
        return np.interp(target, knots, levels, left=np.nan, right=np.nan)

    return interpolate


def huber_levels(time, values, knots):
    left = np.clip(np.searchsorted(knots, time, side="right") - 1, 0, len(knots) - 2)
    fraction = (time - knots[left]) / (knots[left + 1] - knots[left])
    columns = np.column_stack((left, left + 1)).ravel()
    weights = np.column_stack((1 - fraction, fraction)).ravel()
    design = csr_matrix((weights, (np.repeat(np.arange(len(time)), 2), columns)),
                        shape=(len(time), len(knots)))
    gram = (design.T @ design).toarray()
    pilot = np.linalg.lstsq(gram, design.T @ values, rcond=None)[0]
    residual = design @ pilot - values
    mad = np.median(np.abs(residual - np.median(residual)))
    delta = 1.345 * max(1.4826 * mad, HUBER_SCALE_FLOOR)

    def loss(levels):
        residual = design @ levels - values
        influence = np.clip(residual, -delta, delta)
        value = np.sum(influence * (residual - 0.5 * influence))
        return value, design.T @ influence

    result = minimize(loss, np.maximum(pilot, 0), jac=True, method="L-BFGS-B",
                      bounds=[(0, None)] * len(knots),
                      options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 2000})
    if not result.success:
        raise RuntimeError(f"Robust fit failed: {result.message}")
    return result.x


def evaluate_gaps(time, signals):
    rows = []
    for block_size in GAPS:
        interior = np.arange(1, len(time) - 1)
        fold_ids = ((interior - 1) // block_size) % FOLDS
        for signal, values in signals.items():
            for method in METHODS[signal]:
                errors = np.full(len(time), np.nan)
                for fold in range(FOLDS):
                    test = interior[fold_ids == fold]
                    train = np.ones(len(time), dtype=bool)
                    train[test] = False
                    model = fit(method, time[train], values[train])
                    errors[test] = model(time[test]) - values[test]
                residual = errors[interior]
                if not np.isfinite(residual).all():
                    raise RuntimeError("Missing held-out predictions.")
                rows.append({
                    "signal": signal,
                    "method": method,
                    "gap_samples": block_size,
                    "gap_duration_s": block_size * DT,
                    "n_test": len(residual),
                    "rmse": np.sqrt(np.mean(residual**2)),
                    "mae": np.mean(np.abs(residual)),
                })
    return pd.DataFrame(rows)


def grid(start, end):
    return np.linspace(start, end, int(round((end - start) / GRID_STEP)) + 1)


def evaluate_peaks(time, signals, curves):
    rows = []
    for signal in ("co2", "co"):
        values = signals[signal]
        peaks, properties = find_peaks(values, distance=round(PEAK_SEPARATION_S / DT),
                                       prominence=PEAK_PROMINENCE_FRACTION * np.ptp(values))
        for peak, prominence in zip(peaks, properties["prominences"]):
            center = time[peak]
            left, right = center - PEAK_HALF_WIDTH, center + PEAK_HALF_WIDTH
            if left <= time[0] or right >= time[-1]:
                continue
            train = (time < left) | (time > right)
            targets = grid(left, right)
            for method in METHODS[signal]:
                masked = fit(method, time[train], values[train])
                for protocol, model in (("all_observations", curves[signal, method]),
                                        ("peak_window_held_out", masked)):
                    predictions = model(targets)
                    height = predictions.max()
                    # Use the midpoint of the longest contiguous maximum plateau.
                    max_indices = np.flatnonzero(np.abs(predictions - height) <= EPSILON)
                    plateaus = np.split(max_indices, np.flatnonzero(np.diff(max_indices) > 1) + 1)
                    maxima = targets[max(plateaus, key=len)]
                    peak_time = (maxima[0] + maxima[-1]) / 2
                    rows.append({
                        "signal": signal, "method": method, "protocol": protocol,
                        "observed_peak_time_s": center, "observed_peak": values[peak],
                        "observed_prominence": prominence, "window_start_s": left,
                        "window_end_s": right, "n_window_held_out": int((~train).sum())
                        if protocol == "peak_window_held_out" else 0,
                        "reconstructed_peak": height, "peak_time_s": peak_time,
                        "height_error": height - values[peak],
                        "height_error_pct": 100 * (height - values[peak]) / values[peak],
                        "time_error_s": peak_time - center,
                        "maximum_touches_window_edge": bool(
                            maxima[0] == targets[0] or maxima[-1] == targets[-1]),
                    })
    return pd.DataFrame(rows)


def physical_checks(time, dense_time, signals, curves):
    rows = []
    for (signal, method), model in curves.items():
        dense_values = model(dense_time)
        differences = np.diff(dense_values)
        row = {
            "signal": signal, "method": method, "grid_step_s": GRID_STEP,
            "minimum": dense_values.min(),
            "negative_points": int((dense_values < -EPSILON).sum()),
            "largest_drop": None, "decreasing_steps": None,
            "endpoint_error": None, "mean_abs_lance_flow_change": None,
        }
        if signal == "oxygen":
            lance_flow = np.diff(model(time)) / np.diff(time)
            row.update(
                largest_drop=max(0, -differences.min()),
                decreasing_steps=int((differences < -EPSILON).sum()),
                endpoint_error=dense_values[-1] - signals[signal][-1],
                mean_abs_lance_flow_change=np.mean(np.abs(np.diff(lance_flow))),
            )
        row["physical_check_pass"] = row["negative_points"] == 0 and row["decreasing_steps"] in (None, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def style_axes(ax, title, ylabel):
    ax.set_title(title, loc="left", pad=6)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d8dce0", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def draw_curve(ax, time, y, dense_time, signal, curves):
    ax.scatter(time, y, s=9, color="#aab0b8", alpha=0.8, linewidths=0, zorder=2)
    for method in METHODS[signal]:
        label, color, line = STYLES[method]
        ax.plot(dense_time, curves[signal, method](dense_time), ls=line, color=color,
                label=label, zorder=3)


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_signals(out, time, dense_time, signals, curves, peaks):
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 7.6), layout="constrained",
                             gridspec_kw={"width_ratios": (1.47, 1)})
    handles = [Line2D([], [], marker="o", color="#aab0b8", ls="", ms=4, label="Observation")]
    for label, color, line in STYLES.values():
        handles.append(Line2D([], [], color=color, ls=line, label=label))
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), frameon=False)
    for i, (signal, y) in enumerate(signals.items()):
        label, unit = LABELS[signal]
        full, detail = axes[i]
        draw_curve(full, time, y, dense_time, signal, curves)
        style_axes(full, f"({chr(97 + 2 * i)})  {label}", unit)
        full.set_xlim(time[0], time[-1])
        full.xaxis.set_major_locator(MaxNLocator(6))
        if signal == "oxygen":
            center, half = 360.0, 4.0
            detail_label = "Oxygen staircase"
        else:
            features = peaks[peaks["signal"] == signal]
            top = features.loc[features["observed_prominence"].idxmax()]
            center, half = top["observed_peak_time_s"], 12.0
            detail_label = f"{label} near peak"
        lo, hi = max(time[0], center - half), min(time[-1], center + half)
        full.axvspan(lo, hi, color="#dce7ef", lw=0, zorder=0)
        raw_mask = (lo <= time) & (time <= hi)
        dense_mask = (lo <= dense_time) & (dense_time <= hi)
        draw_curve(detail, time[raw_mask], y[raw_mask], dense_time[dense_mask], signal, curves)
        style_axes(detail, f"({chr(98 + 2 * i)})  {detail_label}", unit)
        detail.set_xlim(lo, hi)
        detail.xaxis.set_major_locator(MaxNLocator(5))
        if i == 2:
            full.set_xlabel("Elapsed time (s)")
            detail.set_xlabel("Elapsed time (s)")
    save_figure(fig, out, "figure_1_reconstruction")


def plot_errors(out, scores):
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.0), layout="constrained")
    width = 0.33
    gap_styles = (("#375f80", ""), ("#b9cfdd", "//"))
    for col, (signal, methods) in enumerate(METHODS.items()):
        signal_label, unit = LABELS[signal]
        short_labels = ["PCHIP" if method == "pchip" else
                        {"linear": "Linear", "monotone_linear": "Monotone",
                         "robust_linear": "Robust"}[method] for method in methods]
        positions = np.arange(len(methods))
        for row, metric in enumerate(("rmse", "mae")):
            ax = axes[row, col]
            for i, (block_size, (color, hatch)) in enumerate(zip(GAPS, gap_styles)):
                subset = scores[(scores["signal"] == signal) & (scores["gap_samples"] == block_size)]
                values = subset.set_index("method").loc[list(methods), metric]
                bars = ax.bar(positions + (i - 0.5) * width, values, width, color=color,
                              hatch=hatch, edgecolor="#375f80", linewidth=0.5,
                              label=f"{block_size * DT:g} s gap")
                ax.bar_label(bars, fmt="%.2f" if signal == "oxygen" else "%.3f",
                             fontsize=7, padding=2)
            style_axes(ax, f"({chr(97 + row * 3 + col)})  {signal_label}", f"{metric.upper()} ({unit})")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
            ax.set_xticks(positions, short_labels)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    save_figure(fig, out, "figure_2_holdout_errors")


def plot_diagnostics(out, time, curves, peaks):
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.1), layout="constrained")
    ax = axes[0]
    # Plot interval-average derivatives on the native sampling grid.
    for method in ("linear", "monotone_linear"):
        label, color, line = STYLES[method]
        if method == "linear":
            label = "Linear / PCHIP"
        average_flow = np.diff(curves["oxygen", method](time)) / np.diff(time)
        ax.stairs(average_flow, time, baseline=None, label=label, color=color, ls=line)
    ax.set_xlim(355, 365)
    ax.set_xlabel("Elapsed time (s)")
    style_axes(ax, "(a)  Oxygen-lance flow", r"Flow (m$^3$/s)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    masked = peaks[peaks["protocol"] == "peak_window_held_out"]
    for i, signal in enumerate(("co2", "co"), start=1):
        ax = axes[i]
        for method, marker in (("robust_linear", "o"), ("pchip", "D")):
            rows = masked[(masked["signal"] == signal) & (masked["method"] == method)]
            label, color, _ = STYLES[method]
            ax.scatter(rows["observed_peak_time_s"], rows["height_error_pct"],
                       marker=marker, label=label, color=color, s=20, zorder=3)
        ax.axhline(0, color="#6d737a", lw=0.75)
        ax.set_xlim(0, time[-1])
        ax.set_ylim(masked["height_error_pct"].min() * 1.16, 0.035)
        ax.set_xlabel("Observed peak time (s)")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        style_axes(ax, f"({chr(97 + i)})  {LABELS[signal][0]} peaks", "Held-out height error (%)")
        if i == 1:
            ax.legend(frameon=False, fontsize=7, loc="lower right")
    save_figure(fig, out, "figure_3_diagnostics")


def run(input_path, output_path):
    if output_path.exists():
        raise FileExistsError(f"Use a new output directory: {output_path}")
    if any(folder.casefold() in ("sources", "raw") for folder in output_path.parts):
        raise ValueError("Keep experiment outputs outside the reference and raw-data folders.")

    time, signals = load_heat(input_path)
    dense_time = grid(time[0], time[-1])
    curves = {(signal, method): fit(method, time, values)
              for signal, values in signals.items() for method in METHODS[signal]}
    scores = evaluate_gaps(time, signals)
    peaks = evaluate_peaks(time, signals, curves)
    physical = physical_checks(time, dense_time, signals, curves)
    metadata = {
        "heat": HEAT, "input_filename": input_path.name, "input_sha256": INPUT_HASH,
        "observations": len(time), "sample_step_s": DT, "max_time_s": float(time[-1]),
        "grid_step_s": GRID_STEP, "folds": FOLDS, "gap_sizes_samples": GAPS,
        "nominal_segment_spacing_s": SPACING, "peak_holdout_half_width_s": PEAK_HALF_WIDTH,
        "robust_knot_placement": "snap to nearest retained observation and merge duplicates",
        "huber_scale_floor": HUBER_SCALE_FLOOR,
        "peak_prominence_range_fraction": PEAK_PROMINENCE_FRACTION,
        "peak_minimum_separation_s": PEAK_SEPARATION_S, "physics_tolerance": EPSILON,
        "units": {"oxygen": "m^3", "co2": "m^3/s", "co": "m^3/s"},
        "package_versions": {package: version(package) for package in
                             ("numpy", "scipy", "pandas", "python-calamine", "matplotlib")},
    }
    reconstructed = {"time_s": dense_time}
    for (signal, method), model in curves.items():
        reconstructed[f"{signal}_{method}"] = model(dense_time)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_path.parent, prefix=".heat_01_") as temp:
        out = Path(temp) / "result"
        out.mkdir()
        scores.to_csv(out / "holdout_errors.csv", index=False, float_format="%.10g")
        peaks.to_csv(out / "peak_retention.csv", index=False, float_format="%.10g")
        physical.to_csv(out / "physical_checks.csv", index=False, float_format="%.10g")
        pd.DataFrame(reconstructed).to_csv(out / "reconstructed_curves.csv", index=False,
                                           float_format="%.10g")
        (out / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        with plt.rc_context({
            "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
            "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.linewidth": 0.65,
            "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.25,
            "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
        }):
            plot_signals(out, time, dense_time, signals, curves, peaks)
            plot_errors(out, scores)
            plot_diagnostics(out, time, curves, peaks)
        out.rename(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT, help="Corrected first-heat XLSB")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="New experiment directory")
    args = parser.parse_args()
    run(args.input.resolve(), args.output_dir.resolve())
    print(f"Heat {HEAT} experiment saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
