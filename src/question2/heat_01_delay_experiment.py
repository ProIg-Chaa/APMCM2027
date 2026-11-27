"""Estimate and diagnose effective flue-gas delays for corrected heat 6B03205."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "question2" / "heat_01_reconstruction"
OUTPUT = ROOT / "results" / "question2" / "heat_01_delay"
DT = 0.5
WINDOW_S = 120.0
STEP_S = 30.0
MAX_LAG_S = 60.0
MIN_CORRELATION = 0.45
MIN_PEAK_CONTRAST = 0.08
MIN_FLOW_STD = 0.10
MIN_RELIABLE_WINDOWS = 8
SIGNALS = {
    "co2": ("co2_robust_linear", r"CO$_2$"),
    "co": ("co_robust_linear", "CO"),
}


def load_reconstruction(folder):
    metadata = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))
    if metadata["heat"] != "6B03205" or metadata["sample_step_s"] != DT:
        raise ValueError("This experiment accepts the corrected first-heat reconstruction.")
    dense = pd.read_csv(folder / "reconstructed_curves.csv")
    native = dense.iloc[:: round(DT / metadata["grid_step_s"])].reset_index(drop=True)
    if not np.allclose(np.diff(native["time_s"]), DT):
        raise ValueError("The reconstructed curves do not contain the expected 0.5 s grid.")
    return metadata, native


def interval_signals(native):
    time = native["time_s"].to_numpy()
    midpoint = (time[:-1] + time[1:]) / 2
    oxygen = native["oxygen_monotone_linear"].to_numpy()
    flow = np.diff(oxygen) / DT
    gases = {}
    for name, (column, _) in SIGNALS.items():
        values = native[column].to_numpy()
        gases[name] = (values[:-1] + values[1:]) / 2
    return midpoint, flow, gases


def correlation_curve(flow, gas, indices, lag_samples):
    values = []
    for lag in lag_samples:
        valid = indices[indices + lag < len(flow)]
        source = detrend(flow[valid])
        target = detrend(gas[valid + lag])
        if np.std(source) < 1e-12 or np.std(target) < 1e-12:
            values.append(np.nan)
        else:
            values.append(np.corrcoef(source, target)[0, 1])
    return np.asarray(values)


def summarize_curve(correlations, lags_s):
    best_index = int(np.nanargmax(correlations))
    return {
        "delay_s": lags_s[best_index],
        "correlation": correlations[best_index],
        "peak_contrast": correlations[best_index] - np.nanmedian(correlations),
        "boundary_peak": best_index in (0, len(lags_s) - 1),
    }


def estimate_delays(midpoint, flow, gases):
    lag_samples = np.arange(round(MAX_LAG_S / DT) + 1)
    lags_s = lag_samples * DT
    all_indices = np.arange(len(midpoint))
    fixed_rows = []
    window_rows = []
    curve_rows = []

    for signal, gas in gases.items():
        fixed = summarize_curve(correlation_curve(flow, gas, all_indices, lag_samples), lags_s)
        fixed_rows.append({"signal": signal, **fixed})

        centers = np.arange(WINDOW_S / 2, midpoint[-1] - WINDOW_S / 2 + 1e-9, STEP_S)
        for center in centers:
            indices = np.flatnonzero(
                (midpoint >= center - WINDOW_S / 2) & (midpoint <= center + WINDOW_S / 2)
            )
            correlations = correlation_curve(flow, gas, indices, lag_samples)
            summary = summarize_curve(correlations, lags_s)
            flow_mean = flow[indices].mean()
            flow_std = flow[indices].std()
            reliable = (
                summary["correlation"] >= MIN_CORRELATION
                and summary["peak_contrast"] >= MIN_PEAK_CONTRAST
                and flow_std >= MIN_FLOW_STD
                and not summary["boundary_peak"]
            )
            window_rows.append({
                "signal": signal,
                "window_center_s": center,
                "window_start_s": center - WINDOW_S / 2,
                "window_end_s": center + WINDOW_S / 2,
                "mean_lance_flow": flow_mean,
                "lance_flow_std": flow_std,
                **summary,
                "reliable": reliable,
            })
            curve_rows.extend({
                "signal": signal,
                "window_center_s": center,
                "lag_s": lag_s,
                "correlation": correlation,
            } for lag_s, correlation in zip(lags_s, correlations))

    return (
        pd.DataFrame(fixed_rows),
        pd.DataFrame(window_rows),
        pd.DataFrame(curve_rows),
    )


def model_readiness(windows):
    rows = []
    for signal in SIGNALS:
        accepted = windows[(windows["signal"] == signal) & windows["reliable"]]
        if len(accepted) >= 3:
            relationship = spearmanr(accepted["mean_lance_flow"], accepted["delay_s"])
            rho, p_value = relationship.statistic, relationship.pvalue
        else:
            rho, p_value = np.nan, np.nan
        rows.append({
            "signal": signal,
            "reliable_windows": len(accepted),
            "minimum_required": MIN_RELIABLE_WINDOWS,
            "flow_min": accepted["mean_lance_flow"].min() if len(accepted) else np.nan,
            "flow_max": accepted["mean_lance_flow"].max() if len(accepted) else np.nan,
            "spearman_rho": rho,
            "spearman_p_value": p_value,
            "ready_for_delay_flow_model": len(accepted) >= MIN_RELIABLE_WINDOWS,
        })
    return pd.DataFrame(rows)


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


def plot_overview(folder, midpoint, flow, gases, fixed):
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 6.8), sharex=True, layout="constrained")
    axes[0].plot(midpoint, flow, color="#266b59")
    style(axes[0], "(a)  Oxygen-lance flow from monotone reconstruction", r"m$^3$/s")
    for row, signal in enumerate(("co2", "co"), start=1):
        fixed_delay = fixed.set_index("signal").loc[signal]
        axes[row].plot(midpoint, gases[signal], color="#235883")
        title = f"({chr(97 + row)})  {SIGNALS[signal][1]} robust reconstruction"
        title += f" · global candidate {fixed_delay.delay_s:.1f} s"
        if fixed_delay.boundary_peak:
            title += " (search boundary)"
        style(axes[row], title, r"m$^3$/s")
    axes[-1].set_xlabel("Elapsed time (s)")
    save(fig, folder, "figure_1_flow_and_gases")


def plot_delay_diagnostics(folder, windows, curves, readiness):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2), layout="constrained")
    for column, signal in enumerate(("co2", "co")):
        subset = curves[curves["signal"] == signal]
        matrix = subset.pivot(index="window_center_s", columns="lag_s", values="correlation")
        image = axes[0, column].imshow(
            matrix.to_numpy(), aspect="auto", origin="lower",
            extent=[matrix.columns.min(), matrix.columns.max(),
                    matrix.index.min(), matrix.index.max()],
            cmap="RdBu_r", vmin=-1, vmax=1,
        )
        axes[0, column].set_title(f"({chr(97 + column)})  {SIGNALS[signal][1]} local correlation",
                                  loc="left")
        axes[0, column].set_xlabel("Candidate delay (s)")
        axes[0, column].set_ylabel("Window center (s)")
        fig.colorbar(image, ax=axes[0, column], label="Detrended correlation", shrink=0.82)

        estimates = windows[windows["signal"] == signal]
        accepted = estimates[estimates["reliable"]]
        rejected = estimates[~estimates["reliable"]]
        axes[1, column].scatter(rejected["mean_lance_flow"], rejected["delay_s"],
                                marker="x", color="#9ba1a8", label="Rejected window")
        scatter = axes[1, column].scatter(
            accepted["mean_lance_flow"], accepted["delay_s"],
            c=accepted["correlation"], cmap="Blues", vmin=MIN_CORRELATION, vmax=1,
            edgecolor="#1d405b", linewidth=0.5, label="Reliable window",
        )
        count = readiness.set_index("signal").loc[signal, "reliable_windows"]
        style(axes[1, column],
              f"({chr(99 + column)})  {SIGNALS[signal][1]} delay vs flow · {count} reliable",
              "Estimated effective delay (s)")
        axes[1, column].set_xlabel(r"Mean oxygen-lance flow (m$^3$/s)")
        axes[1, column].legend(frameon=False, fontsize=8)
        if len(accepted):
            fig.colorbar(scatter, ax=axes[1, column], label="Peak correlation", shrink=0.82)
    save(fig, folder, "figure_2_delay_diagnostics")


def run(input_folder, output_folder):
    if output_folder.exists():
        raise FileExistsError(f"Use a new output directory: {output_folder}")
    metadata, native = load_reconstruction(input_folder)
    midpoint, flow, gases = interval_signals(native)
    fixed, windows, curves = estimate_delays(midpoint, flow, gases)
    readiness = model_readiness(windows)

    output_folder.mkdir(parents=True)
    pd.DataFrame({
        "time_s": midpoint,
        "oxygen_lance_flow_m3_s": flow,
        "co2_flow_m3_s": gases["co2"],
        "co_flow_m3_s": gases["co"],
    }).to_csv(output_folder / "aligned_interval_signals.csv", index=False, float_format="%.10g")
    fixed.to_csv(output_folder / "fixed_delay_candidates.csv", index=False, float_format="%.10g")
    windows.to_csv(output_folder / "window_delay_estimates.csv", index=False, float_format="%.10g")
    curves.to_csv(output_folder / "window_correlation_curves.csv", index=False, float_format="%.10g")
    readiness.to_csv(output_folder / "model_readiness.csv", index=False, float_format="%.10g")
    parameters = {
        "heat": metadata["heat"],
        "source_reconstruction": str(input_folder),
        "sample_step_s": DT,
        "window_s": WINDOW_S,
        "window_step_s": STEP_S,
        "lag_search_s": [0.0, MAX_LAG_S],
        "preprocessing": "linear detrend within each window",
        "reliability": {
            "minimum_peak_correlation": MIN_CORRELATION,
            "minimum_peak_minus_median_correlation": MIN_PEAK_CONTRAST,
            "minimum_lance_flow_standard_deviation": MIN_FLOW_STD,
            "exclude_lag_search_boundaries": True,
        },
        "minimum_reliable_windows_for_first_model": MIN_RELIABLE_WINDOWS,
        "interpretation": "effective delay includes reaction, transport, and instrument response",
    }
    (output_folder / "experiment.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with plt.rc_context({
        "font.family": "serif", "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "axes.linewidth": 0.65, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "pdf.fonttype": 42,
    }):
        plot_overview(output_folder, midpoint, flow, gases, fixed)
        plot_delay_diagnostics(output_folder, windows, curves, readiness)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"First-heat delay diagnostics saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
