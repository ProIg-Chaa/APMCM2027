
"""
Ten-heat dynamic delay application.
Preserves all previous experiment results.
"""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .delay_interface import (
        BASELINE_DELAY_S, CO_DYNAMIC_MODEL,
        CO2_VALIDATED_FLOW_RANGE,
        candidate_delay_for_flow,
        rolling_window_mean_flow,
        interpolate_window_delay,
        align_gas_signal_dynamic,
        align_gas_signal,
    )
except ImportError:
    from delay_interface import (
        BASELINE_DELAY_S, CO_DYNAMIC_MODEL,
        CO2_VALIDATED_FLOW_RANGE,
        candidate_delay_for_flow,
        rolling_window_mean_flow,
        interpolate_window_delay,
        align_gas_signal_dynamic,
        align_gas_signal,
    )

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "results" / "question2"
    / "final_process_relationships" / "per_heat_curves"
)

OUTPUT_ROOT = (
    ROOT / "results" / "question2"
    / "dynamic_delay_validation"
)

DT = 0.5
WINDOW_S = 120.0
STEP_S = 30.0


def new_output_dir():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_ROOT / f"run_{stamp}"
    k = 1

    while out.exists():
        out = OUTPUT_ROOT / f"run_{stamp}_{k:02d}"
        k += 1

    out.mkdir(parents=True, exist_ok=False)
    return out


def check_helpers():
    t = np.arange(0.25, 300.0, 0.5)

    centers, means = rolling_window_mean_flow(
        t, np.full(len(t), 4.5), WINDOW_S, STEP_S
    )

    assert len(centers) >= 3
    assert np.allclose(means, 4.5)

    target = np.array([
        centers[0],
        (centers[0] + centers[1]) / 2,
        centers[1]
    ])

    delay, valid = interpolate_window_delay(
        "CO", target, centers[:2], means[:2]
    )

    expected = 83.1775 - 10.5724 * 4.5
    assert valid.all()
    assert np.allclose(delay, expected)

    fallback, flag = interpolate_window_delay(
        "CO",
        np.array([centers[0]]),
        np.array([centers[0]]),
        np.array([7.0])
    )

    assert fallback[0] == BASELINE_DELAY_S["co"]
    assert not flag[0]

    gt = np.arange(0.0, 12.0)
    gv = 2 * gt + 1
    et = np.array([0.0, 1.0, 2.0])
    ds = np.array([1.0, 1.5, 2.0])

    aligned = align_gas_signal_dynamic(gt, gv, et, ds)

    assert np.allclose(aligned, 2 * (et + ds) + 1)

    edge = align_gas_signal_dynamic(
        gt, gv, np.array([10.0]), np.array([3.0])
    )
    assert np.isnan(edge[0])

    return "5 helper checks passed"


def process_one(path, out_dir):
    df = pd.read_csv(path)

    required = [
        "time_s", "oxygen_lance_flow_m3_s",
        "co_flow_m3_s", "co2_flow_m3_s",
        "co_cumulative_m3", "co2_cumulative_m3"
    ]

    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"{path.name}: missing {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    t = df["time_s"].to_numpy(float)

    if not np.isfinite(t).all() or not np.allclose(
        np.diff(t), DT, atol=1e-8, rtol=0
    ):
        raise ValueError(f"{path.name}: invalid time grid")

    oxygen = df["oxygen_lance_flow_m3_s"].to_numpy(float)

    if not np.isfinite(oxygen[1:]).all():
        raise ValueError(f"{path.name}: missing oxygen flow")

    # Oxygen flow is assigned to the right endpoint.
    # Reassign it to the interval midpoint.
    mid = (t[:-1] + t[1:]) / 2
    q = oxygen[1:]

    wc, wq = rolling_window_mean_flow(
        mid, q, WINDOW_S, STEP_S
    )

    # Delay at interval midpoints.
    co_dm, co_vm = interpolate_window_delay(
        "CO", mid, wc, wq
    )
    co2_dm, co2_vm = interpolate_window_delay(
        "CO2", mid, wc, wq
    )

    # Delay at original sample times for cumulative emissions.
    co_ds, co_vs = interpolate_window_delay(
        "CO", t, wc, wq
    )
    co2_ds, co2_vs = interpolate_window_delay(
        "CO2", t, wc, wq
    )

    co = df["co_flow_m3_s"].to_numpy(float)
    co2 = df["co2_flow_m3_s"].to_numpy(float)
    coc = df["co_cumulative_m3"].to_numpy(float)
    co2c = df["co2_cumulative_m3"].to_numpy(float)

    if not all(np.isfinite(x).all() for x in (co, co2, coc, co2c)):
        raise ValueError(f"{path.name}: missing gas signals")

    # Dynamic alignment.
    co_dyn = align_gas_signal_dynamic(t, co, mid, co_dm)
    co2_dyn = align_gas_signal_dynamic(t, co2, mid, co2_dm)

    # Fixed alignment for comparison.
    co_fix = align_gas_signal(
        t, co, mid, BASELINE_DELAY_S["co"]
    )
    co2_fix = align_gas_signal(
        t, co2, mid, BASELINE_DELAY_S["co2"]
    )

    # Cumulative emissions.
    coc_dyn = align_gas_signal_dynamic(t, coc, t, co_ds)
    co2c_dyn = align_gas_signal_dynamic(t, co2c, t, co2_ds)

    coc_fix = align_gas_signal(
        t, coc, t, BASELINE_DELAY_S["co"]
    )
    co2c_fix = align_gas_signal(
        t, co2c, t, BASELINE_DELAY_S["co2"]
    )

    for arr in (coc_dyn, co2c_dyn, coc_fix, co2c_fix):
        if not np.isfinite(arr[0]):
            raise RuntimeError(
                f"{path.name}: cumulative baseline unavailable"
            )

    coc_dyn -= coc_dyn[0]
    co2c_dyn -= co2c_dyn[0]
    coc_fix -= coc_fix[0]
    co2c_fix -= co2c_fix[0]

    def expand(x):
        y = np.full(len(t), np.nan)
        y[1:] = x
        return y

    out = df.copy()

    out["oxygen_interval_midpoint_s"] = expand(mid)

    out["oxygen_flow_120s_mean_m3_s"] = np.interp(
        t, wc, wq, left=np.nan, right=np.nan
    )

    out["co_dynamic_delay_s"] = co_ds
    out["co_dynamic_valid"] = co_vs

    out["co2_delay_s"] = co2_ds
    out["co2_dynamic_valid"] = co2_vs

    out["co_delay_at_interval_midpoint_s"] = expand(co_dm)
    out["co2_delay_at_interval_midpoint_s"] = expand(co2_dm)

    out["co_flow_fixed_aligned_m3_s"] = expand(co_fix)
    out["co_flow_dynamic_aligned_m3_s"] = expand(co_dyn)

    out["co2_flow_fixed_aligned_m3_s"] = expand(co2_fix)
    out["co2_flow_dynamic_aligned_m3_s"] = expand(co2_dyn)

    out["co_cumulative_fixed_aligned_m3"] = coc_fix
    out["co_cumulative_dynamic_aligned_m3"] = coc_dyn

    out["co2_cumulative_fixed_aligned_m3"] = co2c_fix
    out["co2_cumulative_dynamic_aligned_m3"] = co2c_dyn

    match = re.search(r"heat_(\d+)_", path.name)
    heat = int(match.group(1)) if match else -1

    out_name = f"heat_{heat:02d}_dynamic_aligned.csv"

    out.to_csv(
        out_dir / out_name,
        index=False,
        encoding="utf-8-sig"
    )

    # Auditable rolling-window delay estimates.
    wco = [candidate_delay_for_flow("CO", x) for x in wq]
    wco2 = [candidate_delay_for_flow("CO2", x) for x in wq]

    windows = pd.DataFrame({
        "heat_number": heat,
        "window_center_s": wc,
        "oxygen_flow_120s_mean_m3_s": wq,
        "co_delay_s": [x["delay_s"] for x in wco],
        "co_dynamic_valid": [x["valid"] for x in wco],
        "co2_delay_s": [x["delay_s"] for x in wco2],
        "co2_dynamic_valid": [x["valid"] for x in wco2],
    })

    summary = {
        "heat_number": heat,
        "rows": len(df),
        "rolling_windows": len(wc),
        "flow_mean_min": float(np.min(wq)),
        "flow_mean_max": float(np.max(wq)),
        "co_valid_windows": int(sum(x["valid"] for x in wco)),
        "co2_valid_windows": int(sum(x["valid"] for x in wco2)),
        "co_dynamic_aligned_points": int(np.isfinite(co_dyn).sum()),
        "co_fixed_aligned_points": int(np.isfinite(co_fix).sum()),
        "co2_dynamic_aligned_points": int(np.isfinite(co2_dyn).sum()),
        "co2_fixed_aligned_points": int(np.isfinite(co2_fix).sum()),
        "output_file": out_name,
    }

    return summary, windows


def main():
    print("Dynamic helper validation:", check_helpers())

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    files = sorted(INPUT_DIR.glob("heat_*_six_signals.csv"))

    if len(files) != 10:
        raise RuntimeError(f"Expected 10 heats, found {len(files)}")

    out_dir = new_output_dir()

    summaries = []
    all_windows = []

    for path in files:
        print("Processing", path.name)

        summary, windows = process_one(path, out_dir)

        summaries.append(summary)
        all_windows.append(windows)

        print(
            f"  windows={summary['rolling_windows']}; "
            f"CO valid={summary['co_valid_windows']}"
        )

    summary_df = pd.DataFrame(summaries)
    windows_df = pd.concat(all_windows, ignore_index=True)

    summary_df.to_csv(
        out_dir / "dynamic_alignment_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    windows_df.to_csv(
        out_dir / "rolling_window_delay_estimates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Flow-delay graph.
    fig, ax = plt.subplots(figsize=(9, 6))

    good = windows_df[windows_df["co_dynamic_valid"]]
    bad = windows_df[~windows_df["co_dynamic_valid"]]

    ax.scatter(
        good["oxygen_flow_120s_mean_m3_s"],
        good["co_delay_s"],
        s=24, alpha=.65,
        label="CO candidate support"
    )

    ax.scatter(
        bad["oxygen_flow_120s_mean_m3_s"],
        bad["co_delay_s"],
        s=20, marker="x", alpha=.5,
        label="CO fixed fallback"
    )

    qline = np.linspace(
        *CO_DYNAMIC_MODEL["flow_range"], 150
    )

    ax.plot(
        qline,
        CO_DYNAMIC_MODEL["intercept_s"]
        + CO_DYNAMIC_MODEL["slope_s_per_m3_s"] * qline,
        label="CO: 83.1775 - 10.5724 Q"
    )

    ax.axhline(
        BASELINE_DELAY_S["co"],
        linestyle="--", label="CO fixed 32 s"
    )

    ax.axhline(
        BASELINE_DELAY_S["co2"],
        linestyle=":", label="CO2 fixed 34 s"
    )

    ax.set_xlabel("120-second mean oxygen-lance flow (m3/s)")
    ax.set_ylabel("Effective delay (s)")
    ax.set_title("Ten-heat rolling-window flow-delay candidate")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_dir / "figure_flow_delay_candidate.png", dpi=250
    )
    fig.savefig(
        out_dir / "figure_flow_delay_candidate.pdf"
    )
    plt.close(fig)

    metadata = {
        "experiment": "Ten-heat dynamic delay candidate alignment",
        "created_at": datetime.now().astimezone().isoformat(),
        "window_length_s": WINDOW_S,
        "window_step_s": STEP_S,
        "flow_source": "oxygen-lance flow at interval midpoints",
        "window_grid_anchor": (
            "First valid interval midpoint; confirm against V2 training."
        ),
        "flow_unit_note": (
            "Inherited from source/V2 data; source-unit verification remains pending."
        ),
        "CO_candidate": CO_DYNAMIC_MODEL,
        "CO2_model": {
            "type": "fixed baseline",
            "delay_s": BASELINE_DELAY_S["co2"],
            "observed_flow_range": CO2_VALIDATED_FLOW_RANGE,
        },
        "alignment_formula": (
            "analyzer_time = event_time + delay(event_time)"
        ),
        "interpolation": (
            "Linear between adjacent windows only when both "
            "are inside validated flow support."
        ),
        "limitation": (
            "This applies the candidate model; it does not prove "
            "better predictive accuracy. Compare both models "
            "on identical held-out windows."
        ),
        "heat_summaries": summaries,
    }

    (out_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    (out_dir / "README.md").write_text(
        "# Ten-Heat Dynamic Delay Candidate Alignment\n\n"
        f"- Window: {WINDOW_S:.0f} s; step: {STEP_S:.0f} s.\n"
        f"- CO candidate: delay = 83.1775 - 10.5724 × Q.\n"
        f"- CO support: {CO_DYNAMIC_MODEL['flow_range']} m3/s.\n"
        f"- CO outside support: fixed {BASELINE_DELAY_S['co']} s.\n"
        f"- CO2: fixed {BASELINE_DELAY_S['co2']} s.\n"
        "- Positive delay: analyzer_time = event_time + delay(event_time).\n"
        "- No endpoint extrapolation.\n"
        "- Candidate alignment does not prove improved accuracy.\n",
        encoding="utf-8"
    )

    print("\nCOMPLETED:", out_dir)
    print(summary_df.to_string(index=False))
    print("Previous results were not overwritten.")


if __name__ == "__main__":
    main()
