
"""
Question 2 - Corrected Dynamic Delay Validation

Purpose:
1. Apply support-restricted dynamic CO delay.
2. Keep CO2 delay fixed at 34 seconds.
3. Check monotonicity of event-to-analyzer time mapping.
4. Fall back to fixed CO delay for the entire heat
   if dynamic mapping is not strictly increasing.
5. Audit cumulative emissions monotonicity.
6. Preserve all previous experiment results.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .delay_interface import (
        BASELINE_DELAY_S,
        CO_DYNAMIC_MODEL,
        candidate_delay_for_flow,
    )
except ImportError:
    from delay_interface import (
        BASELINE_DELAY_S,
        CO_DYNAMIC_MODEL,
        candidate_delay_for_flow,
    )


ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT / "results" / "question2"
    / "final_process_relationships" / "per_heat_curves"
)

OUTPUT_ROOT = (
    ROOT / "results" / "question2"
    / "dynamic_delay_validation_corrected"
)

DT = 0.5
WINDOW_S = 120.0
STEP_S = 30.0
TOL = 1e-8


# =========================================================
# 1. Rolling window calculation
# =========================================================

def rolling_window_mean_flow(time, flow, window, step):

    time = np.asarray(time, dtype=float)
    flow = np.asarray(flow, dtype=float)

    if len(time) != len(flow):
        raise ValueError("Time and flow lengths differ.")

    if not np.all(np.diff(time) > 0):
        raise ValueError("Time must be strictly increasing.")

    if not np.isfinite(flow).all():
        raise ValueError("Flow contains missing values.")

    starts = np.arange(
        time[0],
        time[-1] - window + 1e-9,
        step
    )

    centers = []
    means = []

    for start in starts:

        mask = (
            (time >= start)
            & (time < start + window)
        )

        if not mask.any():
            continue

        centers.append(start + window / 2)
        means.append(np.mean(flow[mask]))

    return (
        np.asarray(centers),
        np.asarray(means)
    )


# =========================================================
# 2. Calculate delay on arbitrary event-time grids
# =========================================================

def interpolate_window_delay(
    gas,
    target_time,
    window_centers,
    window_flows
):

    target_time = np.asarray(target_time, dtype=float)

    baseline = BASELINE_DELAY_S[gas.lower()]

    delays = np.full(
        len(target_time),
        baseline,
        dtype=float
    )

    valid = np.zeros(
        len(target_time),
        dtype=bool
    )

    flow_estimates = np.full(
        len(target_time),
        np.nan
    )

    supported = (
        (target_time >= window_centers[0])
        & (target_time <= window_centers[-1])
    )

    flow_estimates[supported] = np.interp(
        target_time[supported],
        window_centers,
        window_flows
    )

    for i in np.where(supported)[0]:

        result = candidate_delay_for_flow(
            gas,
            flow_estimates[i]
        )

        delays[i] = result["delay_s"]
        valid[i] = result["valid"]

    return delays, valid, flow_estimates


# =========================================================
# 3. Event-to-analyzer time mapping
# =========================================================

def mapping_diagnostics(event_time, delay):

    event_time = np.asarray(event_time, dtype=float)
    delay = np.asarray(delay, dtype=float)

    analyzer_time = event_time + delay

    differences = np.diff(analyzer_time)

    monotonic = bool(
        np.all(differences > TOL)
    )

    minimum_step = (
        float(np.min(differences))
        if len(differences)
        else None
    )

    return {
        "monotonic": monotonic,
        "minimum_step_s": minimum_step,
        "analyzer_time": analyzer_time
    }


# =========================================================
# 4. Variable-delay interpolation
# =========================================================

def align_variable_delay(
    gas_time,
    gas_values,
    event_time,
    delay
):

    gas_time = np.asarray(gas_time, dtype=float)
    gas_values = np.asarray(gas_values, dtype=float)
    event_time = np.asarray(event_time, dtype=float)
    delay = np.asarray(delay, dtype=float)

    if not (
        len(event_time)
        == len(delay)
    ):
        raise ValueError("Delay and target lengths differ.")

    if not np.all(np.diff(gas_time) > 0):
        raise ValueError("Gas time must be strictly increasing.")

    if not np.isfinite(delay).all():
        raise ValueError("Delay contains non-finite values.")

    diagnostics = mapping_diagnostics(
        event_time,
        delay
    )

    if not diagnostics["monotonic"]:
        raise ValueError(
            "Event-to-analyzer mapping is not monotonic."
        )

    return np.interp(
        diagnostics["analyzer_time"],
        gas_time,
        gas_values,
        left=np.nan,
        right=np.nan
    )


def align_fixed_delay(
    gas_time,
    gas_values,
    event_time,
    delay_s
):

    delay = np.full(
        len(event_time),
        float(delay_s)
    )

    return align_variable_delay(
        gas_time,
        gas_values,
        event_time,
        delay
    )


# =========================================================
# 5. Cumulative emission audit
# =========================================================

def count_cumulative_drops(values):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) < 2:
        return 0

    return int(
        np.sum(np.diff(values) < -TOL)
    )


def validate_source_cumulative(values, name):

    drops = count_cumulative_drops(values)

    if drops > 0:
        raise ValueError(
            f"{name} source cumulative has {drops} drops."
        )


# =========================================================
# 6. Output directory
# =========================================================

def new_output_dir():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out = OUTPUT_ROOT / f"run_{stamp}"

    k = 1

    while out.exists():

        out = OUTPUT_ROOT / f"run_{stamp}_{k:02d}"
        k += 1

    out.mkdir(
        parents=True,
        exist_ok=False
    )

    return out


# =========================================================
# 7. Process one heat
# =========================================================

def process_one(path, out_dir):

    df = pd.read_csv(path)

    required = [
        "time_s",
        "oxygen_lance_flow_m3_s",
        "co_flow_m3_s",
        "co2_flow_m3_s",
        "co_cumulative_m3",
        "co2_cumulative_m3"
    ]

    for col in required:

        if col not in df.columns:
            raise ValueError(
                f"{path.name}: missing {col}"
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    t = df["time_s"].to_numpy(float)

    if not np.isfinite(t).all():
        raise ValueError("Invalid time values.")

    if not np.allclose(
        np.diff(t),
        DT,
        atol=1e-8,
        rtol=0
    ):
        raise ValueError("Invalid time interval.")

    oxygen = df[
        "oxygen_lance_flow_m3_s"
    ].to_numpy(float)

    if not np.isfinite(oxygen[1:]).all():
        raise ValueError("Missing oxygen flow.")

    co = df["co_flow_m3_s"].to_numpy(float)
    co2 = df["co2_flow_m3_s"].to_numpy(float)

    coc = df["co_cumulative_m3"].to_numpy(float)
    co2c = df["co2_cumulative_m3"].to_numpy(float)

    if not all(
        np.isfinite(x).all()
        for x in (co, co2, coc, co2c)
    ):
        raise ValueError("Missing gas observations.")

    validate_source_cumulative(coc, "CO")
    validate_source_cumulative(co2c, "CO2")

    # Oxygen flow is assigned to right endpoints.
    mid = (t[:-1] + t[1:]) / 2
    q = oxygen[1:]

    # 120-second moving window.
    wc, wq = rolling_window_mean_flow(
        mid,
        q,
        WINDOW_S,
        STEP_S
    )

    if len(wc) < 2:
        raise ValueError("Insufficient rolling windows.")

    # Calculate CO delay.
    co_dm, co_vm, co_qm = interpolate_window_delay(
        "co", mid, wc, wq
    )

    co_ds, co_vs, co_qs = interpolate_window_delay(
        "co", t, wc, wq
    )

    # CO2 remains a fixed-delay baseline.
    co2_dm, co2_vm, co2_qm = interpolate_window_delay(
        "co2", mid, wc, wq
    )

    co2_ds, co2_vs, co2_qs = interpolate_window_delay(
        "co2", t, wc, wq
    )

    # -----------------------------------------------------
    # Check monotonicity before applying dynamic CO delay.
    # -----------------------------------------------------

    mid_check = mapping_diagnostics(
        mid,
        co_dm
    )

    sample_check = mapping_diagnostics(
        t,
        co_ds
    )

    has_dynamic_support = bool(
    np.any(co_vm) and np.any(co_vs)
    )

    co_fallback = not (
        has_dynamic_support
        and mid_check["monotonic"]
        and sample_check["monotonic"]
    )

    fallback_reason = None

    if not has_dynamic_support:
        fallback_reason = "no_valid_dynamic_support"

    elif co_fallback:
        fallback_reason = (
            "non_monotonic_event_to_analyzer_mapping"
    )
        co_dm = np.full(
            len(mid),
            BASELINE_DELAY_S["co"]
        )

        co_ds = np.full(
            len(t),
            BASELINE_DELAY_S["co"]
        )

        co_vm = np.zeros(len(mid), dtype=bool)
        co_vs = np.zeros(len(t), dtype=bool)

    # -----------------------------------------------------
    # Dynamic and fixed alignment
    # -----------------------------------------------------

    co_dyn = align_variable_delay(
        t, co, mid, co_dm
    )

    co_fix = align_fixed_delay(
        t, co, mid,
        BASELINE_DELAY_S["co"]
    )

    co2_dyn = align_fixed_delay(
        t, co2, mid,
        BASELINE_DELAY_S["co2"]
    )

    co2_fix = co2_dyn.copy()

    # Cumulative emissions.
    coc_dyn = align_variable_delay(
        t, coc, t, co_ds
    )

    coc_fix = align_fixed_delay(
        t, coc, t,
        BASELINE_DELAY_S["co"]
    )

    co2c_dyn = align_fixed_delay(
        t, co2c, t,
        BASELINE_DELAY_S["co2"]
    )

    co2c_fix = co2c_dyn.copy()

    # Subtract initial aligned cumulative baseline.
    for arr in (
        coc_dyn,
        coc_fix,
        co2c_dyn,
        co2c_fix
    ):

        if not np.isfinite(arr[0]):
            raise RuntimeError(
                "Initial cumulative baseline unavailable."
            )

        arr -= arr[0]

    # If dynamic CO cumulative series still decreases,
    # use fixed delay for the entire heat.
    co_dynamic_drops = count_cumulative_drops(
        coc_dyn
    )

    if co_dynamic_drops > 0 and not co_fallback:

        co_fallback = True

        fallback_reason = (
            "cumulative_emission_monotonicity_failure"
        )

        co_dm = np.full(
            len(mid),
            BASELINE_DELAY_S["co"]
        )

        co_ds = np.full(
            len(t),
            BASELINE_DELAY_S["co"]
        )

        co_vm = np.zeros(len(mid), dtype=bool)
        co_vs = np.zeros(len(t), dtype=bool)

        co_dyn = co_fix.copy()

        coc_dyn = coc_fix.copy()

    # Final cumulative audit.
    drops = {
        "co_dynamic": count_cumulative_drops(coc_dyn),
        "co_fixed": count_cumulative_drops(coc_fix),
        "co2_dynamic": count_cumulative_drops(co2c_dyn),
        "co2_fixed": count_cumulative_drops(co2c_fix)
    }

    if any(v > 0 for v in drops.values()):
        raise RuntimeError(
            f"Cumulative monotonicity failed: {drops}"
        )

    # -----------------------------------------------------
    # Build output
    # -----------------------------------------------------

    def expand(x):

        y = np.full(len(t), np.nan)
        y[1:] = x

        return y

    out = df.copy()

    out["oxygen_interval_midpoint_s"] = expand(mid)

    out["oxygen_flow_120s_mean_m3_s"] = np.interp(
        t,
        wc,
        wq,
        left=np.nan,
        right=np.nan
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

    match = re.search(
        r"heat_(\d+)_",
        path.name
    )

    heat = int(match.group(1)) if match else -1

    output_file = (
        f"heat_{heat:02d}_corrected_aligned.csv"
    )

    out.to_csv(
        out_dir / output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Rolling-window estimates.
    window_results = [
        candidate_delay_for_flow("co", x)
        for x in wq
    ]

    window_results_co2 = [
        candidate_delay_for_flow("co2", x)
        for x in wq
    ]

    windows = pd.DataFrame({

        "heat_number": heat,

        "window_center_s": wc,

        "oxygen_flow_120s_mean_m3_s": wq,

        "co_candidate_delay_s": [
            x["delay_s"] for x in window_results
        ],

        "co_support_valid": [
            x["valid"] for x in window_results
        ],

        "co2_delay_s": [
            x["delay_s"] for x in window_results_co2
        ],

        "co2_support_valid": [
            x["valid"] for x in window_results_co2
        ]
    })

    summary = {

        "heat_number": heat,
        "rows": len(df),
        "rolling_windows": len(wc),

        "flow_mean_min": float(np.min(wq)),
        "flow_mean_max": float(np.max(wq)),

        "co_valid_windows": int(
            sum(x["valid"] for x in window_results)
        ),

        "co2_valid_windows": int(
            sum(x["valid"] for x in window_results_co2)
        ),

        "co_model_used": (
            "fixed_fallback"
            if co_fallback
            else "dynamic_candidate"
        ),

        "co_fallback_reason": fallback_reason,

        "co_mapping_monotonic_before_fallback": (
            mid_check["monotonic"]
            and sample_check["monotonic"]
        ),

        "co_mapping_min_step_before_fallback_s": (
            sample_check["minimum_step_s"]
        ),

        "co_cumulative_drops": drops["co_dynamic"],

        "co2_cumulative_drops": drops["co2_dynamic"],

        "co_dynamic_aligned_points": int(
            np.isfinite(co_dyn).sum()
        ),

        "co_fixed_aligned_points": int(
            np.isfinite(co_fix).sum()
        ),

        "co2_dynamic_aligned_points": int(
            np.isfinite(co2_dyn).sum()
        ),

        "output_file": output_file
    }

    return summary, windows


# =========================================================
# 8. Main
# =========================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    files = sorted(
        INPUT_DIR.glob("heat_*_six_signals.csv")
    )

    if len(files) != 10:
        raise RuntimeError(
            f"Expected 10 heats, found {len(files)}"
        )

    out_dir = new_output_dir()

    summaries = []
    all_windows = []

    print("=" * 65)
    print("Question 2 - Corrected Dynamic Delay Validation")
    print("=" * 65)

    for path in files:

        print("\nProcessing:", path.name)

        summary, windows = process_one(
            path,
            out_dir
        )

        summaries.append(summary)
        all_windows.append(windows)

        print(
            "  CO model:",
            summary["co_model_used"]
        )

        print(
            "  CO cumulative drops:",
            summary["co_cumulative_drops"]
        )

        print(
            "  CO2 cumulative drops:",
            summary["co2_cumulative_drops"]
        )

    summary_df = pd.DataFrame(summaries)

    windows_df = pd.concat(
        all_windows,
        ignore_index=True
    )

    summary_df.to_csv(
        out_dir / "corrected_alignment_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    windows_df.to_csv(
        out_dir / "rolling_window_delay_estimates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Flow-delay figure.
    fig, ax = plt.subplots(figsize=(9, 6))

    good = windows_df[
        windows_df["co_support_valid"]
    ]

    bad = windows_df[
        ~windows_df["co_support_valid"]
    ]

    ax.scatter(
        good["oxygen_flow_120s_mean_m3_s"],
        good["co_candidate_delay_s"],
        label="CO candidate support",
        alpha=0.65
    )

    ax.scatter(
        bad["oxygen_flow_120s_mean_m3_s"],
        bad["co_candidate_delay_s"],
        marker="x",
        label="CO fixed fallback"
    )

    qline = np.linspace(
        *CO_DYNAMIC_MODEL["flow_range"],
        150
    )

    ax.plot(
        qline,
        CO_DYNAMIC_MODEL["intercept_s"]
        + CO_DYNAMIC_MODEL["slope_s_per_m3_s"] * qline,
        label="CO dynamic candidate"
    )

    ax.axhline(
        BASELINE_DELAY_S["co"],
        linestyle="--",
        label="CO fixed 32 s"
    )

    ax.axhline(
        BASELINE_DELAY_S["co2"],
        linestyle=":",
        label="CO2 fixed 34 s"
    )

    ax.set_xlabel(
        "120-second mean oxygen-lance flow"
    )

    ax.set_ylabel("Effective delay (s)")

    ax.set_title(
        "Corrected Ten-Heat Dynamic Delay"
    )

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        out_dir / "figure_corrected_flow_delay.png",
        dpi=250
    )

    fig.savefig(
        out_dir / "figure_corrected_flow_delay.pdf"
    )

    plt.close(fig)

    metadata = {

        "experiment": (
            "Corrected ten-heat dynamic delay validation"
        ),

        "created_at": datetime.now().astimezone().isoformat(),

        "window_length_s": WINDOW_S,
        "window_step_s": STEP_S,

        "CO_model": CO_DYNAMIC_MODEL,

        "CO2_delay_s": BASELINE_DELAY_S["co2"],

        "mapping_formula": (
            "analyzer_time = event_time + delay(event_time)"
        ),

        "monotonicity_policy": (
            "If CO event-to-analyzer mapping is not strictly "
            "increasing, the entire heat uses fixed 32-second "
            "CO delay."
        ),

        "cumulative_policy": (
            "Cumulative emissions must remain non-decreasing. "
            "Dynamic CO alignment falls back to fixed delay "
            "if its cumulative series decreases."
        ),

        "previous_results_preserved": True,

        "heat_summaries": summaries
    }

    with open(
        out_dir / "experiment.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    readme = """# Corrected Dynamic Delay Validation

## Model

CO:
- Dynamic candidate within validated support.
- Fixed 32-second fallback.
- Whole-heat fallback if time mapping is not monotonic.

CO2:
- Fixed 34-second baseline.

## Time mapping

analyzer_time = event_time + delay(event_time)

The mapping must be strictly increasing.

## Cumulative emissions

CO and CO2 cumulative emissions must be non-decreasing.

## Output

- corrected_alignment_summary.csv
- rolling_window_delay_estimates.csv
- heat_XX_corrected_aligned.csv
- figure_corrected_flow_delay.png
- figure_corrected_flow_delay.pdf
- experiment.json

Previous results are preserved.
"""

    (out_dir / "README.md").write_text(
        readme,
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("CORRECTED VALIDATION COMPLETED")
    print("=" * 65)

    print("\nOutput directory:")
    print(out_dir)

    print("\nSummary:")
    print(
        summary_df[
            [
                "heat_number",
                "co_model_used",
                "co_cumulative_drops",
                "co2_cumulative_drops",
                "co_mapping_monotonic_before_fallback"
            ]
        ].to_string(index=False)
    )

    print("\nAll previous results were preserved.")


if __name__ == "__main__":
    main()
