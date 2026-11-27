
"""
Question 2 - Ten-heat delay interface validation.

Reads the existing six-signal CSV files and applies the
fixed effective-delay baselines through delay_interface.py.

Outputs:
    - Ten delay-aligned heat CSV files
    - Alignment summary CSV
    - Heat 01 alignment figure (PNG and PDF)
    - experiment.json
    - README.md

No original data or previous experiment outputs are modified.
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
        BASELINE_DELAY_S,
        OBSERVED_FLOW_RANGE,
        TIME_UNIT,
        FLOW_UNIT,
        delay_for_flow,
        align_gas_signal,
    )
except ImportError:
    from delay_interface import (
        BASELINE_DELAY_S,
        OBSERVED_FLOW_RANGE,
        TIME_UNIT,
        FLOW_UNIT,
        delay_for_flow,
        align_gas_signal,
    )


# ============================================================
# 1. Project paths and configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    ROOT
    / "results"
    / "question2"
    / "final_process_relationships"
    / "per_heat_curves"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "question2"
    / "delay_interface_validation"
)

EXPECTED_DT = 0.5
EXPECTED_HEATS = 10

REQUIRED_COLUMNS = [
    "time_s",
    "oxygen_lance_flow_m3_s",
    "oxygen_cumulative_m3",
    "co2_flow_m3_s",
    "co_flow_m3_s",
    "co2_cumulative_m3",
    "co_cumulative_m3",
]


# ============================================================
# 2. Utility functions
# ============================================================

def create_run_directory() -> Path:
    """Create a unique output directory without overwriting results."""

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{timestamp}"

    suffix = 1
    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"run_{timestamp}_{suffix:02d}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def get_heat_number(path: Path) -> int:
    """Extract heat number from filename."""

    match = re.search(r"heat_(\d+)_", path.name)

    if not match:
        raise ValueError(f"Cannot identify heat number: {path.name}")

    return int(match.group(1))


def load_heat_data(path: Path) -> pd.DataFrame:
    """Read and validate one heat's six-signal CSV."""

    df = pd.read_csv(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    if len(df) < 3:
        raise ValueError(f"Insufficient rows: {path.name}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    time = df["time_s"].to_numpy(dtype=float)

    if not np.isfinite(time).all():
        raise ValueError(f"Non-finite time values: {path.name}")

    if not np.isclose(time[0], 0.0):
        raise ValueError(f"Time does not start at zero: {path.name}")

    dt = np.diff(time)

    if not np.allclose(
        dt,
        EXPECTED_DT,
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError(
            f"Unexpected time interval in {path.name}"
        )

    if not np.all(np.diff(time) > 0):
        raise ValueError(f"Time is not strictly increasing: {path.name}")

    # The first oxygen-flow value is intentionally missing:
    # there is no preceding cumulative-oxygen observation.
    oxygen_flow = df["oxygen_lance_flow_m3_s"].to_numpy(dtype=float)

    if not np.isfinite(oxygen_flow[1:]).all():
        raise ValueError(
            f"Unexpected missing oxygen flow: {path.name}"
        )

    # Gas flow and cumulative signals must be complete.
    gas_columns = [
        "co2_flow_m3_s",
        "co_flow_m3_s",
        "co2_cumulative_m3",
        "co_cumulative_m3",
        "oxygen_cumulative_m3",
    ]

    for col in gas_columns:
        values = df[col].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in {col}: {path.name}"
            )

    return df


def get_flow_range_status(
    gas: str,
    oxygen_flow: np.ndarray,
) -> tuple[list, list]:
    """
    Evaluate whether each oxygen-lance interval lies inside
    the observed reliable-window flow coverage.

    Out-of-range observations are retained.
    """

    status = ["no_interval"]
    in_range = [None]

    for flow in oxygen_flow[1:]:

        if not np.isfinite(flow):
            status.append("missing_lance_flow")
            in_range.append(None)
            continue

        result = delay_for_flow(gas, flow)

        status.append(result["status"])
        in_range.append(result["in_observed_range"])

    return status, in_range


def run_internal_checks() -> dict:
    """Check delay direction and boundary handling."""

    test_time = np.arange(0.0, 5.0 + 1e-9, 0.5)

    # A linear signal makes the expected interpolation exact.
    test_signal = 2.0 * test_time + 3.0

    target_time = np.arange(0.25, 5.0, 0.5)
    test_delay = 1.0

    aligned = align_gas_signal(
        gas_time_s=test_time,
        gas_values=test_signal,
        target_event_time_s=target_time,
        delay_s=test_delay,
    )

    valid = np.isfinite(aligned)

    expected = (
        2.0 * (target_time[valid] + test_delay) + 3.0
    )

    if not np.allclose(aligned[valid], expected):
        raise AssertionError(
            "Delay direction test failed."
        )

    if not np.isnan(aligned[-1]):
        raise AssertionError(
            "Boundary extrapolation test failed."
        )

    if not valid.any():
        raise AssertionError(
            "No valid aligned test points."
        )

    return {
        "delay_direction_test": "passed",
        "boundary_no_extrapolation_test": "passed",
        "interpolation_test": "passed",
    }


# ============================================================
# 3. Core alignment
# ============================================================

def align_one_heat(
    path: Path,
    output_dir: Path,
) -> dict:

    df = load_heat_data(path)

    time = df["time_s"].to_numpy(dtype=float)

    oxygen_flow = df[
        "oxygen_lance_flow_m3_s"
    ].to_numpy(dtype=float)

    co_flow = df["co_flow_m3_s"].to_numpy(dtype=float)
    co2_flow = df["co2_flow_m3_s"].to_numpy(dtype=float)

    co_cumulative = df[
        "co_cumulative_m3"
    ].to_numpy(dtype=float)

    co2_cumulative = df[
        "co2_cumulative_m3"
    ].to_numpy(dtype=float)

    n = len(df)

    # --------------------------------------------------------
    # A. Oxygen-lance interval midpoint
    # --------------------------------------------------------

    interval_midpoint = (
        time[:-1] + time[1:]
    ) / 2.0

    midpoint_full = np.full(n, np.nan)
    midpoint_full[1:] = interval_midpoint

    # --------------------------------------------------------
    # B. Align instantaneous gas flows
    # --------------------------------------------------------

    co_delay = BASELINE_DELAY_S["co"]
    co2_delay = BASELINE_DELAY_S["co2"]

    co_aligned_interval = align_gas_signal(
        gas_time_s=time,
        gas_values=co_flow,
        target_event_time_s=interval_midpoint,
        delay_s=co_delay,
    )

    co2_aligned_interval = align_gas_signal(
        gas_time_s=time,
        gas_values=co2_flow,
        target_event_time_s=interval_midpoint,
        delay_s=co2_delay,
    )

    # The first row has no oxygen-lance interval.
    co_aligned = np.full(n, np.nan)
    co2_aligned = np.full(n, np.nan)

    co_aligned[1:] = co_aligned_interval
    co2_aligned[1:] = co2_aligned_interval

    # --------------------------------------------------------
    # C. Align cumulative gas emissions
    # --------------------------------------------------------

    co_cumulative_shifted = align_gas_signal(
        gas_time_s=time,
        gas_values=co_cumulative,
        target_event_time_s=time,
        delay_s=co_delay,
    )

    co2_cumulative_shifted = align_gas_signal(
        gas_time_s=time,
        gas_values=co2_cumulative,
        target_event_time_s=time,
        delay_s=co2_delay,
    )

    # Set the aligned cumulative emission at event time t=0
    # to zero. This removes the initial offset introduced by
    # shifting a cumulative signal.
    if not (
        np.isfinite(co_cumulative_shifted[0])
        and np.isfinite(co2_cumulative_shifted[0])
    ):
        raise RuntimeError(
            f"Cannot establish cumulative baseline: {path.name}"
        )

    co_cumulative_baseline = float(
        co_cumulative_shifted[0]
    )

    co2_cumulative_baseline = float(
        co2_cumulative_shifted[0]
    )

    co_cumulative_aligned = (
        co_cumulative_shifted - co_cumulative_baseline
    )

    co2_cumulative_aligned = (
        co2_cumulative_shifted - co2_cumulative_baseline
    )

    # --------------------------------------------------------
    # D. Check observed flow coverage
    # --------------------------------------------------------

    co_status, co_in_range = get_flow_range_status(
        "co",
        oxygen_flow,
    )

    co2_status, co2_in_range = get_flow_range_status(
        "co2",
        oxygen_flow,
    )

    # --------------------------------------------------------
    # E. Build output DataFrame
    # --------------------------------------------------------

    result = df.copy()

    result["oxygen_interval_midpoint_s"] = midpoint_full

    result["co_effective_delay_s"] = co_delay
    result["co2_effective_delay_s"] = co2_delay

    result["co_flow_aligned_m3_s"] = co_aligned
    result["co2_flow_aligned_m3_s"] = co2_aligned

    result["co_cumulative_aligned_m3"] = (
        co_cumulative_aligned
    )

    result["co2_cumulative_aligned_m3"] = (
        co2_cumulative_aligned
    )

    result["co_flow_in_observed_range"] = co_in_range
    result["co2_flow_in_observed_range"] = co2_in_range

    result["co_flow_range_status"] = co_status
    result["co2_flow_range_status"] = co2_status

    result["co_flow_alignment_available"] = (
        np.isfinite(co_aligned)
    )

    result["co2_flow_alignment_available"] = (
        np.isfinite(co2_aligned)
    )

    result["co_cumulative_alignment_available"] = (
        np.isfinite(co_cumulative_aligned)
    )

    result["co2_cumulative_alignment_available"] = (
        np.isfinite(co2_cumulative_aligned)
    )

    heat_number = get_heat_number(path)

    output_name = (
        f"heat_{heat_number:02d}_delay_aligned.csv"
    )

    output_path = output_dir / output_name

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # F. Build heat-level summary
    # --------------------------------------------------------

    def range_statistics(flags: list) -> dict:

        assessed = [
            x for x in flags if x is not None
        ]

        inside = sum(x is True for x in assessed)
        outside = sum(x is False for x in assessed)

        return {
            "in_range_intervals": inside,
            "out_of_range_intervals": outside,
            "assessed_intervals": len(assessed),
            "in_range_fraction": (
                inside / len(assessed)
                if assessed else None
            ),
        }

    co_range_stats = range_statistics(co_in_range)
    co2_range_stats = range_statistics(co2_in_range)

    dt = float(np.median(np.diff(time)))

    summary = {
        "heat_number": heat_number,
        "source_file": path.name,
        "output_file": output_name,
        "rows": n,
        "intervals": n - 1,
        "time_start_s": float(time[0]),
        "time_end_s": float(time[-1]),
        "sampling_interval_s": dt,

        "co_delay_s": co_delay,
        "co2_delay_s": co2_delay,

        "co_flow_valid_aligned_points": int(
            np.isfinite(co_aligned).sum()
        ),
        "co_flow_boundary_nan_points": int(
            np.isnan(co_aligned_interval).sum()
        ),

        "co2_flow_valid_aligned_points": int(
            np.isfinite(co2_aligned).sum()
        ),
        "co2_flow_boundary_nan_points": int(
            np.isnan(co2_aligned_interval).sum()
        ),

        "co_cumulative_valid_points": int(
            np.isfinite(co_cumulative_aligned).sum()
        ),
        "co_cumulative_boundary_nan_points": int(
            np.isnan(co_cumulative_aligned).sum()
        ),

        "co2_cumulative_valid_points": int(
            np.isfinite(co2_cumulative_aligned).sum()
        ),
        "co2_cumulative_boundary_nan_points": int(
            np.isnan(co2_cumulative_aligned).sum()
        ),

        "co_cumulative_baseline_m3": (
            co_cumulative_baseline
        ),
        "co2_cumulative_baseline_m3": (
            co2_cumulative_baseline
        ),

        "oxygen_flow_min_m3_s": float(
            np.min(oxygen_flow[1:])
        ),
        "oxygen_flow_max_m3_s": float(
            np.max(oxygen_flow[1:])
        ),

        "co_flow_range_statistics": co_range_stats,
        "co2_flow_range_statistics": co2_range_stats,
    }

    return {
        "summary": summary,
        "data": result,
        "time": time,
        "midpoint": midpoint_full,
        "co_raw": co_flow,
        "co2_raw": co2_flow,
        "co_aligned": co_aligned,
        "co2_aligned": co2_aligned,
        "co_cumulative_raw": co_cumulative,
        "co2_cumulative_raw": co2_cumulative,
        "co_cumulative_aligned": co_cumulative_aligned,
        "co2_cumulative_aligned": co2_cumulative_aligned,
    }


# ============================================================
# 4. Plotting
# ============================================================

def plot_heat_01(
    result: dict,
    output_dir: Path,
) -> None:
    """Plot raw and delay-aligned signals for Heat 01."""

    time = result["time"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
    )

    fig.suptitle(
        "Heat 01 - Raw and Delay-Aligned Gas Signals",
        fontsize=15,
        fontweight="bold",
    )

    # CO2 instantaneous flow
    ax = axes[0, 0]

    ax.plot(
        time,
        result["co2_raw"],
        label="Raw analyzer CO2 flow",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.plot(
        result["midpoint"],
        result["co2_aligned"],
        label="Aligned CO2 flow",
        linewidth=1.2,
    )

    ax.set_title("CO2 Instantaneous Flow")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (source unit)")
    ax.legend()
    ax.grid(alpha=0.25)

    # CO instantaneous flow
    ax = axes[0, 1]

    ax.plot(
        time,
        result["co_raw"],
        label="Raw analyzer CO flow",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.plot(
        result["midpoint"],
        result["co_aligned"],
        label="Aligned CO flow",
        linewidth=1.2,
    )

    ax.set_title("CO Instantaneous Flow")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flow (source unit)")
    ax.legend()
    ax.grid(alpha=0.25)

    # CO2 cumulative emission
    ax = axes[1, 0]

    ax.plot(
        time,
        result["co2_cumulative_raw"],
        label="Raw cumulative CO2",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.plot(
        time,
        result["co2_cumulative_aligned"],
        label="Aligned cumulative CO2 (zero-based)",
        linewidth=1.2,
    )

    ax.set_title("CO2 Cumulative Emission")
    ax.set_xlabel("Event time (s)")
    ax.set_ylabel("Cumulative volume (source unit)")
    ax.legend()
    ax.grid(alpha=0.25)

    # CO cumulative emission
    ax = axes[1, 1]

    ax.plot(
        time,
        result["co_cumulative_raw"],
        label="Raw cumulative CO",
        linewidth=1.0,
        alpha=0.75,
    )

    ax.plot(
        time,
        result["co_cumulative_aligned"],
        label="Aligned cumulative CO (zero-based)",
        linewidth=1.2,
    )

    ax.set_title("CO Cumulative Emission")
    ax.set_xlabel("Event time (s)")
    ax.set_ylabel("Cumulative volume (source unit)")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        output_dir / "heat_01_alignment.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "heat_01_alignment.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 5. Documentation
# ============================================================

def write_documentation(
    output_dir: Path,
    source_files: list,
    summaries: list,
    checks: dict,
) -> None:

    experiment = {
        "experiment_name": "Ten-Heat Delay Interface Validation",
        "created_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),

        "input_directory": str(INPUT_DIR),
        "output_directory": str(output_dir),

        "source_files": [
            path.name for path in source_files
        ],

        "number_of_heats": len(source_files),

        "sampling_interval_s": EXPECTED_DT,

        "time_convention": {
            "positive_delay": (
                "Analyzer observation occurs after "
                "the corresponding oxygen-lance event."
            ),
            "formula": (
                "event_time = analyzer_time - delay"
            ),
            "instantaneous_flow_reference": (
                "oxygen_lance_interval_midpoint"
            ),
            "cumulative_signal_reference": (
                "oxygen_lance_sample_time"
            ),
        },

        "effective_delays": {
            "CO": {
                "delay_s": BASELINE_DELAY_S["co"],
                "observed_flow_range": (
                    OBSERVED_FLOW_RANGE["co"]
                ),
                "uncertainty_s": None,
            },
            "CO2": {
                "delay_s": BASELINE_DELAY_S["co2"],
                "observed_flow_range": (
                    OBSERVED_FLOW_RANGE["co2"]
                ),
                "uncertainty_s": None,
            },
        },

        "units": {
            "time": TIME_UNIT,
            "flow": FLOW_UNIT,
        },

        "alignment_method": (
            "Linear interpolation using delay_interface.py"
        ),

        "boundary_policy": (
            "No extrapolation. Unavailable boundary values "
            "are recorded as NaN."
        ),

        "flow_range_policy": (
            "Out-of-range intervals are retained and flagged. "
            "Observed flow ranges are evidence coverage, "
            "not physical operating limits."
        ),

        "cumulative_policy": (
            "Cumulative signals are shifted on the sample-time "
            "axis and their aligned value at event time t=0 "
            "is subtracted as the baseline."
        ),

        "physical_interpretation": (
            "These are effective delays. The present data "
            "do not separately identify chemical reaction "
            "delay and gas transportation delay."
        ),

        "uncertainty_note": (
            "Delay uncertainty has not yet been quantified."
        ),

        "internal_checks": checks,

        "heat_summaries": summaries,
    }

    with open(
        output_dir / "experiment.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment,
            f,
            ensure_ascii=False,
            indent=2,
        )

    readme = f"""# Ten-Heat Delay Interface Validation

## 1. Input

Existing six-signal CSV files from:

`{INPUT_DIR}`

Number of heats: {len(source_files)}

Sampling interval: {EXPECTED_DT} s

## 2. Effective-delay baselines

| Gas | Effective delay | Observed flow coverage |
|---|---:|---:|
| CO | {BASELINE_DELAY_S['co']} s | {OBSERVED_FLOW_RANGE['co']} |
| CO2 | {BASELINE_DELAY_S['co2']} s | {OBSERVED_FLOW_RANGE['co2']} |

These are fixed effective-delay baselines, not a fitted
dynamic flow-delay function.

The observed flow ranges describe evidence coverage.
They are not physical operating limits.

## 3. Time convention

Positive delay means the analyzer observes a gas signal
after the corresponding oxygen-lance event.

`event_time = analyzer_time - delay`

Instantaneous gas flows are aligned to oxygen-lance
interval midpoints.

The oxygen-lance flow in the source CSV is assigned to
the right endpoint of each interval. This script associates
it with the corresponding interval midpoint.

## 4. Alignment and boundary handling

- Linear interpolation is used.
- No endpoint extrapolation is performed.
- Unavailable boundary values are NaN.
- Out-of-range flow observations are retained and flagged.
- The first oxygen-lance interval is unavailable because
  no preceding cumulative-oxygen observation exists.

## 5. Cumulative emissions

Cumulative CO and CO2 signals are also shifted to event time.
Their values at event time t=0 are subtracted so that the
aligned cumulative series starts at zero.

Original cumulative signals remain unchanged in the output.

## 6. Physical interpretation

The delays are effective delays. The current analysis does
not separately identify chemical reaction delay and gas
transportation delay.

Delay uncertainty has not yet been quantified.

The source flow units still require verification.

## 7. Outputs

- `heat_XX_delay_aligned.csv`: ten aligned heat datasets.
- `alignment_summary.csv`: per-heat alignment statistics.
- `heat_01_alignment.png`: alignment comparison figure.
- `heat_01_alignment.pdf`: vector version of the figure.
- `experiment.json`: reproducibility metadata.

## 8. Internal checks

{json.dumps(checks, ensure_ascii=False, indent=2)}
"""

    (output_dir / "README.md").write_text(
        readme,
        encoding="utf-8",
    )


# ============================================================
# 6. Main
# ============================================================

def main() -> None:

    print("=" * 65)
    print("Question 2 - Ten-Heat Delay Interface Validation")
    print("=" * 65)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    source_files = sorted(
        INPUT_DIR.glob("heat_*_six_signals.csv"),
        key=get_heat_number,
    )

    if len(source_files) != EXPECTED_HEATS:
        raise RuntimeError(
            f"Expected {EXPECTED_HEATS} heat files, "
            f"but found {len(source_files)}."
        )

    checks = run_internal_checks()

    print("\nInternal tests:")
    for name, status in checks.items():
        print(f"  {name}: {status}")

    output_dir = create_run_directory()

    summaries = []
    heat_01_result = None

    print("\nProcessing ten heats...")

    for path in source_files:

        heat_number = get_heat_number(path)

        print(f"  Processing {path.name} ...")

        result = align_one_heat(
            path,
            output_dir,
        )

        summaries.append(result["summary"])

        if heat_number == 1:
            heat_01_result = result

        print(
            f"    CO valid points: "
            f"{result['summary']['co_flow_valid_aligned_points']}"
        )

        print(
            f"    CO2 valid points: "
            f"{result['summary']['co2_flow_valid_aligned_points']}"
        )

    # Summary table
    summary_df = pd.DataFrame(summaries)

    summary_path = output_dir / "alignment_summary.csv"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Heat 01 comparison figure
    if heat_01_result is None:
        raise RuntimeError("Heat 01 output was not generated.")

    plot_heat_01(
        heat_01_result,
        output_dir,
    )

    # Documentation
    write_documentation(
        output_dir,
        source_files,
        summaries,
        checks,
    )

    # Run log
    log_lines = [
        "Ten-heat delay alignment completed.",
        f"Created: {datetime.now().astimezone().isoformat()}",
        f"Input files: {len(source_files)}",
        f"Output directory: {output_dir}",
        f"CO delay: {BASELINE_DELAY_S['co']} s",
        f"CO2 delay: {BASELINE_DELAY_S['co2']} s",
        "No endpoint extrapolation.",
        "Out-of-range samples retained and flagged.",
    ]

    (output_dir / "run_log.txt").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETED")
    print("=" * 65)

    print(f"\nOutput directory:\n{output_dir}")

    print("\nSummary:")
    print(
        summary_df[
            [
                "heat_number",
                "rows",
                "co_delay_s",
                "co2_delay_s",
                "co_flow_valid_aligned_points",
                "co2_flow_valid_aligned_points",
            ]
        ].to_string(index=False)
    )

    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")

    print("\nAll previous results have been preserved.")


if __name__ == "__main__":
    main()
