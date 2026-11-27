
"""
Question 2 - Gas signal delay interface.

Time convention:
    positive delay means analyzer observation occurs AFTER
    the corresponding oxygen-lance event.

    event_time = analyzer_time - delay

The delay is an effective delay, not a separately identified
physical transportation time.
"""

from __future__ import annotations

import numpy as np


# Fixed effective-delay baselines from the existing experiments.
BASELINE_DELAY_S = {
    "co": 32.0,
    "co2": 34.0,
}

# Reliable-window flow coverage from the ten-heat analysis.
# These are evidence ranges, NOT physical operating limits.
OBSERVED_FLOW_RANGE = {
    "co": (3.32, 10.76),
    "co2": (3.32, 8.95),
}

TIME_UNIT = "s"
FLOW_UNIT = "m3/s (subject to source-unit verification)"


def normalize_gas(gas: str) -> str:
    """Normalize gas names."""
    name = gas.strip().lower().replace("₂", "2")

    if name not in BASELINE_DELAY_S:
        raise ValueError(
            f"Unsupported gas: {gas}. Use 'CO' or 'CO2'."
        )

    return name


def delay_for_flow(
    gas: str,
    lance_flow: float,
) -> dict:
    """
    Return the fixed baseline delay and its evidence status.

    The flow input is used for applicability checking only.
    No dynamic flow-delay relationship is assumed.
    """
    name = normalize_gas(gas)
    flow = float(lance_flow)

    if not np.isfinite(flow):
        raise ValueError("lance_flow must be finite.")

    lower, upper = OBSERVED_FLOW_RANGE[name]

    in_range = lower <= flow <= upper

    return {
        "gas": name.upper(),
        "delay_s": BASELINE_DELAY_S[name],
        "model": "fixed_baseline",
        "lance_flow": flow,
        "flow_unit": FLOW_UNIT,
        "observed_flow_range": [lower, upper],
        "in_observed_range": in_range,
        "valid": in_range,
        "status": (
            "within_observed_range"
            if in_range
            else "outside_observed_range"
        ),
        "time_reference": "oxygen_lance_interval_midpoint",
        "delay_definition": (
            "analyzer_observation_time minus "
            "corresponding oxygen_lance_event_time"
        ),
        "uncertainty_s": None,
    }


def align_gas_signal(
    gas_time_s,
    gas_values,
    target_event_time_s,
    delay_s: float,
) -> np.ndarray:
    """
    Shift analyzer observations backward and interpolate
    onto the target event-time axis.

    Formula:
        event_time = analyzer_time - delay_s

    Values outside the available observation range are NaN.
    No endpoint extrapolation is performed.
    """
    gas_time = np.asarray(gas_time_s, dtype=float)
    values = np.asarray(gas_values, dtype=float)
    target = np.asarray(target_event_time_s, dtype=float)

    delay = float(delay_s)

    if not np.isfinite(delay) or delay < 0:
        raise ValueError("delay_s must be finite and non-negative.")

    if gas_time.ndim != 1 or values.ndim != 1:
        raise ValueError("Input time and signal must be 1-D arrays.")

    if target.ndim != 1:
        raise ValueError("Target time must be a 1-D array.")

    if len(gas_time) != len(values) or len(gas_time) < 2:
        raise ValueError("Time and signal lengths must match and be >= 2.")

    if not (
        np.isfinite(gas_time).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("Time arrays must contain finite values.")

    if not np.isfinite(values).all():
        raise ValueError("Gas signal contains non-finite values.")

    if not np.all(np.diff(gas_time) > 0):
        raise ValueError("gas_time_s must be strictly increasing.")

    if not np.all(np.diff(target) > 0):
        raise ValueError("target_event_time_s must be strictly increasing.")

    # Move analyzer timestamps back to inferred event timestamps.
    event_time = gas_time - delay

    # Interpolate only inside the supported time range.
    aligned = np.interp(
        target,
        event_time,
        values,
        left=np.nan,
        right=np.nan,
    )

    return aligned


# Support-restricted dynamic delay candidate
# Derived from the existing V2 window-level LOHO data.

CO_DYNAMIC_MODEL = {
    "intercept_s": 83.1775,
    "slope_s_per_m3_s": -10.5724,
    "flow_range": (3.32062738, 6.064958412),
    "window_length_s": 120,
    "window_step_s": 30,
    "loho_rmse_s": 14.9411,
    "loho_mae_s": 12.7915,
    "fold_slope_mean": -10.5365,
    "fold_slope_sd": 1.1230,
}

CO2_VALIDATED_FLOW_RANGE = (
    3.32062738,
    5.994414817
)

def candidate_delay_for_flow(
    gas: str,
    lance_flow: float
) -> dict:
    """
    Return the support-restricted delay candidate.

    IMPORTANT:
    lance_flow must be the 120-second moving-window
    mean, NOT the instantaneous 0.5-second flow.
    """

    name = normalize_gas(gas)
    flow = float(lance_flow)

    if not np.isfinite(flow):
        raise ValueError("Flow must be finite.")

    if name == "co":
        lower, upper = CO_DYNAMIC_MODEL["flow_range"]
        in_range = lower <= flow <= upper

        if in_range:
            delay = (
                CO_DYNAMIC_MODEL["intercept_s"]
                + CO_DYNAMIC_MODEL["slope_s_per_m3_s"] * flow
            )
            model = "support_restricted_monotone_linear"
            rmse = CO_DYNAMIC_MODEL["loho_rmse_s"]
            mae = CO_DYNAMIC_MODEL["loho_mae_s"]
        else:
            delay = BASELINE_DELAY_S["co"]
            model = "fixed_fallback_outside_support"
            rmse = 15.02
            mae = 13.14

    else:
        lower, upper = CO2_VALIDATED_FLOW_RANGE
        in_range = lower <= flow <= upper

        delay = BASELINE_DELAY_S["co2"]
        model = (
            "fixed_baseline"
            if in_range
            else "fixed_fallback_outside_support"
        )
        rmse = 15.60
        mae = 13.55

    return {
        "gas": name.upper(),
        "delay_s": float(delay),
        "model": model,
        "lance_flow_window_mean": flow,
        "flow_unit": FLOW_UNIT,
        "flow_window_s": 120,
        "flow_step_s": 30,
        "validated_flow_range": [lower, upper],
        "valid": in_range,
        "validation_rmse_s": rmse,
        "validation_mae_s": mae,
        "time_reference": "oxygen_lance_interval_midpoint",
        "delay_interpretation": "effective_delay",
    }


def delay_series_for_flow(
    gas: str,
    flow_series
):
    """Calculate a delay series from 120-second mean flows."""

    flows = np.asarray(flow_series, dtype=float)

    results = [
        candidate_delay_for_flow(gas, q)
        for q in flows
    ]

    delays = np.array([
        r["delay_s"] for r in results
    ])

    valid = np.array([
        r["valid"] for r in results
    ])

    return delays, valid


def rolling_window_mean_flow(
    flow_time_s, lance_flow, window_length_s=120.0, step_s=30.0
):
    """Return complete-window mean flows and their center timestamps."""
    t = np.asarray(flow_time_s, dtype=float)
    q = np.asarray(lance_flow, dtype=float)

    if t.ndim != 1 or q.ndim != 1 or len(t) != len(q):
        raise ValueError("Inputs must be matching 1-D arrays.")
    if len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError("At least three finite observations are required.")
    if np.any(q < 0) or window_length_s <= 0 or step_s <= 0:
        raise ValueError("Invalid flow, window length or step.")

    dtv = np.diff(t)
    if not np.all(dtv > 0) or not np.allclose(
        dtv, np.median(dtv), atol=1e-7, rtol=0
    ):
        raise ValueError("Timestamps must be equally spaced.")

    dt = float(np.median(dtv))
    nwf, nsf = window_length_s / dt, step_s / dt
    nw, ns = int(round(nwf)), int(round(nsf))

    if nw < 1 or ns < 1 or not np.isclose(nwf, nw) or not np.isclose(nsf, ns):
        raise ValueError("Window and step must match sampling interval.")
    if len(q) < nw:
        raise ValueError("Insufficient observations.")

    starts = np.arange(0, len(q) - nw + 1, ns, dtype=int)
    centers = np.array([
        (t[i] + t[i + nw - 1]) / 2 for i in starts
    ])
    means = np.array([
        np.mean(q[i:i + nw]) for i in starts
    ], dtype=float)

    return centers, means


def interpolate_window_delay(
    gas, target_event_time_s, window_centers_s, window_flow_means
):
    """
    Interpolate delay only between adjacent windows inside model support.
    Use fixed baseline outside temporal/flow support.
    """
    name = normalize_gas(gas)
    target = np.asarray(target_event_time_s, dtype=float)
    centers = np.asarray(window_centers_s, dtype=float)
    means = np.asarray(window_flow_means, dtype=float)

    if target.ndim != 1 or centers.ndim != 1 or means.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays.")
    if len(centers) != len(means) or len(centers) == 0:
        raise ValueError("Window arrays must have equal non-zero lengths.")
    if not all(np.isfinite(x).all() for x in (target, centers, means)):
        raise ValueError("Inputs must be finite.")
    if len(centers) > 1 and not np.all(np.diff(centers) > 0):
        raise ValueError("Window centers must increase.")
    if len(target) > 1 and not np.all(np.diff(target) > 0):
        raise ValueError("Target times must increase.")

    estimates = [
        candidate_delay_for_flow(name, q) for q in means
    ]
    wd = np.array([x["delay_s"] for x in estimates])
    wv = np.array([x["valid"] for x in estimates], dtype=bool)

    baseline = float(BASELINE_DELAY_S[name])
    delays = np.full(target.shape, baseline)
    valid = np.zeros(target.shape, dtype=bool)

    for i, t in enumerate(target):
        if t < centers[0] or t > centers[-1]:
            continue

        r = int(np.searchsorted(centers, t, side="left"))

        if r < len(centers) and np.isclose(
            t, centers[r], atol=1e-8, rtol=0
        ):
            if wv[r]:
                delays[i], valid[i] = wd[r], True

        elif 0 < r < len(centers) and wv[r-1] and wv[r]:
            a = (t - centers[r-1]) / (centers[r] - centers[r-1])
            delays[i] = wd[r-1] + a * (wd[r] - wd[r-1])
            valid[i] = True

    return delays, valid


def align_gas_signal_dynamic(
    gas_time_s, gas_values, target_event_time_s, delay_series_s
):
    """
    Time-varying alignment:
        analyzer_time = event_time + delay(event_time)
    """
    gt = np.asarray(gas_time_s, dtype=float)
    gv = np.asarray(gas_values, dtype=float)
    et = np.asarray(target_event_time_s, dtype=float)
    ds = np.asarray(delay_series_s, dtype=float)

    if any(x.ndim != 1 for x in (gt, gv, et, ds)):
        raise ValueError("All inputs must be 1-D arrays.")
    if len(gt) != len(gv) or len(et) != len(ds) or len(gt) < 2:
        raise ValueError("Input lengths do not match.")
    if not all(np.isfinite(x).all() for x in (gt, gv, et, ds)):
        raise ValueError("Inputs must be finite.")
    if np.any(ds < 0):
        raise ValueError("Delays must be non-negative.")
    if not np.all(np.diff(gt) > 0):
        raise ValueError("Gas time must increase.")
    if len(et) > 1 and not np.all(np.diff(et) > 0):
        raise ValueError("Event time must increase.")

    return np.interp(
        et + ds,
        gt,
        gv,
        left=np.nan,
        right=np.nan
    )
