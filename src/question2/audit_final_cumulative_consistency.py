
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "results" / "question2" /
    "final_aligned_dataset" / "run_20261127_162754"
)

INPUT = DATA_DIR / "all_heats_final_aligned.csv"
OUT = DATA_DIR / "cumulative_consistency_audit.csv"

df = pd.read_csv(INPUT)

results = []

print("=" * 70)
print("CUMULATIVE VOLUME CONSISTENCY AUDIT")
print("=" * 70)

for heat, d in df.groupby("heat_number"):

    d = d.sort_values("time_s").reset_index(drop=True)

    t = d["time_s"].to_numpy(dtype=float)
    dt = np.diff(t)

    def integrate_flow(column):
        flow = pd.to_numeric(
            d[column], errors="coerce"
        ).to_numpy(dtype=float)

        # Flow at each row represents its corresponding
        # oxygen-lance interval midpoint.
        valid = np.isfinite(flow[1:]) & (dt > 0)

        integral = np.sum(flow[1:][valid] * dt[valid])

        return float(integral), int(valid.sum())

    def last_value(column):
        s = pd.to_numeric(
            d[column], errors="coerce"
        ).dropna()

        return float(s.iloc[-1]) if len(s) else np.nan

    def stats(column):
        s = pd.to_numeric(
            d[column], errors="coerce"
        ).dropna()

        if len(s) == 0:
            return np.nan, np.nan, np.nan

        return (
            float(s.min()),
            float(s.median()),
            float(s.max())
        )

    signals = [
        ("oxygen_lance_flow_m3_s",
         "oxygen_cumulative_m3",
         "oxygen"),

        ("co_flow_final_aligned_m3_s",
         "co_cumulative_final_aligned_m3",
         "co"),

        ("co2_flow_final_aligned_m3_s",
         "co2_cumulative_final_aligned_m3",
         "co2"),
    ]

    for flow_col, cumulative_col, name in signals:

        integrated, valid_intervals = integrate_flow(flow_col)

        cumulative = last_value(cumulative_col)

        minimum, median, maximum = stats(flow_col)

        ratio = (
            cumulative / integrated
            if np.isfinite(cumulative) and integrated > 0
            else np.nan
        )

        results.append({
            "heat_number": int(heat),
            "signal": name,
            "valid_intervals": valid_intervals,
            "flow_min": minimum,
            "flow_median": median,
            "flow_max": maximum,
            "integrated_flow_volume": integrated,
            "reported_cumulative_volume": cumulative,
            "cumulative_to_integral_ratio": ratio,
            "difference": cumulative - integrated
            if np.isfinite(cumulative) else np.nan,
        })

    # Check missing-value positions
    for name, column in [
        ("CO", "co_flow_final_aligned_m3_s"),
        ("CO2", "co2_flow_final_aligned_m3_s"),
    ]:
        missing = d[column].isna()

        missing_times = d.loc[missing, "time_s"]

        print(
            f"Heat {int(heat):02d} {name}: "
            f"missing={int(missing.sum())}, "
            f"first_missing="
            f"{missing_times.min() if len(missing_times) else 'None'}, "
            f"last_missing="
            f"{missing_times.max() if len(missing_times) else 'None'}"
        )

audit = pd.DataFrame(results)

audit.to_csv(
    OUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("VOLUME CONSISTENCY RESULTS")
print("=" * 70)

print(
    audit.round(4).to_string(index=False)
)

print("\nSaved:")
print(OUT)

print("\nAudit completed. Original data preserved.")
