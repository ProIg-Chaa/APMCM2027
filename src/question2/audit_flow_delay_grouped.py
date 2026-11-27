
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "results" / "question2"
    / "delay_model_v2_2"
    / "loho_predictions.csv"
)

OUTPUT_ROOT = (
    ROOT / "results" / "question2"
    / "flow_delay_final_audit"
)


def main():

    df = pd.read_csv(INPUT)

    required = [
        "signal", "model", "heat",
        "flow", "observed_delay_s",
        "predicted_delay_s"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in ["flow", "observed_delay_s", "predicted_delay_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[
        "signal", "model", "heat",
        "flow", "observed_delay_s",
        "predicted_delay_s"
    ])

    # Use fixed-model rows to avoid counting the same
    # observed window three times.
    actual = df[df["model"] == "fixed"].copy()

    grouped = df[df["model"] == "flow_grouped"].copy()

    if actual.empty or grouped.empty:
        raise ValueError("Missing fixed or flow_grouped predictions.")

    # Keep the same flow boundaries as the previous audit.
    bins = [-np.inf, 4.2, 5.0, np.inf]
    labels = ["Low", "Medium", "High"]

    for data in [actual, grouped]:
        data["flow_band"] = pd.cut(
            data["flow"],
            bins=bins,
            labels=labels,
            right=False
        )

    # -------------------------------------------------
    # 1. Pooled actual delay by flow band
    # -------------------------------------------------

    actual_summary = (
        actual.groupby(
            ["signal", "flow_band"],
            observed=False
        )
        .agg(
            windows=("observed_delay_s", "size"),
            mean_flow=("flow", "mean"),
            mean_delay=("observed_delay_s", "mean"),
            median_delay=("observed_delay_s", "median"),
            q25=("observed_delay_s", lambda x: x.quantile(.25)),
            q75=("observed_delay_s", lambda x: x.quantile(.75))
        )
        .reset_index()
    )

    # -------------------------------------------------
    # 2. Out-of-sample grouped predictions
    # -------------------------------------------------

    prediction_summary = (
        grouped.groupby(
            ["signal", "flow_band"],
            observed=False
        )
        .agg(
            predicted_windows=("predicted_delay_s", "size"),
            mean_predicted_delay=("predicted_delay_s", "mean")
        )
        .reset_index()
    )

    actual_summary = actual_summary.merge(
        prediction_summary,
        on=["signal", "flow_band"],
        how="left"
    )

    # -------------------------------------------------
    # 3. Heat-level stability
    # -------------------------------------------------

    heat_rows = []

    for (signal, heat), g in actual.groupby(["signal", "heat"]):

        low = g[g["flow_band"] == "Low"]
        medium = g[g["flow_band"] == "Medium"]
        high = g[g["flow_band"] == "High"]

        rho = np.nan

        if len(g) >= 4 and g["flow"].nunique() >= 2:
            rho = g["flow"].corr(
                g["observed_delay_s"],
                method="spearman"
            )

        heat_rows.append({

            "signal": signal,
            "heat": heat,

            "windows": len(g),

            "low_n": len(low),
            "medium_n": len(medium),
            "high_n": len(high),

            "low_median_delay": (
                low["observed_delay_s"].median()
                if len(low) else np.nan
            ),

            "medium_median_delay": (
                medium["observed_delay_s"].median()
                if len(medium) else np.nan
            ),

            "high_median_delay": (
                high["observed_delay_s"].median()
                if len(high) else np.nan
            ),

            "high_minus_low": (
                high["observed_delay_s"].median()
                - low["observed_delay_s"].median()
                if len(high) and len(low)
                else np.nan
            ),

            "spearman_rho": rho
        })

    heat_summary = pd.DataFrame(heat_rows)

    # -------------------------------------------------
    # 4. Overall correlation
    # -------------------------------------------------

    correlation_rows = []

    for signal, g in actual.groupby("signal"):

        correlation_rows.append({

            "signal": signal,
            "windows": len(g),

            "spearman_rho": g["flow"].corr(
                g["observed_delay_s"],
                method="spearman"
            ),

            "pearson_r": g["flow"].corr(
                g["observed_delay_s"],
                method="pearson"
            )
        })

    correlation_summary = pd.DataFrame(correlation_rows)

    # -------------------------------------------------
    # 5. Extreme observed delays
    # -------------------------------------------------

    extreme = actual[
        (actual["observed_delay_s"] < 10)
        | (actual["observed_delay_s"] > 55)
    ].copy()

    # -------------------------------------------------
    # 6. Save all results separately
    # -------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = OUTPUT_ROOT / f"run_{stamp}"

    k = 1
    while out.exists():
        out = OUTPUT_ROOT / f"run_{stamp}_{k:02d}"
        k += 1

    out.mkdir()

    actual_summary.to_csv(
        out / "actual_delay_by_flow_band.csv",
        index=False,
        encoding="utf-8-sig"
    )

    heat_summary.to_csv(
        out / "heat_level_stability.csv",
        index=False,
        encoding="utf-8-sig"
    )

    correlation_summary.to_csv(
        out / "flow_delay_correlations.csv",
        index=False,
        encoding="utf-8-sig"
    )

    extreme.to_csv(
        out / "extreme_delay_windows.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 65)
    print("FLOW-DELAY FINAL AUDIT")
    print("=" * 65)

    print("\n1. Actual delay by flow band")
    print(actual_summary.round(3).to_string(index=False))

    print("\n2. Heat-level stability")
    print(heat_summary.round(3).to_string(index=False))

    print("\n3. Overall correlations")
    print(correlation_summary.round(4).to_string(index=False))

    print("\n4. Extreme observed delay windows")
    print(
        extreme[
            ["signal", "heat", "flow", "observed_delay_s"]
        ].to_string(index=False)
    )

    print("\nOutput directory:")
    print(out)

    print("\nPrevious results preserved.")


if __name__ == "__main__":
    main()
