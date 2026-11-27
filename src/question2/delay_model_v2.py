
"""
Question 2 - V2 Dynamic Delay Models

Compare:
1. Fixed delay baseline
2. Monotone decreasing linear regression
3. Low / medium / high oxygen-lance flow grouping

Validation: Leave-One-Heat-Out (LOHO)
Input: V1 all_window_delay_estimates.csv
Output: Independent V2 experiment directory
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "results" / "question2"
    / "ten_heat_reconstruction_delay"
    / "all_window_delay_estimates.csv"
)

OUTPUT = ROOT / "results" / "question2" / "delay_model_v2_2"

SIGNALS = ["co2", "co"]
N_GROUPS = 3
MIN_TRAIN = 12
MIN_GROUP = 4


def fit_fixed(train):
    return float(train["delay_s"].median())


def fit_monotone(train):
    x = train["mean_lance_flow"].to_numpy(float)
    y = train["delay_s"].to_numpy(float)

    if len(x) < MIN_TRAIN or np.ptp(x) < 1e-10:
        return None

    design = np.column_stack([np.ones(len(x)), x])

    result = lsq_linear(
        design,
        y,
        bounds=([-np.inf, -np.inf], [np.inf, 0]),
    )

    if not result.success:
        return None

    return result.x


def predict_monotone(model, x):
    if model is None:
        return np.full(len(x), np.nan)

    return model[0] + model[1] * np.asarray(x)



def fit_grouped(train):
    """
    Genuine three-bin step model.
    Quantile boundaries are estimated from training heats only.
    Each bin predicts its own median delay.
    """
    x = train["mean_lance_flow"].to_numpy(float)
    y = train["delay_s"].to_numpy(float)

    edges = np.quantile(x, [0, 1/3, 2/3, 1])
    edges = np.unique(edges)

    if len(edges) != 4:
        return None

    group_ids = np.digitize(x, edges[1:-1])

    levels = []
    counts = []

    for group_id in range(3):
        mask = group_ids == group_id

        if mask.sum() < MIN_GROUP:
            return None

        levels.append(float(np.median(y[mask])))
        counts.append(int(mask.sum()))

    return {
        "edges": edges,
        "levels": np.asarray(levels),
        "counts": np.asarray(counts),
    }


def predict_grouped(model, x):
    if model is None:
        return np.full(len(x), np.nan)

    x = np.asarray(x, dtype=float)

    group_ids = np.digitize(
        x,
        model["edges"][1:-1]
    )

    return model["levels"][group_ids]


def evaluate(input_path=INPUT, output_dir=OUTPUT):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    data = pd.read_csv(input_path)

    required = {
        "signal",
        "attachment",
        "heat",
        "mean_lance_flow",
        "delay_s",
        "reliable",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Only evaluate the reliable-window population.
    data = data[data["reliable"].astype(str).str.lower()
                .isin(["true", "1"])].copy()

    data = data.dropna(
        subset=["mean_lance_flow", "delay_s"]
    )

    data = data[
        np.isfinite(data["mean_lance_flow"])
        & np.isfinite(data["delay_s"])
    ]

    output_dir.mkdir(parents=True, exist_ok=False)

    prediction_rows = []
    fold_rows = []
    group_rows = []

    for signal in SIGNALS:

        signal_data = data[data["signal"] == signal]

        heats = sorted(signal_data["attachment"].unique())

        for heat_id in heats:

            train = signal_data[
                signal_data["attachment"] != heat_id
            ].copy()

            test = signal_data[
                signal_data["attachment"] == heat_id
            ].copy()

            if len(train) < MIN_TRAIN or len(test) == 0:
                continue

            x_test = test["mean_lance_flow"].to_numpy(float)
            y_test = test["delay_s"].to_numpy(float)

            fixed = fit_fixed(train)
            linear = fit_monotone(train)
            grouped = fit_grouped(train)

            predictions = {
                "fixed": np.full(len(test), fixed),
                "monotone_linear": predict_monotone(
                    linear, x_test
                ),
                "flow_grouped": predict_grouped(
                    grouped, x_test
                ),
            }

            for model_name, predicted in predictions.items():

                valid = np.isfinite(predicted)

                if not valid.any():
                    continue

                errors = predicted[valid] - y_test[valid]

                fold_rows.append({
                    "signal": signal,
                    "model": model_name,
                    "test_attachment": int(heat_id),
                    "test_heat": test["heat"].iloc[0],
                    "test_windows": int(valid.sum()),
                    "rmse_s": float(
                        np.sqrt(np.mean(errors ** 2))
                    ),
                    "mae_s": float(
                        np.mean(np.abs(errors))
                    ),
                })

                for idx in np.where(valid)[0]:
                    prediction_rows.append({
                        "signal": signal,
                        "model": model_name,
                        "test_attachment": int(heat_id),
                        "heat": test["heat"].iloc[idx],
                        "flow": x_test[idx],
                        "observed_delay_s": y_test[idx],
                        "predicted_delay_s": predicted[idx],
                    })

            # Save training-only group information.
            if grouped is not None:
                edges = grouped["edges"]

                for i, (level, count) in enumerate(
                    zip(grouped["levels"], grouped["counts"])
                ):
                    center = (edges[i] + edges[i + 1]) / 2

                    group_rows.append({
                        "signal": signal,
                        "test_attachment": int(heat_id),
                        "group": i + 1,
                        "training_flow_lower": edges[i],
                        "training_flow_upper": edges[i + 1],
                        "training_flow_center": center,
                        "training_median_delay_s": level,
                        "training_windows": count,
                    })

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    groups = pd.DataFrame(group_rows)

    if folds.empty:
        raise RuntimeError("No valid LOHO folds.")

    # Each heat receives equal weight.
    metrics = (
        folds.groupby(["signal", "model"])
        .agg(
            mean_heat_rmse_s=("rmse_s", "mean"),
            mean_heat_mae_s=("mae_s", "mean"),
            test_heats=("test_attachment", "nunique"),
            total_test_windows=("test_windows", "sum"),
        )
        .reset_index()
    )

    predictions.to_csv(
        output_dir / "loho_predictions.csv", index=False
    )
    folds.to_csv(
        output_dir / "loho_per_heat_metrics.csv", index=False
    )
    groups.to_csv(
        output_dir / "training_flow_groups.csv", index=False
    )
    metrics.to_csv(
        output_dir / "model_comparison.csv", index=False
    )

    # Descriptive full-data flow groups, NOT validation predictions.
    descriptive_rows = []

    for signal in SIGNALS:
        subset = data[data["signal"] == signal]

        subset = subset.copy()

        subset["flow_group"] = pd.qcut(
            subset["mean_lance_flow"],
            q=N_GROUPS,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )

        for name, group in subset.groupby(
            "flow_group", observed=True
        ):
            descriptive_rows.append({
                "signal": signal,
                "flow_group": str(name),
                "windows": len(group),
                "heats": group["attachment"].nunique(),
                "mean_flow": group["mean_lance_flow"].mean(),
                "median_delay_s": group["delay_s"].median(),
                "q25_delay_s": group["delay_s"].quantile(0.25),
                "q75_delay_s": group["delay_s"].quantile(0.75),
            })

    descriptive = pd.DataFrame(descriptive_rows)

    descriptive.to_csv(
        output_dir / "descriptive_flow_groups.csv",
        index=False,
    )

    # Model comparison figure.
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained"
    )

    for ax, signal in zip(axes, SIGNALS):
        subset = metrics[metrics["signal"] == signal]

        ax.bar(
            subset["model"],
            subset["mean_heat_rmse_s"],
        )

        ax.set_title(f"{signal.upper()} delay prediction")
        ax.set_ylabel("Equal-weight mean heat RMSE (s)")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(output_dir / "model_comparison.png", dpi=300)
    fig.savefig(output_dir / "model_comparison.pdf")
    plt.close(fig)

    metadata = {
        "input": str(input_path),
        "validation": "leave-one-heat-out",
        "primary_metric": "equal-weight mean heat RMSE",
        "models": [
            "fixed",
            "monotone_linear",
            "flow_grouped",
        ],
        "flow_groups": N_GROUPS,
        "minimum_training_windows": MIN_TRAIN,
        "minimum_group_windows": MIN_GROUP,
        "important_limit": (
            "Evaluation is conditional on V1 reliable windows. "
            "This experiment does not independently validate "
            "the V1 window reliability filter or lag estimation."
        ),
    }

    (output_dir / "experiment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nV2 model comparison:")
    print(metrics.to_string(index=False))

    print("\nDescriptive flow groups:")
    print(descriptive.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluate(args.input.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
