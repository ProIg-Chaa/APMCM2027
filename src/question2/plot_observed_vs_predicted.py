
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT / "results" / "question2" /
    "delay_model_v2_2" / "loho_predictions.csv"
)

OUT_DIR = ROOT / "deliverables" / "question2" / "03_uncertainty"

OUT_PNG = OUT_DIR / "observed_vs_predicted_delay.png"
OUT_PDF = OUT_DIR / "observed_vs_predicted_delay.pdf"


# =========================================================
# 2. DISPLAY CONFIGURATION
# =========================================================

MODEL_ORDER = [
    "fixed",
    "flow_grouped",
    "monotone_linear"
]

MODEL_LABELS = {
    "fixed": "Fixed",
    "flow_grouped": "Flow-grouped",
    "monotone_linear": "Monotone linear"
}

SIGNAL_ORDER = ["co", "co2"]

SIGNAL_TITLES = {
    "co": "(a) CO Delay Prediction",
    "co2": "(b) CO₂ Delay Prediction"
}

# Three models: distinct colors and markers
MODEL_COLORS = {
    "fixed": "#2878B5",
    "flow_grouped": "#E28A43",
    "monotone_linear": "#369779"
}

MODEL_MARKERS = {
    "fixed": "o",
    "flow_grouped": "s",
    "monotone_linear": "^"
}

TITLE_COLOR = "#20364D"
TEXT_COLOR = "#34495E"
SUBTITLE_COLOR = "#65758B"
REFERENCE_COLOR = "#333333"


# =========================================================
# 3. DATA PREPARATION
# =========================================================

def load_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {CSV_FILE}"
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "signal",
        "model",
        "test_attachment",
        "heat",
        "observed_delay_s",
        "predicted_delay_s"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    for col in [
        "observed_delay_s",
        "predicted_delay_s"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "observed_delay_s",
            "predicted_delay_s"
        ]
    ).copy()

    df = df[
        df["signal"].isin(SIGNAL_ORDER)
        & df["model"].isin(MODEL_ORDER)
    ].copy()

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True
    )

    df = df.sort_values(
        ["signal", "test_attachment", "model"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid prediction records.")

    return df


# =========================================================
# 4. AXIS LIMITS
# =========================================================

def get_axis_limits(panel):

    values = np.concatenate([
        panel["observed_delay_s"].to_numpy(),
        panel["predicted_delay_s"].to_numpy()
    ])

    lower = min(0, np.nanmin(values))
    upper = np.nanmax(values)

    margin = max((upper - lower) * 0.08, 2)

    lower = max(0, lower - margin)
    upper = upper + margin

    return lower, upper


# =========================================================
# 5. DRAW ONE PANEL
# =========================================================

def draw_panel(ax, df, signal):

    panel = df[
        df["signal"] == signal
    ].copy()

    if panel.empty:
        raise ValueError(
            f"No data available for {signal}"
        )

    # Scatter points for three models
    for model in MODEL_ORDER:

        subset = panel[
            panel["model"] == model
        ]

        if subset.empty:
            continue

        ax.scatter(
            subset["observed_delay_s"],
            subset["predicted_delay_s"],
            s=38,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.45,
            label=MODEL_LABELS[model],
            zorder=3
        )

    # Ideal prediction reference line
    lower, upper = get_axis_limits(panel)

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color=REFERENCE_COLOR,
        linewidth=1.8,
        label="Ideal prediction (y = x)",
        zorder=2
    )

    # Axis settings
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(
        "Observed delay (s)",
        fontsize=12,
        labelpad=9
    )

    ax.set_ylabel(
        "Predicted delay (s)",
        fontsize=12,
        labelpad=9
    )

    ax.set_title(
        SIGNAL_TITLES[signal],
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=14
    )

    # Grid
    ax.grid(
        True,
        linestyle="-",
        linewidth=0.7,
        alpha=0.22,
        color="#8996A5"
    )

    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=10,
        length=3
    )

    # Sample count
    ax.text(
        0.97,
        0.04,
        f"n = {len(panel)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=SUBTITLE_COLOR
    )

    return panel


# =========================================================
# 6. MAIN
# =========================================================

def main():

    print("=" * 65)
    print("GENERATING OBSERVED VS PREDICTED DELAY FIGURE")
    print("=" * 65)

    df = load_data()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white"
    })

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        constrained_layout=False
    )

    panels = {}

    for i, signal in enumerate(SIGNAL_ORDER):

        panels[signal] = draw_panel(
            axes[i],
            df,
            signal
        )

    # -----------------------------------------------------
    # Shared legend
    # -----------------------------------------------------

    legend_handles = []

    for model in MODEL_ORDER:

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model],
                color="none",
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor="white",
                markersize=8,
                label=MODEL_LABELS[model]
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="Ideal prediction (y = x)"
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.0,
        columnspacing=2.0
    )

    # -----------------------------------------------------
    # Main title
    # -----------------------------------------------------

    fig.suptitle(
        "Observed vs. Predicted Delay",
        fontsize=25,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.985
    )

    fig.text(
        0.5,
        0.915,
        "Leave-one-heat-out predictions across 10 heats",
        ha="center",
        fontsize=13,
        color=SUBTITLE_COLOR
    )

    # -----------------------------------------------------
    # Bottom caption
    # -----------------------------------------------------

    fig.text(
        0.5,
        0.045,
        "The dashed diagonal represents perfect agreement "
        "between observed and predicted delays.",
        ha="center",
        fontsize=10.5,
        color=SUBTITLE_COLOR
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.77,
        bottom=0.14,
        wspace=0.25
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUT_PNG,
        dpi=350,
        bbox_inches="tight"
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight"
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Console report
    # -----------------------------------------------------

    print("\nDATA SUMMARY")
    print("-" * 65)

    for signal in SIGNAL_ORDER:

        panel = panels[signal]

        print(
            f"\n{signal.upper()}: "
            f"{len(panel)} prediction records"
        )

        for model in MODEL_ORDER:

            subset = panel[
                panel["model"] == model
            ]

            if subset.empty:
                continue

            errors = (
                subset["predicted_delay_s"]
                - subset["observed_delay_s"]
            )

            rmse = np.sqrt(
                np.mean(errors ** 2)
            )

            mae = np.mean(
                np.abs(errors)
            )

            print(
                f"  {MODEL_LABELS[model]:18s} "
                f"n={len(subset):4d} "
                f"RMSE={rmse:7.3f}s "
                f"MAE={mae:7.3f}s"
            )

    print("\n" + "=" * 65)
    print("FIGURE GENERATED SUCCESSFULLY")
    print("=" * 65)

    print("Input:", CSV_FILE)
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()
