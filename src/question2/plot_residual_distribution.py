
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT / "results" / "question2" /
    "delay_model_v2_2" / "loho_predictions.csv"
)

OUT_DIR = ROOT / "deliverables" / "question2" / "03_uncertainty"
OUT_PNG = OUT_DIR / "residual_distribution_by_model.png"
OUT_PDF = OUT_DIR / "residual_distribution_by_model.pdf"


# =========================================================
# 2. DISPLAY CONFIGURATION
# =========================================================

MODEL_ORDER = ["fixed", "flow_grouped", "monotone_linear"]
MODEL_LABELS = ["Fixed", "Flow-grouped", "Monotone linear"]

SIGNAL_ORDER = ["co", "co2"]

SIGNAL_TITLES = {
    "co": "(a) CO residual distribution",
    "co2": "(b) CO₂ residual distribution"
}

TITLE_COLOR = "#20364D"
SUBTITLE_COLOR = "#65758B"
TEXT_DARK = "#243447"

MODEL_COLORS = {
    "fixed": "#3B82B8",
    "flow_grouped": "#E19047",
    "monotone_linear": "#4DA37A"
}


# =========================================================
# 3. DATA LOADING
# =========================================================

def load_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "signal",
        "model",
        "observed_delay_s",
        "predicted_delay_s"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["observed_delay_s"] = pd.to_numeric(
        df["observed_delay_s"], errors="raise"
    )

    df["predicted_delay_s"] = pd.to_numeric(
        df["predicted_delay_s"], errors="raise"
    )

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True
    )

    df["signal"] = pd.Categorical(
        df["signal"],
        categories=SIGNAL_ORDER,
        ordered=True
    )

    # Residual = predicted - observed
    df["residual_s"] = (
        df["predicted_delay_s"] - df["observed_delay_s"]
    )

    df["abs_error_s"] = np.abs(df["residual_s"])

    df = df.sort_values(
        ["signal", "model"]
    ).reset_index(drop=True)

    return df


# =========================================================
# 4. PANEL DRAWING
# =========================================================

def draw_panel(ax, df, signal, global_ylim):

    panel = df[df["signal"] == signal].copy()

    positions = np.arange(1, len(MODEL_ORDER) + 1)

    box_data = [
        panel.loc[
            panel["model"] == model,
            "residual_s"
        ].dropna().to_numpy()

        for model in MODEL_ORDER
    ]

    # -----------------------------------------------------
    # Boxplot
    # -----------------------------------------------------

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,

        medianprops=dict(
            color="white",
            linewidth=2.0
        ),

        whiskerprops=dict(
            color="#7A8797",
            linewidth=1.2
        ),

        capprops=dict(
            color="#7A8797",
            linewidth=1.2
        ),

        boxprops=dict(
            edgecolor="white",
            linewidth=1.2
        )
    )

    for patch, model in zip(bp["boxes"], MODEL_ORDER):

        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.85)

    # -----------------------------------------------------
    # Jittered individual observations
    # -----------------------------------------------------

    rng = np.random.default_rng(20261127)

    for i, model in enumerate(MODEL_ORDER, start=1):

        values = panel.loc[
            panel["model"] == model,
            "residual_s"
        ].dropna().to_numpy()

        x = rng.normal(
            loc=i,
            scale=0.06,
            size=len(values)
        )

        ax.scatter(
            x,
            values,
            s=26,
            alpha=0.32,
            color=MODEL_COLORS[model],
            edgecolors="none",
            zorder=2
        )

        # Mean residual marker
        mean_val = np.mean(values)

        ax.scatter(
            [i],
            [mean_val],
            s=95,
            marker="D",
            color="#1F2937",
            edgecolors="white",
            linewidths=1.0,
            zorder=4
        )

        # MAE and bias statistics
        mae_val = panel.loc[
            panel["model"] == model,
            "abs_error_s"
        ].mean()

        bias_val = np.mean(values)

        ax.text(
            i,
            global_ylim[1] - 1.7,
            f"Bias = {bias_val:.2f}\nMAE = {mae_val:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            color=TEXT_DARK,
            linespacing=1.3
        )

    # -----------------------------------------------------
    # Zero reference line
    # -----------------------------------------------------

    ax.axhline(
        0,
        color="#2F2F2F",
        linestyle="--",
        linewidth=1.5,
        alpha=0.85,
        zorder=1
    )

    # -----------------------------------------------------
    # Axis styling
    # -----------------------------------------------------

    ax.set_xticks(positions)

    ax.set_xticklabels(
        MODEL_LABELS,
        fontsize=11
    )

    ax.set_ylabel(
        "Residual (predicted - observed) (s)",
        fontsize=12
    )

    ax.set_title(
        SIGNAL_TITLES[signal],
        fontsize=16,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=14
    )

    ax.set_ylim(global_ylim)

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=1.0
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Number of predictions
    n_points = len(panel)

    ax.text(
        0.97,
        0.05,
        f"n = {n_points}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=SUBTITLE_COLOR
    )


# =========================================================
# 5. SUMMARY TABLE FOR TERMINAL
# =========================================================

def print_summary(df):

    summary = (
        df.groupby(
            ["signal", "model"],
            observed=True
        )
        .agg(
            n=("residual_s", "size"),
            mean_residual_s=("residual_s", "mean"),
            median_residual_s=("residual_s", "median"),
            std_residual_s=("residual_s", "std"),
            mae_s=("abs_error_s", "mean"),

            rmse_s=(
                "residual_s",
                lambda x: np.sqrt(np.mean(np.square(x)))
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 72)
    print("RESIDUAL DISTRIBUTION SUMMARY")
    print("=" * 72)

    print(summary.to_string(index=False))


# =========================================================
# 6. MAIN
# =========================================================

def main():

    df = load_data()

    print_summary(df)

    # Unified y-limits across both panels
    residual_all = df["residual_s"].dropna().to_numpy()

    ymin = np.floor(residual_all.min() - 2.0)
    ymax = np.ceil(residual_all.max() + 2.0)

    global_ylim = (ymin, ymax)

    plt.rcParams["font.family"] = "DejaVu Sans"

    # -----------------------------------------------------
    # Figure layout
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8.8),
        constrained_layout=False
    )

    draw_panel(
        axes[0],
        df,
        "co",
        global_ylim
    )

    draw_panel(
        axes[1],
        df,
        "co2",
        global_ylim
    )

    # -----------------------------------------------------
    # Main title
    # -----------------------------------------------------

    fig.suptitle(
        "Residual Distribution by Model",
        fontsize=26,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.985
    )

    # Subtitle
    fig.text(
        0.5,
        0.925,
        "Leave-one-heat-out prediction errors across 10 heats",
        ha="center",
        fontsize=14,
        color=SUBTITLE_COLOR
    )

    # Explanation
    fig.text(
        0.5,
        0.890,
        "Residual = predicted delay - observed delay | Boxes show median and interquartile range",
        ha="center",
        fontsize=11.5,
        color=SUBTITLE_COLOR
    )

    # -----------------------------------------------------
    # Bottom explanation
    # -----------------------------------------------------

    fig.text(
        0.5,
        0.045,
        "Jittered points represent individual predictions; black diamonds indicate mean residual.",
        ha="center",
        fontsize=11,
        color=SUBTITLE_COLOR
    )

    fig.text(
        0.5,
        0.020,
        "Positive residuals indicate overestimation; negative residuals indicate underestimation.",
        ha="center",
        fontsize=10.5,
        color=SUBTITLE_COLOR
    )

    # -----------------------------------------------------
    # Adjust layout
    # -----------------------------------------------------

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.79,
        bottom=0.14,
        wspace=0.20
    )

    # -----------------------------------------------------
    # Save figures
    # -----------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUT_PNG,
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.15
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        pad_inches=0.15
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Terminal output
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("RESIDUAL DISTRIBUTION FIGURE GENERATED")
    print("=" * 72)

    print("Input:", CSV_FILE)
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)

    print("\nFigure generation completed successfully.")


if __name__ == "__main__":
    main()
