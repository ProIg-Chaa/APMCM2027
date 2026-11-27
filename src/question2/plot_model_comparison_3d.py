
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Patch
from matplotlib import colors as mcolors


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT / "results" / "question2" /
    "delay_model_v2_2" / "model_comparison.csv"
)

OUT_DIR = CSV_FILE.parent
OUT_PNG = OUT_DIR / "model_comparison_3d.png"
OUT_PDF = OUT_DIR / "model_comparison_3d.pdf"


# =========================================================
# 2. DISPLAY CONFIGURATION
# =========================================================

MODEL_ORDER = ["fixed", "flow_grouped", "monotone_linear"]
MODEL_LABELS = ["Fixed", "Flow-grouped", "Monotone linear"]

SIGNAL_TITLES = {
    "co": "(a) CO Delay Prediction",
    "co2": "(b) CO₂ Delay Prediction"
}

COLORS = {
    "RMSE": "#8DB9DC",
    "MAE": "#F2B477",
    "Best RMSE": "#176B9A",
    "Best MAE": "#C96B16"
}


# =========================================================
# 3. COLOR UTILITY
# =========================================================

def adjust_color(color, factor=1.0):
    """
    factor > 1: lighter
    factor < 1: darker
    """
    rgb = np.array(mcolors.to_rgb(color))

    if factor >= 1:
        adjusted = rgb + (1 - rgb) * (factor - 1)
    else:
        adjusted = rgb * factor

    return np.clip(adjusted, 0, 1)


# =========================================================
# 4. DRAW PSEUDO-3D BAR ON 2D AXIS
# =========================================================

def draw_3d_bar(
    ax,
    x,
    height,
    width,
    facecolor,
    dx,
    dy,
    edgecolor="white",
    linewidth=1.0
):
    """
    Draw pseudo-3D bars on a flat 2D axis.
    """

    # Front face
    front = Rectangle(
        (x, 0),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3
    )
    ax.add_patch(front)

    # Right face
    side = Polygon(
        [
            (x + width, 0),
            (x + width + dx, dy),
            (x + width + dx, height + dy),
            (x + width, height)
        ],
        closed=True,
        facecolor=adjust_color(facecolor, 0.78),
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2.8
    )
    ax.add_patch(side)

    # Top face
    top = Polygon(
        [
            (x, height),
            (x + width, height),
            (x + width + dx, height + dy),
            (x + dx, height + dy)
        ],
        closed=True,
        facecolor=adjust_color(facecolor, 1.10),
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3.2
    )
    ax.add_patch(top)


# =========================================================
# 5. PANEL DRAWING
# =========================================================

def draw_panel(ax, df, signal):

    panel = (
        df[df["signal"] == signal]
        .set_index("model")
        .loc[MODEL_ORDER]
    )

    rmse_values = panel["mean_heat_rmse_s"].to_numpy()
    mae_values = panel["mean_heat_mae_s"].to_numpy()

    best_rmse_idx = int(np.argmin(rmse_values))
    best_mae_idx = int(np.argmin(mae_values))

    centers = np.arange(len(MODEL_ORDER))

    bar_width = 0.28
    gap = 0.08

    rmse_left = centers - bar_width - gap / 2
    mae_left = centers + gap / 2

    ymax = max(rmse_values.max(), mae_values.max())

    ax.set_ylim(0, ymax + 4.0)

    # Pseudo-3D depth
    dx = 0.06
    dy = 0.55

    # RMSE bars
    for i, value in enumerate(rmse_values):

        color = (
            COLORS["Best RMSE"]
            if i == best_rmse_idx
            else COLORS["RMSE"]
        )

        draw_3d_bar(
            ax=ax,
            x=rmse_left[i],
            height=value,
            width=bar_width,
            facecolor=color,
            dx=dx,
            dy=dy
        )

        ax.text(
            rmse_left[i] + bar_width / 2 + dx / 2,
            value + dy + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if i == best_rmse_idx else "normal",
            color="#243447",
            zorder=5
        )

    # MAE bars
    for i, value in enumerate(mae_values):

        color = (
            COLORS["Best MAE"]
            if i == best_mae_idx
            else COLORS["MAE"]
        )

        draw_3d_bar(
            ax=ax,
            x=mae_left[i],
            height=value,
            width=bar_width,
            facecolor=color,
            dx=dx,
            dy=dy
        )

        ax.text(
            mae_left[i] + bar_width / 2 + dx / 2,
            value + dy + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if i == best_mae_idx else "normal",
            color="#243447",
            zorder=5
        )

    # Axis styling
    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4 + dx + 0.2)

    ax.set_xticks(centers)
    ax.set_xticklabels(MODEL_LABELS, fontsize=11)

    ax.set_ylabel(
        "Equal-weight mean heat error (s)",
        fontsize=12
    )

    ax.set_xlabel(
        "Prediction model",
        fontsize=12,
        labelpad=3
    )

    ax.set_title(
        SIGNAL_TITLES[signal],
        fontsize=17,
        fontweight="bold",
        color="#20364D",
        pad=14
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=1.0
    )

    ax.set_axisbelow(True)

    # Flat and clean axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================================================
# 6. MAIN
# =========================================================

def main():

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "signal",
        "model",
        "mean_heat_rmse_s",
        "mean_heat_mae_s"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["mean_heat_rmse_s"] = pd.to_numeric(
        df["mean_heat_rmse_s"],
        errors="raise"
    )

    df["mean_heat_mae_s"] = pd.to_numeric(
        df["mean_heat_mae_s"],
        errors="raise"
    )

    # Stable plotting order
    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True
    )

    df = df.sort_values(
        ["signal", "model"]
    ).reset_index(drop=True)

    # Create figure
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7.8),
        constrained_layout=False
    )

    draw_panel(axes[0], df, "co")
    draw_panel(axes[1], df, "co2")

    # Main title
    fig.suptitle(
        "Comparison of Delay Prediction Models",
        fontsize=27,
        fontweight="bold",
        color="#20364D",
        y=0.97
    )

    # Subtitle
    fig.text(
        0.5,
        0.905,
        "Leave-one-heat-out evaluation | Lower values indicate better performance",
        ha="center",
        fontsize=14,
        color="#65758B"
    )

    # Legend
    legend_items = [
        Patch(
            facecolor=COLORS["RMSE"],
            edgecolor="white",
            label="RMSE"
        ),
        Patch(
            facecolor=COLORS["MAE"],
            edgecolor="white",
            label="MAE"
        ),
        Patch(
            facecolor=COLORS["Best RMSE"],
            edgecolor="white",
            label="Best RMSE"
        ),
        Patch(
            facecolor=COLORS["Best MAE"],
            edgecolor="white",
            label="Best MAE"
        ),
    ]

    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=4,
        frameon=False,
        fontsize=12,
        handlelength=1.8,
        columnspacing=2.0
    )

    # Bottom note
    fig.text(
        0.5,
        0.05,
        "Flat axes + pseudo-3D bars. Darker bars indicate the best-performing model under each metric.",
        ha="center",
        fontsize=11,
        color="#65758B"
    )

    # Layout
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.73,
        bottom=0.16,
        wspace=0.18
    )

    # Save figures
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

    print("=" * 60)
    print("MODEL COMPARISON FIGURE GENERATED")
    print("=" * 60)
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()
