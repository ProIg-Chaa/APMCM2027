
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT / "results" / "question2" /
    "delay_model_v2_2" / "loho_per_heat_metrics.csv"
)

OUT_DIR = ROOT / "deliverables" / "question2" / "03_uncertainty"
OUT_PNG = OUT_DIR / "heatwise_loho_error_heatmap.png"
OUT_PDF = OUT_DIR / "heatwise_loho_error_heatmap.pdf"


# =========================================================
# 2. DISPLAY CONFIGURATION
# =========================================================

MODEL_ORDER = ["fixed", "flow_grouped", "monotone_linear"]
MODEL_LABELS = ["Fixed", "Flow-grouped", "Monotone linear"]

SIGNAL_ORDER = ["co", "co2"]

HEAT_LABELS = {
    i: f"H{i:02d}" for i in range(1, 11)
}

TITLE_COLOR = "#20364D"
SUBTITLE_COLOR = "#65758B"
TEXT_DARK = "#243447"
GRID_COLOR = "#FFFFFF"

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue",
    [
        "#F7FBFF",
        "#DCEAF6",
        "#A9CAE4",
        "#6FA4CF",
        "#2E6EAA",
        "#173F6B"
    ]
)


# =========================================================
# 3. DATA PREPARATION
# =========================================================

def load_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "signal",
        "model",
        "test_attachment",
        "test_heat",
        "rmse_s",
        "mae_s"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["test_attachment"] = pd.to_numeric(
        df["test_attachment"], errors="raise"
    ).astype(int)

    df["rmse_s"] = pd.to_numeric(
        df["rmse_s"], errors="raise"
    )

    df["mae_s"] = pd.to_numeric(
        df["mae_s"], errors="raise"
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

    df = df.sort_values(
        ["signal", "test_attachment", "model"]
    ).reset_index(drop=True)

    return df


def build_matrix(df, signal, metric_col):

    panel = df[df["signal"] == signal].copy()

    pivot = panel.pivot_table(
        index="test_attachment",
        columns="model",
        values=metric_col,
        aggfunc="first",
        observed=False
    )

    pivot = pivot.reindex(
        index=sorted(HEAT_LABELS.keys()),
        columns=MODEL_ORDER
    )

    matrix = pivot.to_numpy(dtype=float)

    row_labels = [
        HEAT_LABELS[i] for i in pivot.index
    ]

    return matrix, row_labels


# =========================================================
# 4. DATA CONSISTENCY DIAGNOSTIC
# =========================================================

def check_single_window_metrics(df):

    print("\nMETRIC CONSISTENCY CHECK")
    print("-" * 60)

    if "test_windows" not in df.columns:
        print("test_windows column unavailable; skipped.")
        return

    for _, row in df.iterrows():

        if (
            pd.notna(row["test_windows"])
            and int(row["test_windows"]) == 1
        ):
            if np.isclose(
                row["rmse_s"],
                row["mae_s"],
                rtol=1e-6,
                atol=1e-6
            ):
                print(
                    f"{row['signal'].upper()} "
                    f"{int(row['test_attachment']):02d} "
                    f"{row['model']}: "
                    f"RMSE = MAE = {row['rmse_s']:.4f}s "
                    "(single test window)"
                )


# =========================================================
# 5. DRAW HEATMAP
# =========================================================

def draw_heatmap(
    ax,
    matrix,
    row_labels,
    col_labels,
    title,
    cmap,
    vmin,
    vmax
):

    im = ax.imshow(
        matrix,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    # Axis ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(
        col_labels,
        fontsize=11,
        fontweight="medium"
    )

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(
        row_labels,
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        pad=7
    )

    # Panel title
    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=13
    )

    # Cell borders
    ax.set_xticks(
        np.arange(-0.5, len(col_labels), 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, len(row_labels), 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color=GRID_COLOR,
        linestyle="-",
        linewidth=1.3
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    # Remove outer spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Annotate values and highlight best model
    for i in range(matrix.shape[0]):

        row = matrix[i, :]

        valid_values = row[~np.isnan(row)]

        row_best = (
            np.min(valid_values)
            if len(valid_values) > 0
            else np.nan
        )

        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            if np.isnan(value):

                txt = "NA"
                color = TEXT_DARK
                weight = "normal"

            else:

                txt = f"{value:.2f}"

                normalized = (
                    (value - vmin) / (vmax - vmin)
                    if vmax != vmin
                    else 0
                )

                color = (
                    "white"
                    if normalized > 0.55
                    else TEXT_DARK
                )

                weight = (
                    "bold"
                    if np.isclose(value, row_best)
                    else "normal"
                )

            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=10.5,
                color=color,
                fontweight=weight
            )

    return im


# =========================================================
# 6. MAIN
# =========================================================

def main():

    print("=" * 65)
    print("GENERATING HEAT-WISE LOHO ERROR HEATMAP")
    print("=" * 65)

    df = load_data()

    check_single_window_metrics(df)

    # Build four matrices
    co_rmse, heat_rows = build_matrix(
        df, "co", "rmse_s"
    )

    co_mae, _ = build_matrix(
        df, "co", "mae_s"
    )

    co2_rmse, _ = build_matrix(
        df, "co2", "rmse_s"
    )

    co2_mae, _ = build_matrix(
        df, "co2", "mae_s"
    )

    # Shared color scales for comparable metrics
    rmse_all = np.concatenate([
        co_rmse.flatten(),
        co2_rmse.flatten()
    ])

    mae_all = np.concatenate([
        co_mae.flatten(),
        co2_mae.flatten()
    ])

    rmse_vmin = np.nanmin(rmse_all)
    rmse_vmax = np.nanmax(rmse_all)

    mae_vmin = np.nanmin(mae_all)
    mae_vmax = np.nanmax(mae_all)

    # Font configuration
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": TEXT_DARK,
        "xtick.color": TEXT_DARK,
        "ytick.color": TEXT_DARK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white"
    })

    # -----------------------------------------------------
    # Figure and GridSpec
    # -----------------------------------------------------

    fig = plt.figure(figsize=(17, 10.5))

    # Four heatmaps + two independent colorbar columns
    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=[1.0, 0.035, 1.0, 0.035],
        height_ratios=[1, 1],
        left=0.075,
        right=0.94,
        top=0.825,
        bottom=0.125,
        wspace=0.32,
        hspace=0.34
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 2])

    # Dedicated colorbar axes
    cax_rmse = fig.add_subplot(gs[:, 1])
    cax_mae = fig.add_subplot(gs[:, 3])

    # -----------------------------------------------------
    # Draw four heatmaps
    # -----------------------------------------------------

    im1 = draw_heatmap(
        ax_a,
        co_rmse,
        heat_rows,
        MODEL_LABELS,
        "(a) CO RMSE by held-out heat",
        BLUE_CMAP,
        rmse_vmin,
        rmse_vmax
    )

    im2 = draw_heatmap(
        ax_b,
        co_mae,
        heat_rows,
        MODEL_LABELS,
        "(b) CO MAE by held-out heat",
        BLUE_CMAP,
        mae_vmin,
        mae_vmax
    )

    im3 = draw_heatmap(
        ax_c,
        co2_rmse,
        heat_rows,
        MODEL_LABELS,
        "(c) CO₂ RMSE by held-out heat",
        BLUE_CMAP,
        rmse_vmin,
        rmse_vmax
    )

    im4 = draw_heatmap(
        ax_d,
        co2_mae,
        heat_rows,
        MODEL_LABELS,
        "(d) CO₂ MAE by held-out heat",
        BLUE_CMAP,
        mae_vmin,
        mae_vmax
    )

    # -----------------------------------------------------
    # Axis labels
    # -----------------------------------------------------

    ax_a.set_ylabel(
        "Held-out heat",
        fontsize=12,
        labelpad=9
    )

    ax_c.set_ylabel(
        "Held-out heat",
        fontsize=12,
        labelpad=9
    )

    ax_c.set_xlabel(
        "Prediction model",
        fontsize=12,
        labelpad=9
    )

    ax_d.set_xlabel(
        "Prediction model",
        fontsize=12,
        labelpad=9
    )

    # -----------------------------------------------------
    # Shared colorbars
    # -----------------------------------------------------

    cbar_rmse = fig.colorbar(
        im1,
        cax=cax_rmse
    )

    cbar_rmse.set_label(
        "RMSE (s)",
        fontsize=11,
        labelpad=9
    )

    cbar_rmse.ax.tick_params(
        labelsize=9,
        length=3
    )

    cbar_mae = fig.colorbar(
        im2,
        cax=cax_mae
    )

    cbar_mae.set_label(
        "MAE (s)",
        fontsize=11,
        labelpad=9
    )

    cbar_mae.ax.tick_params(
        labelsize=9,
        length=3
    )

    # -----------------------------------------------------
    # Main title and subtitle
    # -----------------------------------------------------

    fig.suptitle(
        "Heat-wise LOHO Error Heatmap",
        fontsize=25,
        fontweight="bold",
        color=TITLE_COLOR,
        y=0.965
    )

    fig.text(
        0.5,
        0.915,
        "Leave-one-heat-out evaluation across 10 heats",
        ha="center",
        fontsize=13,
        color=SUBTITLE_COLOR
    )

    fig.text(
        0.5,
        0.887,
        "Darker cells indicate larger errors | "
        "Bold numbers mark the best model within each heat",
        ha="center",
        fontsize=11.5,
        color=SUBTITLE_COLOR
    )

    # -----------------------------------------------------
    # Bottom note
    # -----------------------------------------------------

    fig.text(
        0.5,
        0.045,
        "Rows correspond to held-out heats H01-H10. "
        "Columns correspond to Fixed, Flow-grouped "
        "and Monotone linear models.",
        ha="center",
        fontsize=10.5,
        color=SUBTITLE_COLOR
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
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("HEAT-WISE LOHO ERROR HEATMAP GENERATED")
    print("=" * 65)

    print(f"Input CSV : {CSV_FILE}")
    print(f"PNG       : {OUT_PNG}")
    print(f"PDF       : {OUT_PDF}")

    print("\nColor scale:")
    print(
        f"RMSE: {rmse_vmin:.2f} - {rmse_vmax:.2f} s"
    )
    print(
        f"MAE : {mae_vmin:.2f} - {mae_vmax:.2f} s"
    )

    print("\nAll four heatmaps generated successfully.")


if __name__ == "__main__":
    main()
