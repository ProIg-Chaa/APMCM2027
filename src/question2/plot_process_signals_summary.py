from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "deliverables" / "question2" / "01_process_signals"
OUT_DIR = DATA_DIR / "summary_figures"

OUT_PNG = OUT_DIR / "all_heats_process_summary_v1.png"
OUT_PDF = OUT_DIR / "all_heats_process_summary_v1.pdf"

HEAT_ID_MAP = {
    1: "6B03205",
    2: "6B03206",
    3: "6B03270",
    4: "6B03271",
    5: "6C02057",
    6: "6C02058",
    7: "6C02059",
    8: "6C02060",
    9: "6C02107",
    10: "6C02108",
}

HEAT_NUMBERS = range(1, 11)

COMMON_PROGRESS = np.linspace(0, 100, 401)

REQUIRED_COLUMNS = [
    "time_s",
    "oxygen_lance_flow_m3_s",
    "oxygen_cumulative_m3",
    "co_flow_m3_s",
    "co2_flow_m3_s",
    "co_cumulative_m3",
    "co2_cumulative_m3",
]


# ==========================================
# 2. HELPERS
# ==========================================

def load_heat_data(heat_number, heat_id):
    data_file = DATA_DIR / f"heat_{heat_number:02d}_{heat_id}_six_signals.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Missing file: {data_file}")

    df = pd.read_csv(data_file)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Heat {heat_number:02d} missing columns: {missing_cols}"
        )

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["time_s"])
        .sort_values("time_s")
        .reset_index(drop=True)
    )

    t_min = df["time_s"].min()
    t_max = df["time_s"].max()

    if t_max > t_min:
        df["progress_pct"] = (
            (df["time_s"] - t_min) / (t_max - t_min) * 100
        )
    else:
        df["progress_pct"] = 0.0

    return df


def interpolate_signal(df, signal_col, common_progress):
    valid = (
        df[["progress_pct", signal_col]]
        .dropna()
        .sort_values("progress_pct")
    )

    if len(valid) < 2:
        raise ValueError(f"Not enough valid observations for {signal_col}")

    x = valid["progress_pct"].to_numpy(dtype=float)
    y = valid[signal_col].to_numpy(dtype=float)

    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    interpolated = np.interp(common_progress, unique_x, unique_y)
    return interpolated


def final_valid_value(df, col):
    valid = df[col].dropna()
    if valid.empty:
        return np.nan
    return float(valid.iloc[-1])


def format_metric_value(col_name, value):
    if pd.isna(value):
        return "NA"

    if "Duration" in col_name:
        return f"{value:.1f}"

    return f"{value:,.0f}"


def normalize_columns(df):
    normalized = df.copy().astype(float)

    for col in normalized.columns:
        col_min = normalized[col].min()
        col_max = normalized[col].max()

        if np.isclose(col_max, col_min):
            normalized[col] = 0.5
        else:
            normalized[col] = (
                (normalized[col] - col_min) / (col_max - col_min)
            )

    return normalized


# ==========================================
# 3. MAIN
# ==========================================

def main():
    oxygen_curves = []
    co_curves = []
    co2_curves = []

    heat_labels = []
    metrics_rows = []

    print("=" * 60)
    print("GENERATING CROSS-HEAT SUMMARY FIGURE")
    print("=" * 60)

    for heat_number in HEAT_NUMBERS:
        heat_id = HEAT_ID_MAP[heat_number]
        df = load_heat_data(heat_number, heat_id)

        label = f"H{heat_number:02d}"
        heat_labels.append(label)

        oxygen_curves.append(
            interpolate_signal(
                df,
                "oxygen_lance_flow_m3_s",
                COMMON_PROGRESS
            )
        )

        co_curves.append(
            interpolate_signal(
                df,
                "co_flow_m3_s",
                COMMON_PROGRESS
            )
        )

        co2_curves.append(
            interpolate_signal(
                df,
                "co2_flow_m3_s",
                COMMON_PROGRESS
            )
        )

        metrics_rows.append({
            "Heat": label,
            "Duration (s)": float(df["time_s"].max() - df["time_s"].min()),
            "O₂ cumulative (m³)": final_valid_value(df, "oxygen_cumulative_m3"),
            "CO cumulative (m³)": final_valid_value(df, "co_cumulative_m3"),
            "CO₂ cumulative (m³)": final_valid_value(df, "co2_cumulative_m3"),
        })

        print(
            f"Loaded Heat {heat_number:02d} | ID {heat_id} | "
            f"rows={len(df)} | duration={df['time_s'].max() - df['time_s'].min():.1f}s"
        )

    oxygen_curves = np.vstack(oxygen_curves)
    co_curves = np.vstack(co_curves)
    co2_curves = np.vstack(co2_curves)

    oxygen_median = np.median(oxygen_curves, axis=0)
    co_median = np.median(co_curves, axis=0)
    co2_median = np.median(co2_curves, axis=0)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Heat")
    heatmap_df = normalize_columns(metrics_df)

    # ======================================
    # 4. PLOT
    # ======================================

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(
        2, 2,
        left=0.06,
        right=0.95,
        top=0.86,
        bottom=0.12,
        hspace=0.32,
        wspace=0.18
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # (a) Oxygen
    for i, label in enumerate(heat_labels):
        ax1.plot(
            COMMON_PROGRESS,
            oxygen_curves[i],
            color=colors[i],
            linewidth=1.1,
            alpha=0.8,
            label=label
        )

    ax1.plot(
        COMMON_PROGRESS,
        oxygen_median,
        color="black",
        linewidth=2.4,
        label="Median"
    )

    ax1.set_title("(a) Oxygen lance flow comparison", loc="left")
    ax1.set_xlabel("Normalized process progress (%)")
    ax1.set_ylabel("Oxygen flow (m³/s)")
    ax1.grid(alpha=0.25)
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=11,
        frameon=False,
        fontsize=8.5
    )
    # (b) CO
    for i, label in enumerate(heat_labels):
        ax2.plot(
            COMMON_PROGRESS,
            co_curves[i],
            color=colors[i],
            linewidth=1.1,
            alpha=0.8
        )

    ax2.plot(
        COMMON_PROGRESS,
        co_median,
        color="black",
        linewidth=2.4
    )

    ax2.set_title("(b) CO flow comparison", loc="left")
    ax2.set_xlabel("Normalized process progress (%)")
    ax2.set_ylabel("CO flow (m³/s)")
    ax2.grid(alpha=0.25)

    # (c) CO2
    for i, label in enumerate(heat_labels):
        ax3.plot(
            COMMON_PROGRESS,
            co2_curves[i],
            color=colors[i],
            linewidth=1.1,
            alpha=0.8
        )

    ax3.plot(
        COMMON_PROGRESS,
        co2_median,
        color="black",
        linewidth=2.4
    )

    ax3.set_title("(c) CO₂ flow comparison", loc="left")
    ax3.set_xlabel("Normalized process progress (%)")
    ax3.set_ylabel("CO₂ flow (m³/s)")
    ax3.grid(alpha=0.25)

    # (d) Heat-level summary heatmap
    im = ax4.imshow(
        heatmap_df.values,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax4.set_title("(d) Heat-level summary metrics", loc="left")
    ax4.set_xticks(np.arange(len(heatmap_df.columns)))
    ax4.set_xticklabels(
        [
            "Duration\n(s)",
            "O₂ total\n(m³)",
            "CO total\n(m³)",
            "CO₂ total\n(m³)"
        ],
        rotation=0,
        ha="center"
    )
    ax4.set_yticks(np.arange(len(heatmap_df.index)))
    ax4.set_yticklabels(heatmap_df.index)

    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            true_value = metrics_df.iloc[i, j]
            text_value = format_metric_value(metrics_df.columns[j], true_value)

            cell_color = heatmap_df.iloc[i, j]
            text_color = "white" if cell_color > 0.55 else "black"

            ax4.text(
                j, i,
                text_value,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5
            )

    cbar = fig.colorbar(
        im,
        ax=ax4,
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label("Relative magnitude within each metric")

    # Main title + subtitle
    fig.suptitle(
        "Cross-heat Summary of Continuous Process Signals",
        fontsize=20,
        fontweight="bold",
        y=0.965
    )

    fig.text(
        0.5, 0.932,
        "Each colored curve represents one heat. "
        "Panels (a)-(c) use normalized process progress (0%-100%) "
        "to enable cross-heat comparison across unequal durations.",
        ha="center",
        fontsize=10.5
    )

    fig.text(
        0.5, 0.03,
        "Black curves denote the median trend across 10 heats. "
        "Panel (d) summarizes duration and final cumulative volumes.",
        ha="center",
        fontsize=9
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        OUT_PDF,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n" + "=" * 60)
    print("SUMMARY FIGURE GENERATED")
    print("=" * 60)
    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()
