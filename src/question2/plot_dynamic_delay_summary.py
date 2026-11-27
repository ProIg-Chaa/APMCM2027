
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

RUN_DIR = (
    ROOT
    / "results"
    / "question2"
    / "dynamic_delay_validation_corrected"
    / "run_20261127_210856"
)

CSV_FILE = RUN_DIR / "corrected_alignment_summary.csv"

OUT_DIR = RUN_DIR / "summary_figures"

OUT_PNG = (
    OUT_DIR
    / "figure_heatwise_dynamic_delay_summary.png"
)

OUT_PDF = (
    OUT_DIR
    / "figure_heatwise_dynamic_delay_summary.pdf"
)


# ==========================================
# 2. STYLE
# ==========================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#20354A"
BLUE = "#3979A8"
GREEN = "#DCEFE4"
RED = "#F9E3DF"
LIGHT_BLUE = "#E8F0F7"
LIGHT_GRAY = "#F3F5F7"
TEXT = "#263746"


# ==========================================
# 3. DATA LOADING
# ==========================================

def load_data():

    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    required = [
        "heat_number",
        "rolling_windows",
        "co_valid_windows",
        "co2_valid_windows",
        "co_model_used",
        "co_fallback_reason",
        "co_mapping_monotonic_before_fallback",
        "co_cumulative_drops",
        "co2_cumulative_drops",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.sort_values("heat_number").reset_index(drop=True)

    if len(df) != 10:
        raise ValueError(
            f"Expected 10 heats, found {len(df)}"
        )

    return df


# ==========================================
# 4. DATA PROCESSING
# ==========================================

def parse_bool(value):

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in [
        "true", "1", "yes"
    ]


def format_coverage(valid, total):

    if total == 0:
        return "N/A"

    percentage = valid / total * 100

    return (
        f"{valid} / {total}\n"
        f"({percentage:.1f}%)"
    )


def format_reason(reason):

    reason = str(reason).strip()

    if not reason or reason.lower() == "nan":
        return "Not applicable"

    mapping = {
        "non_monotonic_event_to_analyzer_mapping":
            "Non-monotonic event-to-analyzer mapping",

        "no_valid_dynamic_support":
            "No valid dynamic support",
    }

    return mapping.get(
        reason,
        reason.replace("_", " ")
    )


# ==========================================
# 5. SUMMARY FIGURE
# ==========================================

def main():

    df = load_data()

    df["_monotonic"] = (
        df["co_mapping_monotonic_before_fallback"]
        .apply(parse_bool)
    )

    df["_dynamic"] = (
        df["co_model_used"] == "dynamic_candidate"
    )

    dynamic_count = int(df["_dynamic"].sum())

    fallback_count = len(df) - dynamic_count

    monotonic_count = int(
        df["_monotonic"].sum()
    )

    zero_drop_count = int(
        (
            (df["co_cumulative_drops"] == 0)
            &
            (df["co2_cumulative_drops"] == 0)
        ).sum()
    )

    print("=" * 65)
    print("HEAT-WISE DYNAMIC DELAY VALIDATION SUMMARY")
    print("=" * 65)

    print(f"Dynamic candidate: {dynamic_count}")
    print(f"Fixed fallback: {fallback_count}")
    print(f"Monotonic mapping: {monotonic_count}")
    print(f"Zero cumulative drops: {zero_drop_count}")


    # ======================================
    # 6. FIGURE CANVAS
    # ======================================

    fig = plt.figure(
        figsize=(18, 9),
        facecolor="white"
    )

    fig.suptitle(
        "Heat-wise Dynamic Delay Validation Summary",
        fontsize=21,
        fontweight="bold",
        color=NAVY,
        y=0.97
    )

    fig.text(
        0.5,
        0.925,
        "Model selection, mapping monotonicity, "
        "valid-window coverage and cumulative integrity",
        ha="center",
        fontsize=11,
        color="#596B7A"
    )


    # ======================================
    # 7. KPI CARDS
    # ======================================

    cards = [
        ("Dynamic candidate",
         f"{dynamic_count} / 10",
         LIGHT_BLUE),

        ("Fixed fallback",
         f"{fallback_count} / 10",
         RED),

        ("Monotonic mapping",
         f"{monotonic_count} / 10",
         GREEN),

        ("Zero cumulative drops",
         f"{zero_drop_count} / 10",
         LIGHT_GRAY),
    ]

    card_positions = [0.055, 0.285, 0.515, 0.745]

    for x, (label, value, color) in zip(
        card_positions, cards
    ):

        card = FancyBboxPatch(
            (x, 0.79),
            0.20,
            0.095,
            boxstyle="round,pad=0.008",
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none"
        )

        fig.add_artist(card)

        fig.text(
            x + 0.10,
            0.852,
            value,
            ha="center",
            fontsize=19,
            fontweight="bold",
            color=NAVY
        )

        fig.text(
            x + 0.10,
            0.815,
            label,
            ha="center",
            fontsize=9,
            color=TEXT
        )


    # ======================================
    # 8. TABLE DATA
    # ======================================

    headers = [
        "Heat",
        "CO model",
        "Mapping\nmonotonic",
        "CO valid\nwindows",
        "CO₂ valid\nwindows",
        "Cumulative drops\nCO / CO₂",
        "Fallback reason",
    ]

    table_data = []

    for _, row in df.iterrows():

        total = int(row["rolling_windows"])

        co_valid = int(row["co_valid_windows"])

        co2_valid = int(row["co2_valid_windows"])

        model = (
            "Dynamic candidate"
            if row["_dynamic"]
            else "Fixed fallback"
        )

        monotonic = (
            "PASS"
            if row["_monotonic"]
            else "FAIL"
        )

        drops = (
            f"{int(row['co_cumulative_drops'])}"
            f" / "
            f"{int(row['co2_cumulative_drops'])}"
        )

        reason = format_reason(
            row["co_fallback_reason"]
        )

        table_data.append([
            f"H{int(row['heat_number']):02d}",
            model,
            monotonic,
            format_coverage(co_valid, total),
            format_coverage(co2_valid, total),
            drops,
            reason,
        ])


    # ======================================
    # 9. DRAW TABLE
    # ======================================

    ax = fig.add_axes(
        [0.035, 0.19, 0.93, 0.55]
    )

    ax.axis("off")

    col_widths = [
        0.065,
        0.145,
        0.12,
        0.125,
        0.125,
        0.14,
        0.28,
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)

    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():

        cell.set_edgecolor("white")
        cell.set_linewidth(1.2)

        if r == 0:

            cell.set_facecolor(NAVY)

            cell.get_text().set_color("white")

            cell.get_text().set_fontweight("bold")

            cell.get_text().set_fontsize(9)

        else:

            row = df.iloc[r - 1]

            cell.set_facecolor(
                "white" if r % 2 == 0
                else LIGHT_GRAY
            )

            cell.get_text().set_color(TEXT)

            # Model selection
            if c == 1:

                cell.set_facecolor(
                    LIGHT_BLUE
                    if row["_dynamic"]
                    else RED
                )

                cell.get_text().set_fontweight("bold")

            # Monotonicity
            elif c == 2:

                cell.set_facecolor(
                    GREEN
                    if row["_monotonic"]
                    else RED
                )

                cell.get_text().set_fontweight("bold")

            # Cumulative integrity
            elif c == 5:

                no_drop = (
                    row["co_cumulative_drops"] == 0
                    and row["co2_cumulative_drops"] == 0
                )

                cell.set_facecolor(
                    GREEN if no_drop else RED
                )

            # Fallback explanation
            elif c == 6:

                cell.get_text().set_ha("left")

                cell.get_text().set_fontsize(8)


    # ======================================
    # 10. FOOTNOTE
    # ======================================

    fig.text(
        0.5,
        0.105,
        "Valid-window coverage = valid windows / rolling windows. "
        "Mapping monotonicity is evaluated before fallback. "
        "Cumulative drops are reported as CO / CO₂.",
        ha="center",
        fontsize=9,
        color="#596B7A"
    )

    fig.text(
        0.5,
        0.065,
        "Dynamic candidates are retained only when "
        "the mapping passes the required validation checks.",
        ha="center",
        fontsize=9,
        color=NAVY
    )


    # ======================================
    # 11. SAVE
    # ======================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUT_PNG,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print("\n" + "=" * 65)
    print("SUMMARY FIGURE GENERATED")
    print("=" * 65)

    print("PNG:", OUT_PNG)
    print("PDF:", OUT_PDF)


if __name__ == "__main__":
    main()
