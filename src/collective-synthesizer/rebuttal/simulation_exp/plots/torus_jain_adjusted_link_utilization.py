#!/usr/bin/env python3

from __future__ import annotations

import shutil

import numpy as np

from plot_common import PLOTS_DIR, RESULTS_DIR, emit, fnum, read_csv, save_figure, setup_matplotlib, variant_label
from matplotlib.ticker import FixedLocator

VARIANT_ORDER = (
    "balanced",
    "blend_0.75",
    "blend_0.50",
    "blend_0.25",
    "eplb_x2",
    "real",
    "worst_top8",
)

BASELINE_EDGE = "#118ab2"
BASELINE_FILL = "#4cc9f0"
INCREASE_EDGE = "#06d6a0"
INCREASE_FILL = "#80ed99"
DECREASE_EDGE = "#ff5d8f"
DECREASE_FILL = "#ff87ab"


def sort_key(row: dict[str, str]) -> tuple[int, str]:
    variant = row.get("variant", "")
    try:
        return (VARIANT_ORDER.index(variant), variant)
    except ValueError:
        return (len(VARIANT_ORDER), variant)


def selected_sample() -> str | None:
    scores = read_csv(RESULTS_DIR / "torus_link_hotedge_sample_scores.csv")
    for row in scores:
        if str(row.get("selected", "")).lower() == "true":
            return row.get("sample_index")
    return None


def main() -> None:
    plt = setup_matplotlib()
    summary_path = RESULTS_DIR / "torus_link_hotedge_metrics_summary.csv"
    scores_path = RESULTS_DIR / "torus_link_hotedge_sample_scores.csv"
    rows = read_csv(summary_path)
    sample = selected_sample()
    if not rows or sample is None:
        emit({"figure_id": "torus_jain_weighted_link_utilization", "skipped": True, "reason": "missing torus metric CSV"})
        return
    rows = sorted([row for row in rows if row.get("sample_index") == sample], key=sort_key)
    x = np.arange(len(rows))
    baseline = np.asarray(
        [
            (
                fnum(row, "deterministic_baseline_jain_weighted_link_utilization")
                or fnum(row, "deterministic_baseline_jain_adjusted_link_utilization")
            )
            * 100.0
            for row in rows
        ],
        dtype=float,
    )
    delta = np.asarray(
        [
            (fnum(row, "jain_weighted_link_utilization_delta") or fnum(row, "jain_adjusted_link_utilization_delta")) * 100.0
            for row in rows
        ],
        dtype=float,
    )
    glaive = np.asarray(
        [
            (fnum(row, "glaive_jain_weighted_link_utilization") or fnum(row, "glaive_jain_adjusted_link_utilization"))
            * 100.0
            for row in rows
        ],
        dtype=float,
    )
    positive_delta = np.clip(delta, 0.0, None)
    negative_delta = np.clip(delta, None, 0.0)
    labels = [variant_label(row.get("variant", "")) for row in rows]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_axisbelow(True)
    width = 0.7
    bar_lw = 4.5
    ax.bar(
        x,
        baseline,
        width=width,
        color=BASELINE_FILL,
        edgecolor=BASELINE_EDGE,
        linewidth=bar_lw,
        label="HalfR+DR JLU",
        zorder=3,
    )
    positive_mask = positive_delta > 0.0
    negative_mask = negative_delta < 0.0
    ax.bar(
        x[positive_mask],
        positive_delta[positive_mask],
        width=width,
        bottom=baseline[positive_mask],
        color=INCREASE_FILL,
        edgecolor=INCREASE_EDGE,
        linewidth=bar_lw,
        label="Glaive increase",
        zorder=3,
    )
    if np.any(negative_mask):
        ax.bar(
            x[negative_mask],
            negative_delta[negative_mask],
            width=width,
            bottom=baseline[negative_mask],
            color=DECREASE_FILL,
            edgecolor=DECREASE_EDGE,
            linewidth=bar_lw,
            label="Glaive decrease",
            zorder=3,
        )
    for idx, (base, change, total) in enumerate(zip(baseline, delta, glaive)):
        if base >= 8.0:
            ax.text(idx, base / 2.0, f"{base:.1f}%", ha="center", va="center", color="#073b4c", fontsize=20, fontweight="bold")
        if abs(change) >= 2.0:
            ax.text(
                idx,
                base + change / 2.0,
                f"{change:+.1f}%",
                ha="center",
                va="center",
                fontsize=20,
                color="#073b4c",
                fontweight="bold",
            )
    ax.axhline(0.0, color="#333333", linewidth=2)
    ax.set_ylabel("JLU (%)", fontsize=26)
    ax.yaxis.set_label_coords(-0.05, 0.546)
    ax.set_xticks(x)
    ax.tick_params(axis="y", labelsize=22)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=20)
    # ax.set_title("3D Torus 4x4x4", fontsize=22, y=1.08)
    ymin = min(0.0, float(np.min(baseline + negative_delta)) - 3.0)
    ymax = float(np.max(np.maximum(baseline + positive_delta, glaive))) + 10.0
    ax.set_ylim(ymin, ymax)

    ax.grid(False, axis="x")
    ax.grid(True, axis="y", linestyle="-", linewidth=2.0, color="gray", alpha=0.5, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), handletextpad=0.5, columnspacing=1.4, ncol=3, frameon=False, fontsize=23)
    entry = save_figure(
        plt,
        fig,
        "torus_jain_weighted_link_utilization",
        "3D Torus Jain-Weighted Link Utilization (JLU)",
        "JLU is Link Utilization multiplied by Jain fairness, rewarding high utilization without concentrating load.",
        [summary_path, scores_path],
    )
    shutil.copyfile(PLOTS_DIR / "torus_jain_weighted_link_utilization.pdf", PLOTS_DIR / "torus_link_utilization.pdf")
    shutil.copyfile(PLOTS_DIR / "torus_jain_weighted_link_utilization.pdf", PLOTS_DIR / "torus_jain_adjusted_link_utilization.pdf")
    emit(entry)


if __name__ == "__main__":
    main()
