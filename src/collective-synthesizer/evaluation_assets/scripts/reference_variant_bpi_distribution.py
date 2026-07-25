#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict

import numpy as np
from plot_common import (
    RESULTS_DIR,
    emit,
    fnum,
    read_csv,
    save_figure,
    setup_matplotlib,
    skip,
    variant_label,
)


# Preserve the overall canvas aspect ratio while allowing the plot body
# rectangle to be tuned directly.
FIG_SIZE = (26, 10)
LAYOUT_LEFT = 0.075
LAYOUT_RIGHT = 0.990
LAYOUT_BOTTOM = 0.18
LAYOUT_TOP = 0.86


def main_axes_rect() -> list[float]:
    panel_height = LAYOUT_TOP - LAYOUT_BOTTOM
    main_width = LAYOUT_RIGHT - LAYOUT_LEFT
    if main_width <= 0:
        raise ValueError("LAYOUT_LEFT/LAYOUT_RIGHT leave no room for the main plot body")
    return [LAYOUT_LEFT, LAYOUT_BOTTOM, main_width, panel_height]


def main() -> None:
    plt = setup_matplotlib()
    plt.rcParams["figure.autolayout"] = False
    real_metric_path = RESULTS_DIR / "trace_metrics_real_full.csv"
    variant_metric_path = RESULTS_DIR / "trace_metrics_reference_variants.csv"
    official_eplb_metric_path = RESULTS_DIR / "trace_metrics_official_eplb.csv"

    real_rows = [
        row
        for row in read_csv(real_metric_path)
        if fnum(row, "total_tokens") > 0 and row.get("source") == "official"
    ]
    variant_rows = [
        row
        for row in read_csv(variant_metric_path)
        if fnum(row, "total_tokens") > 0
        and row.get("source") == "official"
        and row.get("variant") != "eplb_x2"
    ]
    eplb_rows = [
        row
        for row in read_csv(official_eplb_metric_path)
        if fnum(row, "total_tokens") > 0
        and row.get("source") == "official"
        and row.get("variant") == "eplb_x2"
    ]

    variant_groups: dict[str, list[float]] = defaultdict(list)
    for row in real_rows:
        variant_groups["real"].append(fnum(row, "bimodal_pressure_index"))
    for row in variant_rows:
        variant_groups[row["variant"]].append(fnum(row, "bimodal_pressure_index"))
    for row in eplb_rows:
        variant_groups[row["variant"]].append(fnum(row, "bimodal_pressure_index"))

    variant_order = [
        variant
        for variant in ["balanced", "blend_0.75", "blend_0.50", "blend_0.25", "eplb_x2", "real", "worst_top8"]
        if variant in variant_groups
    ]
    if not variant_order:
        skip("reference_variant_bpi_distribution", "no variant metric rows")
        return

    # Each entry is (fill color, edge color); median lines use edge colors.
    variant_colors = {
        "balanced":    ("#f1fae7", "#bae5bc"),   # (fill, edge)
        "blend_0.25":  ("#2b8cc0", "#004e89"),
        "blend_0.50":  ("#7bccc5", "#2b8cc0"),
        "blend_0.75":  ("#bae5bc", "#7bccc5"),
        "eplb_x2":     ("#c77dff", "#7b2cbf"),
        "real":        ("#db7c26", "#d8572a"),
        "worst_top8":  ("#ff87ab", "#ff5d8f"),
    }

    data = [variant_groups[variant] for variant in variant_order]

    fig = plt.figure(figsize=FIG_SIZE, constrained_layout=False)
    fig.set_tight_layout(False)
    ax = fig.add_axes(main_axes_rect())

    spacing = 0.8
    positions = np.arange(1, len(data) + 1) * spacing
    width = 0.65

    violins = ax.violinplot(
        data,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for i, variant in enumerate(variant_order):
        fill_color, edge_color = variant_colors.get(variant, ("#bbbbbb", "#222222"))

        body = violins["bodies"][i]
        body.set_facecolor(fill_color)
        body.set_alpha(0.85)
        body.set_edgecolor(edge_color)
        body.set_linewidth(9)

        median_val = np.median(data[i])
        half_width = width * 0.4
        ax.plot(
            [positions[i] - half_width, positions[i] + half_width],
            [median_val, median_val],
            color=edge_color,
            linewidth=13,
            solid_capstyle="butt",
        )

    ax.set_ylabel("Bimodal Pressure Index (BPI)", fontsize=48)
    ax.set_ylim(-0.02, 1.03)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=50)
    # ax.tick_params(axis='x', labelsize=42)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [variant_label(variant) for variant in variant_order],
        rotation=0,
        ha="center",
        fontsize=41,
    )
    ax.set_title("BPI under Real and Generated Alltoallv Traces", fontsize=54)

    emit(
        save_figure(
            plt,
            fig,
            "reference_variant_bpi_distribution",
            "Official reference variant BPI distribution",
            "BPI distribution for official real traces, same-total reference controls, and official EPLB device mappings.",
            [real_metric_path, variant_metric_path, official_eplb_metric_path],
            bbox_inches=None,
        )
    )


if __name__ == "__main__":
    main()
