#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict

import numpy as np
from plot_common import RESULTS_DIR, clean_label, emit, fnum, read_csv, save_figure, setup_matplotlib, skip


def main() -> None:
    plt = setup_matplotlib()
    metric_path = RESULTS_DIR / "trace_metrics_real_full.csv"

    real_rows = [
        row
        for row in read_csv(metric_path)
        if fnum(row, "total_tokens") > 0 and row.get("source") == "official"
    ]

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in real_rows:
        groups[(row["model"], row["phase"])].append(fnum(row, "bimodal_pressure_index"))

    ordered_groups = sorted(groups)
    if not ordered_groups:
        skip("trace_bpi_distribution", "no nonzero trace rows")
        return

    data = [groups[key] for key in ordered_groups]
    labels = [f"{clean_label(key[0])}\n{key[1]}" for key in ordered_groups]

    fig, ax = plt.subplots(figsize=(26, 10))
    spacing = 0.65
    positions = np.arange(1, len(data) + 1) * spacing
    width = 0.5

    violins = ax.violinplot(
        data,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    prefill_color = "#53a9d9"
    decode_color = "#bae5bc"
    prefill_fill = "#a8dadc"
    decode_fill = "#f1faee"
    median_color_prefill = "#2b8cc0"
    median_color_decode = "#96da99"

    for i, key in enumerate(ordered_groups):
        phase = key[1]
        edge_color = prefill_color if phase == "prefill" else decode_color
        face_color = prefill_fill if phase == "prefill" else decode_fill
        median_color = median_color_prefill if phase == "prefill" else median_color_decode

        body = violins["bodies"][i]
        body.set_facecolor(face_color)
        body.set_edgecolor(edge_color)
        body.set_alpha(0.85)
        body.set_linewidth(9)

        median_value = np.median(data[i])
        half_width = width * 0.4
        ax.plot(
            [positions[i] - half_width, positions[i] + half_width],
            [median_value, median_value],
            color=median_color,
            linewidth=13,
            solid_capstyle="butt",
        )

    ax.set_ylabel("Bimodal Pressure Index (BPI)", fontsize=46.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticks(positions)
    ax.tick_params(axis="x", labelsize=42)
    ax.tick_params(axis="y", labelsize=48)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_title("BPI Distribution Across MoE Alltoallv Traces", fontsize=54)

    for idx, values in enumerate(data):
        ax.text(
            positions[idx],
            0.99,
            f"n={len(values)}",
            ha="center",
            va="top",
            fontsize=46,
            color="#333333",
        )

    emit(
        save_figure(
            plt,
            fig,
            "trace_bpi_distribution",
            "Official real trace BPI distribution",
            "BPI distribution for nonzero official real traces grouped by model and phase (violin plot).",
            [metric_path],
        )
    )


if __name__ == "__main__":
    main()
