#!/usr/bin/env python3

from __future__ import annotations

from matplotlib.lines import Line2D

from plot_common import (
    RESULTS_DIR,
    TOPOLOGY_JITTER,
    TOPOLOGY_MARKERS,
    emit,
    fnum,
    read_csv,
    save_figure,
    setup_matplotlib,
    skip,
    variant_label,
)


# Keep the total canvas aspect ratio fixed while tuning the plot body and
# right-side legend area independently.
FIG_SIZE = (16, 6)
LAYOUT_LEFT = 0.085
LAYOUT_RIGHT = 0.985
LAYOUT_BOTTOM = 0.14
LAYOUT_TOP = 0.88
PANEL_GAP = 0.002          # smaller value moves the legend closer to the plot body
BODY_WIDTH_RATIO = 0.70    # plot body share within the usable width
LEGEND_WIDTH_RATIO = 0.30  # legend share within the usable width
LEGEND_FONT_SIZE = 22
LEGEND_TITLE_FONT_SIZE = 26
GRID_ZORDER = 0
REFERENCE_LINE_ZORDER = 1
NEGATIVE_POINT_ZORDER = 9
VARIANT_ZORDERS = {
    "balanced": 2,
    "blend_0.75": 4,
    "blend_0.50": 5,
    "blend_0.25": 6,
    "eplb_x2": 3,
    "real": 7,
    "worst_top8": 8,
}


def axes_rects() -> tuple[list[float], list[float]]:
    panel_height = LAYOUT_TOP - LAYOUT_BOTTOM
    available_width = LAYOUT_RIGHT - LAYOUT_LEFT - PANEL_GAP
    total_ratio = BODY_WIDTH_RATIO + LEGEND_WIDTH_RATIO
    if available_width <= 0 or total_ratio <= 0:
        raise ValueError("layout settings leave no room for the plot body and legend")
    main_width = available_width * BODY_WIDTH_RATIO / total_ratio
    legend_width = available_width * LEGEND_WIDTH_RATIO / total_ratio
    main_rect = [LAYOUT_LEFT, LAYOUT_BOTTOM, main_width, panel_height]
    legend_rect = [LAYOUT_LEFT + main_width + PANEL_GAP, LAYOUT_BOTTOM, legend_width, panel_height]
    return main_rect, legend_rect


def main() -> None:
    plt = setup_matplotlib()
    plt.rcParams["figure.autolayout"] = False
    speedup_path = RESULTS_DIR / "performance_speedup_vs_best_baseline.csv"
    excluded_topologies = {"rail_optimized_8x4"}
    speedup_rows = [
        row for row in read_csv(speedup_path)
        if row.get("speedup_vs_best_baseline", "") != ""
        and row.get("topology_key") not in excluded_topologies
    ]
    if not speedup_rows:
        skip("speedup_vs_bpi", "no speedup rows")
        return

    # Sorted list of variants (determines the order of colors)
    sorted_variants = sorted({row["variant"] for row in speedup_rows})

    # ============================================================
    # Color definitions – modify these two lists to change all variant colors globally
    # Both lists must have the same length as sorted_variants (7 items here)
    # ============================================================

    # ============================================================
    # MultiTree_colors = ["#f1fae7", "#bae5bc", "#7bccc5", "#2b8cc0", "#004e89"] # from light to deep
    # PLOT_COLORS = {
    #     "glaive_solver3": "#80ed99",
    #     "teccl_4mb": "#4cc9f0",
    #     "taccl_directmap_nosym_4mb": "#ff87ab",
    #     "msccl_direct": "#c77dff",
    #     "nccl_p2p": "#c77dff",
    #     "nccl_ring": "#f2cc8f",
    #     "nccl_biring": "#db7c26",
    #     "nccl_direct": "#8d99ae",
    #     "nccl_spreadout": "#b1be9d",
    # }
    # LINE_COLORS = {
    #     "glaive_solver3": "#06d6a0",
    #     "teccl_4mb": "#118ab2",
    #     "taccl_directmap_nosym_4mb": "#ff5d8f",
    #     "msccl_direct": "#7b2cbf",
    #     "nccl_p2p": "#7b2cbf",
    #     "nccl_ring": "#bc6c25",
    #     "nccl_biring": "#d8572a",
    #     "nccl_direct": "#4a4e69",
    #     "nccl_spreadout": "#a9a875",
    # }
    # ============================================================

    marker_colors = [
        "#f1fae7",  # variant 0 (e.g., 'balanced')
        "#2b8cc0",  # variant 1 (e.g., 'blend25%')
        "#7bccc5",  # variant 2 (e.g., 'blend50%')
        "#bae5bc",  # variant 3 (e.g., 'blend75%')
        "#c77dff",  # variant 4
        "#db7c26",  # variant 5
        "#ff87ab",  # variant 6
    ]
    # Edge colors – can be identical or different from marker_colors
    edge_colors = [
        "#bae5bc",  # variant 0 (e.g., 'balanced')
        "#004e89",  # variant 1
        "#2b8cc0",  # variant 2
        "#7bccc5",  # variant 3
        "#7b2cbf",  # variant 4
        "#d8572a",  # variant 5
        "#ff5d8f",  # variant 6
    ]

    # Sanity check: ensure lengths match
    if len(marker_colors) != len(sorted_variants):
        raise ValueError(
            f"marker_colors length ({len(marker_colors)}) does not match number of variants ({len(sorted_variants)})"
        )

    fig = plt.figure(figsize=FIG_SIZE, constrained_layout=False)
    fig.set_tight_layout(False)
    main_ax_rect, legend_ax_rect = axes_rects()
    ax = fig.add_axes(main_ax_rect)
    legend_ax = fig.add_axes(legend_ax_rect)
    legend_ax.axis("off")

    # # dedicated markers for this script
    # topology_markers = ['o', 's', '^', 'D', '*']
    # Plot main data points using indexed colors
    for idx, variant in enumerate(sorted_variants):
        zorder_val = VARIANT_ZORDERS.get(variant, VARIANT_ZORDERS["real"])
        rows = [row for row in speedup_rows if row["variant"] == variant]
        for topology in sorted({row["topology_key"] for row in rows}):
            topo_rows = [row for row in rows if row["topology_key"] == topology]
            ax.scatter(
                [
                    max(0.0, min(1.0, fnum(row, "input_bimodal_pressure_index") + TOPOLOGY_JITTER.get(topology, 0.0)))
                    for row in topo_rows
                ],
                [fnum(row, "speedup_vs_best_baseline") for row in topo_rows],
                s=250,
                alpha=0.7,
                marker=TOPOLOGY_MARKERS.get(topology, "o"),
                # marker=marker_style,
                facecolors=marker_colors[idx],
                edgecolors=edge_colors[idx],
                linewidths=4.5,
                zorder=zorder_val,
            )

    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=5.0, zorder=REFERENCE_LINE_ZORDER)

    negative_rows = [
        row for row in speedup_rows
        if fnum(row, "speedup_vs_best_baseline", 1.0) < 1.0
    ]
    if negative_rows:
        ax.scatter(
            [
                max(
                    0.0,
                    min(1.0, fnum(row, "input_bimodal_pressure_index") + TOPOLOGY_JITTER.get(row.get("topology_key", ""), 0.0)),
                )
                for row in negative_rows
            ],
            [fnum(row, "speedup_vs_best_baseline") for row in negative_rows],
            s=250,
            marker="D",
            facecolors="none",
            edgecolors="#B00020",          # fixed red border for speedup < 1
            linewidths=3.5,
            label="speedup < 1",
            zorder=NEGATIVE_POINT_ZORDER,
        )

    ax.set_xlabel("Input Bimodal Pressure Index (BPI)", fontsize=28)
    ax.set_ylabel("Glaive Speedup vs Best Baseline", fontsize=25.5)
    ax.set_title("Glaive Performance across Bimodality Levels", fontsize=30)
    ax.set_ylim(0.4, 6.7)
    ax.set_xlim(-0.025, 0.9)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.6, zorder=GRID_ZORDER)
    for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
        gridline.set_zorder(GRID_ZORDER)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=30)

    # Build variant legend using the same indexed colors
    variant_legend = []
    for idx, variant in enumerate(sorted_variants):
        variant_legend.append(
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=marker_colors[idx],
                markeredgecolor=edge_colors[idx],
                markeredgewidth=2.0,
                markersize=22,
                label=variant_label(variant),
            )
        )
    

    # Topology legend (fixed gray markers, unchanged)
    topology_labels = {
        "mesh_nebula_8x4": "Mesh 8x4",
        "clos_8x4": "Clos 8x4",
        "cm384_16x2": "CM384 2x8x2",
        "torus_tpuv4_4x4x4": "Torus 4x4x4",
    }

    topology_legend_markers = {
        "mesh_nebula_8x4": "o",
        "clos_8x4": "s",
        "cm384_16x2": "^",
        "torus_tpuv4_4x4x4": "D",
    }

    topology_order = [
        "mesh_nebula_8x4",
        "torus_tpuv4_4x4x4",
        "clos_8x4",
        "cm384_16x2",
    ]
    topology_keys = {row["topology_key"] for row in speedup_rows}
    ordered_topologies = [
        topology for topology in topology_order
        if topology in topology_keys
    ] + sorted(topology_keys - set(topology_order))

    topology_legend = [
        Line2D(
            [0], [0],
            marker=topology_legend_markers.get(topology, TOPOLOGY_MARKERS.get(topology, "o")),
            color="none",
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=4.0,
            markersize=20,
            linestyle="none",
            label=topology_labels.get(topology, topology),
        )
        for topology in ordered_topologies
    ]

    # Swap the 2nd (index 1) and 4th (index 3) items
    if len(variant_legend) >= 4:
        variant_legend[1], variant_legend[3] = variant_legend[3], variant_legend[1]

    if negative_rows:
        variant_legend.append(
            Line2D(
                [0], [0],
                marker="D",
                color="#B00020",
                markerfacecolor="none",
                markeredgewidth=4.5,
                markersize=17,
                linestyle="none",
                label="Speedup < 1",
            )
        )

    first_legend = legend_ax.legend(
        handles=variant_legend,
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.15),
        bbox_transform=legend_ax.transAxes,
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        labelspacing=0.27,
    )
    first_legend.set_title("Trace (Color)", prop={"size": LEGEND_TITLE_FONT_SIZE})
    legend_ax.add_artist(first_legend)

    second_legend = legend_ax.legend(
        handles=topology_legend,
        ncol=1,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.2),
        bbox_transform=legend_ax.transAxes,
        fontsize=LEGEND_FONT_SIZE,
        borderaxespad=0.0,
        labelspacing=0.36,
        # fontsize=8
    )
    second_legend.set_title("Topology (Shape)", prop={"size": LEGEND_TITLE_FONT_SIZE})

    # ax.text(
    #     0.01, 0.02,
    #     "Small x-jitter separates topologies with identical input BPI.",
    #     transform=ax.transAxes,
    #     fontsize=8,
    #     color="#555555"
    # )

    emit(
        save_figure(
            plt,
            fig,
            "speedup_vs_bpi",
            "Speedup vs input BPI",
            "Glaive speedup against the best non-Glaive baseline as input BPI changes.",
            [speedup_path],
            bbox_inches=None,
        )
    )


if __name__ == "__main__":
    main()




# # BACK UP
# #!/usr/bin/env python3

# from __future__ import annotations

# from matplotlib.lines import Line2D

# from plot_common import (
#     RESULTS_DIR,
#     TOPOLOGY_JITTER,
#     TOPOLOGY_MARKERS,
#     emit,
#     fnum,
#     read_csv,
#     save_figure,
#     setup_matplotlib,
#     skip,
#     variant_label,
# )


# # Keep the total canvas aspect ratio fixed while tuning the plot body and
# # right-side legend area independently.
# FIG_SIZE = (15, 6)
# LAYOUT_LEFT = 0.085
# LAYOUT_RIGHT = 0.985
# LAYOUT_BOTTOM = 0.14
# LAYOUT_TOP = 0.88
# PANEL_GAP = 0.002          # smaller value moves the legend closer to the plot body
# BODY_WIDTH_RATIO = 0.70    # plot body share within the usable width
# LEGEND_WIDTH_RATIO = 0.30  # legend share within the usable width
# LEGEND_FONT_SIZE = 22
# LEGEND_TITLE_FONT_SIZE = 26
# GRID_ZORDER = 0
# REFERENCE_LINE_ZORDER = 1
# NEGATIVE_POINT_ZORDER = 9
# VARIANT_ZORDERS = {
#     "balanced": 2,
#     "blend_0.75": 4,
#     "blend_0.50": 5,
#     "blend_0.25": 6,
#     "eplb_x2": 3,
#     "real": 7,
#     "worst_top8": 8,
# }


# def axes_rects() -> tuple[list[float], list[float]]:
#     panel_height = LAYOUT_TOP - LAYOUT_BOTTOM
#     available_width = LAYOUT_RIGHT - LAYOUT_LEFT - PANEL_GAP
#     total_ratio = BODY_WIDTH_RATIO + LEGEND_WIDTH_RATIO
#     if available_width <= 0 or total_ratio <= 0:
#         raise ValueError("layout settings leave no room for the plot body and legend")
#     main_width = available_width * BODY_WIDTH_RATIO / total_ratio
#     legend_width = available_width * LEGEND_WIDTH_RATIO / total_ratio
#     main_rect = [LAYOUT_LEFT, LAYOUT_BOTTOM, main_width, panel_height]
#     legend_rect = [LAYOUT_LEFT + main_width + PANEL_GAP, LAYOUT_BOTTOM, legend_width, panel_height]
#     return main_rect, legend_rect


# def main() -> None:
#     plt = setup_matplotlib()
#     plt.rcParams["figure.autolayout"] = False
#     speedup_path = RESULTS_DIR / "performance_speedup_vs_best_baseline.csv"
#     excluded_topologies = {"rail_optimized_8x4"}
#     speedup_rows = [
#         row for row in read_csv(speedup_path)
#         if row.get("speedup_vs_best_baseline", "") != ""
#         and row.get("topology_key") not in excluded_topologies
#     ]
#     if not speedup_rows:
#         skip("speedup_vs_bpi", "no speedup rows")
#         return

#     # Sorted list of variants (determines the order of colors)
#     sorted_variants = sorted({row["variant"] for row in speedup_rows})

#     # ============================================================
#     # Color definitions – modify these two lists to change all variant colors globally
#     # Both lists must have the same length as sorted_variants (7 items here)
#     # ============================================================

#     # ============================================================
#     # MultiTree_colors = ["#f1fae7", "#bae5bc", "#7bccc5", "#2b8cc0", "#004e89"] # from light to deep
#     # PLOT_COLORS = {
#     #     "glaive_solver3": "#80ed99",
#     #     "teccl_4mb": "#4cc9f0",
#     #     "taccl_directmap_nosym_4mb": "#ff87ab",
#     #     "msccl_direct": "#c77dff",
#     #     "nccl_p2p": "#c77dff",
#     #     "nccl_ring": "#f2cc8f",
#     #     "nccl_biring": "#db7c26",
#     #     "nccl_direct": "#8d99ae",
#     #     "nccl_spreadout": "#b1be9d",
#     # }
#     # LINE_COLORS = {
#     #     "glaive_solver3": "#06d6a0",
#     #     "teccl_4mb": "#118ab2",
#     #     "taccl_directmap_nosym_4mb": "#ff5d8f",
#     #     "msccl_direct": "#7b2cbf",
#     #     "nccl_p2p": "#7b2cbf",
#     #     "nccl_ring": "#bc6c25",
#     #     "nccl_biring": "#d8572a",
#     #     "nccl_direct": "#4a4e69",
#     #     "nccl_spreadout": "#a9a875",
#     # }
#     # ============================================================

#     marker_colors = [
#         "#f1fae7",  # variant 0 (e.g., 'balanced')
#         "#2b8cc0",  # variant 1 (e.g., 'blend25%')
#         "#7bccc5",  # variant 2 (e.g., 'blend50%')
#         "#bae5bc",  # variant 3 (e.g., 'blend75%')
#         "#c77dff",  # variant 4
#         "#db7c26",  # variant 5
#         "#ff87ab",  # variant 6
#     ]
#     # Edge colors – can be identical or different from marker_colors
#     edge_colors = [
#         "#bae5bc",  # variant 0 (e.g., 'balanced')
#         "#004e89",  # variant 1
#         "#2b8cc0",  # variant 2
#         "#7bccc5",  # variant 3
#         "#7b2cbf",  # variant 4
#         "#d8572a",  # variant 5
#         "#ff5d8f",  # variant 6
#     ]

#     # Sanity check: ensure lengths match
#     if len(marker_colors) != len(sorted_variants):
#         raise ValueError(
#             f"marker_colors length ({len(marker_colors)}) does not match number of variants ({len(sorted_variants)})"
#         )

#     fig = plt.figure(figsize=FIG_SIZE, constrained_layout=False)
#     fig.set_tight_layout(False)
#     main_ax_rect, legend_ax_rect = axes_rects()
#     ax = fig.add_axes(main_ax_rect)
#     legend_ax = fig.add_axes(legend_ax_rect)
#     legend_ax.axis("off")

#     # # dedicated markers for this script
#     # topology_markers = ['o', 's', '^', 'D', '*']
#     # Plot main data points using indexed colors
#     for idx, variant in enumerate(sorted_variants):
#         zorder_val = VARIANT_ZORDERS.get(variant, VARIANT_ZORDERS["real"])
#         rows = [row for row in speedup_rows if row["variant"] == variant]
#         for topology in sorted({row["topology_key"] for row in rows}):
#             topo_rows = [row for row in rows if row["topology_key"] == topology]
#             ax.scatter(
#                 [
#                     max(0.0, min(1.0, fnum(row, "input_bimodal_pressure_index") + TOPOLOGY_JITTER.get(topology, 0.0)))
#                     for row in topo_rows
#                 ],
#                 [fnum(row, "speedup_vs_best_baseline") for row in topo_rows],
#                 s=250,
#                 alpha=0.7,
#                 marker=TOPOLOGY_MARKERS.get(topology, "o"),
#                 # marker=marker_style,
#                 facecolors=marker_colors[idx],
#                 edgecolors=edge_colors[idx],
#                 linewidths=4.5,
#                 zorder=zorder_val,
#             )

#     ax.axhline(1.0, color="#333333", linestyle="--", linewidth=5.0, zorder=REFERENCE_LINE_ZORDER)

#     negative_rows = [
#         row for row in speedup_rows
#         if fnum(row, "speedup_vs_best_baseline", 1.0) < 1.0
#     ]
#     if negative_rows:
#         ax.scatter(
#             [
#                 max(
#                     0.0,
#                     min(1.0, fnum(row, "input_bimodal_pressure_index") + TOPOLOGY_JITTER.get(row.get("topology_key", ""), 0.0)),
#                 )
#                 for row in negative_rows
#             ],
#             [fnum(row, "speedup_vs_best_baseline") for row in negative_rows],
#             s=250,
#             marker="D",
#             facecolors="none",
#             edgecolors="#B00020",          # fixed red border for speedup < 1
#             linewidths=3.5,
#             label="speedup < 1",
#             zorder=NEGATIVE_POINT_ZORDER,
#         )

#     ax.set_xlabel("Input Bimodal Pressure Index (BPI)", fontsize=28)
#     ax.set_ylabel("Glaive Speedup vs Best Baseline", fontsize=25.5)
#     ax.set_title("Glaive Performance across Bimodality Levels", fontsize=30)
#     ax.set_ylim(0.4, 6.7)
#     ax.set_xlim(-0.025, 0.9)
#     ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
#     ax.set_yticks([1, 2, 3, 4, 5, 6])
#     ax.set_axisbelow(True)
#     ax.grid(True, linestyle='--', linewidth=1.5, alpha=0.6, zorder=GRID_ZORDER)
#     for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
#         gridline.set_zorder(GRID_ZORDER)
#     ax.tick_params(axis='x', labelsize=28)
#     ax.tick_params(axis='y', labelsize=30)

#     # Build variant legend using the same indexed colors
#     variant_legend = []
#     for idx, variant in enumerate(sorted_variants):
#         variant_legend.append(
#             Line2D(
#                 [0], [0],
#                 marker="o",
#                 color="none",
#                 markerfacecolor=marker_colors[idx],
#                 markeredgecolor=edge_colors[idx],
#                 markeredgewidth=2.0,
#                 markersize=22,
#                 label=variant_label(variant),
#             )
#         )
    

#     # Topology legend (fixed gray markers, unchanged)
#     topology_labels = {
#         "mesh_nebula_8x4": "Mesh 8x4",
#         "clos_8x4": "Clos 8x4",
#         "cm384_16x2": "CM384 2x8x2",
#         "torus_tpuv4_4x4x4": "Torus 4x4x4",
#     }

#     topology_legend_markers = {
#         "mesh_nebula_8x4": "o",
#         "clos_8x4": "s",
#         "cm384_16x2": "^",
#         "torus_tpuv4_4x4x4": "D",
#     }

#     topology_order = [
#         "mesh_nebula_8x4",
#         "torus_tpuv4_4x4x4",
#         "clos_8x4",
#         "cm384_16x2",
#     ]
#     topology_keys = {row["topology_key"] for row in speedup_rows}
#     ordered_topologies = [
#         topology for topology in topology_order
#         if topology in topology_keys
#     ] + sorted(topology_keys - set(topology_order))

#     topology_legend = [
#         Line2D(
#             [0], [0],
#             marker=topology_legend_markers.get(topology, TOPOLOGY_MARKERS.get(topology, "o")),
#             color="none",
#             markerfacecolor='white',
#             markeredgecolor='black',
#             markeredgewidth=4.0,
#             markersize=20,
#             linestyle="none",
#             label=topology_labels.get(topology, topology),
#         )
#         for topology in ordered_topologies
#     ]

#     # Swap the 2nd (index 1) and 4th (index 3) items
#     if len(variant_legend) >= 4:
#         variant_legend[1], variant_legend[3] = variant_legend[3], variant_legend[1]

#     if negative_rows:
#         variant_legend.append(
#             Line2D(
#                 [0], [0],
#                 marker="D",
#                 color="#B00020",
#                 markerfacecolor="none",
#                 markeredgewidth=4.5,
#                 markersize=17,
#                 linestyle="none",
#                 label="Speedup < 1",
#             )
#         )

#     first_legend = legend_ax.legend(
#         handles=variant_legend,
#         ncol=1,
#         frameon=False,
#         loc="upper left",
#         bbox_to_anchor=(0.0, 1.15),
#         bbox_transform=legend_ax.transAxes,
#         fontsize=LEGEND_FONT_SIZE,
#         borderaxespad=0.0,
#         labelspacing=0.27,
#     )
#     first_legend.set_title("Trace (Color)", prop={"size": LEGEND_TITLE_FONT_SIZE})
#     legend_ax.add_artist(first_legend)

#     second_legend = legend_ax.legend(
#         handles=topology_legend,
#         ncol=1,
#         frameon=False,
#         loc="lower left",
#         bbox_to_anchor=(0.0, -0.2),
#         bbox_transform=legend_ax.transAxes,
#         fontsize=LEGEND_FONT_SIZE,
#         borderaxespad=0.0,
#         labelspacing=0.36,
#         # fontsize=8
#     )
#     second_legend.set_title("Topology (Shape)", prop={"size": LEGEND_TITLE_FONT_SIZE})

#     # ax.text(
#     #     0.01, 0.02,
#     #     "Small x-jitter separates topologies with identical input BPI.",
#     #     transform=ax.transAxes,
#     #     fontsize=8,
#     #     color="#555555"
#     # )

#     emit(
#         save_figure(
#             plt,
#             fig,
#             "speedup_vs_bpi",
#             "Speedup vs input BPI",
#             "Glaive speedup against the best non-Glaive baseline as input BPI changes.",
#             [speedup_path],
#             bbox_inches=None,
#         )
#     )


# if __name__ == "__main__":
#     main()
