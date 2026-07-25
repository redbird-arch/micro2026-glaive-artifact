#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from plot_common import METHOD_COLORS, RESULTS_DIR, emit, fnum, read_csv, save_figure, setup_matplotlib, skip


def main() -> None:
    plt = setup_matplotlib()
    network_path = RESULTS_DIR / "network_balance_existing_link_256mb_summary.csv"
    rows = read_csv(network_path)
    if not rows:
        skip("network_balance_256mb", "no network summary rows")
        return

    rows = sorted(rows, key=lambda row: (row["topology_key"], row["method"]))
    x = np.arange(len(rows))
    labels = [f"{row['topology_key']}\n{row['method']}" for row in rows]
    colors = [METHOD_COLORS.get(row["method"], "#777777") for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(15.8, 10.2), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "used_links_mean", "Used Links"),
        (axes[1], "link_use_count_sum_mean", "Link Busy Intervals"),
        (axes[2], "busy_time_gini_mean", "Busy-Time Gini"),
    ]:
        ax.bar(x, [fnum(row, metric) for row in rows], color=colors, alpha=0.88)
        ax.set_ylabel(ylabel)
    axes[0].set_title("Existing 256MB Link-Utilization Load Balance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=62, ha="right", fontsize=8)
    legend_methods = sorted({row["method"] for row in rows})
    axes[0].legend(
        [Line2D([0], [0], color=METHOD_COLORS.get(method, "#777777"), lw=8) for method in legend_methods],
        legend_methods,
        ncol=len(legend_methods),
        frameon=False,
        loc="upper right",
    )
    emit(
        save_figure(
            plt,
            fig,
            "network_balance_256mb",
            "Existing 256MB network load balance",
            "Used links, link busy intervals, and busy-time Gini from existing 256MB link-utilization logs.",
            [network_path],
        )
    )


if __name__ == "__main__":
    main()
