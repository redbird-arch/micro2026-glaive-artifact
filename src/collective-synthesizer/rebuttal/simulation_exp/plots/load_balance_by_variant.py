#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from plot_common import RESULTS_DIR, emit, fnum, read_csv, save_figure, setup_matplotlib, skip, variant_label


def main() -> None:
    plt = setup_matplotlib()
    speedup_variant_path = RESULTS_DIR / "performance_speedup_by_variant.csv"
    rows = read_csv(speedup_variant_path)
    if not rows:
        skip("load_balance_by_variant", "no variant speedup rows")
        return

    order = [row["variant"] for row in rows]
    x = np.arange(len(order))
    fig, axes = plt.subplots(4, 1, figsize=(11.8, 10.0), sharex=True)
    load_metrics = [
        ("fabric_active_link_fraction_delta_vs_best_baseline_mean", "Active-link\nfraction delta", "#1F77B4"),
        ("pred_hot_edge_actual_busy_share_delta_vs_best_baseline_mean", "Hot-edge\nshare reduction", "#D62728"),
        ("hot_flow_path_actual_busy_share_delta_vs_best_baseline_mean", "Hot-flow-path\nshare reduction", "#E69F00"),
        ("det_demand_vs_actual_busy_corr_delta_vs_best_baseline_mean", "Deterministic\ncorr. reduction", "#2CA02C"),
    ]
    for ax, (metric, ylabel, color) in zip(axes, load_metrics):
        values = [fnum(row, metric) for row in rows]
        ax.bar(x, values, color=color, alpha=0.84)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_ylabel(ylabel)
    axes[0].set_title("Glaive Load-Balance Effects vs Best Baseline")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([variant_label(variant) for variant in order], rotation=22, ha="right")
    emit(
        save_figure(
            plt,
            fig,
            "load_balance_by_variant",
            "Load-balance effect by variant",
            "Glaive load-balance deltas against the best non-Glaive baseline grouped by traffic variant.",
            [speedup_variant_path],
        )
    )


if __name__ == "__main__":
    main()
