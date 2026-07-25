#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from plot_common import RESULTS_DIR, VARIANT_COLORS, emit, fnum, read_csv, save_figure, setup_matplotlib, skip, variant_label


def main() -> None:
    plt = setup_matplotlib()
    speedup_variant_path = RESULTS_DIR / "performance_speedup_by_variant.csv"
    rows = read_csv(speedup_variant_path)
    if not rows:
        skip("speedup_by_variant", "no variant speedup rows")
        return

    order = [row["variant"] for row in rows]
    x = np.arange(len(order))
    means = np.asarray([fnum(row, "speedup_vs_best_baseline_mean") for row in rows])
    p10 = np.asarray([fnum(row, "speedup_vs_best_baseline_p10") for row in rows])
    p90 = np.asarray([fnum(row, "speedup_vs_best_baseline_p90") for row in rows])
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.bar(x, means, color=[VARIANT_COLORS.get(variant, "#777777") for variant in order], alpha=0.88)
    ax.errorbar(x, means, yerr=[means - p10, p90 - means], fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([variant_label(variant) for variant in order], rotation=22, ha="right")
    ax.set_ylabel("Speedup vs Best Baseline")
    ax.set_title("Aggregate Glaive Speedup by Traffic Variant")
    emit(
        save_figure(
            plt,
            fig,
            "speedup_by_variant",
            "Aggregate speedup by variant",
            "Mean, p10, and p90 Glaive speedup grouped by traffic variant.",
            [speedup_variant_path],
        )
    )


if __name__ == "__main__":
    main()
