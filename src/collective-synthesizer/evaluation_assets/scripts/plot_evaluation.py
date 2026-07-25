#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


TOPOLOGY_DISPLAY = {
    "torus_tpuv4_4x4x4": "3D Torus 4x4x4",
    "mesh_nebula_8x4": "2D Mesh 8x4",
    "cm384_16x2_eval": "3D CM384 2x8x2",
    "cm384_2x8x2_eval": "3D CM384 2x8x2",
    "cm384_8x4_eval": "3D CM384 2x8x2",
    "rail_optimized_8x4_eval": "2D Rail-Optimized 8x4",
    "fattree_8x4_eval": "2D Clos 8x4",
}
TOPOLOGY_GROUPS = [
    ("torus_tpuv4_4x4x4",),
    ("mesh_nebula_8x4",),
    ("cm384_16x2_eval", "cm384_2x8x2_eval", "cm384_8x4_eval"),
    ("rail_optimized_8x4_eval",),
    ("fattree_8x4_eval",),
]
SYNTHETIC_COMBINED_TOPOLOGY_GROUPS = [
    ("mesh_nebula_8x4",),
    ("torus_tpuv4_4x4x4",),
    ("fattree_8x4_eval",),
    ("cm384_16x2_eval", "cm384_2x8x2_eval", "cm384_8x4_eval"),
]
LINK_COMBINED_TOPOLOGY_GROUPS = [
    ("mesh_nebula_8x4",),
    ("torus_tpuv4_4x4x4",),
    ("fattree_8x4_eval",),
    ("cm384_16x2_eval", "cm384_2x8x2_eval", "cm384_8x4_eval"),
]
SYNTHETIC_SPEEDUP_EXCLUDED_TOPOLOGIES = {"rail_optimized_8x4_eval"}
SIZE_ORDER = ["1MB", "8MB", "64MB", "256MB", "2GB"]
METHOD_DISPLAY = {
    "glaive": "Glaive",
    "biring": "BiRing",
    "halfringdr": "HalfR+DR",
    "mpibaseline": "MPICH",
}
METHOD_COLORS = {
    "glaive": "#06d6a0",
    "biring": "#118ab2",
    "halfringdr": "#118ab2",
    "mpibaseline": "#f77f00",
}
METHOD_FILL_COLORS = {
    "glaive": "#80ed99",
    "biring": "#4cc9f0",
    "halfringdr": "#4cc9f0",
    "mpibaseline": "#fcbf49",
}
METHOD_ENDPOINT_MARKER_COLORS = {
    "glaive": "#3a5a40",
    "biring": "#6930c3",
    "halfringdr": "#6930c3",
    "mpibaseline": "#6f4518",
}
METHOD_MARKERS = {
    "glaive": "o",
    "biring": "D",
    "halfringdr": "D",
    "mpibaseline": "s",
}
SYNTHETIC_LINEWIDTH = 9.0
LINK_LINEWIDTH = 6.0
LINK_ENDPOINT_MARKER_Y = 4.0
SCALABILITY_THEORY = {
    "mesh": {
        "label": r"$O(n^{5/2}\log n)$",
        "basis": lambda x: np.power(x, 2.5) * np.log2(np.maximum(x, 2.0)),
    },
    "torus": {
        "label": r"$O(n^{7/3}\log n)$",
        "basis": lambda x: np.power(x, 7.0 / 3.0) * np.log2(np.maximum(x, 2.0)),
    },
    "fullmesh": {
        "label": r"$O(n^2\log n)$",
        "basis": lambda x: np.power(x, 2.0) * np.log2(np.maximum(x, 2.0)),
    },
    "fat-tree": {
        "label": r"$O(n^3\log n)$",
        "basis": lambda x: np.power(x, 3.0) * np.log2(np.maximum(x, 2.0)),
    },
    "cm384": {
        "label": r"$O(n^3\log n)$",
        "basis": lambda x: np.power(x, 3.0) * np.log2(np.maximum(x, 2.0)),
    },
}
SCALABILITY_COMBINED_MAX_DEVICES = 4096
SCALABILITY_COMBINED_XLIM = 4200.0
SCALABILITY_COMBINED_XTICKS = [1000, 2000, 3000, 4000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-style evaluation figures.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def synthetic_methods_for_topology(topology_key: str) -> list[str]:
    if topology_key == "torus_tpuv4_4x4x4":
        return ["glaive", "halfringdr", "mpibaseline"]
    return ["glaive", "biring", "mpibaseline"]


def resolve_topology_groups(
    available_topology_keys: set[str], topology_groups: list[tuple[str, ...]]
) -> list[str]:
    resolved: list[str] = []
    for group in topology_groups:
        for candidate in group:
            if candidate in available_topology_keys:
                resolved.append(candidate)
                break
    return resolved


def link_methods_for_topology(topology_key: str) -> tuple[str, ...]:
    if topology_key == "torus_tpuv4_4x4x4":
        return ("glaive", "halfringdr", "mpibaseline")
    return ("glaive", "biring", "mpibaseline")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 19,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "grid.linewidth": 0.9,
        }
    )


def lineplot(
    ax: plt.Axes,
    x: list[float],
    y: list[float],
    method: str,
    with_markers: bool = True,
    linewidth: float = SYNTHETIC_LINEWIDTH,
    glaive_on_top: bool = False,
) -> None:
    ax.plot(
        x,
        y,
        color=METHOD_COLORS[method],
        marker=METHOD_MARKERS[method] if with_markers else None,
        markersize=20 if with_markers else 0,
        markerfacecolor=METHOD_FILL_COLORS[method],
        markeredgecolor=METHOD_COLORS[method],
        markeredgewidth=6 if with_markers else 0,
        linewidth=linewidth,
        label=METHOD_DISPLAY[method],
        zorder=5 if glaive_on_top and method == "glaive" else 3,
    )


def annotated_subplot_title(index: int, base_title: str) -> str:
    return f"({chr(ord('a') + index)}) {base_title}"


def add_endpoint_star_markers(
    ax: plt.Axes, endpoint_markers: list[tuple[float, str]]
) -> None:
    if not endpoint_markers:
        return
    for x_value, method in endpoint_markers:
        if math.isnan(x_value):
            continue
        ax.scatter(
            x_value,
            LINK_ENDPOINT_MARKER_Y,
            marker="s",
            s=500,
            color=METHOD_ENDPOINT_MARKER_COLORS[method],
            edgecolors=METHOD_ENDPOINT_MARKER_COLORS[method],
            linewidths=1.6,
            zorder=9,
        )


def plot_synthetic_axis(
    ax: plt.Axes,
    grouped: dict[tuple[str, str], dict[str, float]],
    topology_key: str,
    title: str | None = None,
    legend_anchor_y: float = 1.0,
) -> None:
    methods = synthetic_methods_for_topology(topology_key)
    x = np.arange(len(SIZE_ORDER))
    local_max = 0.0
    for method in methods:
        y = [grouped[(topology_key, method)].get(size_label, np.nan) for size_label in SIZE_ORDER]
        finite_values = [value for value in y if not math.isnan(value)]
        if finite_values:
            local_max = max(local_max, max(finite_values))
        lineplot(ax, list(x), y, method, with_markers=True, linewidth=SYNTHETIC_LINEWIDTH)

    ax.set_title(title or TOPOLOGY_DISPLAY[topology_key], fontsize=38, y=1.2)
    ax.set_xlabel("Alltoallv Data Size", fontsize=40)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=40)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER, rotation=30)
    ax.tick_params(axis="x", labelsize=34, rotation=0)
    ax.tick_params(axis="y", labelsize=36)
    if local_max > 0:
        ax.set_ylim(-2, local_max * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.legend(
        loc="upper center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, legend_anchor_y),
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.1,
        borderpad=0.2,
        markerscale=1.0,
        handlelength=1.9,
        fontsize=28,
    )


def print_glaive_speedups_for_topology(
    grouped: dict[tuple[str, str], dict[str, float]], topology_key: str
) -> None:
    methods = synthetic_methods_for_topology(topology_key)
    glaive_values = grouped.get((topology_key, "glaive"), {})
    if not glaive_values:
        return

    for method in methods:
        if method == "glaive":
            continue
        baseline_values = grouped.get((topology_key, method), {})
        speedups: list[float] = []
        for size_label in SIZE_ORDER:
            glaive_bw = glaive_values.get(size_label)
            baseline_bw = baseline_values.get(size_label)
            if (
                glaive_bw is None
                or baseline_bw is None
                or math.isnan(glaive_bw)
                or math.isnan(baseline_bw)
                or baseline_bw <= 0
            ):
                continue
            speedups.append(glaive_bw / baseline_bw)

        if speedups:
            avg_speedup = float(np.mean(speedups))
            print(
                f"[Synthetic][{TOPOLOGY_DISPLAY.get(topology_key, topology_key)}] "
                f"Glaive vs {METHOD_DISPLAY[method]} avg speedup: {avg_speedup:.3f}x "
                f"(n={len(speedups)})"
            )
        else:
            print(
                f"[Synthetic][{TOPOLOGY_DISPLAY.get(topology_key, topology_key)}] "
                f"Glaive vs {METHOD_DISPLAY[method]} avg speedup: N/A (no overlapping points)"
            )


def print_glaive_speedup_summary_all_configs(
    grouped: dict[tuple[str, str], dict[str, float]], topology_keys: list[str]
) -> None:
    baseline_methods: dict[str, str] = {
        "BiRing": "biring",
        "HalfR+DR": "halfringdr",
        "MPICH": "mpibaseline",
    }
    aggregated_speedups: dict[str, list[float]] = {label: [] for label in baseline_methods}

    for topology_key in topology_keys:
        glaive_values = grouped.get((topology_key, "glaive"), {})
        if not glaive_values:
            continue
        for size_label in SIZE_ORDER:
            glaive_bw = glaive_values.get(size_label)
            if glaive_bw is None or math.isnan(glaive_bw):
                continue
            for label, method in baseline_methods.items():
                baseline_bw = grouped.get((topology_key, method), {}).get(size_label)
                if baseline_bw is None or baseline_bw <= 0:
                    continue
                aggregated_speedups[label].append(glaive_bw / baseline_bw)

    print("[Synthetic][Overall] Glaive speedup summary across all configurations:")
    for label, speedups in aggregated_speedups.items():
        if speedups:
            avg_speedup = float(np.mean(speedups))
            print(f"  - vs {label}: {avg_speedup:.3f}x (n={len(speedups)})")
        else:
            print(f"  - vs {label}: N/A (no overlapping points)")


def print_synthetic_experiment_speedup_report(
    grouped: dict[tuple[str, str], dict[str, float]], topology_keys: list[str]
) -> None:
    """Print the speedup numbers used by Synthetic_Experiment.pdf."""
    print("[Synthetic_Experiment] Glaive speedup by topology and baseline:")
    all_speedups: list[float] = []
    by_baseline: dict[str, list[float]] = defaultdict(list)
    for topology_key in topology_keys:
        glaive_values = grouped.get((topology_key, "glaive"), {})
        if not glaive_values:
            continue
        for method in synthetic_methods_for_topology(topology_key):
            if method == "glaive":
                continue
            baseline_values = grouped.get((topology_key, method), {})
            speedups: list[float] = []
            for size_label in SIZE_ORDER:
                glaive_bw = glaive_values.get(size_label)
                baseline_bw = baseline_values.get(size_label)
                if (
                    glaive_bw is None
                    or baseline_bw is None
                    or math.isnan(glaive_bw)
                    or math.isnan(baseline_bw)
                    or baseline_bw <= 0
                ):
                    continue
                speedups.append(glaive_bw / baseline_bw)
            if not speedups:
                continue
            all_speedups.extend(speedups)
            by_baseline[METHOD_DISPLAY[method]].extend(speedups)
            print(
                f"  - {TOPOLOGY_DISPLAY.get(topology_key, topology_key)} vs {METHOD_DISPLAY[method]}: "
                f"{float(np.mean(speedups)):.3f}x "
                f"(n={len(speedups)}, sizes={','.join(SIZE_ORDER[:len(speedups)])})"
            )

    print("[Synthetic_Experiment] Glaive speedup aggregated across displayed topology-size points:")
    for baseline, speedups in sorted(by_baseline.items()):
        print(f"  - vs {baseline}: {float(np.mean(speedups)):.3f}x (n={len(speedups)})")
    if all_speedups:
        print(f"  - all baseline comparisons: {float(np.mean(all_speedups)):.3f}x (n={len(all_speedups)})")


def plot_synthetic(summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in summary_rows:
        grouped[(row["topology_key"], row["method"])][row["size_label"]] = float(row["avg_bandwidth_gbps"])

    available_topology_keys = {row["topology_key"] for row in summary_rows}
    topology_order = resolve_topology_groups(available_topology_keys, TOPOLOGY_GROUPS)
    for topology_key in topology_order:
        fig, ax = plt.subplots(figsize=(9.0, 8.0))
        plot_synthetic_axis(ax, grouped, topology_key)
        fig.tight_layout()
        fig.savefig(output_dir / f"synthetic_bandwidth_{topology_key}.pdf")
        plt.close(fig)
    summary_topology_keys = [
        topology_key
        for topology_key in topology_order
        if topology_key not in SYNTHETIC_SPEEDUP_EXCLUDED_TOPOLOGIES
    ]
    if summary_topology_keys:
        print_glaive_speedup_summary_all_configs(grouped, summary_topology_keys)

    combined_topology_keys = resolve_topology_groups(
        available_topology_keys, SYNTHETIC_COMBINED_TOPOLOGY_GROUPS
    )
    if combined_topology_keys:
        print_synthetic_experiment_speedup_report(grouped, combined_topology_keys)
    if combined_topology_keys:
        fig, axes = plt.subplots(1, len(combined_topology_keys), figsize=(9.2 * len(combined_topology_keys), 8.0))
        if len(combined_topology_keys) == 1:
            axes = [axes]
        for index, (ax, topology_key) in enumerate(zip(axes, combined_topology_keys)):
            plot_synthetic_axis(
                ax,
                grouped,
                topology_key,
                title=annotated_subplot_title(index, TOPOLOGY_DISPLAY[topology_key]),
                legend_anchor_y=1.22,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        plt.subplots_adjust(wspace=0.27)
        fig.savefig(output_dir / "Synthetic_Experiment.pdf")
        plt.close(fig)


def interpolate_y(points: list[tuple[float, float]], target_x: float) -> float:
    if not points:
        return 0.0
    if target_x <= points[0][0]:
        return points[0][1]
    if target_x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x0 <= target_x <= x1:
            if x1 == x0:
                return y1
            ratio = (target_x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def trim_points_to_endpoint(
    points: list[tuple[float, float]], endpoint_x: float
) -> list[tuple[float, float]]:
    if not points or math.isnan(endpoint_x) or endpoint_x <= 0:
        return points

    trimmed = [(x_value, y_value) for x_value, y_value in points if x_value < endpoint_x]
    boundary_y = interpolate_y(points, endpoint_x)

    if trimmed and math.isclose(trimmed[-1][0], endpoint_x, abs_tol=1e-9):
        return trimmed
    trimmed.append((endpoint_x, boundary_y))
    return trimmed


def choose_time_unit(max_microseconds: float) -> tuple[float, str]:
    if max_microseconds >= 1_000_000:
        return 1_000_000.0, "s"
    if max_microseconds >= 1_000:
        return 1_000.0, "ms"
    return 1.0, "us"


def fit_theory_curve(x: np.ndarray, y: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(basis, basis))
    scale = float(np.dot(basis, y) / denominator) if denominator else 0.0
    fitted = scale * basis
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if math.isclose(ss_tot, 0.0) else 1.0 - (ss_res / ss_tot)
    return scale, r_squared


def format_devices_k(value: float, _: object) -> str:
    if math.isclose(value, 0.0, abs_tol=1e-8):
        return "0"
    if abs(value) < 1_000:
        return f"{value:g}"
    value_k = value / 1_000.0
    rounded_k = round(value_k)
    if math.isclose(value_k, rounded_k, abs_tol=1e-8):
        return f"{int(rounded_k)}K"
    return f"{value_k:g}K"


def plot_scalability_axis(
    ax: plt.Axes,
    points: list[tuple[int, float]],
    topology_type: str,
    title: str,
    color: str,
    max_devices: int | None = None,
    fixed_xlim: float | None = None,
    fixed_xticks: list[int] | None = None,
) -> None:
    if max_devices is not None:
        points = [point for point in points if point[0] <= max_devices]

    if not points:
        ax.set_visible(False)
        return

    x = np.array([point[0] for point in points], dtype=float)
    y = np.array([point[1] for point in points], dtype=float)
    theory = SCALABILITY_THEORY[topology_type]
    basis = np.array(theory["basis"](x), dtype=float)
    theory_scale, r_squared = fit_theory_curve(x, y, basis)
    fit_x = np.linspace(0.0, float(x.max()), 400)
    fit_basis = np.array(theory["basis"](fit_x), dtype=float)
    fit_y = theory_scale * fit_basis

    ax.plot(
        fit_x,
        fit_y,
        color="#b3b3b3",
        linestyle="--",
        linewidth=13.0,
        zorder=1,
    )
    ax.plot(
        x,
        y,
        color=color,
        marker="o",
        markersize=22,
        linewidth=11.0,
        markerfacecolor=color,
        markeredgecolor=color,
        zorder=3,
    )
    if fixed_xlim is not None:
        ax.set_xlim(0, fixed_xlim)
    else:
        ax.set_xlim(0, float(x.max()) * 1.04)
    ax.set_ylim(0, max(float(y.max()), float(fit_y.max())) * 1.10)
    if fixed_xticks is not None:
        ax.set_xticks(fixed_xticks)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(format_devices_k))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="x", labelsize=36)
    ax.tick_params(axis="y", labelsize=40)
    ax.set_title(title, fontsize=50)
    ax.set_xlabel("Number of Devices", fontsize=37)
    ax.set_ylabel("Synthesis Time (s)", fontsize=46)
    ax.text(0.05, 0.88, rf"$R^2 = {r_squared:.4f}$", transform=ax.transAxes, fontsize=46)
    # ax.text(0.53, 0.30, theory["label"], transform=ax.transAxes, fontsize=42, color="#5a5a5a")
    ax.text(0.05, 0.7, theory["label"], transform=ax.transAxes, fontsize=43, color="#5a5a5a")


def plot_link_axis(
    ax: plt.Axes,
    method_points: dict[str, list[tuple[float, float]]],
    lifecycle_avg: dict[tuple[str, str], float],
    avg_makespan_us: dict[tuple[str, str], float],
    topology_key: str,
    title: str | None = None,
) -> None:
    endpoint_us_by_method = {
        method: avg_makespan_us.get((topology_key, method), points[-1][0])
        for method, points in method_points.items()
        if points
    }
    max_actual_us = max(endpoint_us_by_method.values(), default=0.0)
    if max_actual_us <= 0:
        max_actual_us = 1.0
    time_divisor, time_unit = choose_time_unit(max_actual_us)

    max_x = 0.0
    legend_methods: list[str] = []
    endpoint_markers: list[tuple[float, str]] = []
    for method, points in method_points.items():
        endpoint_us = endpoint_us_by_method.get(method, points[-1][0])
        trimmed_points = trim_points_to_endpoint(points, endpoint_us)
        if not trimmed_points:
            continue
        x = [point[0] / time_divisor for point in trimmed_points]
        y = [point[1] for point in trimmed_points]
        endpoint_x = endpoint_us / time_divisor
        max_x = max(max_x, endpoint_x)
        lineplot(
            ax,
            x,
            y,
            method,
            with_markers=False,
            linewidth=LINK_LINEWIDTH,
            glaive_on_top=True,
        )
        legend_methods.append(method)
        endpoint_markers.append((endpoint_x, method))

    ax.set_title(title or TOPOLOGY_DISPLAY[topology_key], fontsize=40)
    ax.set_xlabel(f"Alltoallv Time ({time_unit})", fontsize=39)
    ax.set_ylabel("Link Utilization", fontsize=42)
    ax.set_xlim(0, max_x * 1.02 if max_x > 0 else 1.0)
    ax.set_ylim(0, 105)
    add_endpoint_star_markers(ax, endpoint_markers)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "" if math.isclose(value, 0.0, abs_tol=1e-8) else f"{value:g}")
    )
    ax.tick_params(axis="x", labelsize=36)
    ax.tick_params(axis="y", labelsize=32)
    handles = [Line2D([0], [0], color=METHOD_COLORS[method], linewidth=10.0) for method in legend_methods]
    labels = [
        f"{METHOD_DISPLAY[method]}\n{lifecycle_avg[(topology_key, method)]:.1f}%"
        for method in legend_methods
    ]
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=max(1, len(legend_methods)),
        loc="upper center",
        bbox_to_anchor=(0.6, 1.03),
        handletextpad=0.25,
        columnspacing=1.0,
        borderpad=0.2,
        handlelength=1.3,
        labelspacing=0.2,
        fontsize=30,
    )


def plot_link(trace_rows: list[dict[str, str]], summary_rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    lifecycle_avg: dict[tuple[str, str], float] = {}
    avg_makespan_us: dict[tuple[str, str], float] = {}

    for row in trace_rows:
        normalized_time = float(row["normalized_time"])
        reference_us = float(row["reference_makespan_us"])
        actual_us = normalized_time * reference_us
        grouped[(row["topology_key"], row["method"])].append(
            (actual_us, float(row["avg_utilization_pct"]))
        )
        lifecycle_avg[(row["topology_key"], row["method"])] = float(row["lifecycle_avg_utilization_pct"])

    for row in summary_rows:
        lifecycle_avg[(row["topology_key"], row["method"])] = float(row["avg_lifecycle_utilization_pct"])
        avg_makespan_us[(row["topology_key"], row["method"])] = float(row["avg_makespan_us"])

    available_topology_keys = {row["topology_key"] for row in trace_rows} | {
        row["topology_key"] for row in summary_rows
    }
    topology_order = resolve_topology_groups(available_topology_keys, TOPOLOGY_GROUPS)
    combined_topology_keys = resolve_topology_groups(
        available_topology_keys, LINK_COMBINED_TOPOLOGY_GROUPS
    )
    combined_method_points: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for topology_key in topology_order:
        method_points: dict[str, list[tuple[float, float]]] = {}
        for method in link_methods_for_topology(topology_key):
            points = sorted(grouped.get((topology_key, method), []))
            if points:
                method_points[method] = points
        if not method_points:
            continue
        combined_method_points[topology_key] = method_points

        fig, ax = plt.subplots(figsize=(10, 7.5))
        plot_link_axis(ax, method_points, lifecycle_avg, avg_makespan_us, topology_key)
        fig.tight_layout()
        fig.savefig(output_dir / f"link_utilization_{topology_key}.pdf")
        plt.close(fig)

    available_combined_topology_keys = [key for key in combined_topology_keys if key in combined_method_points]
    if available_combined_topology_keys:
        fig, axes = plt.subplots(1, len(available_combined_topology_keys), figsize=(10 * len(available_combined_topology_keys), 6))
        if len(available_combined_topology_keys) == 1:
            axes = [axes]
        for index, (ax, topology_key) in enumerate(zip(axes, available_combined_topology_keys)):
            plot_link_axis(
                ax,
                combined_method_points[topology_key],
                lifecycle_avg,
                avg_makespan_us,
                topology_key,
                title=annotated_subplot_title(index, TOPOLOGY_DISPLAY[topology_key]),
            )
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.46)
        fig.savefig(output_dir / "Link_Utilization.pdf")
        plt.close(fig)


def plot_scalability(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["topology_type"]].append((int(row["npus_count"]), float(row["solver_time_s"])))

    topology_order = ["mesh", "torus", "fullmesh", "fat-tree", "cm384"]
    combined_topology_order = ["fullmesh", "torus", "fat-tree", "cm384"]
    titles = {
        "mesh": "2D Mesh",
        "torus": "3D Torus",
        "fullmesh": "2D FullMesh",
        "fat-tree": "2D Clos",
        "cm384": "3D CM384",
    }
    colors = {
        "mesh": "#06d6a0",
        "torus": "#118ab2",
        "fullmesh": "#f77f00",
        "fat-tree": "#7b2cbf",
        "cm384": "#ff5d8f",
    }

    combined_points = {topology_type: sorted(grouped.get(topology_type, [])) for topology_type in combined_topology_order}
    if any(combined_points.values()):
        fig, axes = plt.subplots(1, 4, figsize=(28, 10))
        for index, (ax, topology_type) in enumerate(zip(axes, combined_topology_order)):
            points = combined_points[topology_type]
            if not points:
                ax.set_visible(False)
                continue
            plot_scalability_axis(
                ax,
                points,
                topology_type,
                annotated_subplot_title(index, titles[topology_type]),
                colors[topology_type],
                max_devices=SCALABILITY_COMBINED_MAX_DEVICES,
                fixed_xlim=SCALABILITY_COMBINED_XLIM,
                fixed_xticks=SCALABILITY_COMBINED_XTICKS,
            )
        fig.tight_layout()
        fig.savefig(output_dir / "Scalability_Study.pdf")
        plt.close(fig)

    for topology_type in topology_order:
        points = sorted(grouped.get(topology_type, []))
        if not points:
            continue

        fig, ax = plt.subplots(figsize=(7, 10))
        plot_scalability_axis(ax, points, topology_type, titles[topology_type], colors[topology_type])
        fig.tight_layout()
        safe_topology_type = topology_type.replace("-", "_")
        fig.savefig(output_dir / f"scalability_solver_time_{safe_topology_type}.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    apply_style()

    parsed_dir = repo_root / "evaluation_assets" / "parsed"
    output_dir = repo_root / "evaluation_assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_summary = read_csv(parsed_dir / "synthetic_summary.csv")
    link_trace = read_csv(parsed_dir / "link_utilization_trace.csv")
    link_summary = read_csv(parsed_dir / "link_utilization_summary.csv")
    scalability_rows = read_csv(parsed_dir / "scalability_raw.csv")

    if synthetic_summary:
        plot_synthetic(synthetic_summary, output_dir)
    if link_trace:
        plot_link(link_trace, link_summary, output_dir)
    if scalability_rows:
        plot_scalability(scalability_rows, output_dir)


if __name__ == "__main__":
    main()
