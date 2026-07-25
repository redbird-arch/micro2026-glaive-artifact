#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = Path(
    os.environ.get(
        "GLAIVE_END2END_SUMMARY",
        REPO_ROOT / "results" / "realmodel_e2e4_analytical" / "summary.json",
    )
).resolve()
OUTPUT_DIR = Path(
    os.environ.get("GLAIVE_END2END_PLOT_DIR", REPO_ROOT / "Pictures")
).resolve()
OUTPUT_PDF = OUTPUT_DIR / "RealModel_Alltoallv_EndToEnd.pdf"

GROUP_ORDER = (
    ("prefill", "small", "Prefill BS=4"),
    ("prefill", "large", "Prefill BS=64"),
    ("decode", "small", "Decode BS=8"),
    ("decode", "large", "Decode BS=128"),
)
CLUSTER_ORDER = (
    ("deepseekv32", "tpuv7", "DS-TPUv7"),
    ("deepseekv32", "h100", "DS-H100"),
    ("qwen3_30b_a3b", "a100", "QW-A100"),
    ("qwen3_30b_a3b", "tpuv4", "QW-TPUv4"),
)
METHODS_BY_HARDWARE = {
    "tpuv7": (("mpi", "MPICH"), ("halfrdr", "HR+DR"), ("glaive", "Glaive")),
    "h100": (("mpi", "MPICH"), ("biring", "BiRing"), ("glaive", "Glaive")),
    "tpuv4": (("mpi", "MPICH"), ("halfrdr", "HR+DR"), ("glaive", "Glaive")),
    "a100": (("mpi", "MPICH"), ("biring", "BiRing"), ("glaive", "Glaive")),
}

COMP_FILL = "#80ed99"
COMP_EDGE = "#06d6a0"
MEM_FILL = "#4cc9f0"
MEM_EDGE = "#118ab2"
COMM_FILL = "#ff87ab"
COMM_EDGE = "#ff5d8f"
LINE_COLOR = "black"


def load_results() -> list[dict]:
    payload = json.loads(SUMMARY_PATH.read_text())
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"summary contains no results: {SUMMARY_PATH}")
    return results


def build_lookup(
    results: list[dict],
) -> dict[tuple[str, str, str, str, str], dict]:
    lookup: dict[tuple[str, str, str, str, str], dict] = {}
    for row in results:
        if row.get("status") not in {"ok", "cached"}:
            continue
        key = (
            str(row["model_key"]),
            str(row["config_key"]),
            str(row["hardware"]),
            str(row["phase"]),
            str(row["method"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate result row: {key}")
        lookup[key] = row
    return lookup


def required_keys() -> set[tuple[str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str]] = set()
    for phase, config_key, _ in GROUP_ORDER:
        for model_key, hardware, _ in CLUSTER_ORDER:
            for method, _ in METHODS_BY_HARDWARE[hardware]:
                keys.add((model_key, config_key, hardware, phase, method))
    return keys


def validate_lookup(
    lookup: dict[tuple[str, str, str, str, str], dict],
) -> None:
    missing = sorted(required_keys() - set(lookup))
    if missing:
        formatted = "\n".join("  " + "/".join(key) for key in missing)
        raise ValueError(f"summary is missing {len(missing)} required rows:\n{formatted}")


def metric_bundle(
    lookup: dict[tuple[str, str, str, str, str], dict],
    model_key: str,
    config_key: str,
    hardware: str,
    phase: str,
    method: str,
) -> dict[str, float]:
    row = lookup[(model_key, config_key, hardware, phase, method)]
    baseline_row = lookup[(model_key, config_key, hardware, phase, "mpi")]
    baseline_total = float(baseline_row["total_time_us"])
    baseline_comm = float(baseline_row["total_exposed_comm_us"])
    row_comm = float(row["total_exposed_comm_us"])
    if baseline_total <= 0.0 or baseline_comm <= 0.0 or row_comm <= 0.0:
        raise ValueError(
            f"non-positive timing for {model_key}/{config_key}/{hardware}/{phase}/{method}"
        )
    return {
        "compute_norm": float(row["total_compute_us"]) / baseline_total,
        "memory_norm": float(row.get("total_memory_us", 0.0)) / baseline_total,
        "comm_norm": row_comm / baseline_total,
        "comm_speedup": baseline_comm / row_comm,
    }


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def report_summary(
    lookup: dict[tuple[str, str, str, str, str], dict],
) -> None:
    comparisons = (
        ("mpi", "MPICH"),
        ("halfrdr", "HalfR+DR"),
        ("biring", "BiRing"),
    )
    for baseline_method, baseline_label in comparisons:
        pure_comm: list[float] = []
        end_to_end: list[float] = []
        for key, glaive_row in lookup.items():
            if key[-1] != "glaive":
                continue
            baseline_key = key[:-1] + (baseline_method,)
            if baseline_key not in lookup:
                continue
            baseline_row = lookup[baseline_key]
            pure_comm.append(
                float(baseline_row["total_exposed_comm_us"])
                / float(glaive_row["total_exposed_comm_us"])
            )
            end_to_end.append(
                float(baseline_row["total_time_us"])
                / float(glaive_row["total_time_us"])
            )
        print(
            f"Glaive vs {baseline_label}: "
            f"PureComm={geometric_mean(pure_comm):.3f}x, "
            f"End2End={geometric_mean(end_to_end):.3f}x, "
            f"n={len(pure_comm)}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lookup = build_lookup(load_results())
    validate_lookup(lookup)
    report_summary(lookup)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 21,
            "xtick.labelsize": 16,
            "ytick.labelsize": 17,
            "legend.fontsize": 20,
        }
    )
    # Match the final paper panel's 1711 x 344 pt canvas.
    fig, ax1 = plt.subplots(figsize=(23.764, 4.778))
    ax2 = ax1.twinx()

    bar_width = 0.11
    method_spacing = 0.015
    cluster_spacing = 0.12
    group_spacing = 0.28

    current_x = 0.0
    first_bar: float | None = None
    last_bar = 0.0
    max_total_norm = 0.0
    max_speedup = 0.0
    group_spans: list[tuple[float, float, str]] = []
    cluster_boundaries: list[float] = []
    group_boundaries: list[float] = []

    for group_index, (phase, config_key, group_label) in enumerate(GROUP_ORDER):
        group_start = current_x - method_spacing / 2.0
        for cluster_index, (model_key, hardware, cluster_label) in enumerate(
            CLUSTER_ORDER
        ):
            positions: list[float] = []
            speedups: list[float] = []

            for method_name, method_label in METHODS_BY_HARDWARE[hardware]:
                x = current_x
                first_bar = x if first_bar is None else first_bar
                last_bar = x
                positions.append(x)
                metrics = metric_bundle(
                    lookup,
                    model_key,
                    config_key,
                    hardware,
                    phase,
                    method_name,
                )
                compute_norm = metrics["compute_norm"]
                memory_norm = metrics["memory_norm"]
                comm_norm = metrics["comm_norm"]
                speedups.append(metrics["comm_speedup"])

                ax1.bar(
                    x,
                    compute_norm,
                    bar_width,
                    color=COMP_FILL,
                    edgecolor=COMP_EDGE,
                    linewidth=1.5,
                    zorder=3,
                )
                ax1.bar(
                    x,
                    memory_norm,
                    bar_width,
                    bottom=compute_norm,
                    color=MEM_FILL,
                    edgecolor=MEM_EDGE,
                    linewidth=1.5,
                    zorder=3,
                )
                ax1.bar(
                    x,
                    comm_norm,
                    bar_width,
                    bottom=compute_norm + memory_norm,
                    color=COMM_FILL,
                    edgecolor=COMM_EDGE,
                    linewidth=1.5,
                    zorder=3,
                )
                ax1.text(
                    x,
                    -0.035,
                    method_label,
                    transform=ax1.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    rotation=90,
                    fontsize=15,
                )
                current_x += bar_width + method_spacing
                max_total_norm = max(
                    max_total_norm, compute_norm + memory_norm + comm_norm
                )
                max_speedup = max(max_speedup, metrics["comm_speedup"])

            ax2.plot(
                positions,
                speedups,
                "o-",
                color=LINE_COLOR,
                zorder=5,
                linewidth=2.0,
                markersize=5,
            )
            ax1.text(
                float(np.mean(positions)),
                -0.33,
                cluster_label,
                transform=ax1.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=16,
            )

            if cluster_index < len(CLUSTER_ORDER) - 1:
                cluster_boundaries.append(current_x - method_spacing / 2.0)
                current_x += cluster_spacing

        group_end = last_bar + bar_width + method_spacing / 2.0
        group_spans.append((group_start, group_end, group_label))
        if group_index < len(GROUP_ORDER) - 1:
            group_boundaries.append(current_x - method_spacing / 2.0)
        current_x += group_spacing

    for boundary in cluster_boundaries:
        ax1.axvline(
            boundary,
            color="black",
            linestyle=(0, (2, 2)),
            linewidth=0.8,
            ymin=-0.35,
            clip_on=False,
            zorder=1,
        )
    for boundary in group_boundaries:
        ax1.axvline(
            boundary,
            color="black",
            linestyle=(0, (6, 4)),
            linewidth=1.0,
            ymin=-0.37,
            clip_on=False,
            zorder=1,
        )

    for group_start, group_end, group_label in group_spans:
        ax1.text(
            (group_start + group_end) / 2.0,
            -0.43,
            group_label,
            transform=ax1.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=21,
        )

    assert first_bar is not None
    ax1.set_xlim(first_bar - bar_width, last_bar + 2 * bar_width)
    ax1.set_ylim(0.0, 1.4)
    ax2.set_ylim(0.0, 14.0)
    ax1.set_yticks(np.arange(0.0, 1.401, 0.2))
    ax2.set_yticks(np.arange(0.0, 14.001, 2.0))
    ax1.set_ylabel("Norm. Time Breakdown")
    ax2.set_ylabel("Alltoallv Speedup")
    ax1.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        color="gray",
        alpha=0.9,
        zorder=0,
    )
    ax1.set_axisbelow(True)
    ax1.set_xticks([])

    if max_total_norm > 1.4 or max_speedup > 14.0:
        print(
            "WARNING: values exceed the paper axes: "
            f"max normalized time={max_total_norm:.3f}, "
            f"max speedup={max_speedup:.3f}"
        )

    handles = [
        Patch(
            facecolor=COMP_FILL,
            edgecolor=COMP_EDGE,
            linewidth=1.5,
            label="Computation",
        ),
        Patch(
            facecolor=MEM_FILL,
            edgecolor=MEM_EDGE,
            linewidth=1.5,
            label="Memory",
        ),
        Patch(
            facecolor=COMM_FILL,
            edgecolor=COMM_EDGE,
            linewidth=1.5,
            label="Communication",
        ),
        Line2D(
            [0],
            [0],
            color=LINE_COLOR,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label="Alltoallv Speedup",
        ),
    ]
    legend = ax1.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=2.2,
        columnspacing=2.1,
        borderpad=0.25,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.305, top=0.877)
    # Preserve the full paper canvas. Cropping the
    # canvas causes the three lower label tiers to overlap in the PDF.
    fig.savefig(OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
