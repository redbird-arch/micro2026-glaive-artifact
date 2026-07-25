#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SPEC = importlib.util.spec_from_file_location("run_standard_studies_base", SCRIPT_DIR / "run_standard_studies.py")
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("Cannot load run_standard_studies.py")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

STUDY_KEY = "h100_clos_path_weight_sensitivity"
RUNTIME_LIBRARY_PATHS = [
    "/hpc2ssd/Edatools/opensource/compile/gcc-12.2.0/lib64",
    "/hpc2ssd/Edatools/opensource/compile/mpfr/lib",
]
SOURCE_16GPU_PREFILL8_L3 = "input/generated/olmoe_h100_glaive_prefill8_remaining39_repick16_20260404/csv/16devices/prefill_bs8_layer3_0_16devices.csv"
SOURCE_16GPU_PREFILL16_L16 = "input/generated/olmoe_h100_glaive_prefill8_remaining39_repick16_20260404/csv/16devices/prefill_bs16_layer16_0_16devices.csv"
SOURCE_32GPU_PREFILL8_L1 = "input/generated/olmoe_h100_glaive_prefill8_remaining7_repick_20260404/csv/32devices/prefill_bs8_layer1_1_32devices.csv"
SOURCE_32GPU_PREFILL16_L16 = "input/generated/olmoe_h100_glaive_prefill8_remaining39_repick16_20260404/csv/32devices/prefill_bs16_layer16_0_32devices.csv"
COMPONENT_STYLES = {
    "thrust": {"fill": "#80ed99", "edge": "#06d6a0"},
    "overlap": {"fill": "#4cc9f0", "edge": "#118ab2"},
    "sweep": {"fill": "#ff87ab", "edge": "#ff5d8f"},
}
EXPERT_COUNT = 64


@dataclass(frozen=True)
class TraceSpec:
    trace_id: int
    trace_key: str
    trace_label: str
    topology_key: str
    topology_json: str
    npu_count: int
    source_csv: str
    size_label: str
    raw_batch_size: int
    block_bytes: int

    @property
    def effective_batch_size(self) -> int:
        return int(self.raw_batch_size * EXPERT_COUNT // self.npu_count)

    @property
    def topology_label(self) -> str:
        return f"Clos 8x{self.npu_count // 8}"

    @property
    def plot_label(self) -> str:
        return f"{self.topology_label} Prefill-BS{self.effective_batch_size}"


TRACE_SPECS = (
    TraceSpec(
        trace_id=0,
        trace_key="clos16_prefill_bs8_l3_repeat",
        trace_label="Clos8x2-Prefill-BS32",
        topology_key="h100_2node_clos",
        topology_json="input/topology/h100_2node.json",
        npu_count=16,
        source_csv=SOURCE_16GPU_PREFILL8_L3,
        size_label="Prefill-BS32",
        raw_batch_size=8,
        block_bytes=8192,
    ),
    TraceSpec(
        trace_id=1,
        trace_key="clos16_prefill_bs16_l16_repeat",
        trace_label="Clos8x2-Prefill-BS64",
        topology_key="h100_2node_clos",
        topology_json="input/topology/h100_2node.json",
        npu_count=16,
        source_csv=SOURCE_16GPU_PREFILL16_L16,
        size_label="Prefill-BS64",
        raw_batch_size=16,
        block_bytes=8192,
    ),
    TraceSpec(
        trace_id=2,
        trace_key="clos32_prefill_bs8_l1_repeat",
        trace_label="Clos8x4-Prefill-BS16",
        topology_key="h100_4node_clos",
        topology_json="input/topology/h100_4node.json",
        npu_count=32,
        source_csv=SOURCE_32GPU_PREFILL8_L1,
        size_label="Prefill-BS16",
        raw_batch_size=8,
        block_bytes=32768,
    ),
    TraceSpec(
        trace_id=3,
        trace_key="clos32_prefill_bs16_l16_repeat",
        trace_label="Clos8x4-Prefill-BS32",
        topology_key="h100_4node_clos",
        topology_json="input/topology/h100_4node.json",
        npu_count=32,
        source_csv=SOURCE_32GPU_PREFILL16_L16,
        size_label="Prefill-BS32",
        raw_batch_size=16,
        block_bytes=32768,
    ),
)


@dataclass(frozen=True)
class WeightSpec:
    weight_id: int
    weight_key: str
    weight_label: str
    sum_weight: float
    max_weight: float
    data_weight: float
    note: str

    @property
    def tuple_label(self) -> str:
        return f"{self.sum_weight:g}/{self.max_weight:g}/{self.data_weight:g}"


WEIGHT_SPECS = (
    WeightSpec(0, "alpha_only", "Alpha only", 1.0, 0.0, 0.0, "only NormSumLoad"),
    WeightSpec(1, "beta_only", "Beta only", 0.0, 1.0, 0.0, "only NormMaxLoad"),
    WeightSpec(2, "gamma_only", "Gamma only", 0.0, 0.0, 1.0, "only NormDataTransfer"),
    WeightSpec(3, "equal", "Equal", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, "equal weights"),
    WeightSpec(4, "current", "Current", 0.3, 0.2, 0.5, "current implementation"),
    WeightSpec(5, "alpha_heavy", "Alpha heavy", 0.8, 0.1, 0.1, "alpha-heavy"),
    WeightSpec(6, "beta_heavy", "Beta heavy", 0.1, 0.8, 0.1, "beta-heavy"),
    WeightSpec(7, "gamma_heavy", "Gamma heavy", 0.1, 0.1, 0.8, "gamma-heavy"),
)


@dataclass(frozen=True)
class WeightTask:
    trace_id: int
    trace_key: str
    size_label: str
    trace_label: str
    topology_key: str
    npu_count: int
    source_csv: str
    raw_batch_size: int
    effective_batch_size: int
    block_bytes: int
    total_input_gib: float
    avg_input_gib_per_gpu: float
    max_input_gib_per_gpu: float
    hot_cap: int
    topology_json: str
    collective_json: str
    weight_id: int
    weight_key: str
    weight_label: str
    weight_tuple_label: str
    weight_note: str
    path_weight_sum: float
    path_weight_max: float
    path_weight_data: float
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Speed path-score weight sensitivity experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", choices=["all", "run", "parse-only"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([*RUNTIME_LIBRARY_PATHS, existing]).rstrip(":")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def collective_path(spec: TraceSpec) -> str:
    return f"evaluation_assets/collectives/weight_sensitivity/{spec.npu_count}devices/{spec.trace_key}_block.json"


def ensure_collective(repo_root: Path, spec: TraceSpec) -> str:
    path = repo_root / collective_path(spec)
    write_json(
        path,
        {
            "collective": "alltoallv",
            "v_datasize": spec.source_csv,
            "block_bytes": spec.block_bytes,
            "chunkfactor": 1,
        },
    )
    return path.relative_to(repo_root).as_posix()


def input_volume_stats(repo_root: Path, spec: TraceSpec) -> tuple[float, float, float]:
    row_totals: list[int] = []
    with (repo_root / spec.source_csv).open(newline="") as handle:
        for row in csv.reader(handle):
            row_totals.append(sum(int(value) for value in row))
    total_bytes = sum(row_totals) * spec.block_bytes
    max_gpu_bytes = max(row_totals, default=0) * spec.block_bytes
    total_gib = total_bytes / (1024.0**3)
    avg_gib_per_gpu = total_gib / spec.npu_count if spec.npu_count else 0.0
    max_gib_per_gpu = max_gpu_bytes / (1024.0**3)
    return total_gib, avg_gib_per_gpu, max_gib_per_gpu


def tasks(repo_root: Path) -> list[WeightTask]:
    rows: list[WeightTask] = []
    for spec in TRACE_SPECS:
        collective_json = ensure_collective(repo_root, spec)
        total_gib, avg_gib_per_gpu, max_gib_per_gpu = input_volume_stats(repo_root, spec)
        hot_cap = 4 * spec.npu_count
        for weight in WEIGHT_SPECS:
            stem = f"{spec.trace_key}_{weight.weight_key}_cap{hot_cap}"
            rows.append(
                WeightTask(
                    trace_id=spec.trace_id,
                    trace_key=spec.trace_key,
                    size_label=spec.size_label,
                    trace_label=spec.trace_label,
                    topology_key=spec.topology_key,
                    npu_count=spec.npu_count,
                    source_csv=spec.source_csv,
                    raw_batch_size=spec.raw_batch_size,
                    effective_batch_size=spec.effective_batch_size,
                    block_bytes=spec.block_bytes,
                    total_input_gib=total_gib,
                    avg_input_gib_per_gpu=avg_gib_per_gpu,
                    max_input_gib_per_gpu=max_gib_per_gpu,
                    hot_cap=hot_cap,
                    topology_json=spec.topology_json,
                    collective_json=collective_json,
                    weight_id=weight.weight_id,
                    weight_key=weight.weight_key,
                    weight_label=weight.weight_label,
                    weight_tuple_label=weight.tuple_label,
                    weight_note=weight.note,
                    path_weight_sum=weight.sum_weight,
                    path_weight_max=weight.max_weight,
                    path_weight_data=weight.data_weight,
                    log_path=repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "speed_weight_sensitivity"
                    / STUDY_KEY
                    / f"{stem}.log",
                )
            )
    return rows


def run_task(repo_root: Path, task: WeightTask, index: int, total: int, force: bool) -> None:
    if not force and base.task_is_complete(task.log_path):
        print(
            f"[skip] {index}/{total} {task.trace_label} {task.weight_label} "
            f"({task.weight_tuple_label})"
        )
        return
    print(
        f"[run] {index}/{total} {task.trace_label} {task.weight_label} "
        f"({task.weight_tuple_label})"
    )
    command = [
        str(repo_root / "build" / "bin" / "tacos"),
        task.topology_json,
        task.collective_json,
        "--solver3",
        "mode=speed",
        f"hot_cap={task.hot_cap}",
        f"path_weight_sum={task.path_weight_sum:.12g}",
        f"path_weight_max={task.path_weight_max:.12g}",
        f"path_weight_data={task.path_weight_data:.12g}",
    ]
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    with task.log_path.open("w") as handle:
        handle.write(f"[WeightSensitivity Host] {os.uname().nodename}\n")
        handle.write(
            "[WeightSensitivity Task] "
            f"trace={task.trace_label},hot_cap={task.hot_cap},"
            f"weights=sum:{task.path_weight_sum:.12g};max:{task.path_weight_max:.12g};"
            f"data:{task.path_weight_data:.12g},mode=speed\n"
        )
        handle.flush()
        subprocess.run(command, cwd=repo_root, env=runtime_env(), stdout=handle, stderr=subprocess.STDOUT, check=True)


def event_span(events: list[dict[str, Any]], cls: str) -> tuple[float, float, float]:
    selected = [event for event in events if event["class"] == cls]
    if not selected:
        return 0.0, 0.0, 0.0
    start = min(float(event["start_us"]) for event in selected)
    end = max(float(event["end_us"]) for event in selected)
    busy = sum(float(event["end_us"]) - float(event["start_us"]) for event in selected)
    return start, end, busy


def active_breakdown_fast(events: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    deltas: dict[float, list[int]] = {}
    for event in events:
        start = round(float(event["start_us"]), 9)
        end = round(float(event["end_us"]), 9)
        if end <= start:
            continue
        if start not in deltas:
            deltas[start] = [0, 0]
        if end not in deltas:
            deltas[end] = [0, 0]
        index = 0 if event["class"] == "hot" else 1
        deltas[start][index] += 1
        deltas[end][index] -= 1
    hot_count = 0
    cold_count = 0
    hot_only = cold_only = overlap = active = 0.0
    last_time: float | None = None
    for time in sorted(deltas):
        if last_time is not None and time > last_time:
            duration = time - last_time
            has_hot = hot_count > 0
            has_cold = cold_count > 0
            if has_hot or has_cold:
                active += duration
            if has_hot and has_cold:
                overlap += duration
            elif has_hot:
                hot_only += duration
            elif has_cold:
                cold_only += duration
        hot_count += deltas[time][0]
        cold_count += deltas[time][1]
        last_time = time
    return hot_only, cold_only, overlap, active


def parse_rows(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for task in tasks(repo_root):
        metrics = base.parse_log_metrics(task.log_path)
        if metrics.get("solver_time_us") is None:
            continue
        events = base.parse_speed_events(task.log_path)
        hot_only, cold_only, overlap, active = active_breakdown_fast(events)
        hot_start, hot_end, hot_busy = event_span(events, "hot")
        cold_start, cold_end, cold_busy = event_span(events, "cold")
        full_makespan = float(metrics.get("speed_full_makespan_us", metrics.get("total_makespan_us", 0.0)) or 0.0)
        summary_rows.append(
            {
                "study_key": STUDY_KEY,
                "topology_key": task.topology_key,
                "npu_count": task.npu_count,
                "trace_id": task.trace_id,
                "trace_key": task.trace_key,
                "trace_label": task.trace_label,
                "size_label": task.size_label,
                "source_csv": task.source_csv,
                "raw_batch_size": task.raw_batch_size,
                "effective_batch_size": task.effective_batch_size,
                "block_bytes": task.block_bytes,
                "total_input_gib": task.total_input_gib,
                "avg_input_gib_per_gpu": task.avg_input_gib_per_gpu,
                "max_input_gib_per_gpu": task.max_input_gib_per_gpu,
                "collective_json": task.collective_json,
                "requested_hot_cap": task.hot_cap,
                "weight_id": task.weight_id,
                "weight_key": task.weight_key,
                "weight_label": task.weight_label,
                "weight_tuple_label": task.weight_tuple_label,
                "weight_note": task.weight_note,
                "path_weight_sum": task.path_weight_sum,
                "path_weight_max": task.path_weight_max,
                "path_weight_data": task.path_weight_data,
                "threshold_bytes": metrics.get("speed_threshold_bytes", ""),
                "hot_flows": metrics.get("speed_hot_flows", ""),
                "cold_flows": metrics.get("speed_cold_flows", ""),
                "hot_bytes": metrics.get("speed_hot_bytes", ""),
                "cold_bytes": metrics.get("speed_cold_bytes", ""),
                "full_solver_time_us": metrics.get("solver_time_us", 0.0),
                "full_makespan_us": full_makespan,
                "hot_only_makespan_us": metrics.get("speed_hot_only_makespan_us", 0.0),
                "cold_only_makespan_us": metrics.get("speed_cold_only_makespan_us", 0.0),
                "hot_only_active_us": hot_only,
                "overlap_us": overlap,
                "cold_only_active_us": cold_only,
                "active_us": active,
                "hot_span_start_us": hot_start,
                "hot_span_end_us": hot_end,
                "cold_span_start_us": cold_start,
                "cold_span_end_us": cold_end,
                "hot_event_busy_us": hot_busy,
                "cold_event_busy_us": cold_busy,
                "demand_fingerprint": metrics.get("demand_fingerprint", ""),
                "log_path": task.log_path.relative_to(repo_root).as_posix(),
            }
        )
        for event in events:
            event_rows.append(
                {
                    "study_key": STUDY_KEY,
                    "topology_key": task.topology_key,
                    "npu_count": task.npu_count,
                    "trace_id": task.trace_id,
                    "trace_key": task.trace_key,
                    "trace_label": task.trace_label,
                    "size_label": task.size_label,
                    "source_csv": task.source_csv,
                    "raw_batch_size": task.raw_batch_size,
                    "effective_batch_size": task.effective_batch_size,
                    "block_bytes": task.block_bytes,
                    "requested_hot_cap": task.hot_cap,
                    "weight_id": task.weight_id,
                    "weight_key": task.weight_key,
                    "weight_label": task.weight_label,
                    "weight_tuple_label": task.weight_tuple_label,
                    "path_weight_sum": task.path_weight_sum,
                    "path_weight_max": task.path_weight_max,
                    "path_weight_data": task.path_weight_data,
                    "class": event["class"],
                    "link_src": event["link_src"],
                    "link_dst": event["link_dst"],
                    "flow_src": event["flow_src"],
                    "flow_dst": event["flow_dst"],
                    "chunk_id": event["chunk_id"],
                    "bytes": event["bytes"],
                    "start_us": event["start_us"],
                    "end_us": event["end_us"],
                    "path": event.get("path", ""),
                }
            )
    return summary_rows, event_rows


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_order = {spec.trace_id: idx for idx, spec in enumerate(TRACE_SPECS)}
    weight_order = {spec.weight_id: idx for idx, spec in enumerate(WEIGHT_SPECS)}
    return sorted(rows, key=lambda r: (trace_order[int(r["trace_id"])], weight_order[int(r["weight_id"])]))


def plot_weight_label(weight_label: str) -> str:
    return {
        "Alpha only": "W1",
        "Beta only": "W2",
        "Gamma only": "W3",
        "Equal": "W4",
        "Current": "W5",
        "Alpha heavy": "W6",
        "Beta heavy": "W7",
        "Gamma heavy": "W8",
    }.get(weight_label, weight_label)


def grouped_positions(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[tuple[float, str]], list[float]]:
    x_positions: list[float] = []
    x_labels: list[str] = []
    group_centers: list[tuple[float, str]] = []
    boundaries: list[float] = []
    position = 0.0
    group_gap = 0.85
    by_trace: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_trace.setdefault(int(row["trace_id"]), []).append(row)
    for spec in TRACE_SPECS:
        trace_rows = by_trace.get(spec.trace_id, [])
        if not trace_rows:
            continue
        start = position
        for row in trace_rows:
            x_positions.append(position)
            x_labels.append(plot_weight_label(str(row["weight_label"])))
            position += 1.0
        end = position - 1.0
        group_centers.append(((start + end) / 2.0, spec.plot_label))
        boundaries.append(position - 0.5 + group_gap / 2.0)
        position += group_gap
    return np.array(x_positions), x_labels, group_centers, boundaries


def choose_time_unit(max_us: float) -> tuple[float, str]:
    if max_us >= 1000.0:
        return 1000.0, "ms"
    return 1.0, "us"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_active_breakdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    apply_plot_style()
    rows = ordered_rows(rows)
    x, x_labels, group_centers, boundaries = grouped_positions(rows)
    hot = np.array([float(row["hot_only_active_us"]) for row in rows])
    overlap = np.array([float(row["overlap_us"]) for row in rows])
    cold = np.array([float(row["cold_only_active_us"]) for row in rows])
    total = hot + overlap + cold
    divisor, unit = choose_time_unit(float(np.max(total)) if len(total) else 1.0)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_axisbelow(True)
    width = 0.76
    bar_lw = 4
    ax.bar(
        x,
        hot / divisor,
        width=width,
        color=COMPONENT_STYLES["thrust"]["fill"],
        edgecolor=COMPONENT_STYLES["thrust"]["edge"],
        linewidth=bar_lw,
        zorder=3,
    )
    ax.bar(
        x,
        overlap / divisor,
        width=width,
        bottom=hot / divisor,
        color=COMPONENT_STYLES["overlap"]["fill"],
        edgecolor=COMPONENT_STYLES["overlap"]["edge"],
        linewidth=bar_lw,
        zorder=3,
    )
    cold_mask = cold > 0
    if np.any(cold_mask):
        ax.bar(
            x[cold_mask],
            cold[cold_mask] / divisor,
            width=width,
            bottom=((hot + overlap) / divisor)[cold_mask],
            color=COMPONENT_STYLES["sweep"]["fill"],
            edgecolor=COMPONENT_STYLES["sweep"]["edge"],
            linewidth=bar_lw,
            zorder=3,
        )
    ax.set_ylabel(f"Collective Time ({unit})", fontsize=28)
    ax.yaxis.set_label_coords(-0.04, 0.43)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=20)
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == "W5":
            tick_label.set_color("#ef476f")
            tick_label.set_fontweight("bold")
    ax.tick_params(axis="y", labelsize=24)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", zorder=0, linewidth=3)
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100, 150, 200]))
    handles = [
        Patch(
            facecolor=COMPONENT_STYLES["thrust"]["fill"],
            edgecolor=COMPONENT_STYLES["thrust"]["edge"],
            linewidth=bar_lw,
            label="Thrust-only",
        ),
        Patch(
            facecolor=COMPONENT_STYLES["overlap"]["fill"],
            edgecolor=COMPONENT_STYLES["overlap"]["edge"],
            linewidth=bar_lw,
            label="Overlap",
        ),
        Patch(
            facecolor=COMPONENT_STYLES["sweep"]["fill"],
            edgecolor=COMPONENT_STYLES["sweep"]["edge"],
            linewidth=bar_lw,
            label="Sweep-only",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fontsize=26,
        handletextpad=0.45,
        columnspacing=2.5,
    )
    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="#b8b8b8", linewidth=0.8, linestyle=":", alpha=0.8)
    for center, label in group_centers:
        ax.text(center, -0.15, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=23)
    ymax = float(np.max(total)) / divisor * 1.14 if len(total) else 1.0
    ax.set_ylim(0, ymax)
    if len(x):
        edge_pad = 0.2
        ax.set_xlim(float(x[0]) - width / 2 - edge_pad, float(x[-1]) + width / 2 + edge_pad)
    fig.subplots_adjust(top=0.84, bottom=0.34, left=0.085, right=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def fmt_us(value: Any) -> str:
    value = float(value)
    if value >= 1000:
        return f"{value / 1000:.3f} ms"
    return f"{value:.2f} us"


def trace_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trace.setdefault(str(row["trace_label"]), []).append(row)
    for trace_label, trace_rows in by_trace.items():
        active_values = [float(row["active_us"]) for row in trace_rows]
        best = min(trace_rows, key=lambda row: float(row["active_us"]))
        current = next(row for row in trace_rows if row["weight_key"] == "current")
        min_active = float(best["active_us"])
        current_active = float(current["active_us"])
        summaries.append(
            {
                "trace_label": trace_label,
                "best_weight": best["weight_label"],
                "best_tuple": best["weight_tuple_label"],
                "current_weight": current["weight_label"],
                "current_tuple": current["weight_tuple_label"],
                "best_active_us": min_active,
                "current_active_us": current_active,
                "current_gap_pct": (current_active - min_active) / min_active * 100.0 if min_active else 0.0,
                "spread_pct": (max(active_values) - min_active) / min_active * 100.0 if min_active else 0.0,
            }
        )
    return summaries


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = ordered_rows(rows)
    lines = ["# Speed path-score weight sensitivity", ""]
    lines.append("This experiment fixes the hotspot threshold at `4N` and varies only the three HotSpot/Thrust path-scoring weights from Algorithm 2. In paper-formula order, `alpha = path_weight_sum`, `beta = path_weight_max`, and `gamma = path_weight_data`.")
    lines.append("")
    lines.append("The implementation defaults are `alpha=0.3, beta=0.2, gamma=0.5`, equivalently `DataTransfer=0.5, SumLoad=0.3, MaxLoad=0.2`. Algorithm 2 in `Glaive_submitted.pdf` gives the symbolic formula but does not hard-code these values in the paper text.")
    lines.append("")
    lines.append("The experiment reuses the H100 Clos/FatTree topologies and real traces selected for threshold sensitivity: `Prefill-BS32/BS64` on 16-GPU `Clos 8x2` and `Prefill-BS16/BS32` on 32-GPU `Clos 8x4`. It tests whether collective time remains stable across alpha/beta/gamma settings. The three breakdown segments are Thrust-only, Overlap, and Sweep-only, using the same principal style as `Speed_Torus_Breakdown`.")
    lines.append("")
    lines.append("Weights are recorded as `(alpha, beta, gamma) = (NormSumLoad, NormMaxLoad, NormDataTransfer)`. The x-axis uses `W1` through `W8` to avoid repeating long tuples across 32 bars; red `W5` is the implementation default.")
    lines.append("")
    lines.append("| tick | label | alpha | beta | gamma | note |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for weight in WEIGHT_SPECS:
        tick = plot_weight_label(weight.weight_label)
        lines.append(
            f"| {tick} | {weight.weight_label} | {weight.sum_weight:.6g} | {weight.max_weight:.6g} | "
            f"{weight.data_weight:.6g} | {weight.note} |"
        )
    lines.append("")
    lines.append("Output figures:")
    lines.append("")
    lines.append("- `evaluation_assets/plots/Speed_Weight_Sensitivity_Breakdown.pdf`")
    lines.append("- `evaluation_assets/parsed/speed_weight_sensitivity_summary.csv`")
    lines.append("- `evaluation_assets/parsed/speed_weight_sensitivity_events.csv`")
    lines.append("")
    lines.append("## Trace-level result")
    lines.append("")
    lines.append("| trace | topology | raw BS | effective BS | best weight | current gap | spread | current collective time | best collective time |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for summary in trace_summary_rows(rows):
        first = next(row for row in rows if row["trace_label"] == summary["trace_label"])
        spec = next(spec for spec in TRACE_SPECS if spec.trace_id == int(first["trace_id"]))
        lines.append(
            "| "
            f"{summary['trace_label']} | "
            f"{spec.topology_label} | "
            f"{int(first['raw_batch_size'])} | "
            f"{int(first['effective_batch_size'])} | "
            f"{summary['best_weight']} ({summary['best_tuple']}) | "
            f"{summary['current_gap_pct']:.3f}% | "
            f"{summary['spread_pct']:.3f}% | "
            f"{fmt_us(summary['current_active_us'])} | "
            f"{fmt_us(summary['best_active_us'])} |"
        )
    lines.append("")
    lines.append("## Raw rows")
    lines.append("")
    lines.append("| trace | weight | alpha | beta | gamma | hot flows | cold flows | Thrust-only | Overlap | Sweep-only | collective time | full makespan |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            f"{row['trace_label']} | "
            f"{row['weight_label']} | "
            f"{float(row['path_weight_sum']):.6g} | "
            f"{float(row['path_weight_max']):.6g} | "
            f"{float(row['path_weight_data']):.6g} | "
            f"{int(float(row.get('hot_flows') or 0))} | "
            f"{int(float(row.get('cold_flows') or 0))} | "
            f"{fmt_us(row['hot_only_active_us'])} | "
            f"{fmt_us(row['overlap_us'])} | "
            f"{fmt_us(row['cold_only_active_us'])} | "
            f"{fmt_us(row['active_us'])} | "
            f"{fmt_us(row['full_makespan_us'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    all_tasks = tasks(repo_root)
    if args.stage in {"all", "run"}:
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            futures = {
                executor.submit(
                    run_task,
                    repo_root,
                    task,
                    index,
                    len(all_tasks),
                    args.force,
                ): task.log_path.name
                for index, task in enumerate(all_tasks, 1)
            }
            for future in as_completed(futures):
                future.result()
    if args.stage in {"all", "parse-only"}:
        rows, event_rows = parse_rows(repo_root)
        rows = ordered_rows(rows)
        parsed_dir = repo_root / "evaluation_assets" / "parsed"
        plots_dir = repo_root / "evaluation_assets" / "plots"
        write_csv(parsed_dir / "speed_weight_sensitivity_summary.csv", rows)
        write_csv(parsed_dir / "speed_weight_sensitivity_events.csv", event_rows)
        write_markdown(parsed_dir / "speed_weight_sensitivity.md", rows)
        plot_active_breakdown(rows, plots_dir / "Speed_Weight_Sensitivity_Breakdown.pdf")
        print(f"[done] parsed {len(rows)} rows and {len(event_rows)} events")
        for summary in trace_summary_rows(rows):
            print(
                "[summary] "
                f"{summary['trace_label']}: best={summary['best_weight']} ({summary['best_tuple']}), "
                f"current_gap={summary['current_gap_pct']:.3f}%, spread={summary['spread_pct']:.3f}%"
            )


if __name__ == "__main__":
    main()
