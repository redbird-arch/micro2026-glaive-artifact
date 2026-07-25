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
from matplotlib.ticker import FixedLocator, FuncFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SPEC = importlib.util.spec_from_file_location("run_standard_studies_base", SCRIPT_DIR / "run_standard_studies.py")
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("Cannot load run_standard_studies.py")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

STUDY_KEY = "h100_clos_batch_16x32_sensitivity"
STANDARD_REPEAT_COUNT = 5
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
BAR_WIDTH = 0.76


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
        # return f"{self.topology_label}\nPrefill-BS{self.effective_batch_size}"
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
class SensitivityTask:
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
    policy_key: str
    policy_label: str
    hot_cap: int
    topology_json: str
    collective_json: str
    log_path: Path
    standard_log_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Speed hot-flow cap sensitivity experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", choices=["all", "run", "parse-only"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--standard-repeats", type=int, default=STANDARD_REPEAT_COUNT)
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


def threshold_policies(n: int) -> list[tuple[str, str, int]]:
    return [
        ("const2", "2", 2),
        ("const4", "4", 4),
        ("const8", "8", 8),
        ("n", "N", n),
        ("2n", "2N", 2 * n),
        ("4n", "4N", 4 * n),
        ("n2_over_4", "N^2/4", n * n // 4),
        ("n2_over_2", "N^2/2", n * n // 2),
    ]


def display_policy_label(policy_label: str) -> str:
    if policy_label == "N^2/4":
        return r"$N^{2}/4$"
    if policy_label == "N^2/2":
        return r"$N^{2}/2$"
    return policy_label


def collective_path(spec: TraceSpec) -> str:
    return f"evaluation_assets/collectives/sensitivity/{spec.npu_count}devices/{spec.trace_key}_block.json"


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


def tasks(repo_root: Path, standard_repeats: int = STANDARD_REPEAT_COUNT) -> list[SensitivityTask]:
    rows: list[SensitivityTask] = []
    for spec in TRACE_SPECS:
        collective_json = ensure_collective(repo_root, spec)
        total_gib, avg_gib_per_gpu, max_gib_per_gpu = input_volume_stats(repo_root, spec)
        for policy_key, policy_label, hot_cap in threshold_policies(spec.npu_count):
            stem = f"{spec.trace_key}_{policy_key}_cap{hot_cap}"
            rows.append(
                SensitivityTask(
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
                    policy_key=policy_key,
                    policy_label=policy_label,
                    hot_cap=hot_cap,
                    topology_json=spec.topology_json,
                    collective_json=collective_json,
                    log_path=repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "speed_sensitivity"
                    / STUDY_KEY
                    / f"{stem}.log",
                    standard_log_paths=tuple(
                        repo_root
                        / "evaluation_assets"
                        / "raw_logs"
                        / "speed_sensitivity_standard_compare"
                        / STUDY_KEY
                        / f"{stem}_rep{repeat}.log"
                        for repeat in range(max(1, standard_repeats))
                    ),
                )
            )
    return rows

def run_task(repo_root: Path, task: SensitivityTask, index: int, total: int, force: bool) -> None:
    if not force and base.task_is_complete(task.log_path) and all(base.task_is_complete(path) for path in task.standard_log_paths):
        print(f"[skip] {index}/{total} {task.trace_label} {task.policy_label} cap={task.hot_cap}")
        return
    print(f"[run] {index}/{total} {task.trace_label} {task.policy_label} cap={task.hot_cap}")
    command_base = [
        str(repo_root / "build" / "bin" / "tacos"),
        task.topology_json,
        task.collective_json,
        "--solver3",
    ]
    if force or not base.task_is_complete(task.log_path):
        task.log_path.parent.mkdir(parents=True, exist_ok=True)
        with task.log_path.open("w") as handle:
            handle.write(f"[Sensitivity Host] {os.uname().nodename}\n")
            handle.write(f"[Sensitivity Task] trace={task.trace_label},policy={task.policy_key},hot_cap={task.hot_cap},mode=speed\n")
            handle.flush()
            subprocess.run(
                [*command_base, "mode=speed", f"hot_cap={task.hot_cap}"],
                cwd=repo_root,
                env=runtime_env(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
    for repeat, standard_log_path in enumerate(task.standard_log_paths):
        if not force and base.task_is_complete(standard_log_path):
            continue
        standard_log_path.parent.mkdir(parents=True, exist_ok=True)
        with standard_log_path.open("w") as handle:
            handle.write(f"[Sensitivity Host] {os.uname().nodename}\n")
            handle.write(f"[Sensitivity Task] trace={task.trace_label},policy={task.policy_key},hot_cap={task.hot_cap},mode=standard,repeat={repeat}\n")
            handle.flush()
            subprocess.run(
                [*command_base, "mode=standard", f"hot_cap={task.hot_cap}"],
                cwd=repo_root,
                env=runtime_env(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

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


def parse_standard_samples(task: SensitivityTask, repo_root: Path) -> tuple[float, list[float], str, str]:
    samples: list[tuple[float, Path]] = []
    for path in task.standard_log_paths:
        metrics = base.parse_log_metrics(path)
        solver_time = metrics.get("solver_time_us")
        if solver_time is None:
            continue
        samples.append((float(solver_time), path))
    if not samples:
        return 0.0, [], "", ""
    fastest_solver, fastest_path = min(samples, key=lambda item: item[0])
    return (
        fastest_solver,
        [solver for solver, _path in samples],
        fastest_path.relative_to(repo_root).as_posix(),
        ";".join(path.relative_to(repo_root).as_posix() for _solver, path in samples),
    )


def parse_rows(repo_root: Path, standard_repeats: int = STANDARD_REPEAT_COUNT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for task in tasks(repo_root, standard_repeats):
        metrics = base.parse_log_metrics(task.log_path)
        if metrics.get("solver_time_us") is None:
            continue
        events = base.parse_speed_events(task.log_path)
        hot_only, cold_only, overlap, active = active_breakdown_fast(events)
        hot_start, hot_end, hot_busy = event_span(events, "hot")
        cold_start, cold_end, cold_busy = event_span(events, "cold")
        standard_solver, standard_samples, fastest_standard_log_path, standard_log_paths = parse_standard_samples(task, repo_root)
        speed_full_solver = float(metrics.get("speed_full_solver_time_us", metrics.get("solver_time_us", 0.0)) or 0.0)
        speed_hot_solver = float(metrics.get("speed_hot_solver_time_us", 0.0) or 0.0)
        speed_cold_solver = float(metrics.get("speed_cold_solver_time_us", 0.0) or 0.0)
        diagnostic_total = speed_hot_solver + speed_cold_solver
        if diagnostic_total > 0.0:
            scaled_hot_solver = standard_solver * speed_hot_solver / diagnostic_total
            scaled_cold_solver = standard_solver * speed_cold_solver / diagnostic_total
        else:
            scaled_hot_solver = 0.0
            scaled_cold_solver = standard_solver
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
                "policy_key": task.policy_key,
                "policy_label": task.policy_label,
                "requested_hot_cap": task.hot_cap,
                "threshold_bytes": metrics.get("speed_threshold_bytes", ""),
                "hot_flows": metrics.get("speed_hot_flows", ""),
                "cold_flows": metrics.get("speed_cold_flows", ""),
                "hot_bytes": metrics.get("speed_hot_bytes", ""),
                "cold_bytes": metrics.get("speed_cold_bytes", ""),
                "standard_solver_time_us": standard_solver,
                "full_solver_time_us": standard_solver,
                "solver_pct_of_makespan": (standard_solver / full_makespan * 100.0) if full_makespan else 0.0,
                "standard_solver_time_samples_us": ";".join(f"{value:.2f}" for value in standard_samples),
                "standard_repeat_count": len(standard_samples),
                "speed_full_solver_time_us": speed_full_solver,
                "speed_hot_solver_time_us": speed_hot_solver,
                "speed_cold_solver_time_us": speed_cold_solver,
                "scaled_hot_solver_time_us": scaled_hot_solver,
                "scaled_cold_solver_time_us": scaled_cold_solver,
                "hot_solver_time_us": scaled_hot_solver,
                "cold_solver_time_us": scaled_cold_solver,
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
                "standard_log_path": fastest_standard_log_path,
                "standard_log_paths": standard_log_paths,
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
                    "total_input_gib": task.total_input_gib,
                    "avg_input_gib_per_gpu": task.avg_input_gib_per_gpu,
                    "max_input_gib_per_gpu": task.max_input_gib_per_gpu,
                    "policy_key": task.policy_key,
                    "policy_label": task.policy_label,
                    "requested_hot_cap": task.hot_cap,
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
    policy_order = {key: idx for idx, (key, _, _) in enumerate(threshold_policies(16))}
    return sorted(rows, key=lambda r: (trace_order[int(r["trace_id"])], policy_order[str(r["policy_key"])]))


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
            x_labels.append(display_policy_label(str(row["policy_label"])))
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
    width = BAR_WIDTH
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
    if len(x):
        edge_pad = 0.2
        ax.set_xlim(float(x[0]) - width / 2 - edge_pad, float(x[-1]) + width / 2 + edge_pad)
    ax.set_ylabel(f"Collective Time ({unit})", fontsize=28)
    ax.yaxis.set_label_coords(-0.04, 0.43)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=17)
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == "4N":
            tick_label.set_color("#ef476f")
            tick_label.set_fontweight("bold")

    ax.tick_params(axis="y", labelsize=24)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", zorder=0, linewidth=3)
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
        bbox_to_anchor=(0.5, 1.15),
        fontsize=26,
        handletextpad=0.45,
        columnspacing=2,
    )
    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="#b8b8b8", linewidth=0.8, linestyle=":", alpha=0.8)
    for center, label in group_centers:
        ax.text(center, -0.15, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=23)
    ymax = float(np.max(total)) / divisor * 1.14 if len(total) else 1.0
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(bottom=0.34, top=0.84, left=0.085, right=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_solver_breakdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    apply_plot_style()
    rows = ordered_rows(rows)
    x, x_labels, group_centers, boundaries = grouped_positions(rows)
    hot = np.array([float(row["scaled_hot_solver_time_us"]) for row in rows])
    cold = np.array([float(row["scaled_cold_solver_time_us"]) for row in rows])
    total = hot + cold
    max_us = float(np.max(total)) if len(rows) else 1.0
    divisor, unit = choose_time_unit(max_us)
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_axisbelow(True)
    width = BAR_WIDTH
    bar_lw = 4
    ax.bar(
        x,
        hot / divisor,
        width=width,
        color=COMPONENT_STYLES["thrust"]["fill"],
        edgecolor=COMPONENT_STYLES["thrust"]["edge"],
        linewidth=bar_lw,
        label="Thrust",
        zorder=3,
    )
    ax.bar(
        x,
        cold / divisor,
        width=width,
        bottom=hot / divisor,
        color=COMPONENT_STYLES["sweep"]["fill"],
        edgecolor=COMPONENT_STYLES["sweep"]["edge"],
        linewidth=bar_lw,
        label="Sweep",
        zorder=3,
    )
    if len(x):
        edge_pad = 0.2
        ax.set_xlim(float(x[0]) - width / 2 - edge_pad, float(x[-1]) + width / 2 + edge_pad)
    ax.set_ylabel(f"Synthesis Time ({unit})", fontsize=28)
    ax.yaxis.set_label_coords(-0.04, 0.42)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=17)
    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() == "4N":
            tick_label.set_color("#ef476f")
            tick_label.set_fontweight("bold")
            
    ax.tick_params(axis="y", labelsize=24)
    ax.yaxis.set_major_locator(FixedLocator([0, 100, 200, 300]))
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", zorder=0, linewidth=3)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        fontsize=26,
        handletextpad=0.45,
        columnspacing=2,
    )
    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="#b8b8b8", linewidth=0.8, linestyle=":", alpha=0.8)
    for center, label in group_centers:
        ax.text(center, -0.15, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=23)
    ymax = max_us / divisor * 1.14 if len(total) else 1.0
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(bottom=0.34, top=0.84, left=0.085, right=0.995)
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
        makespans = [float(row["full_makespan_us"]) for row in trace_rows]
        solver_pcts = [float(row["solver_pct_of_makespan"]) for row in trace_rows]
        best = min(trace_rows, key=lambda row: float(row["full_makespan_us"]))
        npu_count = int(trace_rows[0]["npu_count"])
        four_cap = 4 * npu_count
        four = next(row for row in trace_rows if int(float(row["requested_hot_cap"])) == four_cap)
        min_makespan = float(best["full_makespan_us"])
        four_makespan = float(four["full_makespan_us"])
        summaries.append(
            {
                "trace_label": trace_label,
                "best_policy": best["policy_label"],
                "four_n_policy": four["policy_label"],
                "four_n_cap": four_cap,
                "best_makespan_us": min_makespan,
                "makespan_4n_us": four_makespan,
                "gap_4n_pct": (four_makespan - min_makespan) / min_makespan * 100.0 if min_makespan else 0.0,
                "spread_pct": (max(makespans) - min_makespan) / min_makespan * 100.0 if min_makespan else 0.0,
                "max_solver_pct": max(solver_pcts) if solver_pcts else 0.0,
                "solver_pct_4n": float(four["solver_pct_of_makespan"]),
            }
        )
    return summaries


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = ordered_rows(rows)
    lines = ["# Speed hot-flow cap sensitivity", ""]
    lines.append("This experiment uses two H100 Clos/FatTree scales: `Clos 8x2` on `h100_2node` and `Clos 8x4` on `h100_4node`. Figures and text use effective batch sizes: `Prefill-BS32` / `Prefill-BS64` for 16 GPUs and `Prefill-BS16` / `Prefill-BS32` for 32 GPUs. The byte threshold follows the bandwidth-latency formula, and the cap limits how many flows above that threshold enter the thrust set.")
    lines.append("")
    lines.append("Batch-size convention: `prefill_bs*` in the source CSV/manifest is the raw batch size of the 64-expert trace. With N GPUs, each GPU hosts `64/N` experts, so the displayed effective batch size is `raw_bs * 64 / N`. The four mappings are 16-GPU raw BS8 -> effective BS32, 16-GPU raw BS16 -> effective BS64, 32-GPU raw BS8 -> effective BS16, and 32-GPU raw BS16 -> effective BS32.")
    lines.append("")
    lines.append("Topology convention: the earlier `h100_2node_railonly.json` / `h100_4node_railonly.json` files use `topology=rail-optimized`, which partitions scale-out traffic into eight rails by local GPU index. The final experiment instead uses `h100_2node.json` / `h100_4node.json` with `topology=fat-tree` for the Clos/FatTree results.")
    lines.append("")
    lines.append("The H100 Clos prefill candidates were rescanned for a 4N makespan at or near the optimum, a visible makespan spread, and a short absolute 4N Standard synthesis time with limited growth over the smallest cap. The selected traces are raw `BS8-L3` and `BS16-L16` at 16 GPUs, and raw `BS8-L1` and `BS16-L16` at 32 GPUs. Across five final repeats, the 4N makespan gap is 0.0% for all four groups. Smaller caps can still rank higher in solver time because they have smaller search spaces; the intended result is near-optimal collective time at 4N with microsecond-scale synthesis time.")
    lines.append("")
    lines.append("Both scales use the semantic thresholds `2/4/8/N/2N/4N/N^2/4/N^2/2`. These expand to `2/4/8/16/32/64/64/128` at 16 GPUs and `2/4/8/32/64/128/256/512` at 32 GPUs. At 16 GPUs, `4N` and `N^2/4` both equal 64 but remain separate semantic x-axis points.")
    lines.append("")
    lines.append("The solver-time plot uses the full solve time from `mode=standard` as its total height. Each cap runs Standard five times on a compute node, and the fastest repeat represents the method's fast path. Thrust/Sweep segment proportions come from thrust-only and sweep-only diagnostic times in `mode=speed`; the stacked bar partitions the fastest Standard total time using those diagnostic proportions.")
    lines.append("")
    lines.append("The CSV retains `total_input_gib`, `avg_input_gib_per_gpu`, and `max_input_gib_per_gpu` for validation, but figures and conclusions identify traces by batch size.")
    lines.append("")
    lines.append("Output figures:")
    lines.append("")
    lines.append("- `evaluation_assets/plots/Speed_Sensitivity_Breakdown.pdf`")
    lines.append("- `evaluation_assets/plots/Speed_Sensitivity_Solver_Time.pdf`")
    lines.append("- `evaluation_assets/parsed/speed_sensitivity_summary.csv`")
    lines.append("- `evaluation_assets/parsed/speed_sensitivity_events.csv`")
    lines.append("## Trace-level result")
    lines.append("")
    lines.append("| trace | topology | raw BS | effective BS | best cap | 4N cap | 4N gap | makespan spread | max solver/makespan | 4N solver/makespan | 4N makespan | 4N Standard solve |")
    lines.append("| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for summary in trace_summary_rows(rows):
        first = next(row for row in rows if row["trace_label"] == summary["trace_label"])
        four = next(
            row
            for row in rows
            if row["trace_label"] == summary["trace_label"]
            and int(float(row["requested_hot_cap"])) == int(summary["four_n_cap"])
        )
        lines.append(
            f"| {summary['trace_label']} | {first['topology_key']} | {int(first['raw_batch_size'])} | "
            f"{int(first['effective_batch_size'])} | {summary['best_policy']} | {summary['four_n_policy']} | "
            f"{summary['gap_4n_pct']:.3f}% | {summary['spread_pct']:.3f}% | "
            f"{summary['max_solver_pct']:.2f}% | {summary['solver_pct_4n']:.2f}% | "
            f"{fmt_us(summary['makespan_4n_us'])} | {fmt_us(four['standard_solver_time_us'] or 0.0)} |"
        )
    lines.append("")
    lines.append("## Per-cap data")
    lines.append("")
    lines.append("| trace | topology | cap | requested cap | thrust flows | sweep flows | fastest Standard solve | solve/makespan | scaled thrust solve | scaled sweep solve | Speed full solve | full makespan | active | thrust-only | overlap | sweep-only |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            f"| {row['trace_label']} | {row['topology_key']} | {row['policy_label']} | {int(row['requested_hot_cap'])} | "
            f"{int(float(row['hot_flows']))} | {int(float(row['cold_flows']))} | "
            f"{fmt_us(row['standard_solver_time_us'] or 0.0)} | {float(row['solver_pct_of_makespan']):.2f}% | "
            f"{fmt_us(row['scaled_hot_solver_time_us'])} | {fmt_us(row['scaled_cold_solver_time_us'])} | "
            f"{fmt_us(row['speed_full_solver_time_us'])} | "
            f"{fmt_us(row['full_makespan_us'])} | {fmt_us(row['active_us'])} | {fmt_us(row['hot_only_active_us'])} | "
            f"{fmt_us(row['overlap_us'])} | {fmt_us(row['cold_only_active_us'])} |"
        )
    lines.append("")
    lines.append("The black line in the earlier solver-time plot represented a separate full solve in `mode=speed` with event capture enabled; it was not the sum of thrust-only and sweep-only diagnostics. The final plot removes this line to prevent diagnostic Speed overhead from being mistaken for the production Standard solve time.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_and_write(repo_root: Path, standard_repeats: int = STANDARD_REPEAT_COUNT) -> dict[str, int]:
    summary_rows, event_rows = parse_rows(repo_root, standard_repeats)
    parsed_dir = repo_root / "evaluation_assets" / "parsed"
    plot_dir = repo_root / "evaluation_assets" / "plots"
    write_csv(parsed_dir / "speed_sensitivity_summary.csv", ordered_rows(summary_rows))
    write_csv(parsed_dir / "speed_sensitivity_events.csv", event_rows)
    if summary_rows:
        plot_active_breakdown(
            summary_rows, plot_dir / "Speed_Sensitivity_Breakdown.pdf"
        )
        plot_solver_breakdown(
            summary_rows, plot_dir / "Speed_Sensitivity_Solver_Time.pdf"
        )
        write_markdown(parsed_dir / "speed_sensitivity.md", summary_rows)
    return {"summary_rows": len(summary_rows), "event_rows": len(event_rows)}


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    all_tasks = tasks(repo_root, args.standard_repeats)
    if args.stage in {"all", "run"}:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "task_count": len(all_tasks),
                    "max_workers": args.max_workers,
                    "standard_repeats": args.standard_repeats,
                },
                indent=2,
            )
        )
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
        result = parse_and_write(repo_root, args.standard_repeats)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
