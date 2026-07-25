#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_evaluation import (
    SCALABILITY_COMBINED_MAX_DEVICES,
    SCALABILITY_COMBINED_XLIM,
    SCALABILITY_COMBINED_XTICKS,
    annotated_subplot_title,
    apply_style as apply_paper_style,
    plot_scalability_axis,
)

SCALABILITY_THEORY = {
    "mesh": lambda x: np.power(x, 2.5) * np.log2(np.maximum(x, 2.0)),
    "torus": lambda x: np.power(x, 7.0 / 3.0) * np.log2(np.maximum(x, 2.0)),
    "fullmesh": lambda x: np.power(x, 2.0) * np.log2(np.maximum(x, 2.0)),
    "fat-tree": lambda x: np.power(x, 3.0) * np.log2(np.maximum(x, 2.0)),
    "cm384": lambda x: np.power(x, 3.0) * np.log2(np.maximum(x, 2.0)),
}
SCALABILITY_LABELS = {
    "fullmesh": "2D FullMesh",
    "torus": "3D Torus",
    "fat-tree": "2D Clos",
    "cm384": "3D CM384",
    "mesh": "2D Mesh",
}
SCALABILITY_COLORS = {
    "fullmesh": "#f77f00",
    "torus": "#118ab2",
    "fat-tree": "#7b2cbf",
    "cm384": "#ff5d8f",
    "mesh": "#06d6a0",
}
SCALABILITY_COMBINED_ORDER = ["fullmesh", "torus", "fat-tree", "cm384"]
SCALABILITY_MAX_DEVICES = 4096


@dataclass(frozen=True)
class Task:
    name: str
    topology_json: str
    collective_json: str
    args: tuple[str, ...]
    log_path: Path
    stage: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Standard scalability and Speed studies.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--stage",
        choices=["all", "fairness", "scalability", "speed", "breakeven", "parse-only"],
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--limit-scalability", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def runtime_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def task_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = path.read_text(errors="replace")
    return "[TACOS Solver] Done!" in text or "[TACOS] Done!" in text or "[TACOS Baseline] Done!" in text


def run_local_task(repo_root: Path, task: Task) -> None:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(repo_root / "build" / "bin" / "tacos"),
        task.topology_json,
        task.collective_json,
        *task.args,
    ]
    with task.log_path.open("w") as handle:
        subprocess.run(command, cwd=repo_root, env=runtime_env(repo_root), stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_task(repo_root: Path, task: Task, force: bool) -> str:
    if not force and task_is_complete(task.log_path):
        return f"[skip] {task.name}"
    run_local_task(repo_root, task)
    return f"[done] {task.name}"


def run_tasks(
    repo_root: Path,
    tasks: list[Task],
    force: bool,
    max_workers: int,
) -> None:
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(run_task, repo_root, task, force): task.name
            for task in tasks
        }
        for future in as_completed(futures):
            print(future.result())


def parse_log_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace") if path.exists() else ""
    def first_float(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    def first_int(pattern: str) -> int | None:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    summary: dict[str, object] = {
        "solver_time_us": first_float(r"Solver Time:\s*([0-9.]+)\s*us"),
        "total_makespan_us": first_float(r"Total Makespan:\s*([0-9.]+)\s*us"),
        "collective_time_us": first_float(r"Collective Time:\s*([0-9.]+)\s*us"),
        "time_to_solve_us": first_float(r"Time to solve:\s*([0-9.]+)\s*us"),
        "demand_rows": first_int(r"Demand Rows:\s*([0-9]+)"),
        "demand_cols": first_int(r"Demand Cols:\s*([0-9]+)"),
        "demand_nonzeros": first_int(r"Demand Nonzeros:\s*([0-9]+)"),
        "demand_total_bytes": first_int(r"Demand Total Bytes:\s*([0-9]+)"),
        "demand_fingerprint": "",
    }
    fingerprint = re.search(r"Demand Fingerprint:\s*(0x[0-9a-fA-F]+)", text)
    if fingerprint:
        summary["demand_fingerprint"] = fingerprint.group(1).lower()

    speed = re.search(r"^\[Speed Summary CSV\]\s*(.+)$", text, re.MULTILINE)
    if speed:
        for item in speed.group(1).split(","):
            key, value = item.split("=", 1)
            try:
                summary[f"speed_{key}"] = float(value)
            except ValueError:
                summary[f"speed_{key}"] = value
    return summary


def parse_speed_events(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"^\[Speed Event\]\s+class=(?P<class>\w+)\s+"
        r"link_src=(?P<link_src>-?\d+)\s+link_dst=(?P<link_dst>-?\d+)\s+"
        r"flow_src=(?P<flow_src>-?\d+)\s+flow_dst=(?P<flow_dst>-?\d+)\s+"
        r"chunk_id=(?P<chunk_id>-?\d+)\s+bytes=(?P<bytes>\d+)\s+"
        r"start_us=(?P<start_us>[0-9.eE+-]+)\s+end_us=(?P<end_us>[0-9.eE+-]+)\s+path=(?P<path>[0-9>.-]*)"
    )
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        item = match.groupdict()
        rows.append(
            {
                "class": item["class"],
                "link_src": int(item["link_src"]),
                "link_dst": int(item["link_dst"]),
                "flow_src": int(item["flow_src"]),
                "flow_dst": int(item["flow_dst"]),
                "chunk_id": int(item["chunk_id"]),
                "bytes": int(item["bytes"]),
                "start_us": float(item["start_us"]),
                "end_us": float(item["end_us"]),
                "path": item["path"],
            }
        )
    return rows


def fairness_tasks(repo_root: Path) -> list[Task]:
    base = repo_root / "evaluation_assets" / "raw_logs" / "standard_fairness"
    cases = [
        (
            "cm384_1node_16",
            "input/topology/cm384_1node_16.json",
            "input/collective/alltoallv512_16devices.json",
            ("original", "clean", "standard"),
        ),
        (
            "cm384_2node_32",
            "input/topology/cm384_2node.json",
            "input/collective/alltoallv512_32devices.json",
            ("original", "clean", "standard"),
        ),
        (
            "fattree_8x4_decode_tiled",
            "evaluation_assets/topologies/scalability/fattree_8x4.json",
            "evaluation_assets/collectives/scalability/fattree_8x4_decode_bs64_layer8_56_tiled.json",
            ("clean", "standard"),
        ),
    ]
    modes = {
        "original": ("--solver", "mode=complete"),
        "clean": ("--solver", "mode=clean"),
        "standard": ("--solver", "mode=standard"),
    }
    tasks: list[Task] = []
    for case_name, topology_json, collective_json, case_modes in cases:
        for mode in case_modes:
            args = modes[mode]
            tasks.append(
                Task(
                    name=f"fairness::{case_name}::{mode}",
                    topology_json=topology_json,
                    collective_json=collective_json,
                    args=args,
                    log_path=base / case_name / f"{mode}.log",
                    stage="standard_fairness",
                )
            )
    return tasks


def build_fairness_rows(repo_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for task in fairness_tasks(repo_root):
        metrics = parse_log_metrics(task.log_path)
        case_name = task.log_path.parent.name
        mode = task.log_path.stem
        row = {
            "case": case_name,
            "mode": mode,
            "demand_rows": metrics.get("demand_rows"),
            "demand_cols": metrics.get("demand_cols"),
            "demand_nonzeros": metrics.get("demand_nonzeros"),
            "demand_total_bytes": metrics.get("demand_total_bytes"),
            "demand_fingerprint": metrics.get("demand_fingerprint", ""),
            "solver_time_us": metrics.get("solver_time_us") or metrics.get("time_to_solve_us"),
            "makespan_us": metrics.get("total_makespan_us") or metrics.get("collective_time_us"),
            "log_path": task.log_path.relative_to(repo_root).as_posix(),
        }
        by_case[case_name].append(row)
        rows.append(row)

    for case_rows in by_case.values():
        fingerprints = {row["demand_fingerprint"] for row in case_rows if row["demand_fingerprint"]}
        fair = len(fingerprints) == 1 and all(row["demand_fingerprint"] for row in case_rows)
        for row in case_rows:
            row["fair_same_demand"] = fair
    return rows


def scalability_tasks(repo_root: Path, limit: int = 0) -> list[Task]:
    manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "scalability_cases.json")
    tasks: list[Task] = []
    for case in manifest["cases"]:
        if str(case.get("topology_type")) == "mesh" or int(case.get("npus_count", 0)) > SCALABILITY_MAX_DEVICES:
            continue
        log_stem = case.get("log_stem", Path(str(case["topology_json"])).stem)
        tasks.append(
            Task(
                name=f"standard_scalability::{case['topology_type']}::{log_stem}",
                topology_json=str(case["topology_json"]),
                collective_json=str(case["collective_json"]),
                args=("--solver", "mode=standard"),
                log_path=repo_root
                / "evaluation_assets"
                / "raw_logs"
                / "standard_scalability"
                / str(case["topology_type"])
                / f"{log_stem}.log",
                stage="standard_scalability",
            )
        )
    return tasks[:limit] if limit > 0 else tasks


def build_scalability_rows(repo_root: Path) -> list[dict[str, object]]:
    manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "scalability_cases.json")
    rows: list[dict[str, object]] = []
    for case in manifest["cases"]:
        if str(case.get("topology_type")) == "mesh" or int(case.get("npus_count", 0)) > SCALABILITY_MAX_DEVICES:
            continue
        log_stem = case.get("log_stem", Path(str(case["topology_json"])).stem)
        log_path = (
            repo_root
            / "evaluation_assets"
            / "raw_logs"
            / "standard_scalability"
            / str(case["topology_type"])
            / f"{log_stem}.log"
        )
        metrics = parse_log_metrics(log_path)
        solver_time_us = metrics.get("solver_time_us")
        if solver_time_us is None:
            continue
        rows.append(
            {
                "case_id": case.get("case_id", log_stem),
                "topology_type": case["topology_type"],
                "point": case["point"],
                "npus_count": case["npus_count"],
                "workload_mode": case.get("workload_mode", ""),
                "workload_tag": case.get("workload_tag", ""),
                "workload_source_csv": case.get("workload_source_csv", ""),
                "block_bytes": case.get("block_bytes", ""),
                "source_matrix_sparsity": case.get("source_matrix_sparsity", ""),
                "solver_time_us": solver_time_us,
                "solver_time_s": float(solver_time_us) / 1e6,
                "makespan_us": metrics.get("total_makespan_us"),
                "demand_fingerprint": metrics.get("demand_fingerprint", ""),
                "topology_json": case["topology_json"],
                "collective_json": case["collective_json"],
                "log_path": log_path.relative_to(repo_root).as_posix(),
            }
        )
    return rows


def format_devices(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g}K"
    return f"{value:g}"


def fit_theory(x: np.ndarray, y: np.ndarray, basis: np.ndarray) -> tuple[float, float]:
    denom = float(np.dot(basis, basis))
    scale = float(np.dot(basis, y) / denom) if denom > 0 else 0.0
    pred = scale * basis
    ss_res = float(np.sum(np.square(y - pred)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return scale, r2


def plot_scalability(rows: list[dict[str, object]], output_path: Path) -> None:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if int(row["npus_count"]) <= SCALABILITY_MAX_DEVICES:
            grouped[str(row["topology_type"])].append((int(row["npus_count"]), float(row["solver_time_s"])))

    titles = {
        "fullmesh": "2D FullMesh",
        "torus": "3D Torus",
        "fat-tree": "2D Clos",
        "cm384": "3D CM384",
    }
    apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    for index, (ax, topology_type) in enumerate(zip(axes, SCALABILITY_COMBINED_ORDER)):
        points = sorted(grouped.get(topology_type, []))
        if not points:
            ax.set_visible(False)
            continue
        plot_scalability_axis(
            ax,
            points,
            topology_type,
            annotated_subplot_title(index, titles[topology_type]),
            SCALABILITY_COLORS[topology_type],
            max_devices=SCALABILITY_COMBINED_MAX_DEVICES,
            fixed_xlim=SCALABILITY_COMBINED_XLIM,
            fixed_xticks=SCALABILITY_COMBINED_XTICKS,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_matrix_from_collective(repo_root: Path, collective_json: str) -> list[list[int]]:
    cfg = load_json(repo_root / collective_json)
    block_bytes = int(cfg.get("block_bytes", 4096))
    if "v_datasize" not in cfg:
        raise ValueError(f"collective has no v_datasize: {collective_json}")
    csv_path = repo_root / str(cfg["v_datasize"])
    matrix: list[list[int]] = []
    with csv_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            matrix.append([int(cell.strip()) * block_bytes for cell in line.split(",") if cell.strip()])
    return matrix


def cold_matrix_from_events(events: list[dict[str, object]], n: int) -> list[list[int]]:
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    seen: set[tuple[int, int, int, int]] = set()
    for event in events:
        if event["class"] != "cold":
            continue
        key = (
            int(event["flow_src"]),
            int(event["flow_dst"]),
            int(event["chunk_id"]),
            int(event["bytes"]),
        )
        if key in seen:
            continue
        seen.add(key)
        matrix[key[0]][key[1]] += key[3]
    return matrix


def clos_path(src: int, dst: int, n: int) -> list[int]:
    if src == dst:
        return [src]
    gpus_per_node = 8
    scale_up_base = n
    scale_out = n + math.ceil(n / gpus_per_node)
    if src // gpus_per_node == dst // gpus_per_node:
        return [src, scale_up_base + (src // gpus_per_node), dst]
    return [src, scale_out, dst]


def transfer_time_us(topo: dict[str, object], src: int, dst: int, bytes_count: int, n: int) -> float:
    bandwidths = list(topo["bandwidth"])
    latencies = list(topo["latency"])
    gpus_per_node = int(topo["shape"][0])
    is_same_node_gpu_to_switch = (
        (src < n and dst >= n and dst < n + math.ceil(n / gpus_per_node) and src // gpus_per_node == dst - n)
        or (dst < n and src >= n and src < n + math.ceil(n / gpus_per_node) and dst // gpus_per_node == src - n)
    )
    dim = 0 if is_same_node_gpu_to_switch else min(1, len(bandwidths) - 1)
    bandwidth_bytes_per_us = float(bandwidths[dim]) * (1 << 30) / 1e6
    return float(latencies[dim]) / 1000.0 + bytes_count / bandwidth_bytes_per_us


def first_gap(intervals: list[tuple[float, float]], ready: float, duration: float) -> tuple[float, float]:
    start = ready
    for busy_start, busy_end in sorted(intervals):
        if start + duration <= busy_start + 1e-9:
            break
        if start < busy_end - 1e-9:
            start = busy_end
    return start, start + duration


def schedule_path(
    topo: dict[str, object],
    n: int,
    slots: dict[tuple[int, int], list[tuple[float, float]]],
    path: list[int],
    bytes_count: int,
    earliest: float,
) -> list[dict[str, object]]:
    ready = earliest
    events: list[dict[str, object]] = []
    for src, dst in zip(path, path[1:]):
        duration = transfer_time_us(topo, src, dst, bytes_count, n)
        start, end = first_gap(slots[(src, dst)], ready, duration)
        slots[(src, dst)].append((start, end))
        slots[(src, dst)].sort()
        events.append(
            {
                "class": "cold",
                "link_src": src,
                "link_dst": dst,
                "bytes": bytes_count,
                "start_us": start,
                "end_us": end,
                "path": ">".join(str(node) for node in path),
            }
        )
        ready = end
    return events


def initial_hot_slots(events: list[dict[str, object]]) -> dict[tuple[int, int], list[tuple[float, float]]]:
    slots: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if event["class"] == "hot":
            slots[(int(event["link_src"]), int(event["link_dst"]))].append(
                (float(event["start_us"]), float(event["end_us"]))
            )
    for key in list(slots):
        slots[key].sort()
    return slots


def schedule_cold_pairwise(
    topo: dict[str, object],
    cold_matrix: list[list[int]],
    hot_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    n = len(cold_matrix)
    slots = initial_hot_slots(hot_events)
    current_time = 0.0
    all_events: list[dict[str, object]] = []
    for round_index in range(n - 1):
        step_start = current_time
        step_end = step_start
        xor_distance = round_index + 1
        transfers: list[tuple[int, int, int]] = []
        for src in range(n):
            dst = src ^ xor_distance
            if src < dst:
                if cold_matrix[src][dst] > 0:
                    transfers.append((src, dst, cold_matrix[src][dst]))
                if cold_matrix[dst][src] > 0:
                    transfers.append((dst, src, cold_matrix[dst][src]))
        for src, dst, bytes_count in transfers:
            events = schedule_path(topo, n, slots, clos_path(src, dst, n), bytes_count, step_start)
            all_events.extend(events)
            if events:
                step_end = max(step_end, float(events[-1]["end_us"]))
        current_time = step_end
    return all_events


def schedule_cold_biring(
    topo: dict[str, object],
    cold_matrix: list[list[int]],
    hot_events: list[dict[str, object]],
) -> list[dict[str, object]]:
    n = len(cold_matrix)
    slots = initial_hot_slots(hot_events)
    clockwise: list[list[dict[str, int]]] = [[] for _ in range(n)]
    counter: list[list[dict[str, int]]] = [[] for _ in range(n)]
    max_rounds = 0
    for src in range(n):
        for dst in range(n):
            bytes_count = cold_matrix[src][dst]
            if src == dst or bytes_count <= 0:
                continue
            cw_hops = (dst - src + n) % n
            ccw_hops = (src - dst + n) % n
            use_cw = cw_hops < ccw_hops or (cw_hops == ccw_hops and src % 2 == 0)
            hops = cw_hops if use_cw else ccw_hops
            max_rounds = max(max_rounds, hops)
            msg = {"final_dest": dst, "bytes": bytes_count, "remaining": hops}
            (clockwise if use_cw else counter)[src].append(msg)

    current_time = 0.0
    all_events: list[dict[str, object]] = []
    for _ in range(max_rounds):
        step_start = current_time
        step_end = step_start
        transfers: list[tuple[int, int, int, bool, list[dict[str, int]]]] = []
        for node in range(n):
            if clockwise[node]:
                transfers.append((node, (node + 1) % n, sum(m["bytes"] for m in clockwise[node]), True, clockwise[node]))
            if counter[node]:
                transfers.append((node, (node - 1 + n) % n, sum(m["bytes"] for m in counter[node]), False, counter[node]))
        transfers.sort(key=lambda item: (item[0], not item[3], item[1]))
        next_clockwise: list[list[dict[str, int]]] = [[] for _ in range(n)]
        next_counter: list[list[dict[str, int]]] = [[] for _ in range(n)]
        for src, dst, bytes_count, is_clockwise, payloads in transfers:
            events = schedule_path(topo, n, slots, clos_path(src, dst, n), bytes_count, step_start)
            all_events.extend(events)
            if events:
                step_end = max(step_end, float(events[-1]["end_us"]))
            for payload in payloads:
                if payload["remaining"] <= 1:
                    continue
                forwarded = dict(payload)
                forwarded["remaining"] -= 1
                (next_clockwise if is_clockwise else next_counter)[dst].append(forwarded)
        clockwise, counter = next_clockwise, next_counter
        current_time = step_end
    return all_events


def active_breakdown(events: list[dict[str, object]]) -> tuple[float, float, float, float]:
    cuts: list[float] = []
    for event in events:
        cuts.extend([float(event["start_us"]), float(event["end_us"])])
    cuts = sorted(set(round(cut, 9) for cut in cuts))
    hot_only = cold_only = overlap = active = 0.0
    for start, end in zip(cuts, cuts[1:]):
        if end <= start:
            continue
        has_hot = False
        has_cold = False
        for event in events:
            if float(event["start_us"]) < end - 1e-9 and float(event["end_us"]) > start + 1e-9:
                if event["class"] == "hot":
                    has_hot = True
                else:
                    has_cold = True
            if has_hot and has_cold:
                break
        duration = end - start
        if has_hot or has_cold:
            active += duration
        if has_hot and has_cold:
            overlap += duration
        elif has_hot:
            hot_only += duration
        elif has_cold:
            cold_only += duration
    return hot_only, cold_only, overlap, active


def speed_tasks(repo_root: Path) -> list[Task]:
    return [
        Task(
            name="speed::fattree_8x4_eval::layer1_group0_256MB",
            topology_json="evaluation_assets/topologies/synthetic/fattree_8x4_eval.json",
            collective_json="evaluation_assets/collectives/synthetic/32devices/layer1_group0_32devices_256MB.json",
            args=("--solver", "mode=speed"),
            log_path=repo_root / "evaluation_assets" / "raw_logs" / "speed" / "fattree_8x4_eval_layer1_group0_256MB.log",
            stage="speed",
        )
    ]


def build_speed_rows(repo_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    topo = load_json(repo_root / "evaluation_assets/topologies/synthetic/fattree_8x4_eval.json")
    task = speed_tasks(repo_root)[0]
    metrics = parse_log_metrics(task.log_path)
    standard_events = parse_speed_events(task.log_path)
    if not standard_events:
        return [], []
    n = int(topo["shape"][0]) * int(topo["shape"][1])
    cold_matrix = cold_matrix_from_events(standard_events, n)
    hot_events = [event for event in standard_events if event["class"] == "hot"]

    methods = {
        "standard": [event for event in standard_events if event["class"] in {"hot", "cold"}],
        "cold_biring": [*hot_events, *schedule_cold_biring(topo, cold_matrix, hot_events)],
        "cold_mpibaseline": [*hot_events, *schedule_cold_pairwise(topo, cold_matrix, hot_events)],
    }
    summary_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for method, events in methods.items():
        hot_only, cold_only, overlap, active = active_breakdown(events)
        makespan = max((float(event["end_us"]) for event in events), default=0.0)
        cold_events = [event for event in events if event["class"] == "cold"]
        cold_makespan = max((float(event["end_us"]) for event in cold_events), default=0.0)
        summary_rows.append(
            {
                "topology_key": "fattree_8x4_eval",
                "method": method,
                "hot_solver_time_us": metrics.get("speed_hot_solver_time_us", 0.0),
                "cold_solver_time_us": metrics.get("speed_cold_solver_time_us", 0.0) if method == "standard" else 0.0,
                "full_solver_time_us": metrics.get("speed_full_solver_time_us", 0.0) if method == "standard" else metrics.get("speed_hot_solver_time_us", 0.0),
                "makespan_us": makespan,
                "hot_only_active_us": hot_only,
                "cold_only_active_us": cold_only,
                "overlap_us": overlap,
                "active_us": active,
                "hot_only_makespan_us": metrics.get("speed_hot_only_makespan_us", 0.0),
                "cold_only_makespan_us": cold_makespan,
                "demand_fingerprint": metrics.get("demand_fingerprint", ""),
                "log_path": task.log_path.relative_to(repo_root).as_posix(),
            }
        )
        for event in events:
            event_rows.append(
                {
                    "method": method,
                    "class": event["class"],
                    "link_src": event["link_src"],
                    "link_dst": event["link_dst"],
                    "bytes": event["bytes"],
                    "start_us": event["start_us"],
                    "end_us": event["end_us"],
                    "path": event.get("path", ""),
                }
            )
    return summary_rows, event_rows


def plot_speed(summary_rows: list[dict[str, object]], output_path: Path) -> None:
    if not summary_rows:
        return
    methods = [str(row["method"]) for row in summary_rows]
    labels = {
        "standard": "Cold: Standard",
        "cold_biring": "Cold: BiRing",
        "cold_mpibaseline": "Cold: MPICH",
    }
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    hot_solver = [float(row["hot_solver_time_us"]) for row in summary_rows]
    cold_solver = [float(row["cold_solver_time_us"]) for row in summary_rows]
    axes[0].bar(x, hot_solver, label="Hot solve", color="#06d6a0")
    axes[0].bar(x, cold_solver, bottom=hot_solver, label="Cold solve", color="#118ab2")
    axes[0].set_ylabel("Solve Time (us)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[m] for m in methods], rotation=12, ha="right")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25, linestyle="--")

    hot_only = [float(row["hot_only_active_us"]) for row in summary_rows]
    cold_only = [float(row["cold_only_active_us"]) for row in summary_rows]
    overlap = [float(row["overlap_us"]) for row in summary_rows]
    axes[1].bar(x, hot_only, label="Hot-only", color="#06d6a0")
    axes[1].bar(x, cold_only, bottom=hot_only, label="Cold-only", color="#118ab2")
    bottom = [h + c for h, c in zip(hot_only, cold_only)]
    axes[1].bar(x, overlap, bottom=bottom, label="Overlap", color="#f77f00")
    axes[1].set_ylabel("Active Makespan Breakdown (us)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[m] for m in methods], rotation=12, ha="right")
    axes[1].legend(frameon=False)
    axes[1].grid(True, axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


BREAKEVEN_SIZE_BYTES = {
    "64MB": 64 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
}
BREAKEVEN_PREFILL_SOURCE_CSV = "input/Prefill_Group4_Layer5.csv"


def load_token_matrix_csv(path: Path) -> list[list[int]]:
    with path.open(newline="") as handle:
        return [[int(cell) for cell in row] for row in csv.reader(handle) if row]


def merge_prefill_to_32_devices(matrix: list[list[int]]) -> list[list[int]]:
    if len(matrix) != 64 or any(len(row) != 64 for row in matrix):
        raise ValueError("prefill breakeven source must be a 64x64 token matrix")
    merged: list[list[int]] = []
    for row in range(0, 64, 2):
        merged_row: list[int] = []
        for col in range(0, 64, 2):
            merged_row.append(
                matrix[row][col]
                + matrix[row][col + 1]
                + matrix[row + 1][col]
                + matrix[row + 1][col + 1]
            )
        merged.append(merged_row)
    return merged


def scale_token_matrix_to_nominal_size(
    matrix: list[list[int]], target_devices: int, nominal_bytes_per_rank: int
) -> list[list[int]]:
    total_tokens = sum(sum(row) for row in matrix)
    target_tokens = max(1, round(target_devices * nominal_bytes_per_rank / 4096))
    factor = target_tokens / total_tokens if total_tokens > 0 else 1.0
    scaled: list[list[int]] = []
    for row in matrix:
        scaled_row: list[int] = []
        for value in row:
            if value <= 0:
                scaled_row.append(0)
                continue
            scaled_value = int(round(value * factor))
            scaled_row.append(max(1, scaled_value))
        scaled.append(scaled_row)
    return scaled


def write_token_matrix_csv(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix)


def ensure_breakeven_prefill_collectives(repo_root: Path) -> dict[tuple[int, str], dict[str, object]]:
    source_matrix_64 = load_token_matrix_csv(repo_root / BREAKEVEN_PREFILL_SOURCE_CSV)
    source_by_devices = {
        64: source_matrix_64,
        32: merge_prefill_to_32_devices(source_matrix_64),
    }
    records: dict[tuple[int, str], dict[str, object]] = {}
    for target_devices, base_matrix in source_by_devices.items():
        for size_label, nominal_bytes in BREAKEVEN_SIZE_BYTES.items():
            matrix = scale_token_matrix_to_nominal_size(base_matrix, target_devices, nominal_bytes)
            case_id = f"prefill_group4_layer5_{target_devices}devices_{size_label}"
            csv_path = (
                repo_root
                / "evaluation_assets"
                / "csv"
                / "speed_breakeven"
                / f"{target_devices}devices"
                / f"{case_id}.csv"
            )
            collective_path = (
                repo_root
                / "evaluation_assets"
                / "collectives"
                / "speed_breakeven"
                / f"{target_devices}devices"
                / f"{case_id}.json"
            )
            write_token_matrix_csv(csv_path, matrix)
            collective_path.parent.mkdir(parents=True, exist_ok=True)
            collective_path.write_text(
                json.dumps(
                    {
                        "collective": "alltoallv",
                        "block_bytes": 4096,
                        "v_datasize": csv_path.relative_to(repo_root).as_posix(),
                        "chunkfactor": 1,
                    },
                    indent=2,
                )
                + "\\n"
            )
            records[(target_devices, size_label)] = {
                "case_id": case_id,
                "trace": "prefill_group4_layer5",
                "target_devices": target_devices,
                "size_label": size_label,
                "source_csv": BREAKEVEN_PREFILL_SOURCE_CSV,
                "collective_json": collective_path.relative_to(repo_root).as_posix(),
            }
    return records


def breakeven_tasks(repo_root: Path) -> list[Task]:
    topo_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json")
    cases = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.json")["cases"]
    prefill_cases = ensure_breakeven_prefill_collectives(repo_root)
    topology_keys = ["mesh_nebula_8x4", "torus_tpuv4_4x4x4", "fattree_8x4_eval", "cm384_16x2_eval"]
    topologies = {item["topology_key"]: item for item in topo_manifest["topologies"]}
    selected_decode_cases: list[dict[str, object]] = []
    for target_devices in (32, 64):
        for size_label in ("64MB", "256MB"):
            case = next(
                c for c in cases
                if int(c["target_devices"]) == target_devices and c["size_label"] == size_label and int(c["sample_index"]) == 0
            )
            selected_decode_cases.append({**case, "trace": "decode"})

    tasks: list[Task] = []
    for topology_key in topology_keys:
        topo = topologies[topology_key]
        target_devices = int(topo["target_devices"])
        for case in selected_decode_cases:
            if int(case["target_devices"]) != target_devices:
                continue
            tasks.append(
                Task(
                    name=f"breakeven::decode::{topology_key}::{case['case_id']}",
                    topology_json=str(topo["topology_json"]),
                    collective_json=str(case["collective_json"]),
                    args=("--solver", "mode=standard"),
                    log_path=repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "speed_breakeven"
                    / "decode"
                    / topology_key
                    / f"{case['case_id']}.log",
                    stage="speed_breakeven",
                )
            )
        for size_label in ("64MB", "256MB"):
            case = prefill_cases[(target_devices, size_label)]
            tasks.append(
                Task(
                    name=f"breakeven::prefill::{topology_key}::{case['case_id']}",
                    topology_json=str(topo["topology_json"]),
                    collective_json=str(case["collective_json"]),
                    args=("--solver", "mode=standard"),
                    log_path=repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "speed_breakeven"
                    / "prefill_group4_layer5"
                    / topology_key
                    / f"{case['case_id']}.log",
                    stage="speed_breakeven",
                )
            )
    return tasks


def build_breakeven_rows(repo_root: Path) -> list[dict[str, object]]:
    synthetic_summary = read_csv(repo_root / "evaluation_assets" / "parsed" / "synthetic_summary.csv")
    baseline_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in synthetic_summary:
        baseline_by_key[(row["topology_key"], row["size_label"])].append(row)

    rows: list[dict[str, object]] = []
    for task in breakeven_tasks(repo_root):
        metrics = parse_log_metrics(task.log_path)
        topology_key = task.log_path.parent.name
        trace = task.log_path.parent.parent.name
        case_id = task.log_path.stem
        size_label = "256MB" if "256MB" in case_id else "64MB"
        baseline_rows = baseline_by_key.get((topology_key, size_label), [])
        other_rows = [row for row in baseline_rows if row["method"] != "glaive"]
        if not other_rows or metrics.get("solver_time_us") is None or metrics.get("total_makespan_us") is None:
            continue
        glaive_makespan = float(metrics["total_makespan_us"])
        best_baseline = min(other_rows, key=lambda row: float(row["avg_makespan_us"]))
        baseline_makespan = float(best_baseline["avg_makespan_us"])
        delta = baseline_makespan - glaive_makespan
        solver_time = float(metrics["solver_time_us"])
        batch_min = math.floor(solver_time / delta) + 1 if delta > 0 else math.inf
        rows.append(
            {
                "trace": trace,
                "topology_key": topology_key,
                "size_label": size_label,
                "glaive_makespan_us": glaive_makespan,
                "best_baseline": best_baseline["method"],
                "best_baseline_makespan_us": baseline_makespan,
                "standard_solver_time_us": solver_time,
                "per_batch_gain_us": delta,
                "break_even_batch_size": "inf" if math.isinf(batch_min) else int(batch_min),
                "demand_fingerprint": metrics.get("demand_fingerprint", ""),
                "log_path": task.log_path.relative_to(repo_root).as_posix(),
            }
        )
    return rows


def write_breakeven_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "trace",
        "topology_key",
        "size_label",
        "best_baseline",
        "per_batch_gain_us",
        "standard_solver_time_us",
        "break_even_batch_size",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{float(row[h]):.2f}" if h.endswith("_us") else str(row[h])
                for h in headers
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if args.stage in {"all", "fairness"}:
        run_tasks(repo_root, fairness_tasks(repo_root), args.force, args.max_workers)
    if args.stage in {"all", "scalability"}:
        run_tasks(
            repo_root,
            scalability_tasks(repo_root, args.limit_scalability),
            args.force,
            args.max_workers,
        )
    if args.stage in {"all", "speed"}:
        run_tasks(repo_root, speed_tasks(repo_root), args.force, args.max_workers)
    if args.stage in {"all", "breakeven"}:
        run_tasks(repo_root, breakeven_tasks(repo_root), args.force, args.max_workers)

    parsed_dir = repo_root / "evaluation_assets" / "parsed"
    plot_dir = repo_root / "evaluation_assets" / "plots"

    fairness_rows = build_fairness_rows(repo_root)
    write_csv(parsed_dir / "standard_fairness.csv", fairness_rows)

    scalability_rows = build_scalability_rows(repo_root)
    write_csv(parsed_dir / "standard_scalability_raw.csv", scalability_rows)
    if scalability_rows:
        plot_scalability(scalability_rows, plot_dir / "Scalability_Study_Standard.pdf")

    speed_summary, speed_events = build_speed_rows(repo_root)
    write_csv(parsed_dir / "speed_clos_summary.csv", speed_summary)
    write_csv(parsed_dir / "speed_clos_events.csv", speed_events)
    if speed_summary:
        plot_speed(speed_summary, plot_dir / "Speed_Clos_Breakdown.pdf")

    breakeven_rows = build_breakeven_rows(repo_root)
    write_csv(parsed_dir / "speed_breakeven.csv", breakeven_rows)
    write_breakeven_markdown(parsed_dir / "speed_breakeven.md", breakeven_rows)

    print(
        json.dumps(
            {
                "fairness_rows": len(fairness_rows),
                "scalability_rows": len(scalability_rows),
                "speed_summary_rows": len(speed_summary),
                "speed_event_rows": len(speed_events),
                "breakeven_rows": len(breakeven_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
