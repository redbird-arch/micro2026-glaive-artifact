#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SPEC = importlib.util.spec_from_file_location("run_standard_studies_base", SCRIPT_DIR / "run_standard_studies.py")
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("Cannot load run_standard_studies.py")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)

Task = base.Task

LINK_BREAKDOWN_TOPOLOGIES = [
    "mesh_nebula_8x4",
    "torus_tpuv4_4x4x4",
    "fattree_8x4_eval",
    "cm384_16x2_eval",
]
TOPOLOGY_DISPLAY = {
    "mesh_nebula_8x4": "2D Mesh 8x4",
    "torus_tpuv4_4x4x4": "3D Torus 4x4x4",
    "fattree_8x4_eval": "2D Clos 8x4",
    "cm384_16x2_eval": "3D CM384 2x8x2",
    "fullmesh": "2D FullMesh",
    "torus": "3D Torus",
    "fat-tree": "2D Clos",
    "cm384": "3D CM384",
}
CACHE_TOPOLOGY_ORDER = ["fullmesh", "torus", "fat-tree", "cm384"]
CACHE_SCALE_SHAPES = {
    "fullmesh": [(8, 2), (8, 4), (8, 8), (8, 16), (8, 32), (8, 64), (8, 128)],
    "fat-tree": [(8, 2), (8, 4), (8, 8), (8, 16), (8, 32), (8, 64), (8, 128)],
    "cm384": [(16, 1), (16, 2), (16, 4), (16, 8), (16, 16), (16, 32), (16, 64)],
    "torus": [(2, 2, 4), (2, 4, 4), (4, 4, 4), (4, 4, 8), (4, 8, 8), (8, 8, 8), (8, 8, 16)],
}
FIGURE9_REFERENCE = Path("evaluation_assets/parsed/figure9_h100_glaive_reference.csv")
FIGURE9_NAMESPACE = "input/generated/olmoe_h100_batch_figure9_source_railonly_20260606"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Speed_OPT3 follow-up experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--stage",
        choices=["all", "scalability", "cache", "speed", "figure9", "parse-only"],
        default="all",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_task(repo_root: Path, task: Task, force: bool) -> str:
    if not force and base.task_is_complete(task.log_path):
        return f"[skip] {task.name}"
    base.run_local_task(repo_root, task)
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


def case_id_for_shape(topology_type: str, shape: tuple[int, ...]) -> str:
    if topology_type == "torus":
        return f"torus_{shape[0]}x{shape[1]}x{shape[2]}"
    if topology_type == "fat-tree":
        return f"fattree_{shape[0]}x{shape[1]}"
    return f"{topology_type}_{shape[0]}x{shape[1]}"


def topology_payload(topology_type: str, shape: tuple[int, ...]) -> dict[str, Any]:
    if topology_type == "fullmesh":
        return {"topology": "fullmesh", "dimension": 2, "shape": list(shape), "bandwidth": [300, 300], "latency": [500, 500]}
    if topology_type == "torus":
        return {"topology": "torus", "dimension": 3, "shape": list(shape), "bandwidth": [56, 56, 56], "latency": [500, 500, 500]}
    if topology_type == "fat-tree":
        return {
            "topology": "fat-tree",
            "dimension": 2,
            "shape": list(shape),
            "switch-dimension": 2,
            "switch-shape": [1, 1],
            "link-count": [1, 1],
            "bandwidth": [300, 25],
            "latency": [500, 500],
        }
    if topology_type == "cm384":
        return {
            "topology": "cm384",
            "dimension": 2,
            "shape": list(shape),
            "switch-shape": [1, 1],
            "link-count": [1, 1],
            "direct-bandwidth": 270,
            "direct-latency": 200,
            "bandwidth": [196, 196],
            "latency": [200, 200],
        }
    raise ValueError(f"unsupported topology type {topology_type}")


def ensure_cache_probe_case(repo_root: Path, topology_type: str, shape: tuple[int, ...], block_bytes: int = 4096) -> tuple[str, str, str, int]:
    case_id = case_id_for_shape(topology_type, shape)
    suffix = "" if block_bytes == 4096 else f"_{block_bytes}B"
    topology_path = repo_root / "evaluation_assets" / "topologies" / "opt3_cache" / f"{case_id}.json"
    collective_path = repo_root / "evaluation_assets" / "collectives" / "opt3_cache" / f"{case_id}{suffix}_decode_bs64_layer8_56_tiled.json"
    write_json(topology_path, topology_payload(topology_type, shape))
    write_json(
        collective_path,
        {
            "collective": "alltoallv",
            "block_bytes": block_bytes,
            "synthetic_v_datasize": {"pattern": "tiled-csv", "base_csv": "input/Decode_BS64_Layer8_56.csv"},
        },
    )
    n = math.prod(shape)
    return (
        topology_path.relative_to(repo_root).as_posix(),
        collective_path.relative_to(repo_root).as_posix(),
        f"{case_id}{suffix}_decode_bs64_layer8_56_tiled",
        int(n),
    )


def cache_scale_tasks(repo_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for topology_type in CACHE_TOPOLOGY_ORDER:
        for shape in CACHE_SCALE_SHAPES[topology_type]:
            topo, coll, stem, n = ensure_cache_probe_case(repo_root, topology_type, shape)
            tasks.append(
                Task(
                    name=f"opt3_cache_scale::{topology_type}::{n}",
                    topology_json=topo,
                    collective_json=coll,
                    args=("--solver", "mode=standard"),
                    log_path=repo_root / "evaluation_assets" / "raw_logs" / "standard_cache" / "scale" / topology_type / f"{stem}.log",
                    stage="standard_cache",
                )
            )
    return tasks


def cache_data_volume_tasks(repo_root: Path) -> list[Task]:
    tasks: list[Task] = []
    representative = {
        "fullmesh": (8, 32),
        "torus": (4, 8, 8),
        "fat-tree": (8, 32),
        "cm384": (16, 16),
    }
    for topology_type, shape in representative.items():
        for block_bytes in (4096, 65536):
            topo, coll, stem, n = ensure_cache_probe_case(repo_root, topology_type, shape, block_bytes=block_bytes)
            tasks.append(
                Task(
                    name=f"opt3_cache_data_volume::{topology_type}::{n}::{block_bytes}",
                    topology_json=topo,
                    collective_json=coll,
                    args=("--solver", "mode=standard"),
                    log_path=repo_root / "evaluation_assets" / "raw_logs" / "standard_cache" / "data_volume" / topology_type / f"{stem}.log",
                    stage="standard_cache",
                )
            )
    return tasks


def parse_cache_csv_rows(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("[Standard Cache CSV]"):
            continue
        payload = line.split("]", 1)[1].strip()
        row: dict[str, Any] = {}
        for item in payload.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"phase", "routing_kind"}:
                row[key] = value
            elif "." in value:
                row[key] = float(value)
            else:
                try:
                    row[key] = int(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def cache_rows_for_task(task: Task, source: str, topology_type: str, point: str, npus_count: int, block_bytes: int = 4096) -> list[dict[str, Any]]:
    metrics = base.parse_log_metrics(task.log_path)
    rows = []
    for row in parse_cache_csv_rows(task.log_path):
        rows.append(
            {
                "source": source,
                "topology_type": topology_type,
                "point": point,
                "npus_count": npus_count,
                "block_bytes": block_bytes,
                "phase": row.get("phase", ""),
                "cache_init_time_us": row.get("init_time_us", ""),
                "approx_cache_bytes": row.get("approx_cache_bytes", ""),
                "approx_cache_mib": float(row.get("approx_cache_bytes", 0)) / (1024 * 1024),
                "routing_kind": row.get("routing_kind", ""),
                "connection_edges": row.get("connection_edges", ""),
                "distance_matrix_entries": row.get("distance_matrix_entries", ""),
                "link_cache_entries": row.get("link_cache_entries", ""),
                "small_switch_candidate_paths": row.get("small_switch_candidate_paths", ""),
                "small_switch_dominant_paths": row.get("small_switch_dominant_paths", ""),
                "gpu_pair_min_chunk_entries": row.get("gpu_pair_min_chunk_entries", ""),
                "shortest_path_cache_entries": row.get("shortest_path_cache_entries", ""),
                "time_shortest_path_cache_entries": row.get("time_shortest_path_cache_entries", ""),
                "solver_time_us": metrics.get("solver_time_us"),
                "makespan_us": metrics.get("total_makespan_us"),
                "log_path": task.log_path.as_posix(),
            }
        )
    return rows


def build_scalability_cache_rows(repo_root: Path) -> list[dict[str, Any]]:
    manifest = base.load_json(repo_root / "evaluation_assets" / "manifests" / "scalability_cases.json")
    by_log: dict[str, dict[str, Any]] = {}
    for case in manifest["cases"]:
        if str(case.get("topology_type")) == "mesh" or int(case.get("npus_count", 0)) > base.SCALABILITY_MAX_DEVICES:
            continue
        stem = case.get("log_stem", Path(str(case["topology_json"])).stem)
        log_path = repo_root / "evaluation_assets" / "raw_logs" / "standard_scalability" / str(case["topology_type"]) / f"{stem}.log"
        task = Task(
            name=f"scalability_cache::{case['topology_type']}::{stem}",
            topology_json=str(case["topology_json"]),
            collective_json=str(case["collective_json"]),
            args=("--solver", "mode=standard"),
            log_path=log_path,
            stage="standard_scalability",
        )
        rows = cache_rows_for_task(
            task,
            "scalability_figure",
            str(case["topology_type"]),
            str(case["point"]),
            int(case["npus_count"]),
            int(case.get("block_bytes", 4096)),
        )
        for row in rows:
            row["log_path"] = log_path.relative_to(repo_root).as_posix()
        by_log[str(log_path)] = {"rows": rows}
    return [row for item in by_log.values() for row in item["rows"]]


def build_cache_scale_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in cache_scale_tasks(repo_root):
        parts = task.name.split("::")
        topology_type = parts[1]
        n = int(parts[2])
        stem = task.log_path.stem.replace("_decode_bs64_layer8_56_tiled", "")
        task_rows = cache_rows_for_task(task, "scale_16_to_1k", topology_type, stem, n, 4096)
        for row in task_rows:
            row["log_path"] = task.log_path.relative_to(repo_root).as_posix()
        rows.extend(task_rows)
    return rows


def build_cache_data_volume_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in cache_data_volume_tasks(repo_root):
        parts = task.name.split("::")
        topology_type = parts[1]
        n = int(parts[2])
        block_bytes = int(parts[3])
        stem = task.log_path.stem.replace("_decode_bs64_layer8_56_tiled", "")
        task_rows = cache_rows_for_task(task, "data_volume_probe", topology_type, stem, n, block_bytes)
        for row in task_rows:
            row["log_path"] = task.log_path.relative_to(repo_root).as_posix()
        rows.extend(task_rows)
    return rows


def build_precompute_cache_rows(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initial_by_key: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    post_by_key: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for row in cache_rows:
        key = (
            str(row["source"]),
            str(row["topology_type"]),
            int(row["npus_count"]),
            int(row["block_bytes"]),
            str(row["point"]),
        )
        if row["phase"] == "initial":
            initial_by_key[key] = row
        elif row["phase"] == "post_solve":
            post_by_key[key] = row

    rows: list[dict[str, Any]] = []
    for key, initial in sorted(initial_by_key.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4])):
        post = post_by_key.get(key, {})
        static_bytes = int(float(initial.get("approx_cache_bytes", 0) or 0))
        post_bytes = int(float(post.get("approx_cache_bytes", static_bytes) or static_bytes))
        lazy_bytes = max(0, post_bytes - static_bytes)
        rows.append(
            {
                "source": initial["source"],
                "topology_type": initial["topology_type"],
                "point": initial["point"],
                "npus_count": initial["npus_count"],
                "block_bytes": initial["block_bytes"],
                "routing_kind": initial["routing_kind"],
                "topology_static_cache_init_time_us": initial["cache_init_time_us"],
                "topology_static_cache_bytes": static_bytes,
                "topology_static_cache_mib": static_bytes / (1024 * 1024),
                "solve_lazy_cache_bytes": lazy_bytes,
                "solve_lazy_cache_mib": lazy_bytes / (1024 * 1024),
                "post_solve_total_cache_bytes": post_bytes,
                "post_solve_total_cache_mib": post_bytes / (1024 * 1024),
                "connection_edges": initial.get("connection_edges", ""),
                "distance_matrix_entries": initial.get("distance_matrix_entries", ""),
                "link_cache_entries": initial.get("link_cache_entries", ""),
                "small_switch_candidate_paths": initial.get("small_switch_candidate_paths", ""),
                "small_switch_dominant_paths": initial.get("small_switch_dominant_paths", ""),
                "gpu_pair_min_chunk_entries": initial.get("gpu_pair_min_chunk_entries", ""),
                "post_shortest_path_cache_entries": post.get("shortest_path_cache_entries", ""),
                "post_time_shortest_path_cache_entries": post.get("time_shortest_path_cache_entries", ""),
                "solver_time_us": initial.get("solver_time_us", ""),
                "makespan_us": initial.get("makespan_us", ""),
                "log_path": initial.get("log_path", ""),
            }
        )
    return rows


def fmt_us(value: Any) -> str:
    if value in (None, ""):
        return ""
    value = float(value)
    if value >= 1e6:
        return f"{value / 1e6:.3f}s"
    if value >= 1000:
        return f"{value / 1000:.3f}ms"
    return f"{value:.2f}us"


def fmt_mib(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.3f}"


def write_cache_markdown(path: Path, scalability_rows: list[dict[str, Any]], scale_rows: list[dict[str, Any]], data_volume_rows: list[dict[str, Any]]) -> None:
    initial_scalability = [row for row in scalability_rows if row["phase"] == "initial"]
    post_scalability = {(row["topology_type"], row["npus_count"], row["point"]): row for row in scalability_rows if row["phase"] == "post_solve"}
    initial_scale = [row for row in scale_rows if row["phase"] == "initial"]
    data_initial = [row for row in data_volume_rows if row["phase"] == "initial"]
    data_post = [row for row in data_volume_rows if row["phase"] == "post_solve"]

    lines: list[str] = []
    lines.append("# Standard Cache-Overhead Audit")
    lines.append("")
    lines.append("## Measurement convention")
    lines.append("")
    lines.append("`cache_init_time_us` measures construction of `StandardSynthesizer`, including static caches for the connectivity graph, topology distances, link bandwidth/latency tables, small-switch candidate paths, and GPU-pair minimum chunks. `approx_cache_bytes` at `initial` is the topology-static cache that can be precomputed and stored for a topology; `post_solve` additionally includes path-cache entries loaded lazily for actual flows and chunks.")
    lines.append("")
    lines.append("Use `topology_static_cache_mib = initial approx_cache_bytes / MiB` for the portion that can be precomputed for a topology and excluded from production solve time. `solve_lazy_cache_mib = post_solve - initial` depends on the trace's flow/chunk sizes and is built during solving, so its construction cost is included in `Solver Time`.")
    lines.append("")
    lines.append("## Raw initial/post convention: Scalability_Study_Standard")
    lines.append("")
    lines.append("| Topology | Point | NPU | cache init | initial MiB | post MiB | solver time | post shortest/path entries |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(initial_scalability, key=lambda r: (CACHE_TOPOLOGY_ORDER.index(r["topology_type"]), int(r["npus_count"]))):
        post = post_scalability.get((row["topology_type"], row["npus_count"], row["point"]), {})
        entries = f"{post.get('shortest_path_cache_entries', '')}/{post.get('time_shortest_path_cache_entries', '')}"
        lines.append(
            f"| {TOPOLOGY_DISPLAY.get(row['topology_type'], row['topology_type'])} | {row['point']} | {row['npus_count']} | "
            f"{fmt_us(row['cache_init_time_us'])} | {fmt_mib(row['approx_cache_mib'])} | {fmt_mib(post.get('approx_cache_mib', ''))} | "
            f"{fmt_us(row['solver_time_us'])} | {entries} |"
        )
    lines.append("")
    lines.append("This raw convention keeps both `initial MiB` and `post MiB` to show growth from the lazily loaded path cache. `post MiB` is not precomputable from topology alone because it contains entries produced for actual traces, flows, and chunks during solving.")
    lines.append("")
    lines.append("## Static-precomputation convention: Scalability_Study_Standard")
    lines.append("")
    lines.append("| Topology | Point | NPU | static cache init | topology-static MiB | solve-lazy MiB | solver time | post shortest/path entries |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(initial_scalability, key=lambda r: (CACHE_TOPOLOGY_ORDER.index(r["topology_type"]), int(r["npus_count"]))):
        post = post_scalability.get((row["topology_type"], row["npus_count"], row["point"]), {})
        entries = f"{post.get('shortest_path_cache_entries', '')}/{post.get('time_shortest_path_cache_entries', '')}"
        lines.append(
            f"| {TOPOLOGY_DISPLAY.get(row['topology_type'], row['topology_type'])} | {row['point']} | {row['npus_count']} | "
            f"{fmt_us(row['cache_init_time_us'])} | {fmt_mib(row['approx_cache_mib'])} | {fmt_mib(max(0.0, float(post.get('approx_cache_mib', row['approx_cache_mib']) or 0.0) - float(row['approx_cache_mib'])))} | "
            f"{fmt_us(row['solver_time_us'])} | {entries} |"
        )
    lines.append("")
    lines.append("## Cache size from 16 to 1K NPUs")
    lines.append("")
    lines.append("| Topology | NPU | point | routing | static cache init | topology-static MiB |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: |")
    for row in sorted(initial_scale, key=lambda r: (CACHE_TOPOLOGY_ORDER.index(r["topology_type"]), int(r["npus_count"]))):
        lines.append(
            f"| {TOPOLOGY_DISPLAY.get(row['topology_type'], row['topology_type'])} | {row['npus_count']} | {row['point']} | "
            f"{row['routing_kind']} | {fmt_us(row['cache_init_time_us'])} | {fmt_mib(row['approx_cache_mib'])} |"
        )
    lines.append("")
    lines.append("## Raw initial/post convention: communication-volume independence")
    lines.append("")
    lines.append("| Topology | NPU | block_bytes | initial MiB | post MiB | init entries | post entries |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    post_by = {(r["topology_type"], r["npus_count"], r["block_bytes"]): r for r in data_post}
    for row in sorted(data_initial, key=lambda r: (CACHE_TOPOLOGY_ORDER.index(r["topology_type"]), int(r["block_bytes"]))):
        post = post_by.get((row["topology_type"], row["npus_count"], row["block_bytes"]), {})
        lines.append(
            f"| {TOPOLOGY_DISPLAY.get(row['topology_type'], row['topology_type'])} | {row['npus_count']} | {row['block_bytes']} | "
            f"{fmt_mib(row['approx_cache_mib'])} | {fmt_mib(post.get('approx_cache_mib', ''))} | "
            f"{row.get('shortest_path_cache_entries', '')}/{row.get('time_shortest_path_cache_entries', '')} | "
            f"{post.get('shortest_path_cache_entries', '')}/{post.get('time_shortest_path_cache_entries', '')} |"
        )
    lines.append("")
    lines.append("For a fixed topology, `initial MiB` should remain constant as `block_bytes` changes. Any change in `post MiB` or post path entries comes from the cache loaded lazily during solving.")
    lines.append("")
    lines.append("## Static-precomputation convention: communication-volume independence")
    lines.append("")
    lines.append("| Topology | NPU | block_bytes | topology-static MiB | solve-lazy MiB | init entries | post entries |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(data_initial, key=lambda r: (CACHE_TOPOLOGY_ORDER.index(r["topology_type"]), int(r["block_bytes"]))):
        post = post_by.get((row["topology_type"], row["npus_count"], row["block_bytes"]), {})
        lines.append(
            f"| {TOPOLOGY_DISPLAY.get(row['topology_type'], row['topology_type'])} | {row['npus_count']} | {row['block_bytes']} | "
            f"{fmt_mib(row['approx_cache_mib'])} | {fmt_mib(max(0.0, float(post.get('approx_cache_mib', row['approx_cache_mib']) or 0.0) - float(row['approx_cache_mib'])))} | "
            f"{row.get('shortest_path_cache_entries', '')}/{row.get('time_shortest_path_cache_entries', '')} | "
            f"{post.get('shortest_path_cache_entries', '')}/{post.get('time_shortest_path_cache_entries', '')} |"
        )
    lines.append("")
    lines.append("For a fixed topology, `topology-static MiB` is identical across `block_bytes`. Differences in `solve-lazy MiB` or post path entries come from flow/chunk-specific paths loaded during solving and are not topology-static.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def speed_link_tasks(repo_root: Path) -> list[Task]:
    topo_manifest = base.load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json")
    topologies = {item["topology_key"]: item for item in topo_manifest["topologies"]}
    tasks: list[Task] = []
    for topology_key in LINK_BREAKDOWN_TOPOLOGIES:
        target_devices = int(topologies[topology_key]["target_devices"])
        collective = f"evaluation_assets/collectives/synthetic/{target_devices}devices/layer1_group0_{target_devices}devices_256MB.json"
        tasks.append(
            Task(
                name=f"speed_link::{topology_key}::layer1_group0_256MB",
                topology_json=str(topologies[topology_key]["topology_json"]),
                collective_json=collective,
                args=("--solver", "mode=speed"),
                log_path=repo_root / "evaluation_assets" / "raw_logs" / "speed" / f"{topology_key}_layer1_group0_256MB.log",
                stage="speed",
            )
        )
    return tasks



@dataclass(frozen=True)
class TopologyModel:
    key: str
    config: dict[str, Any]
    topology: str
    shape: tuple[int, ...]
    n: int
    gpus_per_node: int
    num_nodes: int
    total_nodes: int


def build_topology_model(topology_key: str, topo: dict[str, Any]) -> TopologyModel:
    topology = str(topo["topology"])
    shape = tuple(int(item) for item in topo["shape"])
    n = int(math.prod(shape))
    if topology in {"fat-tree", "cm", "cm384"}:
        gpus_per_node = int(shape[0])
        num_nodes = int(math.ceil(n / gpus_per_node))
        total_nodes = n + num_nodes + 1
    else:
        gpus_per_node = n
        num_nodes = 1
        total_nodes = n
    return TopologyModel(topology_key, topo, topology, shape, n, gpus_per_node, num_nodes, total_nodes)


def rank_to_coords(rank: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords = []
    value = rank
    for size in reversed(shape):
        coords.append(value % size)
        value //= size
    return tuple(reversed(coords))


def coords_to_rank(coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
    value = 0
    for coord, size in zip(coords, shape):
        value = value * size + coord
    return value


def mesh_path(model: TopologyModel, src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    current = list(rank_to_coords(src, model.shape))
    target = rank_to_coords(dst, model.shape)
    path = [src]
    for dim, target_coord in enumerate(target):
        while current[dim] != target_coord:
            current[dim] += 1 if current[dim] < target_coord else -1
            path.append(coords_to_rank(tuple(current), model.shape))
    return path


def torus_path(model: TopologyModel, src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    current = list(rank_to_coords(src, model.shape))
    target = rank_to_coords(dst, model.shape)
    path = [src]
    for dim, size in enumerate(model.shape):
        forward = (target[dim] - current[dim]) % size
        backward = (current[dim] - target[dim]) % size
        if forward <= backward:
            step = 1
            steps = forward
        else:
            step = -1
            steps = backward
        for _ in range(steps):
            current[dim] = (current[dim] + step) % size
            path.append(coords_to_rank(tuple(current), model.shape))
    return path


def local_switch(model: TopologyModel, rank: int) -> int:
    return model.n + rank // model.gpus_per_node


def global_switch(model: TopologyModel) -> int:
    return model.n + model.num_nodes


def fat_tree_path(model: TopologyModel, src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    if src // model.gpus_per_node == dst // model.gpus_per_node:
        return [src, local_switch(model, src), dst]
    return [src, global_switch(model), dst]


def cm384_direct_pair(model: TopologyModel, src: int, dst: int) -> bool:
    return (
        src < model.n
        and dst < model.n
        and src != dst
        and src // model.gpus_per_node == dst // model.gpus_per_node
        and src // 2 == dst // 2
    )


def cm384_path(model: TopologyModel, src: int, dst: int) -> list[int]:
    if src == dst:
        return [src]
    if cm384_direct_pair(model, src, dst):
        return [src, dst]
    if src // model.gpus_per_node == dst // model.gpus_per_node:
        return [src, local_switch(model, src), dst]
    return [src, local_switch(model, src), global_switch(model), local_switch(model, dst), dst]


def topology_path(model: TopologyModel, src: int, dst: int) -> list[int]:
    if model.topology == "mesh":
        return mesh_path(model, src, dst)
    if model.topology == "torus":
        return torus_path(model, src, dst)
    if model.topology == "fullmesh":
        return [src] if src == dst else [src, dst]
    if model.topology == "fat-tree":
        return fat_tree_path(model, src, dst)
    if model.topology in {"cm", "cm384"}:
        return cm384_path(model, src, dst)
    raise ValueError(f"unsupported topology for Speed baseline replay: {model.topology}")


def direct_edge_dim(model: TopologyModel, src: int, dst: int) -> int:
    if src >= model.n or dst >= model.n:
        return 0
    src_coords = rank_to_coords(src, model.shape)
    dst_coords = rank_to_coords(dst, model.shape)
    for dim, (left, right, size) in enumerate(zip(src_coords, dst_coords, model.shape)):
        if left == right:
            continue
        if abs(left - right) == 1 or abs(left - right) == size - 1:
            return min(dim, len(model.config.get("bandwidth", [1])) - 1)
    return 0


def edge_profile(model: TopologyModel, src: int, dst: int) -> tuple[float, float]:
    if model.topology in {"cm", "cm384"} and src < model.n and dst < model.n:
        return float(model.config.get("direct-bandwidth", model.config["bandwidth"][0])), float(
            model.config.get("direct-latency", model.config["latency"][0])
        )
    bandwidths = [float(item) for item in model.config["bandwidth"]]
    latencies = [float(item) for item in model.config["latency"]]
    if model.topology in {"mesh", "torus", "fullmesh"}:
        dim = direct_edge_dim(model, src, dst)
    elif model.topology == "fat-tree":
        one_gpu = (src < model.n) != (dst < model.n)
        switch = dst if src < model.n else src
        dim = 0 if one_gpu and switch != global_switch(model) else min(1, len(bandwidths) - 1)
    elif model.topology in {"cm", "cm384"}:
        one_gpu = (src < model.n) != (dst < model.n)
        dim = 0 if one_gpu else min(1, len(bandwidths) - 1)
    else:
        dim = 0
    dim = min(dim, len(bandwidths) - 1)
    return bandwidths[dim], latencies[dim]


def edge_time_us(model: TopologyModel, src: int, dst: int, bytes_count: int) -> float:
    bandwidth_gib_s, latency_ns = edge_profile(model, src, dst)
    bandwidth_bytes_per_us = bandwidth_gib_s * (1 << 30) / 1e6
    return latency_ns / 1000.0 + bytes_count / bandwidth_bytes_per_us


def schedule_path_generic(
    model: TopologyModel,
    slots: dict[tuple[int, int], list[tuple[float, float]]],
    path: list[int],
    bytes_count: int,
    earliest: float,
    flow_src: int,
    flow_dst: int,
    chunk_id: int,
) -> list[dict[str, Any]]:
    ready = earliest
    events: list[dict[str, Any]] = []
    path_text = ">".join(str(node) for node in path)
    for src, dst in zip(path, path[1:]):
        duration = edge_time_us(model, src, dst, bytes_count)
        start, end = base.first_gap(slots[(src, dst)], ready, duration)
        slots[(src, dst)].append((start, end))
        slots[(src, dst)].sort()
        events.append(
            {
                "class": "cold",
                "link_src": src,
                "link_dst": dst,
                "flow_src": flow_src,
                "flow_dst": flow_dst,
                "chunk_id": chunk_id,
                "bytes": bytes_count,
                "start_us": start,
                "end_us": end,
                "path": path_text,
            }
        )
        ready = end
    return events


def schedule_cold_pairwise_generic(
    model: TopologyModel,
    cold_matrix: list[list[int]],
    hot_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = len(cold_matrix)
    slots = base.initial_hot_slots(hot_events)
    current_time = 0.0
    all_events: list[dict[str, Any]] = []
    next_chunk_id = 0
    for round_index in range(n - 1):
        step_start = current_time
        step_end = step_start
        xor_distance = round_index + 1
        transfers: list[tuple[int, int, int]] = []
        for src in range(n):
            dst = src ^ xor_distance
            if dst >= n or src >= dst:
                continue
            if cold_matrix[src][dst] > 0:
                transfers.append((src, dst, cold_matrix[src][dst]))
            if cold_matrix[dst][src] > 0:
                transfers.append((dst, src, cold_matrix[dst][src]))
        for src, dst, bytes_count in transfers:
            path = topology_path(model, src, dst)
            events = schedule_path_generic(model, slots, path, bytes_count, step_start, src, dst, next_chunk_id)
            next_chunk_id += 1
            all_events.extend(events)
            if events:
                step_end = max(step_end, float(events[-1]["end_us"]))
        current_time = step_end
    return all_events


def schedule_cold_biring_generic(
    model: TopologyModel,
    cold_matrix: list[list[int]],
    hot_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = len(cold_matrix)
    slots = base.initial_hot_slots(hot_events)
    clockwise: list[list[dict[str, int]]] = [[] for _ in range(n)]
    counter: list[list[dict[str, int]]] = [[] for _ in range(n)]
    max_rounds = 0
    for src in range(n):
        for dst in range(n):
            bytes_count = int(cold_matrix[src][dst])
            if src == dst or bytes_count <= 0:
                continue
            cw_hops = (dst - src + n) % n
            ccw_hops = (src - dst + n) % n
            use_cw = cw_hops < ccw_hops or (cw_hops == ccw_hops and src % 2 == 0)
            hops = cw_hops if use_cw else ccw_hops
            max_rounds = max(max_rounds, hops)
            payload = {"final_dest": dst, "bytes": bytes_count, "remaining": hops}
            (clockwise if use_cw else counter)[src].append(payload)

    current_time = 0.0
    all_events: list[dict[str, Any]] = []
    next_chunk_id = 0
    for _round in range(max_rounds):
        step_start = current_time
        step_end = step_start
        transfers: list[tuple[int, int, int, bool, list[dict[str, int]]]] = []
        for node in range(n):
            if clockwise[node]:
                transfers.append((node, (node + 1) % n, sum(item["bytes"] for item in clockwise[node]), True, clockwise[node]))
            if counter[node]:
                transfers.append((node, (node - 1 + n) % n, sum(item["bytes"] for item in counter[node]), False, counter[node]))
        transfers.sort(key=lambda item: (item[0], not item[3], item[1]))
        next_clockwise: list[list[dict[str, int]]] = [[] for _ in range(n)]
        next_counter: list[list[dict[str, int]]] = [[] for _ in range(n)]
        for src, dst, bytes_count, is_clockwise, payloads in transfers:
            path = topology_path(model, src, dst)
            events = schedule_path_generic(model, slots, path, bytes_count, step_start, src, dst, next_chunk_id)
            next_chunk_id += 1
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


def event_span(events: list[dict[str, Any]], cls: str) -> tuple[float, float, float]:
    selected = [event for event in events if event["class"] == cls]
    if not selected:
        return 0.0, 0.0, 0.0
    start = min(float(event["start_us"]) for event in selected)
    end = max(float(event["end_us"]) for event in selected)
    busy = sum(float(event["end_us"]) - float(event["start_us"]) for event in selected)
    return start, end, busy


def active_breakdown_fast(events: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    deltas: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for event in events:
        start = round(float(event["start_us"]), 9)
        end = round(float(event["end_us"]), 9)
        if end <= start:
            continue
        index = 0 if event["class"] == "hot" else 1
        deltas[start][index] += 1
        deltas[end][index] -= 1
    hot_count = 0
    cold_count = 0
    hot_only = 0.0
    cold_only = 0.0
    overlap = 0.0
    active = 0.0
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


def cold_ring_method(topology_key: str) -> str:
    return "cold_halfringdr" if topology_key == "torus_tpuv4_4x4x4" else "cold_biring"


METHOD_DISPLAY = {
    "standard": "Glaive",
    "cold_biring": "BiRing",
    "cold_halfringdr": "HalfRingDR",
    "cold_mpibaseline": "MPICH",
}


def method_order_for_topology(topology_key: str) -> list[str]:
    return ["standard", cold_ring_method(topology_key), "cold_mpibaseline"]


def summarize_speed_method(
    topology_key: str,
    method: str,
    events: list[dict[str, Any]],
    metrics: dict[str, Any],
    log_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    hot_only, cold_only, overlap, active = active_breakdown_fast(events)
    makespan = max((float(event["end_us"]) for event in events), default=0.0)
    hot_start, hot_end, hot_busy = event_span(events, "hot")
    cold_start, cold_end, cold_busy = event_span(events, "cold")
    hot_solver = float(metrics.get("speed_hot_solver_time_us", 0.0) or 0.0)
    cold_solver = float(metrics.get("speed_cold_solver_time_us", 0.0) or 0.0) if method == "standard" else 0.0
    full_solver = (
        float(metrics.get("speed_full_solver_time_us", metrics.get("solver_time_us", 0.0)) or 0.0)
        if method == "standard"
        else hot_solver
    )
    try:
        log_path_value = log_path.relative_to(repo_root).as_posix()
    except ValueError:
        # Breakdown logs may be generated under the run output root rather
        # than the immutable source tree.  Keep the absolute path in that
        # case so the provenance remains inspectable instead of failing the
        # entire post-processing step.
        log_path_value = str(log_path)
    return {
        "topology_key": topology_key,
        "method": method,
        "method_label": METHOD_DISPLAY[method],
        "full_solver_time_us": full_solver,
        "hot_solver_time_us": hot_solver,
        "cold_solver_time_us": cold_solver,
        "makespan_us": makespan,
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
        "log_path": log_path_value,
    }


def append_speed_events(
    event_rows: list[dict[str, Any]],
    topology_key: str,
    method: str,
    events: list[dict[str, Any]],
) -> None:
    for event in events:
        event_rows.append(
            {
                "topology_key": topology_key,
                "method": method,
                "method_label": METHOD_DISPLAY[method],
                "class": event["class"],
                "link_src": event["link_src"],
                "link_dst": event["link_dst"],
                "flow_src": event.get("flow_src", ""),
                "flow_dst": event.get("flow_dst", ""),
                "chunk_id": event.get("chunk_id", ""),
                "bytes": event["bytes"],
                "start_us": event["start_us"],
                "end_us": event["end_us"],
                "path": event.get("path", ""),
            }
        )


def build_speed_link_rows(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for task in speed_link_tasks(repo_root):
        topology_key = task.log_path.stem.replace("_layer1_group0_256MB", "")
        metrics = base.parse_log_metrics(task.log_path)
        standard_events = base.parse_speed_events(task.log_path)
        if not standard_events:
            continue
        topo = base.load_json(repo_root / task.topology_json)
        model = build_topology_model(topology_key, topo)
        hot_events = [event for event in standard_events if event["class"] == "hot"]
        cold_matrix = base.cold_matrix_from_events(standard_events, model.n)
        ring_method = cold_ring_method(topology_key)
        methods = {
            "standard": [event for event in standard_events if event["class"] in {"hot", "cold"}],
            ring_method: [*hot_events, *schedule_cold_biring_generic(model, cold_matrix, hot_events)],
            "cold_mpibaseline": [*hot_events, *schedule_cold_pairwise_generic(model, cold_matrix, hot_events)],
        }
        for method in method_order_for_topology(topology_key):
            events = methods[method]
            summary_rows.append(summarize_speed_method(topology_key, method, events, metrics, task.log_path, repo_root))
            append_speed_events(event_rows, topology_key, method, events)
    return summary_rows, event_rows


def choose_time_unit(max_us: float) -> tuple[float, str]:
    if max_us >= 1000:
        return 1000.0, "ms"
    return 1.0, "us"


def plot_speed_link(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    by_key = {(row["topology_key"], row["method"]): row for row in summary_rows}
    rows: list[dict[str, Any]] = []
    x_positions: list[float] = []
    group_centers: list[tuple[float, str]] = []
    group_boundaries: list[float] = []
    position = 0.0
    for topology_key in LINK_BREAKDOWN_TOPOLOGIES:
        start_position = position
        for method in method_order_for_topology(topology_key):
            row = by_key.get((topology_key, method))
            if row is None:
                continue
            rows.append(row)
            x_positions.append(position)
            position += 1.0
        end_position = position - 1.0
        if end_position >= start_position:
            group_centers.append(((start_position + end_position) / 2.0, topology_key))
            group_boundaries.append(position - 0.5)
        position += 0.65

    if not rows:
        return
    x = np.array(x_positions)
    hot = np.array([float(row["hot_only_active_us"]) for row in rows])
    overlap = np.array([float(row["overlap_us"]) for row in rows])
    cold = np.array([float(row["cold_only_active_us"]) for row in rows])
    total = hot + overlap + cold
    max_us = float(np.max(total)) if len(rows) else 1.0
    divisor, unit = choose_time_unit(max_us)

    fig, ax = plt.subplots(figsize=(15.8, 6.4))
    width = 0.72
    ax.bar(x, hot / divisor, width=width, label="Hot-only", color="#06d6a0")
    ax.bar(x, overlap / divisor, width=width, bottom=hot / divisor, label="Overlap", color="#f77f00")
    ax.bar(x, cold / divisor, width=width, bottom=(hot + overlap) / divisor, label="Cold-only", color="#118ab2")
    ax.set_ylabel(f"Active Makespan Breakdown ({unit})", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[row["method"]] for row in rows], rotation=28, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=12)
    for boundary in group_boundaries[:-1]:
        ax.axvline(boundary, color="#b8b8b8", linewidth=0.8, linestyle=":", alpha=0.8)
    for center, topology_key in group_centers:
        ax.text(center, -0.22, TOPOLOGY_DISPLAY.get(topology_key, topology_key), transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)
    for idx, row in enumerate(rows):
        active = float(row["active_us"])
        if active <= 0:
            continue
        overlap_pct = 100.0 * float(row["overlap_us"]) / active
        ax.text(x[idx], total[idx] / divisor, f"{overlap_pct:.0f}%", ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(bottom=0.25, top=0.88, left=0.08, right=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_speed_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Speed Hot/Cold Overlap Analysis", ""]
    lines.append("The table expands each of the four topologies from the combined `Link_Utilization.pdf` panel into three cold-traffic policies, yielding `4 x 3 = 12` stacked bars. All policies share the same Glaive hot events; baseline rows replace only cold-traffic scheduling to isolate its effect on active makespan.")
    lines.append("")
    lines.append("The `Glaive` row uses the original Speed/Glaive cold scheduler. The ring baseline is `BiRing` for Mesh, Clos, and CM384, and `HalfRingDR` for Torus. The `MPICH` row uses pairwise/XOR-round cold scheduling.")
    lines.append("")
    lines.append("| Topology | Cold policy | makespan(us) | hot-only(us) | overlap(us) | cold-only(us) | hot span(us) | cold span(us) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    by_key = {(row["topology_key"], row["method"]): row for row in summary_rows}
    for topology_key in LINK_BREAKDOWN_TOPOLOGIES:
        for method in method_order_for_topology(topology_key):
            row = by_key.get((topology_key, method))
            if row is None:
                continue
            lines.append(
                f"| {TOPOLOGY_DISPLAY.get(row['topology_key'], row['topology_key'])} | {METHOD_DISPLAY[row['method']]} | {float(row['makespan_us']):.2f} | "
                f"{float(row['hot_only_active_us']):.2f} | {float(row['overlap_us']):.2f} | {float(row['cold_only_active_us']):.2f} | "
                f"{float(row['hot_span_start_us']):.2f}-{float(row['hot_span_end_us']):.2f} | "
                f"{float(row['cold_span_start_us']):.2f}-{float(row['cold_span_end_us']):.2f} |"
            )
    lines.append("")
    lines.append("Baseline cold scheduling is a replay analysis over demand aggregated from the original Glaive cold events. Hot-link occupancy windows are unchanged, so the figure isolates how cold-policy choices alter hot/cold overlap under the same hotspot algorithm.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def figure9_reference_rows(repo_root: Path) -> list[dict[str, str]]:
    rows = read_csv(repo_root / FIGURE9_REFERENCE)
    if not rows:
        raise FileNotFoundError(f"missing Figure 9 reference CSV: {repo_root / FIGURE9_REFERENCE}")
    return rows


def figure9_tasks(repo_root: Path) -> list[Task]:
    rows = figure9_reference_rows(repo_root)
    tasks: list[Task] = []
    for row in rows:
        devices = int(row["target_devices"])
        topology_json = "input/topology/h100_1node.json" if devices == 8 else "input/topology/h100_2node.json"
        collective_json = f"{FIGURE9_NAMESPACE}/collective/{row['case_id']}_{devices}devices.json"
        tasks.append(
            Task(
                name=f"figure9_standard::{devices}::{row['case_id']}",
                topology_json=topology_json,
                collective_json=collective_json,
                args=("--solver", "mode=standard"),
                log_path=repo_root / "evaluation_assets" / "raw_logs" / "speed_breakeven" / "figure9_h100_clos" / f"{devices}devices" / f"{row['case_id']}.log",
                stage="speed_breakeven",
            )
        )
    return tasks


def build_figure9_breakeven_rows(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    refs = figure9_reference_rows(repo_root)
    ref_by_key = {(int(row["target_devices"]), row["case_id"]): row for row in refs}
    for task in figure9_tasks(repo_root):
        _, devices_s, case_id = task.name.split("::")
        devices = int(devices_s)
        ref = ref_by_key[(devices, case_id)]
        metrics = base.parse_log_metrics(task.log_path)
        solver_time = metrics.get("solver_time_us")
        if solver_time is None:
            continue
        real_makespan_us = float(ref["real_makespan_us"])
        total_with_solver = float(solver_time) + real_makespan_us
        solver_share = 100.0 * float(solver_time) / total_with_solver if total_with_solver > 0 else 0.0
        rows.append(
            {
                "case_id": case_id,
                "phase": ref["phase"],
                "nominal_batch_size": int(ref["nominal_batch_size"]),
                "target_devices": devices,
                "nodes": int(ref["nodes"]),
                "topology_key": "h100_1node_railonly" if devices == 8 else "h100_2node_railonly",
                "chunk_label": ref["chunk_label"],
                "real_makespan_us": real_makespan_us,
                "real_total_us": float(ref["real_total_us"]),
                "standard_solver_time_us": float(solver_time),
                "standard_sim_makespan_us": metrics.get("total_makespan_us"),
                "total_with_solver_us": total_with_solver,
                "solver_share_pct": solver_share,
                "runtime_share_pct": 100.0 - solver_share,
                "algorithm_bytes_per_rank": int(float(ref["algorithm_bytes_per_rank"])),
                "demand_fingerprint": metrics.get("demand_fingerprint", ""),
                "log_path": task.log_path.relative_to(repo_root).as_posix(),
            }
        )
    return sorted(rows, key=lambda r: (int(r["target_devices"]), str(r["phase"]), int(r["nominal_batch_size"])))


def figure9_case_label(row: dict[str, Any]) -> str:
    prefix = "D" if row["phase"] == "decode" else "P"
    return f"{prefix}{row['nominal_batch_size']}"


def plot_figure9_solver_share(rows: list[dict[str, Any]], devices: int, output_path: Path) -> None:
    selected = [row for row in rows if int(row["target_devices"]) == devices]
    if not selected:
        return
    x = np.arange(len(selected))
    runtime = np.array([float(row["runtime_share_pct"]) for row in selected])
    solver = np.array([float(row["solver_share_pct"]) for row in selected])
    labels = [figure9_case_label(row) for row in selected]
    fig, ax = plt.subplots(figsize=(max(12.0, 0.72 * len(selected)), 5.4))
    ax.bar(x, runtime, label="Measured makespan", color="#118ab2")
    ax.bar(x, solver, bottom=runtime, label="Standard solve", color="#f77f00")
    for idx, pct in enumerate(solver):
        ax.text(idx, min(99.0, runtime[idx] + solver[idx] + 1.0), f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, rotation=90)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Share of solve + measured makespan", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_title(f"H100 {devices}-GPU Figure 9 Cases", fontsize=15)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_figure9_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Figure 9 H100 Standard Solver Share", ""]
    lines.append("The measured-system makespan uses `avg_collective_ms` from the Figure 9 bandwidth data, which is the collective time used to compute bandwidth. `real_total_us` additionally retains `avg_ms`, including pack/unpack.")
    lines.append("")
    for devices in (8, 16):
        selected = [row for row in rows if int(row["target_devices"]) == devices]
        if not selected:
            continue
        lines.append(f"## {devices} GPU")
        lines.append("")
        lines.append("| case | real makespan(us) | standard solve(us) | solve share |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in selected:
            lines.append(
                f"| {figure9_case_label(row)} `{row['case_id']}` | {float(row['real_makespan_us']):.2f} | "
                f"{float(row['standard_solver_time_us']):.2f} | {float(row['solver_share_pct']):.2f}% |"
            )
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def selected_tasks(repo_root: Path, stage: str) -> list[Task]:
    tasks: list[Task] = []
    if stage in {"all", "scalability"}:
        tasks.extend(base.scalability_tasks(repo_root))
    if stage in {"all", "cache"}:
        tasks.extend(cache_scale_tasks(repo_root))
        tasks.extend(cache_data_volume_tasks(repo_root))
    if stage in {"all", "speed"}:
        tasks.extend(speed_link_tasks(repo_root))
    if stage in {"all", "figure9"}:
        tasks.extend(figure9_tasks(repo_root))
    return tasks


def parse_and_write(repo_root: Path) -> dict[str, int]:
    parsed_dir = repo_root / "evaluation_assets" / "parsed"
    plot_dir = repo_root / "evaluation_assets" / "plots"

    scalability_rows = base.build_scalability_rows(repo_root)
    write_csv(parsed_dir / "standard_scalability_raw.csv", scalability_rows)
    if scalability_rows:
        base.plot_scalability(scalability_rows, plot_dir / "Scalability_Study_Standard.pdf")

    scalability_cache_rows = build_scalability_cache_rows(repo_root)
    cache_scale_rows = build_cache_scale_rows(repo_root)
    cache_data_volume = build_cache_data_volume_rows(repo_root)
    write_csv(parsed_dir / "standard_scalability_cache.csv", scalability_cache_rows)
    write_csv(parsed_dir / "standard_cache_scale.csv", cache_scale_rows)
    write_csv(parsed_dir / "standard_cache_data_volume.csv", cache_data_volume)
    precompute_cache_rows = build_precompute_cache_rows([*scalability_cache_rows, *cache_scale_rows, *cache_data_volume])
    write_csv(parsed_dir / "standard_static_precompute_cache.csv", precompute_cache_rows)
    write_cache_markdown(parsed_dir / "standard_cache_overhead.md", scalability_cache_rows, cache_scale_rows, cache_data_volume)

    speed_summary, speed_events = build_speed_link_rows(repo_root)
    write_csv(parsed_dir / "speed_clos_summary.csv", speed_summary)
    write_csv(parsed_dir / "speed_clos_events.csv", speed_events)
    write_csv(parsed_dir / "speed_link_summary.csv", speed_summary)
    write_csv(parsed_dir / "speed_link_events.csv", speed_events)
    if speed_summary:
        plot_speed_link(speed_summary, plot_dir / "Speed_Clos_Breakdown.pdf")
        plot_speed_link(speed_summary, plot_dir / "Speed_Link_Breakdown.pdf")
        write_speed_markdown(parsed_dir / "speed_link_breakdown.md", speed_summary)

    figure9_rows = build_figure9_breakeven_rows(repo_root)
    write_csv(parsed_dir / "speed_breakeven.csv", figure9_rows)
    write_figure9_markdown(parsed_dir / "speed_breakeven.md", figure9_rows)
    plot_figure9_solver_share(figure9_rows, 8, plot_dir / "Speed_Breakeven_8GPU.pdf")
    plot_figure9_solver_share(figure9_rows, 16, plot_dir / "Speed_Breakeven_16GPU.pdf")

    return {
        "scalability_rows": len(scalability_rows),
        "scalability_cache_rows": len(scalability_cache_rows),
        "cache_scale_rows": len(cache_scale_rows),
        "cache_data_volume_rows": len(cache_data_volume),
        "speed_summary_rows": len(speed_summary),
        "speed_event_rows": len(speed_events),
        "figure9_breakeven_rows": len(figure9_rows),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.stage != "parse-only":
        tasks = selected_tasks(repo_root, args.stage)
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "task_count": len(tasks),
                    "max_workers": args.max_workers,
                    "force": args.force,
                },
                indent=2,
            )
        )
        run_tasks(repo_root, tasks, args.force, args.max_workers)
    result = parse_and_write(repo_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
