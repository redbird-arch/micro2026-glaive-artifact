#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


METHOD_DISPLAY = {
    "glaive": "Glaive",
    "biring": "BiRing",
    "halfringdr": "HalfR+DR",
    "mpibaseline": "MPICH",
}
MAKESPAN_RE = re.compile(r"^\s*(?:Total Makespan|Makespan):\s*([0-9.]+)\s*us\b", re.MULTILINE)
SOLVER_RE = re.compile(r"^\s*(?:Solver Time|Algorithm Time):\s*([0-9.]+)\s*us\b", re.MULTILINE)
BASELINE_RESULT_HEADINGS = {
    "biring": "BiRing Algorithm Results",
    "halfringdr": "HalfRing+DimRotation Algorithm Results",
    "mpibaseline": "MPI Baseline (Pairwise Exchange) Results",
}
LINK_RE = re.compile(r"Link\((.+?)->(.+?)\): intervals=\[(.*)\], utilization=([0-9.]+)\s*%")
INTERVAL_RE = re.compile(r"\[(\d+),\s*(\d+)\]")


def selected_topologies_for_case(target_devices: int) -> list[str]:
    if target_devices == 64:
        return ["torus_tpuv4_4x4x4"]
    return [
        "mesh_nebula_8x4",
        "cm384_16x2_eval",
        "fattree_8x4_eval",
    ]


def synthetic_methods_for_topology(topology_key: str) -> list[str]:
    if topology_key == "torus_tpuv4_4x4x4":
        return ["glaive", "halfringdr", "mpibaseline"]
    return ["glaive", "biring", "mpibaseline"]


def link_methods_for_topology(topology_key: str) -> tuple[str, ...]:
    if topology_key == "torus_tpuv4_4x4x4":
        return ("glaive", "halfringdr", "mpibaseline")
    return ("glaive", "biring", "mpibaseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse evaluation logs into CSV summaries.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", choices=["all", "synthetic", "link", "scalability"], default="all")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def parse_log_metrics(path: Path, method: str = "") -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    text = path.read_text(errors="replace")
    if method in BASELINE_RESULT_HEADINGS:
        heading = BASELINE_RESULT_HEADINGS[method]
        block = re.search(
            rf"\[{re.escape(heading)}\](.*?)(?=\n\[|\Z)",
            text,
            flags=re.DOTALL,
        )
        block_text = block.group(1) if block else ""
        makespan_match = re.search(r"^\s*Makespan:\s*([0-9.]+)\s*us\b", block_text, re.MULTILINE)
        solver_match = re.search(r"^\s*Algorithm Time:\s*([0-9.]+)\s*us\b", block_text, re.MULTILINE)
    else:
        makespan_match = MAKESPAN_RE.search(text)
        solver_match = SOLVER_RE.search(text)
    makespan = float(makespan_match.group(1)) if makespan_match else None
    solver_time = float(solver_match.group(1)) if solver_match else None
    return makespan, solver_time


def parse_link_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    text = path.read_text(errors="replace")
    intervals: dict[str, list[tuple[int, int]]] = {}
    for line in text.splitlines():
        match = LINK_RE.search(line)
        if not match:
            continue
        edge_key = f"{match.group(1)}->{match.group(2)}"
        edge_intervals = [(int(start), int(end)) for start, end in INTERVAL_RE.findall(match.group(3))]
        intervals[edge_key] = edge_intervals
    return intervals


def topology_directed_links(config: dict[str, object]) -> int:
    topology = str(config["topology"])
    shape = [int(v) for v in config["shape"]]
    if topology == "mesh":
        total = 0
        for dim, dim_size in enumerate(shape):
            other = math.prod(shape) // dim_size
            total += other * (dim_size - 1)
        return total * 2
    if topology == "torus":
        total = 0
        for dim, dim_size in enumerate(shape):
            other = math.prod(shape) // dim_size
            total += other * dim_size
        return total * 2
    if topology == "fullmesh":
        total = 0
        for dim, dim_size in enumerate(shape):
            other = math.prod(shape) // dim_size
            total += other * (dim_size * (dim_size - 1) // 2)
        return total * 2
    if topology == "cm384":
        gpus_per_node, num_nodes = shape
        switches_per_node, switches_per_rail = [int(v) for v in config["switch-shape"]]
        direct_pairs = num_nodes * (gpus_per_node // 2)
        tier1_pairs = num_nodes * gpus_per_node * switches_per_node
        tier2_pairs = num_nodes * switches_per_node * switches_per_rail
        return 2 * (direct_pairs + tier1_pairs + tier2_pairs)
    if topology == "rail-optimized":
        gpus_per_node, num_nodes = shape
        switch_shape = [int(v) for v in config["switch-shape"]]
        scale_up_pairs = num_nodes * gpus_per_node * switch_shape[0]
        gpu_to_rail_pairs = num_nodes * gpus_per_node
        layer_pairs = 0
        if len(switch_shape) > 1:
            current = switch_shape[1]
            for next_count in switch_shape[2:]:
                layer_pairs += current * next_count
                current = next_count
        return 2 * (scale_up_pairs + gpu_to_rail_pairs + layer_pairs)
    if topology == "fat-tree":
        gpus_per_node, num_nodes = shape
        switch_shape = [int(v) for v in config["switch-shape"]]
        scale_up_pairs = num_nodes * gpus_per_node * switch_shape[0]
        scale_out_pairs = 0
        if len(switch_shape) > 1:
            scale_out_pairs += gpus_per_node * num_nodes * switch_shape[1]
            current = switch_shape[1]
            for next_count in switch_shape[2:]:
                scale_out_pairs += current * next_count
                current = next_count
        return 2 * (scale_up_pairs + scale_out_pairs)
    raise ValueError(f"Unsupported topology for link counting: {topology}")


def build_synthetic_rows(repo_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    topo_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json")
    case_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.json")
    topologies = {item["topology_key"]: item for item in topo_manifest["topologies"]}

    raw_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)

    for case in case_manifest["cases"]:
        for topology_key in selected_topologies_for_case(int(case["target_devices"])):
            topo = topologies[topology_key]
            for method in synthetic_methods_for_topology(topology_key):
                log_path = (
                    repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "synthetic"
                    / topology_key
                    / method
                    / case["size_label"]
                    / f"sample{case['sample_index']}.log"
                )
                makespan_us, solver_time_us = parse_log_metrics(log_path, method)
                if makespan_us is None:
                    continue
                bandwidth_gbps = float(case["nominal_bytes_per_rank"]) / makespan_us / 1000.0
                row = {
                    "topology_key": topology_key,
                    "topology_type": topo["topology_type"],
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "size_label": case["size_label"],
                    "sample_index": case["sample_index"],
                    "case_id": case["case_id"],
                    "target_devices": case["target_devices"],
                    "nominal_bytes_per_rank": case["nominal_bytes_per_rank"],
                    "avg_row_bytes": case["avg_row_bytes"],
                    "total_bytes": case["total_bytes"],
                    "makespan_us": makespan_us,
                    "solver_time_us": solver_time_us,
                    "bandwidth_gbps": bandwidth_gbps,
                    "log_path": log_path.relative_to(repo_root).as_posix(),
                }
                raw_rows.append(row)
                grouped[(topology_key, method, case["size_label"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for (topology_key, method, size_label), rows in sorted(grouped.items()):
        topo = topologies[topology_key]
        summary_rows.append(
            {
                "topology_key": topology_key,
                "topology_type": topo["topology_type"],
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "size_label": size_label,
                "sample_count": len(rows),
                "avg_makespan_us": mean(float(row["makespan_us"]) for row in rows),
                "avg_solver_time_us": mean(float(row["solver_time_us"]) for row in rows if row["solver_time_us"] is not None),
                "avg_bandwidth_gbps": mean(float(row["bandwidth_gbps"]) for row in rows),
                "nominal_bytes_per_rank": rows[0]["nominal_bytes_per_rank"],
            }
        )

    return raw_rows, summary_rows


def link_segments(intervals: dict[str, list[tuple[int, int]]], total_links: int) -> list[tuple[int, int, float]]:
    boundaries = sorted({point for edge_intervals in intervals.values() for interval in edge_intervals for point in interval})
    segments: list[tuple[int, int, float]] = []
    if len(boundaries) < 2:
        return segments

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        active = 0
        for edge_intervals in intervals.values():
            if any(interval_start <= start and interval_end >= end for interval_start, interval_end in edge_intervals):
                active += 1
        segments.append((start, end, active / total_links if total_links else 0.0))
    return segments


def sample_segments(segments: list[tuple[int, int, float]], time_ns: float) -> float:
    for start, end, util in segments:
        if start <= time_ns < end:
            return util
    return 0.0


def average_lifecycle_utilization_pct(segments: list[tuple[int, int, float]], makespan_us: float) -> float:
    makespan_ns = makespan_us * 1000.0
    if makespan_ns <= 0:
        return 0.0
    weighted = 0.0
    for start_ns, end_ns, util in segments:
        clipped_start = max(0.0, float(start_ns))
        clipped_end = min(makespan_ns, float(end_ns))
        if clipped_end > clipped_start:
            weighted += (clipped_end - clipped_start) * util
    return 100.0 * weighted / makespan_ns


def build_link_rows(repo_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    topo_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json")
    case_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.json")
    topologies = {item["topology_key"]: item for item in topo_manifest["topologies"]}
    topology_configs = {
        key: load_json(repo_root / item["topology_json"]) for key, item in topologies.items()
    }

    raw_rows: list[dict[str, object]] = []
    grouped_runs: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    for case in case_manifest["cases"]:
        if case["size_label"] != "256MB":
            continue

        for topology_key in selected_topologies_for_case(int(case["target_devices"])):
            total_links = topology_directed_links(topology_configs[topology_key])
            for method in link_methods_for_topology(topology_key):
                log_path = (
                    repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "link"
                    / topology_key
                    / method
                    / f"sample{case['sample_index']}.log"
                )
                if not log_path.exists():
                    continue
                makespan_us, solver_time_us = parse_log_metrics(log_path)
                if makespan_us is None:
                    continue
                intervals = parse_link_intervals(log_path)
                segments = link_segments(intervals, total_links)
                lifecycle_avg_utilization_pct = average_lifecycle_utilization_pct(segments, makespan_us)
                row = {
                    "topology_key": topology_key,
                    "method": method,
                    "method_display": METHOD_DISPLAY[method],
                    "sample_index": case["sample_index"],
                    "case_id": case["case_id"],
                    "makespan_us": makespan_us,
                    "solver_time_us": solver_time_us,
                    "total_directed_links": total_links,
                    "lifecycle_avg_utilization_pct": lifecycle_avg_utilization_pct,
                    "segments": segments,
                    "log_path": log_path.relative_to(repo_root).as_posix(),
                }
                raw_rows.append(row)
                grouped_runs[(topology_key, method)].append(row)

    reference_makespan_ns = {
        topology_key: mean(run["makespan_us"] * 1000.0 for run in runs)
        for (topology_key, method), runs in grouped_runs.items()
        if method == "glaive"
    }
    lifecycle_summary = {
        (topology_key, method): mean(float(run["lifecycle_avg_utilization_pct"]) for run in runs)
        for (topology_key, method), runs in grouped_runs.items()
    }

    trace_rows: list[dict[str, object]] = []
    for topology_key in sorted({key for key, _ in grouped_runs}):
        ref_ns = reference_makespan_ns[topology_key]
        max_norm = max(
            (run["makespan_us"] * 1000.0) / ref_ns
            for (key, _), runs in grouped_runs.items()
            if key == topology_key
            for run in runs
        )
        grid = [idx * max_norm / 250.0 for idx in range(251)]
        for method in link_methods_for_topology(topology_key):
            runs = grouped_runs.get((topology_key, method), [])
            if not runs:
                continue
            for x in grid:
                time_ns = x * ref_ns
                utilization = mean(sample_segments(run["segments"], time_ns) for run in runs)
                trace_rows.append(
                    {
                        "topology_key": topology_key,
                        "method": method,
                        "method_display": METHOD_DISPLAY[method],
                        "normalized_time": x,
                        "avg_utilization": utilization,
                        "avg_utilization_pct": utilization * 100.0,
                        "lifecycle_avg_utilization_pct": lifecycle_summary[(topology_key, method)],
                        "reference_makespan_us": ref_ns / 1000.0,
                    }
                )

    summary_rows: list[dict[str, object]] = []
    for (topology_key, method), runs in sorted(grouped_runs.items()):
        summary_rows.append(
            {
                "topology_key": topology_key,
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "sample_count": len(runs),
                "avg_makespan_us": mean(float(run["makespan_us"]) for run in runs),
                "avg_solver_time_us": mean(
                    float(run["solver_time_us"]) for run in runs if run["solver_time_us"] is not None
                ),
                "avg_lifecycle_utilization_pct": lifecycle_summary[(topology_key, method)],
            }
        )

    raw_segment_rows: list[dict[str, object]] = []
    for row in raw_rows:
        for start_ns, end_ns, util in row["segments"]:
            raw_segment_rows.append(
                {
                    "topology_key": row["topology_key"],
                    "method": row["method"],
                    "method_display": row["method_display"],
                    "sample_index": row["sample_index"],
                    "segment_start_ns": start_ns,
                    "segment_end_ns": end_ns,
                    "avg_utilization": util,
                    "avg_utilization_pct": util * 100.0,
                    "lifecycle_avg_utilization_pct": row["lifecycle_avg_utilization_pct"],
                    "total_directed_links": row["total_directed_links"],
                    "makespan_us": row["makespan_us"],
                    "log_path": row["log_path"],
                }
            )

    return raw_segment_rows, trace_rows, summary_rows


def build_scalability_rows(repo_root: Path) -> list[dict[str, object]]:
    manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "scalability_cases.json")
    rows: list[dict[str, object]] = []
    for case in manifest["cases"]:
        # Keep workload-specific log stems strict. Older far-end logs can coexist
        # under the bare topology stem and must not be mixed into this rerun.
        log_stem = case.get("log_stem", Path(case["topology_json"]).stem)
        log_path = (
            repo_root
            / "evaluation_assets"
            / "raw_logs"
            / "scalability"
            / case["topology_type"]
            / f"{log_stem}.log"
        )
        makespan_us, solver_time_us = parse_log_metrics(log_path)
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
                "solver_time_s": solver_time_us / 1e6,
                "makespan_us": makespan_us,
                "topology_json": case["topology_json"],
                "collective_json": case["collective_json"],
                "log_path": log_path.relative_to(repo_root).as_posix(),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    synthetic_raw: list[dict[str, object]] = []
    synthetic_summary: list[dict[str, object]] = []
    link_raw: list[dict[str, object]] = []
    link_trace: list[dict[str, object]] = []
    link_summary: list[dict[str, object]] = []
    scalability_rows: list[dict[str, object]] = []

    if args.stage in {"all", "synthetic"}:
        synthetic_raw, synthetic_summary = build_synthetic_rows(repo_root)
        write_csv(repo_root / "evaluation_assets" / "parsed" / "synthetic_raw.csv", synthetic_raw)
        write_csv(repo_root / "evaluation_assets" / "parsed" / "synthetic_summary.csv", synthetic_summary)

    if args.stage in {"all", "link"}:
        link_raw, link_trace, link_summary = build_link_rows(repo_root)
        write_csv(repo_root / "evaluation_assets" / "parsed" / "link_utilization_raw.csv", link_raw)
        write_csv(repo_root / "evaluation_assets" / "parsed" / "link_utilization_trace.csv", link_trace)
        write_csv(repo_root / "evaluation_assets" / "parsed" / "link_utilization_summary.csv", link_summary)

    if args.stage in {"all", "scalability"}:
        scalability_rows = build_scalability_rows(repo_root)
        write_csv(repo_root / "evaluation_assets" / "parsed" / "scalability_raw.csv", scalability_rows)

    print(
        json.dumps(
            {
                "stage": args.stage,
                "synthetic_raw_rows": len(synthetic_raw),
                "synthetic_summary_rows": len(synthetic_summary),
                "link_raw_rows": len(link_raw),
                "link_trace_rows": len(link_trace),
                "link_summary_rows": len(link_summary),
                "scalability_rows": len(scalability_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
