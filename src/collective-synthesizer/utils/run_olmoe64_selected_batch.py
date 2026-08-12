#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMBER = r"([0-9.eE+-]+)"
TOTAL_MAKESPAN_RE = re.compile(rf"Total Makespan:\s*{NUMBER}\s*us")
SOLVER_TIME_RE = re.compile(rf"Solver Time:\s*{NUMBER}\s*us")
BASELINE_PATTERNS = {
    "bruck_makespan_us": re.compile(rf"Bruck Makespan:\s*{NUMBER}\s*us"),
    "spreadout_makespan_us": re.compile(rf"Spreadout Makespan:\s*{NUMBER}\s*us"),
    "pairwise_makespan_us": re.compile(rf"Pairwise Makespan:\s*{NUMBER}\s*us"),
    "biring_makespan_us": re.compile(rf"BiRing Makespan:\s*{NUMBER}\s*us"),
    "halfrdr_makespan_us": re.compile(
        rf"HalfRing\+DimRotation Makespan:\s*{NUMBER}\s*us"
    ),
    "reported_mpi_makespan_us": re.compile(
        rf"MPI Baseline Makespan:\s*{NUMBER}\s*us"
    ),
}
MPI_WINNER_RE = re.compile(r"MPI Baseline Winner:\s*(.+?)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Glaive clean mode and the communication baselines for a "
            "Figure 12 collective manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collective-root", type=Path, default=Path("."))
    parser.add_argument("--solver-bin", type=Path, default=Path("build/bin/tacos"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing-logs-root",
        type=Path,
        help="read logs from another result tree instead of invoking the solver",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun tasks whose existing logs already contain complete results",
    )
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def parse_glaive_log(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        makespan_match = TOTAL_MAKESPAN_RE.search(line)
        if makespan_match:
            metrics["glaive_makespan_us"] = float(makespan_match.group(1))
        solver_match = SOLVER_TIME_RE.search(line)
        if solver_match:
            metrics["glaive_solver_time_us"] = float(solver_match.group(1))
    return metrics


def parse_baseline_log(path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in path.read_text(errors="replace").splitlines():
        for metric_name, pattern in BASELINE_PATTERNS.items():
            match = pattern.search(line)
            if match:
                metrics[metric_name] = float(match.group(1))
        winner_match = MPI_WINNER_RE.search(line)
        if winner_match:
            metrics["reported_mpi_winner"] = winner_match.group(1).strip()

    classic_candidates = (
        ("Bruck", "bruck_makespan_us"),
        ("Spreadout", "spreadout_makespan_us"),
        ("Pairwise", "pairwise_makespan_us"),
    )
    available = [
        (name, float(metrics[key]))
        for name, key in classic_candidates
        if key in metrics
    ]
    if available:
        winner, makespan = min(available, key=lambda item: item[1])
        metrics["mpi_winner"] = winner
        metrics["mpi_makespan_us"] = makespan
    elif "reported_mpi_makespan_us" in metrics:
        metrics["mpi_makespan_us"] = metrics["reported_mpi_makespan_us"]
        if "reported_mpi_winner" in metrics:
            metrics["mpi_winner"] = metrics["reported_mpi_winner"]
    return metrics


def log_is_complete(path: Path, parser, required_metrics: tuple[str, ...]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    metrics = parser(path)
    return all(metric in metrics for metric in required_metrics)


def run_and_log(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def topology_baseline_method(topology_name: str) -> tuple[str, str]:
    if "torus" in topology_name:
        return ("halfrdr", "halfrdr_makespan_us")
    return ("biring", "biring_makespan_us")


def default_layer_id(layer: int, direction: str) -> str:
    suffix = "Dispatch" if direction == "dispatch" else "Combine"
    return f"OLMoEBlock{layer:02d}{suffix}"


def astra_layer_id(item: dict[str, Any]) -> str:
    direction = str(item["direction"])
    key = (
        "astra_dispatch_layer_name"
        if direction == "dispatch"
        else "astra_combine_layer_name"
    )
    return str(item.get(key) or default_layer_id(int(item["layer"]), direction))


def write_astra_external_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "phase", "duration_us"])
        for layer_name, duration_us in rows:
            writer.writerow([layer_name, "fwd", f"{duration_us:.6f}"])


def export_astra_csvs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    grouped: dict[
        tuple[str, str, str], list[tuple[int, int, str, float]]
    ] = defaultdict(list)
    for item in results:
        if item.get("status") != "ok":
            continue
        topology_name = str(item["topology_name"])
        phase = str(item["phase"])
        direction_order = 0 if item["direction"] == "dispatch" else 1
        baseline_method, baseline_metric = topology_baseline_method(topology_name)
        for method_name, metric_name in (
            ("glaive", "glaive_makespan_us"),
            (baseline_method, baseline_metric),
            ("mpi", "mpi_makespan_us"),
        ):
            grouped[(topology_name, phase, method_name)].append(
                (
                    int(item["layer"]),
                    direction_order,
                    astra_layer_id(item),
                    float(item[metric_name]),
                )
            )

    for (topology_name, phase, method_name), rows in grouped.items():
        rows.sort(key=lambda item: (item[0], item[1]))
        write_astra_external_csv(
            output_dir
            / "astra_external"
            / f"{topology_name}_{phase}_{method_name}.csv",
            [(layer_name, duration_us) for _, _, layer_name, duration_us in rows],
        )


def task_log_paths(
    root: Path,
    topology_name: str,
    phase: str,
    layer: int,
    case_stem: str,
) -> tuple[Path, Path]:
    case_dir = root / topology_name / phase / f"layer{layer}"
    return (
        case_dir / f"{case_stem}.glaive_clean.log",
        case_dir / f"{case_stem}.baselines.log",
    )


def main() -> None:
    args = parse_args()
    repo_root = args.collective_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    solver_bin = resolve_repo_path(repo_root, str(args.solver_bin))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing_logs_root is None and not solver_bin.is_file():
        raise FileNotFoundError(f"solver executable not found: {solver_bin}")

    log_root = (
        args.reuse_existing_logs_root.resolve()
        if args.reuse_existing_logs_root is not None
        else output_dir
    )
    results: list[dict[str, Any]] = []
    summary_path = output_dir / "summary.json"

    for topology_name, topology_relative in manifest["topologies"].items():
        topology_path = resolve_repo_path(repo_root, str(topology_relative))
        if not topology_path.is_file():
            raise FileNotFoundError(f"topology input not found: {topology_path}")
        baseline_method, baseline_metric = topology_baseline_method(topology_name)
        for case in manifest["cases"]:
            layer = int(case["layer"])
            phase = str(case["phase"])
            for direction, collective_key in (
                ("dispatch", "dispatch_collective"),
                ("combine", "combine_collective"),
            ):
                collective_path = resolve_repo_path(
                    repo_root, str(case[collective_key])
                )
                if not collective_path.is_file():
                    raise FileNotFoundError(
                        f"collective input not found: {collective_path}"
                    )
                case_stem = f"{phase}_layer{layer}_{direction}"
                solver_log, baseline_log = task_log_paths(
                    log_root, topology_name, phase, layer, case_stem
                )

                solver_complete = log_is_complete(
                    solver_log,
                    parse_glaive_log,
                    ("glaive_makespan_us", "glaive_solver_time_us"),
                )
                baseline_complete = log_is_complete(
                    baseline_log,
                    parse_baseline_log,
                    (baseline_metric, "mpi_makespan_us"),
                )
                solver_rc = 0 if solver_complete else 1
                baseline_rc = 0 if baseline_complete else 1

                if args.reuse_existing_logs_root is None:
                    output_solver_log, output_baseline_log = task_log_paths(
                        output_dir, topology_name, phase, layer, case_stem
                    )
                    solver_log = output_solver_log
                    baseline_log = output_baseline_log
                    if args.force or not solver_complete:
                        solver_rc = run_and_log(
                            [
                                str(solver_bin),
                                str(topology_path),
                                str(collective_path),
                                "--solver",
                                "mode=clean",
                            ],
                            repo_root,
                            solver_log,
                        )
                    if args.force or not baseline_complete:
                        baseline_rc = run_and_log(
                            [
                                str(solver_bin),
                                str(topology_path),
                                str(collective_path),
                                "--baselines",
                            ],
                            repo_root,
                            baseline_log,
                        )

                record: dict[str, Any] = {
                    "topology_name": topology_name,
                    "topology_relative_json": topology_relative,
                    "phase": phase,
                    "layer": layer,
                    "direction": direction,
                    "nominal_batch_size": int(case["nominal_batch_size"]),
                    "local_batch_size": int(case["local_batch_size"]),
                    "dispatch_collective": case["dispatch_collective"],
                    "combine_collective": case["combine_collective"],
                    "solver_log": str(solver_log),
                    "baseline_log": str(baseline_log),
                    "solver_returncode": solver_rc,
                    "baseline_returncode": baseline_rc,
                    "astra_dispatch_layer_name": case.get(
                        "astra_dispatch_layer_name"
                    ),
                    "astra_combine_layer_name": case.get(
                        "astra_combine_layer_name"
                    ),
                    "strong_baseline_method": baseline_method,
                    "mpi_semantics": "classic_baseline_only",
                }
                if solver_rc == 0 and baseline_rc == 0:
                    record.update(parse_glaive_log(solver_log))
                    record.update(parse_baseline_log(baseline_log))
                    required = (
                        "glaive_makespan_us",
                        baseline_metric,
                        "mpi_makespan_us",
                    )
                    if all(metric in record for metric in required):
                        record["strong_baseline_makespan_us"] = record[
                            baseline_metric
                        ]
                        record["status"] = "ok"
                    else:
                        record["status"] = "parse_failed"
                else:
                    record["status"] = "failed"

                results.append(record)
                summary_path.write_text(
                    json.dumps({"results": results}, indent=2) + "\n"
                )

    export_astra_csvs(output_dir, results)
    failures = [item for item in results if item["status"] != "ok"]
    print(f"Wrote {summary_path}")
    if failures:
        raise SystemExit(f"{len(failures)} Figure 12 collective tasks failed")


if __name__ == "__main__":
    main()
