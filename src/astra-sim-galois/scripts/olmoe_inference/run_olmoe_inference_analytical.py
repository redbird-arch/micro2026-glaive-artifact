#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = (
    REPO_ROOT / "build" / "astra_analytical" / "build" / "AnalyticalAstra" / "bin" / "AnalyticalAstra"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "inputs"
    / "metadata"
    / "DeepSeekV3Proxy"
    / "deepseekv3_proxy_experiments.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "deepseekv3_proxy_inference_analytical"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AnalyticalAstra for the generated OLMoE inference scenarios and "
            "summarize EndToEnd.csv metrics."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def rel_to_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def parse_end_to_end_csv(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))

    header = rows[0]
    header_index = {name.strip(): idx for idx, name in enumerate(header) if name.strip()}
    data_rows = [row for row in rows[1:] if row and len(row) > 1 and row[0].strip()]
    if not data_rows:
        raise ValueError(f"{path} does not contain data rows")

    def value(row: list[str], column: str, default: float = 0.0) -> float:
        idx = header_index.get(column)
        if idx is None or idx >= len(row) or not row[idx]:
            return default
        return float(row[idx])

    first = data_rows[0]
    total_time_us = value(first, "workload finished at")
    total_compute_us = value(first, "total comp")
    total_memory_us = value(first, "total memory")
    total_exposed_comm_us = value(
        first,
        "total exposed comm",
        total_time_us - total_compute_us - total_memory_us,
    )
    total_fwd_comm_us = sum(
        value(row, "fwd total comm") for row in data_rows
    )

    return {
        "total_time_us": total_time_us,
        "total_compute_us": total_compute_us,
        "total_memory_us": total_memory_us,
        "total_exposed_comm_us": total_exposed_comm_us,
        "total_fwd_comm_us": total_fwd_comm_us,
    }


def run_one(
    binary: Path,
    scenario: dict[str, object],
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    end_to_end_csv = output_dir / "EndToEnd.csv"
    log_path = output_dir / "stdout.log"

    result: dict[str, object] = {
        "run_name": str(scenario["run_name"]),
        "hardware": str(scenario["hardware"]),
        "hardware_label": str(scenario["hardware_label"]),
        "topology_label": str(scenario["topology_label"]),
        "phase": str(scenario["phase"]),
        "method": str(scenario["method"]),
        "output_dir": rel_to_repo(output_dir),
        "log_path": rel_to_repo(log_path),
    }
    for field in ("model_key", "model_label", "config_key", "config_label"):
        if field in scenario:
            result[field] = scenario[field]

    if skip_existing and end_to_end_csv.exists():
        result.update(parse_end_to_end_csv(end_to_end_csv))
        result["status"] = "cached"
        return result

    network = (REPO_ROOT / str(scenario["network"])).resolve()
    system = (REPO_ROOT / str(scenario["system"])).resolve()
    workload = (REPO_ROOT / str(scenario["workload"])).resolve()
    units_count = [str(value) for value in scenario["units_count"]]
    num_queues = [str(value) for value in scenario["num_queues_per_dim"]]

    cmd = [
        str(binary.resolve()),
        f"--network-configuration={network}",
        f"--system-configuration={system}",
        f"--workload-configuration={workload}",
        f"--path={output_dir.resolve()}/",
        f"--run-name={scenario['run_name']}",
        "--num-passes=1",
        "--total-stat-rows=1",
        "--stat-row=0",
        "--rendezvous-protocol=false",
        "--comm-scale=1",
        "--compute-scale=1",
        "--injection-scale=1",
        "--units-count",
        *units_count,
        "--num-queues-per-dim",
        *num_queues,
    ]

    with log_path.open("w") as handle:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    result["returncode"] = proc.returncode
    if proc.returncode != 0:
        result["status"] = "failed"
        return result

    result.update(parse_end_to_end_csv(end_to_end_csv))
    result["status"] = "ok"
    return result


def write_summary_csv(path: Path, results: list[dict[str, object]]) -> None:
    headers = [
        "run_name",
        "model_key",
        "model_label",
        "config_key",
        "config_label",
        "hardware",
        "hardware_label",
        "topology_label",
        "phase",
        "method",
        "status",
        "returncode",
        "total_time_us",
        "total_compute_us",
        "total_memory_us",
        "total_exposed_comm_us",
        "total_fwd_comm_us",
        "output_dir",
        "log_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    scenarios = manifest.get("scenarios", [])
    if not scenarios:
        raise ValueError(f"manifest has no scenarios: {args.manifest}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results_by_index: dict[int, dict[str, object]] = {}
    summary_json = args.output_root / "summary.json"
    summary_csv = args.output_root / "summary.csv"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                run_one,
                args.binary,
                scenario,
                args.output_root / str(scenario["run_name"]),
                args.skip_existing,
            ): index
            for index, scenario in enumerate(scenarios)
        }
        for future in as_completed(futures):
            index = futures[future]
            results_by_index[index] = future.result()
            partial = [
                results_by_index[item]
                for item in sorted(results_by_index)
            ]
            summary_json.write_text(json.dumps({"results": partial}, indent=2) + "\n")

    results = [results_by_index[index] for index in range(len(scenarios))]
    summary_json.write_text(json.dumps({"results": results}, indent=2) + "\n")
    write_summary_csv(summary_csv, results)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
