#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

METHOD_ARGS = {
    "glaive": ["--solver3", "mode=standard"],
    "biring": ["--baseline-method", "biring"],
    "halfringdr": ["--baseline-method", "halfringdr"],
    "mpibaseline": ["--baseline-method", "mpibaseline"],
}
SYNTHETIC_TOPOLOGIES_32 = [
    "mesh_nebula_8x4",
    "cm384_16x2_eval",
    "fattree_8x4_eval",
]


def link_methods_for_topology(topology_key: str) -> tuple[str, ...]:
    if topology_key == "torus_tpuv4_4x4x4":
        return ("glaive", "halfringdr", "mpibaseline")
    return ("glaive", "biring", "mpibaseline")


@dataclass(frozen=True)
class Task:
    stage: str
    name: str
    topology_json: str
    collective_json: str
    args: tuple[str, ...]
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Collective-Synthesizer evaluation workloads.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage", choices=["all", "synthetic", "link", "scalability"], default="all")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--task-name-regex", default="")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def task_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = path.read_text(errors="replace")
    return (
        "[TACOS Solver3] Done!" in text
        or "[TACOS Baseline] Done!" in text
        or "[TACOS Baselines] Done!" in text
        or "[TACOS] Done!" in text
    )


def selected_topologies_for_case(target_devices: int) -> list[str]:
    if target_devices == 64:
        return ["torus_tpuv4_4x4x4"]
    return list(SYNTHETIC_TOPOLOGIES_32)


def synthetic_methods_for_topology(topology_key: str) -> tuple[str, ...]:
    if topology_key == "torus_tpuv4_4x4x4":
        return ("glaive", "halfringdr", "mpibaseline")
    return ("glaive", "biring", "mpibaseline")


def run_task_local(repo_root: Path, task: Task) -> None:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    tacos_bin = repo_root / "build" / "bin" / "tacos"
    command = [
        str(tacos_bin),
        task.topology_json,
        task.collective_json,
        *task.args,
    ]
    env = os.environ.copy()
    if task.stage in {"synthetic", "scalability"}:
        env["TACOS_SUPPRESS_EVENT_LOG"] = "1"
    with task.log_path.open("w") as handle:
        subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_task(repo_root: Path, args: argparse.Namespace, task: Task) -> str:
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.force and task_is_complete(task.log_path):
        return f"skip {task.name}"

    try:
        run_task_local(repo_root, task)
    except subprocess.CalledProcessError:
        if task_is_complete(task.log_path):
            return f"done {task.name} (post-check)"
        raise
    return f"done {task.name}"


def synthetic_tasks(repo_root: Path) -> list[Task]:
    topo_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json")
    case_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.json")

    topologies = {item["topology_key"]: item for item in topo_manifest["topologies"]}
    tasks: list[Task] = []

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
                tasks.append(
                    Task(
                        stage="synthetic",
                        name=f"synthetic::{topology_key}::{method}::{case['case_id']}",
                        topology_json=topo["topology_json"],
                        collective_json=case["collective_json"],
                        args=tuple(METHOD_ARGS[method]),
                        log_path=log_path,
                    )
                )
    return tasks


def link_tasks(repo_root: Path) -> list[Task]:
    topo_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json")
    case_manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.json")
    topologies = {item["topology_key"]: item for item in topo_manifest["topologies"]}

    tasks: list[Task] = []
    for case in case_manifest["cases"]:
        if case["size_label"] != "256MB":
            continue

        for topology_key in selected_topologies_for_case(int(case["target_devices"])):
            topo = topologies[topology_key]
            for method in link_methods_for_topology(topology_key):
                extra_args = list(METHOD_ARGS[method])
                if method == "glaive":
                    extra_args.append("--print-schedule")
                log_path = (
                    repo_root
                    / "evaluation_assets"
                    / "raw_logs"
                    / "link"
                    / topology_key
                    / method
                    / f"sample{case['sample_index']}.log"
                )
                tasks.append(
                    Task(
                        stage="link",
                        name=f"link::{topology_key}::{method}::{case['case_id']}",
                        topology_json=topo["topology_json"],
                        collective_json=case["collective_json"],
                        args=tuple(extra_args),
                        log_path=log_path,
                    )
                )
    return tasks


def scalability_tasks(repo_root: Path) -> list[Task]:
    manifest = load_json(repo_root / "evaluation_assets" / "manifests" / "scalability_cases.json")
    # Figure 8's paper panel fixes the x-axis at 0--4096 devices.  The source
    # manifest also contains larger stress points (6400--9216 devices); keep
    # those available in the manifest but do not make them part of the
    # default artifact reproduction because they are outside the paper panel
    # and can take tens of minutes each on a single CPU slot.
    max_devices = int(os.environ.get("GLAIVE_SCALABILITY_MAX_DEVICES", "4096"))
    sorted_cases = sorted(
        manifest["cases"],
        key=lambda case: (int(case["npus_count"]), str(case["topology_type"]), str(case["point"])),
    )
    tasks: list[Task] = []
    for case in sorted_cases:
        # The paper's Figure 8 contains FullMesh, Torus, Clos, and CM384;
        # Mesh is retained in the manifest for optional stress testing but is
        # not part of the target panel.
        if str(case["topology_type"]) == "mesh":
            continue
        if max_devices > 0 and int(case["npus_count"]) > max_devices:
            continue
        log_stem = case.get("log_stem", Path(case["topology_json"]).stem)
        log_path = (
            repo_root
            / "evaluation_assets"
            / "raw_logs"
            / "scalability"
            / case["topology_type"]
            / f"{log_stem}.log"
        )
        tasks.append(
            Task(
                stage="scalability",
                name=f"scalability::{case['topology_type']}::{case.get('workload_tag', 'default')}::{case['npus_count']}",
                topology_json=case["topology_json"],
                collective_json=case["collective_json"],
                args=tuple(METHOD_ARGS["glaive"]),
                log_path=log_path,
            )
        )
    return tasks


def selected_tasks(repo_root: Path, stage: str) -> list[Task]:
    tasks: list[Task] = []
    if stage in {"all", "synthetic"}:
        tasks.extend(synthetic_tasks(repo_root))
    if stage in {"all", "link"}:
        tasks.extend(link_tasks(repo_root))
    if stage in {"all", "scalability"}:
        tasks.extend(scalability_tasks(repo_root))
    return tasks


def filter_tasks(tasks: list[Task], task_name_regex: str, max_tasks: int) -> list[Task]:
    selected = tasks
    if task_name_regex:
        pattern = re.compile(task_name_regex)
        selected = [task for task in selected if pattern.search(task.name)]
    if max_tasks > 0:
        selected = selected[:max_tasks]
    return selected


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    tasks = filter_tasks(selected_tasks(repo_root, args.stage), args.task_name_regex, args.max_tasks)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "task_count": len(tasks),
                "task_name_regex": args.task_name_regex,
                "max_tasks": args.max_tasks,
                "max_workers": args.max_workers,
                "force": args.force,
            },
            indent=2,
        )
    )

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(run_task, repo_root, args, task): task.name
            for task in tasks
        }
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()
