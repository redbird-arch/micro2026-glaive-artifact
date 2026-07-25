#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
EVALUATION_ASSETS_OUT_ROOT = Path(
    os.environ.get("GLAIVE_EVAL_ASSETS_OUT_ROOT", REPO_ROOT / "evaluation_assets")
).resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import run_trace_studies as trace_studies  # noqa: E402


SPEED_OPT3_PATH = REPO_ROOT / "evaluation_assets" / "scripts" / "run_speed_opt3.py"
SPEC = importlib.util.spec_from_file_location("run_speed_opt3_speed", SPEED_OPT3_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {SPEED_OPT3_PATH}")
speed_opt3 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = speed_opt3
SPEC.loader.exec_module(speed_opt3)


TOPOLOGY_KEY = "torus_tpuv4_4x4x4"
SELECTION_LABEL = "256MB"
SAMPLE_INDEX = "0"
VARIANT_ORDER = (
    "balanced",
    "blend_0.75",
    "blend_0.50",
    "blend_0.25",
    "eplb_x2",
    "real",
    "worst_top8",
)
VARIANT_LABELS = {
    "balanced": "Balanced",
    "blend_0.75": "Blend 75%",
    "blend_0.50": "Blend 50%",
    "blend_0.25": "Blend 25%",
    "eplb_x2": "EPLB x2",
    "real": "Real",
    "worst_top8": "Worst-top8",
}
METHODS = ("standard", "cold_halfringdr")
METHOD_LABELS = {
    "standard": "Glaive",
    "cold_halfringdr": "HalfR+DR",
}
BASELINE_MAKESPAN_RE = re.compile(r"Makespan:\s*([0-9.eE+-]+)\s*us")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_cases() -> list[dict[str, str]]:
    manifest = trace_studies.RESULTS_DIR / "generated_trace_variant_manifest.csv"
    by_variant: dict[str, dict[str, str]] = {}
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["topology_key"] == TOPOLOGY_KEY
                and row["selection_label"] == SELECTION_LABEL
                and row["sample_index"] == SAMPLE_INDEX
                and row["variant"] in VARIANT_ORDER
            ):
                by_variant[row["variant"]] = row
    missing = [variant for variant in VARIANT_ORDER if variant not in by_variant]
    if missing:
        raise RuntimeError(f"Missing manifest variants: {missing}")
    return [by_variant[variant] for variant in VARIANT_ORDER]


def run_speed(case: dict[str, str], log_path: Path, force: bool) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and log_path.exists() and log_path.stat().st_size > 0 and speed_opt3.base.task_is_complete(log_path):
        return "skip"
    command = trace_studies.runtime_command() + [
        str(REPO_ROOT / case["topology_json"]),
        str(REPO_ROOT / case["collective_json"]),
        "--solver",
        "mode=speed",
    ]
    env = os.environ.copy()
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False, env=env)
    return "done" if result.returncode == 0 else f"failed:{result.returncode}"


def write_cold_inputs(case: dict[str, str], cold_matrix: list[list[int]]) -> tuple[Path, Path]:
    original_collective = json.loads((REPO_ROOT / case["collective_json"]).read_text())
    block_bytes = int(original_collective.get("block_bytes", 4096))
    trace_path = (
        trace_studies.GENERATED_DIR
        / "speed_cold_traces"
        / f"{case['case_id']}_cold.csv"
    )
    collective_path = (
        trace_studies.GENERATED_DIR
        / "speed_cold_collectives"
        / f"{case['case_id']}_cold.json"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    collective_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", newline="") as handle:
        for row in cold_matrix:
            cells: list[str] = []
            for value in row:
                if value % block_bytes != 0:
                    raise ValueError(f"cold bytes {value} is not divisible by block_bytes={block_bytes}")
                cells.append(str(value // block_bytes))
            handle.write(",".join(cells) + "\n")
    payload = dict(original_collective)
    payload["v_datasize"] = trace_path.resolve().as_posix()
    collective_path.write_text(json.dumps(payload, indent=2) + "\n")
    return trace_path, collective_path


def run_cold_halfringdr(case: dict[str, str], collective_path: Path, log_path: Path, force: bool) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and log_path.exists() and log_path.stat().st_size > 0 and speed_opt3.base.task_is_complete(log_path):
        return "skip"
    command = trace_studies.runtime_command() + [
        str(REPO_ROOT / case["topology_json"]),
        str(collective_path),
        "--baseline-method",
        "halfringdr",
    ]
    env = os.environ.copy()
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False, env=env)
    return "done" if result.returncode == 0 else f"failed:{result.returncode}"


def parse_baseline_makespan_us(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    match = BASELINE_MAKESPAN_RE.search(log_path.read_text(errors="replace"))
    return float(match.group(1)) if match else 0.0


def first_gap(slots: list[tuple[float, float]], earliest: float, duration: float) -> tuple[float, float]:
    start = earliest
    for busy_start, busy_end in sorted(slots):
        if start + duration <= busy_start:
            return start, start + duration
        if start < busy_end and start + duration > busy_start:
            start = busy_end
    return start, start + duration


def initial_hot_slots(hot_events: list[dict[str, Any]]) -> dict[tuple[int, int], list[tuple[float, float]]]:
    slots: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for event in hot_events:
        start = float(event["start_us"])
        end = float(event["end_us"])
        if end > start:
            slots[(int(event["link_src"]), int(event["link_dst"]))].append((start, end))
    for edge_slots in slots.values():
        edge_slots.sort()
    return slots


def torus_id_to_coord(rank: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coords: list[int] = []
    value = rank
    for size in shape:
        coords.append(value % size)
        value //= size
    return tuple(coords)


def torus_coord_to_id(coords: tuple[int, ...], shape: tuple[int, ...]) -> int:
    rank = 0
    stride = 1
    for coord, size in zip(coords, shape):
        rank += coord * stride
        stride *= size
    return rank


def neighbor_along_dimension(rank: int, shape: tuple[int, ...], dimension: int, delta: int) -> int:
    coords = list(torus_id_to_coord(rank, shape))
    coords[dimension] = (coords[dimension] + delta + shape[dimension]) % shape[dimension]
    return torus_coord_to_id(tuple(coords), shape)


def halfringdr_edge_time_us(model: Any, src: int, dst: int, bytes_count: int) -> float:
    _, latency_ns = speed_opt3.edge_profile(model, src, dst)
    return speed_opt3.edge_time_us(model, src, dst, bytes_count) + 5.0 * latency_ns / 1000.0


def schedule_cold_halfringdr_dimrotation(
    model: Any,
    cold_matrix: list[list[int]],
    hot_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if model.topology != "torus":
        raise ValueError(f"HalfR+DR replay requires torus topology, got {model.topology}")
    shape = tuple(int(item) for item in model.shape)
    dims = len(shape)
    n = model.n
    coords = [torus_id_to_coord(node, shape) for node in range(n)]
    chunk_resident: list[list[list[dict[str, int]]]] = [[[] for _ in range(n)] for _ in range(dims)]
    for src in range(n):
        for dst in range(n):
            bytes_count = int(cold_matrix[src][dst])
            if src == dst or bytes_count <= 0:
                continue
            base_chunk = bytes_count // dims
            remainder = bytes_count % dims
            for chunk in range(dims):
                chunk_bytes = base_chunk + (1 if chunk < remainder else 0)
                if chunk_bytes > 0:
                    chunk_resident[chunk][src].append(
                        {"final_dest": dst, "bytes": chunk_bytes, "remaining": 0}
                    )

    slots = initial_hot_slots(hot_events)
    current_time = 0.0
    next_event_id = 0
    events: list[dict[str, Any]] = []

    for phase in range(dims):
        phase_states: list[dict[str, Any]] = []
        max_phase_stages = 0
        for chunk in range(dims):
            dimension = (chunk + phase) % dims
            dim_size = shape[dimension]
            stage_num = 0 if dim_size <= 1 else dim_size // 2
            state = {
                "dimension": dimension,
                "stage_num": stage_num,
                "positive": [[[] for _ in range(n)] for _ in range(stage_num + 1)],
                "negative": [[[] for _ in range(n)] for _ in range(stage_num + 1)],
                "settled": [[] for _ in range(n)],
            }
            max_phase_stages = max(max_phase_stages, stage_num)
            for node in range(n):
                for payload in chunk_resident[chunk][node]:
                    current_coord = coords[node]
                    final_coord = coords[int(payload["final_dest"])]
                    positive_hops = (final_coord[dimension] - current_coord[dimension] + dim_size) % dim_size
                    negative_hops = (current_coord[dimension] - final_coord[dimension] + dim_size) % dim_size
                    if positive_hops == 0:
                        state["settled"][node].append(payload)
                    elif positive_hops < negative_hops:
                        state["positive"][positive_hops][node].append(
                            {
                                "final_dest": int(payload["final_dest"]),
                                "bytes": int(payload["bytes"]),
                                "remaining": positive_hops,
                            }
                        )
                    elif negative_hops < positive_hops:
                        state["negative"][negative_hops][node].append(
                            {
                                "final_dest": int(payload["final_dest"]),
                                "bytes": int(payload["bytes"]),
                                "remaining": negative_hops,
                            }
                        )
                    else:
                        positive_bytes = (int(payload["bytes"]) + 1) // 2
                        negative_bytes = int(payload["bytes"]) // 2
                        if positive_bytes > 0:
                            state["positive"][positive_hops][node].append(
                                {
                                    "final_dest": int(payload["final_dest"]),
                                    "bytes": positive_bytes,
                                    "remaining": positive_hops,
                                }
                            )
                        if negative_bytes > 0:
                            state["negative"][negative_hops][node].append(
                                {
                                    "final_dest": int(payload["final_dest"]),
                                    "bytes": negative_bytes,
                                    "remaining": negative_hops,
                                }
                            )
            phase_states.append(state)

        for stage in range(1, max_phase_stages + 1):
            for _substage in range(stage):
                transfers: list[dict[str, Any]] = []
                for chunk, state in enumerate(phase_states):
                    if stage > int(state["stage_num"]):
                        continue
                    dimension = int(state["dimension"])
                    for node in range(n):
                        positive_payloads = state["positive"][stage][node]
                        if positive_payloads:
                            bytes_count = sum(int(payload["bytes"]) for payload in positive_payloads)
                            if bytes_count > 0:
                                transfers.append(
                                    {
                                        "chunk": chunk,
                                        "src": node,
                                        "dst": neighbor_along_dimension(node, shape, dimension, +1),
                                        "bytes": bytes_count,
                                        "positive": True,
                                        "payloads": positive_payloads,
                                    }
                                )
                        negative_payloads = state["negative"][stage][node]
                        if negative_payloads:
                            bytes_count = sum(int(payload["bytes"]) for payload in negative_payloads)
                            if bytes_count > 0:
                                transfers.append(
                                    {
                                        "chunk": chunk,
                                        "src": node,
                                        "dst": neighbor_along_dimension(node, shape, dimension, -1),
                                        "bytes": bytes_count,
                                        "positive": False,
                                        "payloads": negative_payloads,
                                    }
                                )
                if not transfers:
                    continue

                step_start = current_time
                step_end = step_start
                next_positive = [[[] for _ in range(n)] for _ in range(dims)]
                next_negative = [[[] for _ in range(n)] for _ in range(dims)]

                for transfer in transfers:
                    src = int(transfer["src"])
                    dst = int(transfer["dst"])
                    bytes_count = int(transfer["bytes"])
                    duration = halfringdr_edge_time_us(model, src, dst, bytes_count)
                    edge_slots = slots[(src, dst)]
                    start, end = first_gap(edge_slots, step_start, duration)
                    edge_slots.append((start, end))
                    edge_slots.sort()
                    step_end = max(step_end, end)
                    events.append(
                        {
                            "class": "cold",
                            "link_src": src,
                            "link_dst": dst,
                            "flow_src": src,
                            "flow_dst": dst,
                            "chunk_id": next_event_id,
                            "bytes": bytes_count,
                            "start_us": start,
                            "end_us": end,
                            "path": f"{src}>{dst}",
                        }
                    )
                    next_event_id += 1

                    chunk = int(transfer["chunk"])
                    destination_queue = next_positive if bool(transfer["positive"]) else next_negative
                    destination_settled = phase_states[chunk]["settled"][dst]
                    for payload in transfer["payloads"]:
                        if int(payload["remaining"]) <= 1:
                            destination_settled.append(
                                {
                                    "final_dest": int(payload["final_dest"]),
                                    "bytes": int(payload["bytes"]),
                                    "remaining": 0,
                                }
                            )
                        else:
                            destination_queue[chunk][dst].append(
                                {
                                    "final_dest": int(payload["final_dest"]),
                                    "bytes": int(payload["bytes"]),
                                    "remaining": int(payload["remaining"]) - 1,
                                }
                            )

                current_time = step_end
                for chunk, state in enumerate(phase_states):
                    if stage > int(state["stage_num"]):
                        continue
                    state["positive"][stage] = next_positive[chunk]
                    state["negative"][stage] = next_negative[chunk]

        for chunk, state in enumerate(phase_states):
            chunk_resident[chunk] = state["settled"]

    return events


def build_rows(cases: list[dict[str, str]], force: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    log_root = trace_studies.LOGS_DIR / "torus_speed_breakdown"
    cold_log_root = trace_studies.LOGS_DIR / "torus_speed_breakdown_cold_halfringdr"
    for case in cases:
        log_path = log_root / f"{case['case_id']}.log"
        status = run_speed(case, log_path, force)
        metrics = speed_opt3.base.parse_log_metrics(log_path)
        standard_events = speed_opt3.base.parse_speed_events(log_path)
        if not standard_events:
            status_rows.append(
                {
                    "case_id": case["case_id"],
                    "variant": case["variant"],
                    "standard_status": status,
                    "cold_halfringdr_status": "not_run",
                    "standard_log_path": trace_studies.rel(log_path),
                    "cold_halfringdr_log_path": "",
                    "cold_trace_path": "",
                    "cold_collective_path": "",
                    "backend_cold_makespan_us": "",
                    "model_cold_no_hot_makespan_us": "",
                }
            )
            continue
        topology = speed_opt3.base.load_json(REPO_ROOT / case["topology_json"])
        model = speed_opt3.build_topology_model(TOPOLOGY_KEY, topology)
        hot_events = [event for event in standard_events if event["class"] == "hot"]
        cold_matrix = speed_opt3.base.cold_matrix_from_events(standard_events, model.n)
        cold_trace_path, cold_collective_path = write_cold_inputs(case, cold_matrix)
        cold_log_path = cold_log_root / f"{case['case_id']}_cold_halfringdr.log"
        cold_status = run_cold_halfringdr(case, cold_collective_path, cold_log_path, force)
        backend_cold_makespan = parse_baseline_makespan_us(cold_log_path)
        no_hot_halfringdr_events = schedule_cold_halfringdr_dimrotation(model, cold_matrix, [])
        model_cold_no_hot_makespan = max((float(event["end_us"]) for event in no_hot_halfringdr_events), default=0.0)
        status_rows.append(
            {
                "case_id": case["case_id"],
                "variant": case["variant"],
                "standard_status": status,
                "cold_halfringdr_status": cold_status,
                "standard_log_path": trace_studies.rel(log_path),
                "cold_halfringdr_log_path": trace_studies.rel(cold_log_path),
                # Cold inputs are generated below the run output root, not in
                # the immutable source tree.  Use trace_studies.rel() so both
                # source inputs and generated run artifacts are represented
                # consistently in the status CSV.
                "cold_trace_path": trace_studies.rel(cold_trace_path),
                "cold_collective_path": trace_studies.rel(cold_collective_path),
                "backend_cold_makespan_us": backend_cold_makespan,
                "model_cold_no_hot_makespan_us": model_cold_no_hot_makespan,
            }
        )
        method_events = {
            "standard": [event for event in standard_events if event["class"] in {"hot", "cold"}],
            "cold_halfringdr": [*hot_events, *schedule_cold_halfringdr_dimrotation(model, cold_matrix, hot_events)],
        }
        for method in METHODS:
            events = method_events[method]
            row = speed_opt3.summarize_speed_method(TOPOLOGY_KEY, method, events, metrics, log_path, REPO_ROOT)
            row.update(
                {
                    "case_id": case["case_id"],
                    "variant": case["variant"],
                    "variant_label": VARIANT_LABELS[case["variant"]],
                    "method_label": METHOD_LABELS[method],
                    "cold_halfringdr_backend_cold_only_makespan_us": backend_cold_makespan if method == "cold_halfringdr" else "",
                    "cold_halfringdr_model_cold_only_makespan_us": model_cold_no_hot_makespan
                    if method == "cold_halfringdr"
                    else "",
                }
            )
            summary_rows.append(row)
            for event in events:
                event_rows.append(
                    {
                        "case_id": case["case_id"],
                        "variant": case["variant"],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
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
    return summary_rows, event_rows, status_rows


def choose_time_unit(max_us: float) -> tuple[float, str]:
    if max_us >= 1000:
        return 1000.0, "ms"
    return 1.0, "us"


def plot_torus_breakdown(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

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
    by_key = {(row["variant"], row["method"]): row for row in summary_rows}
    group_x = np.arange(len(VARIANT_ORDER), dtype=float)
    bar_width = 0.30
    offsets = {"standard": -0.19, "cold_halfringdr": 0.19}
    component_styles = {
        "hot_only_active_us": {"fill": "#80ed99", "edge": "#06d6a0"},
        "overlap_us": {"fill": "#4cc9f0", "edge": "#118ab2"},
        "cold_only_active_us": {"fill": "#ff87ab", "edge": "#ff5d8f"},
    }
    method_hatches = {"standard": "", "cold_halfringdr": "//"}
    bar_lw = 2.7

    totals = []
    for variant in VARIANT_ORDER:
        for method in METHODS:
            row = by_key.get((variant, method))
            if row:
                totals.append(float(row["hot_only_active_us"]) + float(row["overlap_us"]) + float(row["cold_only_active_us"]))
    divisor, unit = choose_time_unit(max(totals) if totals else 1.0)

    fig, ax = plt.subplots(figsize=(11.5, 4))
    ax.set_axisbelow(True)
    for method in METHODS:
        x = group_x + offsets[method]
        hot = np.array([float(by_key[(variant, method)]["hot_only_active_us"]) / divisor for variant in VARIANT_ORDER])
        overlap = np.array([float(by_key[(variant, method)]["overlap_us"]) / divisor for variant in VARIANT_ORDER])
        cold = np.array([float(by_key[(variant, method)]["cold_only_active_us"]) / divisor for variant in VARIANT_ORDER])
        hatch = method_hatches[method]
        ax.bar(
            x,
            hot,
            width=bar_width,
            color=component_styles["hot_only_active_us"]["fill"],
            edgecolor=component_styles["hot_only_active_us"]["edge"],
            linewidth=bar_lw,
            hatch=hatch,
            zorder=3,
        )
        ax.bar(
            x,
            overlap,
            width=bar_width,
            bottom=hot,
            color=component_styles["overlap_us"]["fill"],
            edgecolor=component_styles["overlap_us"]["edge"],
            linewidth=bar_lw,
            hatch=hatch,
            zorder=3,
        )
        ax.bar(
            x,
            cold,
            width=bar_width,
            bottom=hot + overlap,
            color=component_styles["cold_only_active_us"]["fill"],
            edgecolor=component_styles["cold_only_active_us"]["edge"],
            linewidth=bar_lw,
            hatch=hatch,
            zorder=3,
        )

    ax.set_title("3D Torus 4x4x4", fontsize=22, y=1.08)
    ax.set_ylabel(f"Collective Time ({unit})", fontsize=21)
    ax.yaxis.set_label_coords(-0.055, 0.52)
    ax.set_xticks(group_x)
    ax.set_xticklabels([VARIANT_LABELS[variant] for variant in VARIANT_ORDER], rotation=0, ha="center", fontsize=17)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", zorder=0)
    component_handles = [
        Patch(
            facecolor=component_styles["hot_only_active_us"]["fill"],
            edgecolor=component_styles["hot_only_active_us"]["edge"],
            linewidth=bar_lw,
            label="Thrust",
        ),
        Patch(
            facecolor=component_styles["overlap_us"]["fill"],
            edgecolor=component_styles["overlap_us"]["edge"],
            linewidth=bar_lw,
            label="Overlap",
        ),
        Patch(
            facecolor=component_styles["cold_only_active_us"]["fill"],
            edgecolor=component_styles["cold_only_active_us"]["edge"],
            linewidth=bar_lw,
            label="Sweep",
        ),
    ]
    method_handles = [
        Patch(facecolor="white", edgecolor="#073b4c", linewidth=2.4, label="Glaive cold"),
        Patch(facecolor="white", edgecolor="#073b4c", linewidth=2.4, hatch="//", label="HalfR+DR cold"),
    ]
    ax.legend(
        handles=[*component_handles, *method_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=5,
        frameon=False,
        fontsize=15.5,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    ymax = max(totals) / divisor * 1.18 if totals else 1.0
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(top=0.73, bottom=0.2, left=0.13, right=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun and plot torus Speed hot/cold replay breakdown.")
    parser.add_argument("--force", action="store_true", help="rerun mode=speed logs")
    args = parser.parse_args()

    cases = load_cases()
    summary_rows, event_rows, status_rows = build_rows(cases, args.force)
    parsed_dir = EVALUATION_ASSETS_OUT_ROOT / "parsed"
    plot_dir = EVALUATION_ASSETS_OUT_ROOT / "plots"
    write_csv(parsed_dir / "speed_torus_summary.csv", summary_rows)
    write_csv(parsed_dir / "speed_torus_events.csv", event_rows)
    write_csv(parsed_dir / "speed_torus_status.csv", status_rows)
    plot_torus_breakdown(summary_rows, plot_dir / "Speed_Torus_Breakdown.pdf")
    print(
        json.dumps(
            {
                "summary": str((parsed_dir / "speed_torus_summary.csv").resolve()),
                "events": str((parsed_dir / "speed_torus_events.csv").resolve()),
                "status": str((parsed_dir / "speed_torus_status.csv").resolve()),
                "plot": str((plot_dir / "Speed_Torus_Breakdown.pdf").resolve()),
                "rows": len(summary_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
