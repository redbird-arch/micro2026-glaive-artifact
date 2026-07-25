#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np


TOKEN_BYTES = 4096
TOP_K_DEFAULT = 8
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(os.environ.get("GLAIVE_SIM_EXP_OUT_ROOT", REPO_ROOT / "rebuttal" / "simulation_exp")).resolve()
RESULTS_DIR = OUT_ROOT / "results"
GENERATED_DIR = OUT_ROOT / "generated"
LOGS_DIR = OUT_ROOT / "logs"
PLOTS_DIR = OUT_ROOT / "plots"
PLOT_SCRIPT_DIR = REPO_ROOT / "rebuttal" / "simulation_exp" / "plots"

METHODS_32 = ("glaive", "biring", "mpibaseline")
METHODS_64 = ("glaive", "halfringdr", "mpibaseline")
METHOD_ARGS = {
    "glaive": ["--solver3", "mode=clean", "--print-schedule"],
    "biring": ["--baseline-method", "biring"],
    "halfringdr": ["--baseline-method", "halfringdr"],
    "mpibaseline": ["--baseline-method", "mpibaseline"],
}
TOPOLOGIES = {
    "mesh_nebula_8x4": {
        "target_devices": 32,
        "path": "evaluation_assets/topologies/synthetic/mesh_nebula_8x4.json",
    },
    "clos_8x4": {
        "target_devices": 32,
        "path": "evaluation_assets/topologies/synthetic/fattree_8x4_eval.json",
    },
    "cm384_16x2": {
        "target_devices": 32,
        "path": "evaluation_assets/topologies/synthetic/cm384_16x2_eval.json",
    },
    "rail_optimized_8x4": {
        "target_devices": 32,
        "path": "evaluation_assets/topologies/synthetic/rail_optimized_8x4_eval.json",
    },
    "torus_tpuv4_4x4x4": {
        "target_devices": 64,
        "path": "evaluation_assets/topologies/synthetic/torus_tpuv4_4x4x4.json",
    },
}
VARIANT_LABELS = {
    "real": "Real",
    "balanced": "Balanced",
    "blend_0.25": "Blend 25%",
    "blend_0.50": "Blend 50%",
    "blend_0.75": "Blend 75%",
    "eplb_x2": "EPLB x2",
    "worst_top8": "Worst-top8",
}
MAKESPAN_RE = re.compile(r"^\s*(?:Total Makespan|Makespan):\s*([0-9.]+)\s*us\b", re.MULTILINE)
SOLVER_RE = re.compile(r"^\s*(?:Solver Time|Algorithm Time):\s*([0-9.]+)\s*us\b", re.MULTILINE)
TOTAL_ROUNDS_RE = re.compile(r"^\s*Total rounds:\s*([0-9]+)\s*$", re.MULTILINE)
TOTAL_STEPS_RE = re.compile(r"^\s*Total steps with traffic:\s*([0-9]+)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"Link\((.+?)->(.+?)\): intervals=\[(.*)\], utilization=([0-9.]+)\s*%")
INTERVAL_RE = re.compile(r"\[(\d+),\s*(\d+)\]")
NUMERIC_NODE_RE = re.compile(r"^\d+$")
SCALE_UP_SWITCH_RE = re.compile(r"^scale_up_switch\((\d+)\)$")
SCALE_OUT_SWITCH_RE = re.compile(r"^scale_out_switch\((\d+)\)$")
CM_NODE_SWITCH_RE = re.compile(r"^node_switch\(node=(\d+),sw=(\d+)\)$")
CM_RAIL_SWITCH_RE = re.compile(r"^rail_switch\(rail=(\d+),sw=(\d+)\)$")

TOPOLOGY_CONFIG_CACHE: dict[str, dict[str, object]] = {}
TOPOLOGY_GRAPH_CACHE: dict[str, tuple[list[str], dict[str, list[str]]]] = {}
DETERMINISTIC_ROUTING_CACHE: dict[tuple[str, str], dict[str, object]] = {}


def ensure_dirs() -> None:
    for path in (
        RESULTS_DIR,
        GENERATED_DIR / "traces",
        GENERATED_DIR / "collectives",
        GENERATED_DIR / "eplb_maps",
        LOGS_DIR,
        PLOTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def variant_label(value: str) -> str:
    if value in VARIANT_LABELS:
        return VARIANT_LABELS[value]
    if value.startswith("blend_"):
        try:
            return f"Blend {float(value.split('_', 1)[1]) * 100:.0f}%"
        except (IndexError, ValueError):
            pass
    return value.replace("_", " ")


def load_matrix(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append([int(float(cell)) for cell in row])
    if not rows:
        return np.zeros((0, 0), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.astype(np.int64).tolist())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def gini(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    total = float(np.sum(flat))
    if total <= 0:
        return 0.0
    sorted_values = np.sort(flat)
    n = sorted_values.size
    weighted = float(np.sum(np.arange(1, n + 1, dtype=np.float64) * sorted_values))
    return max(0.0, min(1.0, (2.0 * weighted) / (n * total) - (n + 1.0) / n))


def cv(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    avg = float(np.mean(arr)) if arr.size else 0.0
    return float(np.std(arr) / avg) if avg > 0 else 0.0


def top_share(values: np.ndarray, fraction: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0 or arr.size == 0:
        return 0.0
    k = max(1, int(math.ceil(arr.size * fraction)))
    return float(np.sum(np.sort(arr)[-k:]) / total)


def top_count_for_share(values: np.ndarray, target_share: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0 or arr.size == 0:
        return 0.0
    desc = np.sort(arr)[::-1]
    threshold = target_share * total
    cumsum = np.cumsum(desc)
    count = int(np.searchsorted(cumsum, threshold, side="left")) + 1
    return float(count / arr.size)


def normalized_entropy_concentration(values: np.ndarray) -> float:
    """0 means uniform over all possible entries; 1 means one entry carries all mass."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    total = float(np.sum(arr))
    if n <= 1 or total <= 0:
        return 0.0
    p = arr[arr > 0] / total
    entropy = -float(np.sum(p * np.log(p))) if p.size else 0.0
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


def normalized_hhi_concentration(values: np.ndarray) -> float:
    """Normalized Simpson/HHI concentration, size-corrected to [0, 1]."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    total = float(np.sum(arr))
    if n <= 1 or total <= 0:
        return 0.0
    p = arr / total
    hhi = float(np.sum(p * p))
    uniform_hhi = 1.0 / n
    return max(0.0, min(1.0, (hhi - uniform_hhi) / (1.0 - uniform_hhi)))


def normalized_lorenz80_concentration(sorted_values: np.ndarray, total: float) -> float:
    """Pareto-style concentration: 0 for uniform, 1 when one entry explains 80% bytes."""
    n = sorted_values.size
    if n <= 1 or total <= 0:
        return 0.0
    frac80 = top_count_for_share_sorted(sorted_values, total, 0.80)
    single_entry_fraction = 1.0 / n
    denom = 0.80 - single_entry_fraction
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (0.80 - min(frac80, 0.80)) / denom))


def balanced_adjusted_lorenz80_concentration(sorted_values: np.ndarray, total: float) -> float:
    """Lorenz-80 concentration normalized against a same-size, same-total integer-balanced matrix."""
    n = sorted_values.size
    total_int = int(round(total))
    if n <= 1 or total_int <= 0:
        return 0.0
    actual_frac80 = top_count_for_share_sorted(sorted_values, total, 0.80)
    balanced = np.full(n, total_int // n, dtype=np.float64)
    rem = total_int % n
    if rem:
        balanced[-rem:] += 1.0
    balanced_frac80 = top_count_for_share_sorted(np.sort(balanced), float(total_int), 0.80)
    single_entry_fraction = 1.0 / n
    denom = balanced_frac80 - single_entry_fraction
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (balanced_frac80 - actual_frac80) / denom))


def sorted_distribution(values: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    sorted_arr = np.sort(arr) if arr.size else arr
    return arr, total, sorted_arr


def gini_sorted(sorted_values: np.ndarray, total: float) -> float:
    n = sorted_values.size
    if n == 0 or total <= 0:
        return 0.0
    weighted = float(np.sum(np.arange(1, n + 1, dtype=np.float64) * sorted_values))
    return max(0.0, min(1.0, (2.0 * weighted) / (n * total) - (n + 1.0) / n))


def top_share_sorted(sorted_values: np.ndarray, total: float, fraction: float) -> float:
    if total <= 0 or sorted_values.size == 0:
        return 0.0
    k = max(1, int(math.ceil(sorted_values.size * fraction)))
    return float(np.sum(sorted_values[-k:]) / total)


def top_count_for_share_sorted(sorted_values: np.ndarray, total: float, target_share: float) -> float:
    if total <= 0 or sorted_values.size == 0:
        return 0.0
    cumsum = np.cumsum(sorted_values[::-1])
    count = int(np.searchsorted(cumsum, target_share * total, side="left")) + 1
    return float(count / sorted_values.size)


def offdiag_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.int64)
    n, m = matrix.shape
    if n != m:
        return matrix.reshape(-1)
    mask = ~np.eye(n, dtype=bool)
    return matrix[mask]


def communication_matrix(matrix: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    if result.ndim == 2 and result.shape[0] == result.shape[1]:
        np.fill_diagonal(result, 0)
    return result


def infer_dataset(path: Path) -> tuple[str, str, str]:
    text = path.as_posix().lower()
    if "deepseek" in text:
        model = "deepseekv3"
    elif "qwen" in text:
        model = "qwen"
    elif "olmoe" in text:
        model = "olmoe"
    else:
        model = "unknown"
    phase = "prefill" if "prefill" in text else "decode" if "decode" in text else "unknown"
    source = "official" if "input/official_data" in text else "raw" if "input/raw_data" in text else "generated"
    return model, phase, source


def parse_case_metadata(path: Path) -> dict[str, object]:
    name = path.stem
    batch_match = re.search(r"BS(\d+)", name, re.IGNORECASE)
    actual_match = re.search(r"actual(\d+)", name, re.IGNORECASE)
    layer_match = re.search(r"Layer(\d+)", name, re.IGNORECASE)
    sample_match = re.search(r"_(\d+)$", name)
    return {
        "batch_size": int(batch_match.group(1)) if batch_match else "",
        "actual_batch_size": int(actual_match.group(1)) if actual_match else "",
        "layer": int(layer_match.group(1)) if layer_match else "",
        "sample_index": int(sample_match.group(1)) if sample_match else "",
    }


def balanced_matrix(shape: tuple[int, int], total: int) -> np.ndarray:
    n, m = shape
    result = np.zeros(shape, dtype=np.int64)
    cells = [(i, j) for i in range(n) for j in range(m) if i != j]
    if not cells:
        return result
    base = total // len(cells)
    rem = total % len(cells)
    for idx, (i, j) in enumerate(cells):
        result[i, j] = base + (1 if idx < rem else 0)
    return result


def worst_matrix(shape: tuple[int, int], total: int, top_k: int) -> np.ndarray:
    n, m = shape
    result = np.zeros(shape, dtype=np.int64)
    hot = list(range(min(top_k, m)))
    if not hot:
        return result
    row_base = total // n if n else 0
    row_rem = total % n if n else 0
    for i in range(n):
        row_total = row_base + (1 if i < row_rem else 0)
        dests = [d for d in hot if d != i]
        if len(dests) < top_k:
            for d in range(m):
                if d != i and d not in dests:
                    dests.append(d)
                if len(dests) >= top_k:
                    break
        dests = dests[: max(1, min(top_k, len(dests)))]
        base = row_total // len(dests)
        rem = row_total % len(dests)
        for idx, d in enumerate(dests):
            result[i, d] = base + (1 if idx < rem else 0)
    return result


def replicate_experts_by_load(load: np.ndarray, num_physical_experts: int) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy version of DeepSeek EPLB's greedy expert replication step."""
    logical_load = np.asarray(load, dtype=np.float64).reshape(-1)
    num_logical = logical_load.size
    if num_physical_experts < num_logical:
        raise ValueError("EPLB requires at least one physical copy per logical expert")
    phy2log = list(range(num_logical))
    logcnt = np.ones(num_logical, dtype=np.int64)
    while len(phy2log) < num_physical_experts:
        per_copy_load = logical_load / np.maximum(logcnt, 1)
        logical = int(np.argmax(per_copy_load))
        phy2log.append(logical)
        logcnt[logical] += 1
    return np.asarray(phy2log, dtype=np.int64), logcnt


def pack_physical_experts_to_devices(weights: np.ndarray, num_devices: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy balanced packing with equal physical-expert capacity per device."""
    phy_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    num_physical = phy_weights.size
    if num_devices <= 0 or num_physical % num_devices != 0:
        raise ValueError("number of physical experts must be divisible by number of devices")
    capacity = num_physical // num_devices
    device_load = np.zeros(num_devices, dtype=np.float64)
    device_count = np.zeros(num_devices, dtype=np.int64)
    device_of_phy = np.full(num_physical, -1, dtype=np.int64)
    for phy in np.argsort(phy_weights)[::-1]:
        candidates = np.flatnonzero(device_count < capacity)
        device = int(min(candidates, key=lambda d: (device_load[d], device_count[d], d)))
        device_of_phy[int(phy)] = device
        device_load[device] += phy_weights[int(phy)]
        device_count[device] += 1
    return device_of_phy, device_load


def split_integer_by_replicas(amount: int, replicas: list[int], running_load: np.ndarray) -> list[tuple[int, int]]:
    if amount <= 0 or not replicas:
        return []
    base = amount // len(replicas)
    rem = amount % len(replicas)
    ordered = sorted(replicas, key=lambda phy: (running_load[phy], phy))
    allocations: list[tuple[int, int]] = []
    for idx, phy in enumerate(ordered):
        share = base + (1 if idx < rem else 0)
        if share <= 0:
            continue
        running_load[phy] += share
        allocations.append((phy, share))
    return allocations


def deepseek_eplb_device_matrix(
    expert_matrix: np.ndarray,
    target_devices: int,
    replica_factor: int = 2,
    map_path: Path | None = None,
) -> np.ndarray:
    """Map a logical expert trace to devices using DeepSeek EPLB-style global placement.

    DeepSeek EPLB first replicates hot logical experts and then packs physical
    expert copies to devices. The trace values are not regenerated; each logical
    flow is split over the source and destination replicas implied by the map.
    """
    logical = communication_matrix(expert_matrix)
    n, m = logical.shape
    if n != m:
        raise ValueError("EPLB mapping expects a square logical expert matrix")
    num_physical = n * replica_factor
    if num_physical % target_devices != 0:
        raise ValueError("physical expert count must be divisible by target device count")

    expert_load = logical.sum(axis=0).astype(np.float64)
    phy2log, logcnt = replicate_experts_by_load(expert_load, num_physical)
    phy_weight = np.asarray([expert_load[log] / max(1, logcnt[log]) for log in phy2log], dtype=np.float64)
    device_of_phy, packed_device_load = pack_physical_experts_to_devices(phy_weight, target_devices)
    log2phys: list[list[int]] = [[] for _ in range(n)]
    for phy, logical_expert in enumerate(phy2log.tolist()):
        log2phys[logical_expert].append(phy)

    result = np.zeros((target_devices, target_devices), dtype=np.int64)
    source_running_load = np.zeros(num_physical, dtype=np.float64)
    dest_running_load = np.zeros(num_physical, dtype=np.float64)
    for src in range(n):
        src_replicas = log2phys[src]
        for dst in range(n):
            amount = int(logical[src, dst])
            if amount <= 0:
                continue
            dst_replicas = log2phys[dst]
            for src_phy, src_amount in split_integer_by_replicas(amount, src_replicas, source_running_load):
                for dst_phy, dst_amount in split_integer_by_replicas(src_amount, dst_replicas, dest_running_load):
                    result[int(device_of_phy[src_phy]), int(device_of_phy[dst_phy])] += dst_amount
    result = communication_matrix(result)

    if map_path is not None:
        write_json(
            map_path,
            {
                "algorithm": "DeepSeek EPLB global policy, NumPy reimplementation",
                "official_reference": "https://github.com/deepseek-ai/EPLB",
                "load_definition": "column sums of the diagonal-zeroed logical expert matrix",
                "replica_factor": replica_factor,
                "target_devices": target_devices,
                "num_logical_experts": n,
                "num_physical_experts": num_physical,
                "physical_experts_per_device": num_physical // target_devices,
                "input_offdiag_tokens": int(logical.sum()),
                "output_offdiag_tokens": int(result.sum()),
                "local_tokens_after_mapping": int(logical.sum()) - int(result.sum()),
                "physical_to_logical": phy2log.astype(int).tolist(),
                "logical_replica_count": logcnt.astype(int).tolist(),
                "logical_to_physical": [replicas for replicas in log2phys],
                "device_of_physical": device_of_phy.astype(int).tolist(),
                "packed_device_load_estimate": packed_device_load.tolist(),
            },
        )
    return result


def deepseek_eplb_device_matrix_proportional(
    expert_matrix: np.ndarray,
    target_devices: int,
    replica_factor: int = 2,
) -> np.ndarray:
    """Fast device-level EPLB aggregation for large metric-only sweeps.

    This uses the same load-based replication and balanced physical-expert
    packing as ``deepseek_eplb_device_matrix``.  Instead of materializing each
    integer sub-flow over physical replicas, it aggregates logical flows
    proportionally by the fraction of each logical expert's replicas placed on
    each target device.  The result is rounded back to an integer matrix while
    preserving the rounded off-device total.
    """
    logical = communication_matrix(expert_matrix)
    n, m = logical.shape
    if n != m:
        raise ValueError("EPLB mapping expects a square logical expert matrix")
    num_physical = n * replica_factor
    if num_physical % target_devices != 0:
        raise ValueError("physical expert count must be divisible by target device count")

    expert_load = logical.sum(axis=0).astype(np.float64)
    phy2log, logcnt = replicate_experts_by_load(expert_load, num_physical)
    phy_weight = np.asarray([expert_load[log] / max(1, logcnt[log]) for log in phy2log], dtype=np.float64)
    device_of_phy, _ = pack_physical_experts_to_devices(phy_weight, target_devices)

    logical_to_device = np.zeros((n, target_devices), dtype=np.float64)
    np.add.at(logical_to_device, (phy2log, device_of_phy), 1.0)
    logical_to_device /= np.maximum(logcnt, 1)[:, None]

    mapped = logical_to_device.T @ logical.astype(np.float64, copy=False) @ logical_to_device
    np.fill_diagonal(mapped, 0.0)
    target_total = int(round(float(mapped.sum())))
    result = np.rint(mapped).astype(np.int64)
    np.fill_diagonal(result, 0)
    preserve_offdiag_total(result, target_total)
    return result


def blend_with_balanced(matrix: np.ndarray, alpha: float) -> np.ndarray:
    total = int(matrix.sum())
    balanced = balanced_matrix(matrix.shape, total)
    blended = (1.0 - alpha) * matrix.astype(np.float64) + alpha * balanced.astype(np.float64)
    rounded = np.rint(blended).astype(np.int64)
    np.fill_diagonal(rounded, 0)
    preserve_offdiag_total(rounded, total)
    return rounded


def preserve_offdiag_total(matrix: np.ndarray, target_total: int) -> None:
    if matrix.size == 0:
        return
    n, m = matrix.shape
    offdiag_mask = np.ones(matrix.shape, dtype=bool)
    if n == m:
        np.fill_diagonal(offdiag_mask, False)
    indices = np.flatnonzero(offdiag_mask.reshape(-1))
    if indices.size == 0:
        return
    flat = matrix.reshape(-1)
    diff = int(target_total) - int(flat[indices].sum())
    if diff != 0:
        if diff > 0:
            base = diff // indices.size
            rem = diff % indices.size
            if base:
                flat[indices] += base
            if rem:
                flat[indices[:rem]] += 1
        else:
            remaining = -diff
            order = indices[np.argsort(flat[indices])[::-1]]
            while remaining > 0:
                progressed = False
                for idx in order:
                    if flat[idx] <= 0:
                        continue
                    flat[idx] -= 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
                if not progressed:
                    break


def matrix_metrics(matrix: np.ndarray, top_k: int = TOP_K_DEFAULT) -> dict[str, object]:
    matrix = communication_matrix(matrix)
    n, m = matrix.shape if matrix.size else (0, 0)
    offdiag = offdiag_values(matrix).astype(np.float64)
    offdiag_arr, total, offdiag_sorted = sorted_distribution(offdiag)
    nonzero = offdiag_arr[offdiag_arr > 0]
    nonzero_arr, nonzero_total, nonzero_sorted = sorted_distribution(nonzero)
    row_sums = matrix.sum(axis=1).astype(np.float64) if n else np.array([], dtype=np.float64)
    col_sums = matrix.sum(axis=0).astype(np.float64) if m else np.array([], dtype=np.float64)
    row_arr, row_total, row_sorted = sorted_distribution(row_sums)
    col_arr, col_total, col_sorted = sorted_distribution(col_sums)
    total_cells = offdiag.size
    nonzero_count = int(nonzero.size)
    top_dest_k = max(1, min(top_k, m))
    top_dest_share = top_share_sorted(col_sorted, col_total, top_dest_k / m) if m else 0.0
    top_source_share = top_share_sorted(row_sorted, row_total, top_dest_k / n) if n else 0.0
    flow_top20 = top_share_sorted(offdiag_sorted, total, 0.20)
    flow_skew20 = max(0.0, min(1.0, (flow_top20 - 0.20) / 0.80))
    dest_pressure = 0.0
    if m > top_dest_k and total > 0:
        uniform_dest_share = top_dest_k / m
        dest_pressure = max(0.0, min(1.0, (top_dest_share - uniform_dest_share) / (1.0 - uniform_dest_share)))
    sparsity = 1.0 - (nonzero_count / total_cells) if total_cells else 0.0
    bpi_legacy = 0.50 * flow_skew20 + 0.30 * dest_pressure + 0.20 * sparsity
    bpi_flow_entropy = normalized_entropy_concentration(offdiag_arr)
    bpi_dest_entropy = normalized_entropy_concentration(col_arr)
    bpi_source_entropy = normalized_entropy_concentration(row_arr)
    bpi_flow_hhi = normalized_hhi_concentration(offdiag_arr)
    bpi_dest_hhi = normalized_hhi_concentration(col_arr)
    bpi_source_hhi = normalized_hhi_concentration(row_arr)
    bpi_flow_lorenz80_uniform = normalized_lorenz80_concentration(offdiag_sorted, total)
    bpi_flow_lorenz80 = balanced_adjusted_lorenz80_concentration(offdiag_sorted, total)
    bpi_entropy_max = max(bpi_flow_entropy, bpi_dest_entropy)
    bpi_lorenz_destmax = max(bpi_flow_lorenz80, bpi_dest_entropy)
    return {
        "rows": n,
        "cols": m,
        "total_tokens": int(total),
        "nonzero_offdiag": nonzero_count,
        "offdiag_cells": total_cells,
        "sparsity": sparsity,
        "flow_gini_all": gini_sorted(offdiag_sorted, total),
        "flow_gini_nonzero": gini_sorted(nonzero_sorted, nonzero_total),
        "flow_cv_all": cv(offdiag_arr),
        "flow_cv_nonzero": cv(nonzero_arr),
        "flow_max_to_mean_nonzero": float(np.max(nonzero_arr) / np.mean(nonzero_arr)) if nonzero_arr.size else 0.0,
        "flow_top1pct_share": top_share_sorted(offdiag_sorted, total, 0.01),
        "flow_top5pct_share": top_share_sorted(offdiag_sorted, total, 0.05),
        "flow_top10pct_share": top_share_sorted(offdiag_sorted, total, 0.10),
        "flow_top20pct_share": flow_top20,
        "flow_frac_for_50pct_bytes": top_count_for_share_sorted(offdiag_sorted, total, 0.50),
        "flow_frac_for_80pct_bytes": top_count_for_share_sorted(offdiag_sorted, total, 0.80),
        "flow_frac_for_90pct_bytes": top_count_for_share_sorted(offdiag_sorted, total, 0.90),
        "row_cv": cv(row_arr),
        "col_cv": cv(col_arr),
        "row_gini": gini_sorted(row_sorted, row_total),
        "col_gini": gini_sorted(col_sorted, col_total),
        "topk_source_share": top_source_share,
        "topk_dest_share": top_dest_share,
        "flow_skew20": flow_skew20,
        "dest_pressure_topk": dest_pressure,
        "bpi_legacy_weighted": bpi_legacy,
        "bpi_flow_entropy": bpi_flow_entropy,
        "bpi_dest_entropy": bpi_dest_entropy,
        "bpi_source_entropy": bpi_source_entropy,
        "bpi_entropy_max": bpi_entropy_max,
        "bpi_lorenz_destmax": bpi_lorenz_destmax,
        "bpi_flow_hhi": bpi_flow_hhi,
        "bpi_dest_hhi": bpi_dest_hhi,
        "bpi_source_hhi": bpi_source_hhi,
        "bpi_flow_lorenz80_uniform": bpi_flow_lorenz80_uniform,
        "bpi_flow_lorenz80": bpi_flow_lorenz80,
        "bimodal_pressure_index": bpi_flow_lorenz80,
    }


def discover_trace_files(include_raw_olmoe: bool = True) -> list[Path]:
    roots = [
        REPO_ROOT / "input" / "official_data" / "olmoe_cases",
        REPO_ROOT / "input" / "official_data" / "qwen_cases",
        REPO_ROOT / "input" / "official_data" / "DeepSeekv3.2_cases",
    ]
    if include_raw_olmoe:
        roots.extend(
            [
                REPO_ROOT / "input" / "raw_data" / "olmoe_inf" / "OLMoE_Inference",
                REPO_ROOT / "input" / "raw_data" / "olmoe_inf" / "OLMoE_Inference_BS1024_8192",
            ]
        )
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.rglob("*.csv")))
    return files


def synthetic_case_files() -> list[dict[str, object]]:
    manifest_path = REPO_ROOT / "evaluation_assets" / "manifests" / "synthetic_cases.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text())
    rows: list[dict[str, object]] = []
    for case in payload.get("cases", []):
        generated = REPO_ROOT / case["generated_csv"]
        if generated.exists():
            rows.append({"case": case, "path": generated})
    return rows


def analyze_traces(args: argparse.Namespace) -> None:
    ensure_dirs()
    files = discover_trace_files(include_raw_olmoe=not args.no_raw_olmoe)
    if args.limit:
        files = files[: args.limit]
    rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        matrix = communication_matrix(load_matrix(path))
        model, phase, source = infer_dataset(path)
        meta = parse_case_metadata(path)
        base = {
            "trace_id": path.stem,
            "model": model,
            "phase": phase,
            "source": source,
            "path": rel(path),
            **meta,
        }
        metrics = matrix_metrics(matrix, TOP_K_DEFAULT)
        rows.append({**base, "variant": "real", **metrics})
        total = int(matrix.sum())
        variant_mats = {
            "balanced": balanced_matrix(matrix.shape, total),
            "worst_top8": worst_matrix(matrix.shape, total, TOP_K_DEFAULT),
            "blend_0.25": blend_with_balanced(matrix, 0.25),
            "blend_0.50": blend_with_balanced(matrix, 0.50),
            "blend_0.75": blend_with_balanced(matrix, 0.75),
        }
        for variant, variant_matrix in variant_mats.items():
            variant_rows.append({**base, "variant": variant, **matrix_metrics(variant_matrix, TOP_K_DEFAULT)})
        if index % 1000 == 0:
            print(f"analyzed {index}/{len(files)} trace files", flush=True)

    synthetic_rows: list[dict[str, object]] = []
    for item in synthetic_case_files():
        case = item["case"]
        path = item["path"]
        matrix = communication_matrix(load_matrix(path))
        base = {
            "trace_id": case["case_id"],
            "model": "olmoe",
            "phase": "decode",
            "source": "evaluation_synthetic",
            "path": rel(path),
            "batch_size": case.get("raw_batch_size", ""),
            "actual_batch_size": "",
            "layer": case.get("layer", ""),
            "sample_index": case.get("sample_index", ""),
            "target_devices": case.get("target_devices", ""),
            "size_label": case.get("size_label", ""),
            "source_expert_trace": case.get("source_csv", ""),
            "generated_device_trace": case.get("generated_csv", ""),
            "merge_adjacent_pairs": case.get("merge_adjacent_pairs", ""),
            "scale_factor": case.get("scale_factor", 1),
        }
        synthetic_rows.append({**base, "variant": "real", **matrix_metrics(matrix, TOP_K_DEFAULT)})
        total = int(matrix.sum())
        synthetic_variant_mats = {
            "balanced": balanced_matrix(matrix.shape, total),
            "worst_top8": worst_matrix(matrix.shape, total, TOP_K_DEFAULT),
            "blend_0.25": blend_with_balanced(matrix, 0.25),
            "blend_0.50": blend_with_balanced(matrix, 0.50),
            "blend_0.75": blend_with_balanced(matrix, 0.75),
        }
        source_csv = case.get("source_csv", "")
        target_devices = int(case.get("target_devices", matrix.shape[0]) or matrix.shape[0])
        if source_csv:
            source_path = REPO_ROOT / str(source_csv)
            if source_path.exists():
                expert_matrix = communication_matrix(load_matrix(source_path)) * int(case.get("scale_factor", 1) or 1)
                synthetic_variant_mats["eplb_x2"] = deepseek_eplb_device_matrix(expert_matrix, target_devices, 2)
        for variant, variant_matrix in synthetic_variant_mats.items():
            variant_rows.append({**base, "variant": variant, **matrix_metrics(variant_matrix, TOP_K_DEFAULT)})

    all_metric_rows = rows + synthetic_rows
    write_csv(RESULTS_DIR / "trace_metrics_real_full.csv", all_metric_rows)
    write_csv(RESULTS_DIR / "trace_metrics_reference_variants.csv", variant_rows)
    write_csv(RESULTS_DIR / "trace_metrics_summary.csv", summarize_metrics(all_metric_rows))
    write_csv(RESULTS_DIR / "trace_metrics_reference_variant_summary.csv", summarize_metrics(variant_rows))
    selected = select_trace_suite(all_metric_rows, args.per_group)
    write_csv(RESULTS_DIR / "selected_trace_suite.csv", selected)
    print(
        json.dumps(
            {
                "real_rows": len(all_metric_rows),
                "reference_variant_rows": len(variant_rows),
                "selected_suite_rows": len(selected),
            },
            indent=2,
        )
    )


def parse_target_devices(value: str) -> list[int]:
    targets: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        targets.append(int(item))
    return targets


def official_eplb_metrics(args: argparse.Namespace) -> None:
    ensure_dirs()
    metric_path = RESULTS_DIR / "trace_metrics_real_full.csv"
    if not metric_path.exists():
        raise FileNotFoundError(f"Missing {metric_path}; run metrics first")
    targets = parse_target_devices(args.targets)
    if not targets:
        raise ValueError("at least one target device count is required")

    real_rows = [
        row
        for row in read_csv(metric_path)
        if row.get("source") == "official" and fnum(row, "total_tokens") > 0
    ]
    if args.limit:
        real_rows = real_rows[: args.limit]

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for index, row in enumerate(real_rows, start=1):
        source_path = REPO_ROOT / row["path"]
        matrix = communication_matrix(load_matrix(source_path))
        n, m = matrix.shape if matrix.size else (0, 0)
        base = {
            "trace_id": row.get("trace_id", source_path.stem),
            "model": row.get("model", ""),
            "phase": row.get("phase", ""),
            "source": "official",
            "path": row.get("path", rel(source_path)),
            "batch_size": row.get("batch_size", ""),
            "actual_batch_size": row.get("actual_batch_size", ""),
            "layer": row.get("layer", ""),
            "sample_index": row.get("sample_index", ""),
            "source_expert_trace": row.get("path", rel(source_path)),
            "generated_device_trace": "",
            "merge_adjacent_pairs": "",
            "scale_factor": 1,
        }
        for target in targets:
            if n != m or n <= 0 or (n * 2) % target != 0:
                skipped.append(
                    {
                        "trace_id": base["trace_id"],
                        "path": base["path"],
                        "rows": n,
                        "cols": m,
                        "target_devices": target,
                        "reason": "incompatible square matrix or target device count",
                    }
                )
                continue
            eplb_matrix = deepseek_eplb_device_matrix_proportional(matrix, target, 2)
            rows.append(
                {
                    **base,
                    "variant": "eplb_x2",
                    **matrix_metrics(eplb_matrix, TOP_K_DEFAULT),
                    "target_devices": target,
                    "size_label": "",
                }
            )
        if index % 500 == 0:
            print(f"official EPLB metrics {index}/{len(real_rows)} traces", flush=True)

    output_path = RESULTS_DIR / "trace_metrics_official_eplb.csv"
    write_csv(output_path, rows)
    if skipped:
        write_csv(RESULTS_DIR / "trace_metrics_official_eplb_skipped.csv", skipped)
    print(
        json.dumps(
            {
                "input_official_nonzero_rows": len(real_rows),
                "targets": targets,
                "eplb_rows": len(rows),
                "skipped_rows": len(skipped),
                "output": rel(output_path),
            },
            indent=2,
        )
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model", ""),
            row.get("phase", ""),
            row.get("source", ""),
            row.get("variant", "real"),
        )
        groups[key].append(row)
    metrics = [
        "bimodal_pressure_index",
        "bpi_flow_entropy",
        "bpi_dest_entropy",
        "bpi_entropy_max",
        "bpi_lorenz_destmax",
        "bpi_flow_hhi",
        "bpi_dest_hhi",
        "bpi_flow_lorenz80",
        "bpi_flow_lorenz80_uniform",
        "bpi_legacy_weighted",
        "flow_gini_all",
        "flow_top10pct_share",
        "flow_top20pct_share",
        "flow_frac_for_80pct_bytes",
        "sparsity",
        "col_gini",
        "topk_dest_share",
        "row_cv",
        "col_cv",
    ]
    out: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        model, phase, source, variant = key
        row: dict[str, object] = {
            "model": model,
            "phase": phase,
            "source": source,
            "variant": variant,
            "count": len(items),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in items if item.get(metric, "") != ""]
            row[f"{metric}_mean"] = mean(vals) if vals else 0.0
            row[f"{metric}_p50"] = percentile(vals, 50)
            row[f"{metric}_p90"] = percentile(vals, 90)
            row[f"{metric}_min"] = min(vals) if vals else 0.0
            row[f"{metric}_max"] = max(vals) if vals else 0.0
        out.append(row)
    return out


def select_trace_suite(rows: list[dict[str, object]], per_group: int) -> list[dict[str, object]]:
    del per_group
    selected: list[dict[str, object]] = []
    for row in rows:
        if row.get("variant") != "real":
            continue
        if row.get("model") != "olmoe" or row.get("phase") != "decode":
            continue
        if row.get("source") != "evaluation_synthetic":
            continue
        if int(float(row.get("target_devices", 0) or 0)) not in {32, 64}:
            continue
        if float(row.get("total_tokens", 0) or 0) <= 0:
            continue
        out = dict(row)
        out["selection_label"] = str(row.get("size_label", "synthetic32"))
        selected.append(out)
    size_order = {"1MB": 0, "8MB": 1, "64MB": 2, "256MB": 3, "2GB": 4}
    return sorted(
        selected,
        key=lambda row: (
            size_order.get(str(row.get("size_label", "")), 99),
            int(float(row.get("sample_index", 0) or 0)),
            str(row.get("trace_id", "")),
        ),
    )


def select_suite_from_existing(args: argparse.Namespace) -> None:
    ensure_dirs()
    metrics_path = RESULTS_DIR / "trace_metrics_real_full.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}; run metrics first")
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = select_trace_suite(rows, args.per_group)
    write_csv(RESULTS_DIR / "selected_trace_suite.csv", selected)
    print(json.dumps({"selected_suite_rows": len(selected)}, indent=2))


def resize_matrix(matrix: np.ndarray, target: int) -> np.ndarray | None:
    n, m = matrix.shape
    if n != m:
        return None
    if n == target:
        return matrix.copy()
    if n < target or n % target != 0:
        return None
    factor = n // target
    return matrix.reshape(target, factor, target, factor).sum(axis=(1, 3)).astype(np.int64)


def materialize_suite(args: argparse.Namespace) -> None:
    ensure_dirs()
    suite_path = RESULTS_DIR / "selected_trace_suite.csv"
    if not suite_path.exists():
        raise FileNotFoundError(f"Missing {suite_path}; run metrics first")
    with suite_path.open(newline="") as handle:
        suite = list(csv.DictReader(handle))
    if args.max_suite:
        suite = suite[: args.max_suite]

    rows: list[dict[str, object]] = []
    variants = ["real", "balanced", "worst_top8", "eplb_x2", "blend_0.25", "blend_0.50", "blend_0.75"]
    for entry in suite:
        source_path = REPO_ROOT / entry["path"]
        matrix = communication_matrix(load_matrix(source_path))
        for topology_key, topo in TOPOLOGIES.items():
            target = int(topo["target_devices"])
            if int(float(entry.get("target_devices", 0) or 0)) != target:
                continue
            resized = resize_matrix(matrix, target)
            if resized is None:
                continue
            resized = communication_matrix(resized)
            if int(resized.sum()) <= 0:
                continue
            variant_mats = {
                "real": resized,
                "balanced": balanced_matrix(resized.shape, int(resized.sum())),
                "worst_top8": worst_matrix(resized.shape, int(resized.sum()), TOP_K_DEFAULT),
                "blend_0.25": blend_with_balanced(resized, 0.25),
                "blend_0.50": blend_with_balanced(resized, 0.50),
                "blend_0.75": blend_with_balanced(resized, 0.75),
            }
            for variant in variants:
                case_id = sanitize(
                    f"{entry['model']}_{entry['phase']}_{entry['selection_label']}_{entry['trace_id']}_{target}d_{variant}"
                )
                if variant == "eplb_x2":
                    expert_trace = entry.get("source_expert_trace", "")
                    if not expert_trace:
                        continue
                    expert_path = REPO_ROOT / str(expert_trace)
                    if not expert_path.exists():
                        continue
                    scale_factor = int(float(entry.get("scale_factor", 1) or 1))
                    expert_matrix = communication_matrix(load_matrix(expert_path)) * scale_factor
                    eplb_map_path = GENERATED_DIR / "eplb_maps" / f"{case_id}.json"
                    variant_mats[variant] = deepseek_eplb_device_matrix(expert_matrix, target, 2, eplb_map_path)
                variant_matrix = variant_mats[variant]
                csv_path = GENERATED_DIR / "traces" / f"{case_id}.csv"
                collective_path = GENERATED_DIR / "collectives" / f"{case_id}.json"
                write_matrix(csv_path, variant_matrix)
                write_json(
                    collective_path,
                    {
                        "collective": "alltoallv",
                        "v_datasize": rel(csv_path),
                        "chunkfactor": 1,
                        "block_bytes": TOKEN_BYTES,
                    },
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "model": entry["model"],
                        "phase": entry["phase"],
                        "source": entry["source"],
                        "selection_label": entry["selection_label"],
                        "source_trace": entry["path"],
                        "source_expert_trace": entry.get("source_expert_trace", ""),
                        "merge_adjacent_pairs": entry.get("merge_adjacent_pairs", ""),
                        "scale_factor": entry.get("scale_factor", ""),
                        "size_label": entry.get("size_label", ""),
                        "sample_index": entry.get("sample_index", ""),
                        "source_bpi": entry["bimodal_pressure_index"],
                        "variant": variant,
                        "topology_key": topology_key,
                        "target_devices": target,
                        "topology_json": topo["path"],
                        "csv_path": rel(csv_path),
                        "collective_json": rel(collective_path),
                        "eplb_map_json": (
                            rel(GENERATED_DIR / "eplb_maps" / f"{case_id}.json") if variant == "eplb_x2" else ""
                        ),
                        **matrix_metrics(variant_matrix, TOP_K_DEFAULT),
                    }
                )
    write_csv(RESULTS_DIR / "generated_trace_variant_manifest.csv", rows)
    print(json.dumps({"generated_cases": len(rows)}, indent=2))


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def display_topology_key(value: str) -> str:
    return value.replace("fattree", "clos").replace("fat-tree", "clos")


def parse_log_metrics(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    text = path.read_text(errors="replace")
    makespan = MAKESPAN_RE.search(text)
    solver = SOLVER_RE.search(text)
    return (
        float(makespan.group(1)) if makespan else None,
        float(solver.group(1)) if solver else None,
    )


def parse_schedule_stats(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "halfring_total_rounds": "",
            "halfring_total_steps": "",
            "halfring_stage_substage_events": "",
        }
    text = path.read_text(errors="replace")
    rounds = TOTAL_ROUNDS_RE.search(text)
    steps = TOTAL_STEPS_RE.search(text)
    stage_substage_events = len(re.findall(r"\bStage\s+[0-9]+,\s+Sub-stage\s+[0-9]+", text))
    return {
        "halfring_total_rounds": int(rounds.group(1)) if rounds else "",
        "halfring_total_steps": int(steps.group(1)) if steps else "",
        "halfring_stage_substage_events": stage_substage_events if stage_substage_events else "",
    }


def parse_link_intervals(path: Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    if not path.exists():
        return intervals
    for line in path.read_text(errors="replace").splitlines():
        match = LINK_RE.search(line)
        if not match:
            continue
        edge = f"{match.group(1)}->{match.group(2)}"
        intervals[edge] = [(int(start), int(end)) for start, end in INTERVAL_RE.findall(match.group(3))]
    return intervals


def load_topology_config(topology_path: str | Path) -> dict[str, object]:
    path = REPO_ROOT / str(topology_path)
    key = path.as_posix()
    if key not in TOPOLOGY_CONFIG_CACHE:
        TOPOLOGY_CONFIG_CACHE[key] = json.loads(path.read_text())
    return TOPOLOGY_CONFIG_CACHE[key]


def shape_product(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def endpoint_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("node "):
        return (0, int(name.split()[1]), name)
    if name.startswith("switch "):
        return (1, int(name.split()[1]), name)
    return (2, 0, name)


def normalize_endpoint(name: str, config: dict[str, object]) -> str:
    value = name.strip()
    if value.startswith("node "):
        return value
    if value.startswith("switch "):
        return value
    if NUMERIC_NODE_RE.match(value):
        return f"node {int(value)}"

    topology_type = str(config.get("topology", ""))
    shape = [int(v) for v in config.get("shape", [])]
    switch_shape = [int(v) for v in config.get("switch-shape", [])]

    scale_up = SCALE_UP_SWITCH_RE.match(value)
    if scale_up:
        return f"switch {int(scale_up.group(1))}"
    scale_out = SCALE_OUT_SWITCH_RE.match(value)
    if scale_out:
        if topology_type in {"fat-tree", "rail-optimized"} and shape and switch_shape:
            num_nodes = shape[1] if len(shape) >= 2 else 1
            scale_up_switches = num_nodes * switch_shape[0]
            return f"switch {scale_up_switches + int(scale_out.group(1))}"
        return f"switch {int(scale_out.group(1))}"

    node_switch = CM_NODE_SWITCH_RE.match(value)
    if node_switch and topology_type in {"cm", "cm384"} and switch_shape:
        node_idx = int(node_switch.group(1))
        sw = int(node_switch.group(2))
        return f"switch {node_idx * switch_shape[0] + sw}"
    rail_switch = CM_RAIL_SWITCH_RE.match(value)
    if rail_switch and topology_type in {"cm", "cm384"} and shape and switch_shape:
        num_nodes = shape[1]
        num_node_switches = num_nodes * switch_shape[0]
        rail = int(rail_switch.group(1))
        sw = int(rail_switch.group(2))
        switches_per_rail = switch_shape[1]
        return f"switch {num_node_switches + rail * switches_per_rail + sw}"

    return value


def normalize_edge_name(edge: str, config: dict[str, object]) -> str:
    if "->" not in edge:
        return edge
    src, dst = edge.split("->", 1)
    return f"{normalize_endpoint(src, config)}->{normalize_endpoint(dst, config)}"


def add_directed_edge(graph: dict[str, set[str]], src: str, dst: str) -> None:
    graph[src].add(dst)
    graph.setdefault(dst, set())


def add_bidirectional_edge(graph: dict[str, set[str]], src: str, dst: str) -> None:
    add_directed_edge(graph, src, dst)
    add_directed_edge(graph, dst, src)


def build_topology_graph_from_config(config: dict[str, object]) -> tuple[list[str], dict[str, list[str]]]:
    graph: dict[str, set[str]] = defaultdict(set)
    topology_type = str(config.get("topology", ""))
    shape = [int(v) for v in config.get("shape", [])]
    switch_shape = [int(v) for v in config.get("switch-shape", [])]

    if topology_type == "mesh":
        num_gpus = shape_product(shape)
        for gpu in range(num_gpus):
            graph.setdefault(f"node {gpu}", set())

        # Glaive's direct mesh solver uses row-major coordinates, while the
        # baseline Topology::Mesh object historically expands strides from the
        # first dimension.  The analysis graph uses their union so fabric-level
        # denominators stay fixed across methods in this rebuttal suite.
        strides = []
        for dim in range(len(shape)):
            stride = 1
            for later in range(dim + 1, len(shape)):
                stride *= shape[later]
            strides.append(stride)
        for src in range(num_gpus):
            remaining = src
            coord = []
            for stride, dim_size in zip(strides, shape):
                coord.append(remaining // stride)
                remaining %= stride
            for dim, dim_size in enumerate(shape):
                if coord[dim] >= dim_size - 1:
                    continue
                neighbor_coord = list(coord)
                neighbor_coord[dim] += 1
                dst = sum(value * stride for value, stride in zip(neighbor_coord, strides))
                add_bidirectional_edge(graph, f"node {src}", f"node {dst}")

        stride = 1
        for dim_size in shape:
            for src in range(num_gpus):
                if (src // stride) % dim_size < dim_size - 1:
                    add_bidirectional_edge(graph, f"node {src}", f"node {src + stride}")
            stride *= dim_size

    elif topology_type == "fat-tree":
        gpus_per_node = shape[0]
        num_nodes = shape[1] if len(shape) >= 2 else 1
        num_gpus = gpus_per_node * num_nodes
        scale_up_per_node = switch_shape[0]
        total_scale_up = num_nodes * scale_up_per_node
        total_scale_out = sum(switch_shape[1:])
        for gpu in range(num_gpus):
            graph.setdefault(f"node {gpu}", set())
        for switch in range(total_scale_up + total_scale_out):
            graph.setdefault(f"switch {switch}", set())

        for node_idx in range(num_nodes):
            node_gpu_base = node_idx * gpus_per_node
            node_switch_base = node_idx * scale_up_per_node
            for gpu_offset in range(gpus_per_node):
                gpu = f"node {node_gpu_base + gpu_offset}"
                for sw_offset in range(scale_up_per_node):
                    add_bidirectional_edge(graph, gpu, f"switch {node_switch_base + sw_offset}")

        if len(switch_shape) > 1:
            layer_base = total_scale_up
            layer_count = switch_shape[1]
            for gpu in range(num_gpus):
                for sw in range(layer_count):
                    add_bidirectional_edge(graph, f"node {gpu}", f"switch {layer_base + sw}")
            current_base = layer_base
            current_count = layer_count
            for layer_idx in range(2, len(switch_shape)):
                next_base = current_base + current_count
                next_count = switch_shape[layer_idx]
                for current in range(current_count):
                    for nxt in range(next_count):
                        add_bidirectional_edge(
                            graph,
                            f"switch {current_base + current}",
                            f"switch {next_base + nxt}",
                        )
                current_base = next_base
                current_count = next_count

    elif topology_type == "rail-optimized":
        gpus_per_node = shape[0]
        num_nodes = shape[1] if len(shape) >= 2 else 1
        num_gpus = gpus_per_node * num_nodes
        scale_up_per_node = switch_shape[0]
        total_scale_up = num_nodes * scale_up_per_node
        total_scale_out = sum(switch_shape[1:])
        for gpu in range(num_gpus):
            graph.setdefault(f"node {gpu}", set())
        for switch in range(total_scale_up + total_scale_out):
            graph.setdefault(f"switch {switch}", set())

        for node_idx in range(num_nodes):
            node_gpu_base = node_idx * gpus_per_node
            node_switch_base = node_idx * scale_up_per_node
            for gpu_offset in range(gpus_per_node):
                gpu = f"node {node_gpu_base + gpu_offset}"
                for sw_offset in range(scale_up_per_node):
                    add_bidirectional_edge(graph, gpu, f"switch {node_switch_base + sw_offset}")

        if len(switch_shape) > 1:
            scale_out_base = total_scale_up
            scale_out_count = switch_shape[1]
            for node_idx in range(num_nodes):
                node_gpu_base = node_idx * gpus_per_node
                for gpu_offset in range(gpus_per_node):
                    if gpu_offset < scale_out_count:
                        add_bidirectional_edge(
                            graph,
                            f"node {node_gpu_base + gpu_offset}",
                            f"switch {scale_out_base + gpu_offset}",
                        )
            current_base = scale_out_base
            current_count = scale_out_count
            for layer_idx in range(2, len(switch_shape)):
                next_base = current_base + current_count
                next_count = switch_shape[layer_idx]
                for current in range(current_count):
                    for nxt in range(next_count):
                        add_bidirectional_edge(
                            graph,
                            f"switch {current_base + current}",
                            f"switch {next_base + nxt}",
                        )
                current_base = next_base
                current_count = next_count

    elif topology_type in {"cm", "cm384"}:
        gpus_per_node = shape[0]
        num_nodes = shape[1]
        num_gpus = gpus_per_node * num_nodes
        switches_per_node = switch_shape[0]
        switches_per_rail = switch_shape[1]
        num_node_switches = num_nodes * switches_per_node
        num_rail_switches = switches_per_node * switches_per_rail
        for gpu in range(num_gpus):
            graph.setdefault(f"node {gpu}", set())
        for switch in range(num_node_switches + num_rail_switches):
            graph.setdefault(f"switch {switch}", set())

        for node_idx in range(num_nodes):
            node_gpu_base = node_idx * gpus_per_node
            for gpu_offset in range(0, gpus_per_node - 1, 2):
                add_bidirectional_edge(
                    graph,
                    f"node {node_gpu_base + gpu_offset}",
                    f"node {node_gpu_base + gpu_offset + 1}",
                )
            node_switch_base = node_idx * switches_per_node
            for gpu_offset in range(gpus_per_node):
                for sw in range(switches_per_node):
                    add_bidirectional_edge(
                        graph,
                        f"node {node_gpu_base + gpu_offset}",
                        f"switch {node_switch_base + sw}",
                    )
        rail_switch_base = num_node_switches
        for node_idx in range(num_nodes):
            for rail_idx in range(switches_per_node):
                node_switch = f"switch {node_idx * switches_per_node + rail_idx}"
                for sw in range(switches_per_rail):
                    add_bidirectional_edge(
                        graph,
                        node_switch,
                        f"switch {rail_switch_base + rail_idx * switches_per_rail + sw}",
                    )

    nodes = sorted(graph.keys(), key=endpoint_sort_key)
    adjacency = {node: sorted(neighbors, key=endpoint_sort_key) for node, neighbors in graph.items()}
    return nodes, adjacency


def topology_graph(topology_path: str | Path) -> tuple[list[str], dict[str, list[str]]]:
    key = (REPO_ROOT / str(topology_path)).as_posix()
    if key not in TOPOLOGY_GRAPH_CACHE:
        TOPOLOGY_GRAPH_CACHE[key] = build_topology_graph_from_config(load_topology_config(topology_path))
    return TOPOLOGY_GRAPH_CACHE[key]


def topology_edges(topology_path: str | Path) -> list[str]:
    _, graph = topology_graph(topology_path)
    return [f"{src}->{dst}" for src in sorted(graph, key=endpoint_sort_key) for dst in graph[src]]


def normalized_edge_stats(
    intervals: dict[str, list[tuple[int, int]]],
    topology_path: str | Path,
) -> tuple[dict[str, dict[str, float]], int]:
    config = load_topology_config(topology_path)
    edges = topology_edges(topology_path)
    stats_by_edge = {
        edge: {
            "busy_ns": 0.0,
            "interval_count": 0.0,
            "last_finish_ns": 0.0,
        }
        for edge in edges
    }
    unknown_edges = 0
    for raw_edge, edge_intervals in intervals.items():
        edge = normalize_edge_name(raw_edge, config)
        if edge not in stats_by_edge:
            stats_by_edge[edge] = {
                "busy_ns": 0.0,
                "interval_count": 0.0,
                "last_finish_ns": 0.0,
            }
            unknown_edges += 1
        for start, end in edge_intervals:
            duration = max(0, end - start)
            if duration <= 0:
                continue
            stats_by_edge[edge]["busy_ns"] += float(duration)
            stats_by_edge[edge]["interval_count"] += 1.0
            stats_by_edge[edge]["last_finish_ns"] = max(stats_by_edge[edge]["last_finish_ns"], float(end))
    return stats_by_edge, unknown_edges


def normalized_busy_by_edge(
    intervals: dict[str, list[tuple[int, int]]],
    topology_path: str | Path,
) -> tuple[dict[str, float], int]:
    stats_by_edge, unknown_edges = normalized_edge_stats(intervals, topology_path)
    return {edge: stats["busy_ns"] for edge, stats in stats_by_edge.items()}, unknown_edges


def jain_index(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    denom = float(np.sum(arr * arr))
    if arr.size == 0 or denom <= 0:
        return 0.0
    return float((np.sum(arr) ** 2) / (arr.size * denom))


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64).reshape(-1)
    rhs = np.asarray(right, dtype=np.float64).reshape(-1)
    if lhs.size != rhs.size or lhs.size <= 1:
        return 0.0
    lhs = lhs - float(lhs.mean())
    rhs = rhs - float(rhs.mean())
    denom = float(np.sqrt(np.sum(lhs * lhs) * np.sum(rhs * rhs)))
    if denom <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(np.sum(lhs * rhs) / denom)))


def weighted_percentile(values: list[float], weights: list[float], q: float) -> float:
    if not values or not weights:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    total = float(weight.sum())
    if total <= 0:
        return 0.0
    order = np.argsort(arr)
    arr = arr[order]
    weight = weight[order]
    cdf = np.cumsum(weight)
    return float(arr[np.searchsorted(cdf, q / 100.0 * total, side="left")])


def temporal_link_activity_metrics(
    intervals: dict[str, list[tuple[int, int]]],
    fabric_link_count: int,
) -> dict[str, object]:
    if fabric_link_count <= 0 or not intervals:
        return {
            "fabric_active_link_fraction_mean": 0,
            "fabric_active_link_fraction_p50": 0,
            "fabric_active_link_fraction_p95": 0,
            "fabric_active_link_fraction_max": 0,
            "fabric_active_link_count_mean": 0,
            "fabric_active_link_count_p95": 0,
        }
    events: dict[int, int] = defaultdict(int)
    for edge_intervals in intervals.values():
        for start, end in edge_intervals:
            if end <= start:
                continue
            events[start] += 1
            events[end] -= 1
    if not events:
        return {
            "fabric_active_link_fraction_mean": 0,
            "fabric_active_link_fraction_p50": 0,
            "fabric_active_link_fraction_p95": 0,
            "fabric_active_link_fraction_max": 0,
            "fabric_active_link_count_mean": 0,
            "fabric_active_link_count_p95": 0,
        }
    active = 0
    previous: int | None = None
    counts: list[float] = []
    durations: list[float] = []
    for timestamp in sorted(events):
        if previous is not None and timestamp > previous:
            duration = float(timestamp - previous)
            counts.append(float(active))
            durations.append(duration)
        active += events[timestamp]
        previous = timestamp
    total_time = float(sum(durations))
    if total_time <= 0:
        return {
            "fabric_active_link_fraction_mean": 0,
            "fabric_active_link_fraction_p50": 0,
            "fabric_active_link_fraction_p95": 0,
            "fabric_active_link_fraction_max": 0,
            "fabric_active_link_count_mean": 0,
            "fabric_active_link_count_p95": 0,
        }
    weighted_mean = float(np.average(np.asarray(counts), weights=np.asarray(durations)))
    return {
        "fabric_active_link_fraction_mean": weighted_mean / fabric_link_count,
        "fabric_active_link_fraction_p50": weighted_percentile(
            [count / fabric_link_count for count in counts], durations, 50
        ),
        "fabric_active_link_fraction_p95": weighted_percentile(
            [count / fabric_link_count for count in counts], durations, 95
        ),
        "fabric_active_link_fraction_max": max(counts) / fabric_link_count if counts else 0.0,
        "fabric_active_link_count_mean": weighted_mean,
        "fabric_active_link_count_p95": weighted_percentile(counts, durations, 95),
    }


def fabric_balance_metrics(
    intervals: dict[str, list[tuple[int, int]]],
    makespan_us: float | None,
    topology_path: str | Path | None,
) -> dict[str, object]:
    if not topology_path:
        return {}
    busy_by_edge, unknown_edges = normalized_busy_by_edge(intervals, topology_path)
    busy = np.asarray(list(busy_by_edge.values()), dtype=np.float64)
    fabric_link_count = int(busy.size)
    if fabric_link_count == 0:
        return {}
    used = int(np.count_nonzero(busy > 0))
    makespan_ns = (makespan_us or 0.0) * 1000.0
    util = busy / makespan_ns * 100.0 if makespan_ns > 0 else np.zeros_like(busy)
    max_busy = float(busy.max()) if busy.size else 0.0
    total_busy = float(busy.sum())
    metrics: dict[str, object] = {
        "fabric_link_count": fabric_link_count,
        "fabric_unknown_link_count": unknown_edges,
        "fabric_used_links": used,
        "fabric_used_link_ratio": used / fabric_link_count if fabric_link_count else 0.0,
        "fabric_zero_busy_link_ratio": 1.0 - used / fabric_link_count if fabric_link_count else 0.0,
        "fabric_busy_time_total_ns": total_busy,
        "fabric_busy_time_max_ns": max_busy,
        "fabric_busy_time_mean_ns": float(busy.mean()) if busy.size else 0.0,
        "fabric_busy_time_p95_ns": percentile(busy.tolist(), 95),
        "fabric_busy_time_cv": cv(busy),
        "fabric_busy_time_gini": gini(busy),
        "fabric_busy_time_jain": jain_index(busy),
        "fabric_busy_time_top10_share": top_share(busy, 0.10),
        "fabric_utilization_max_pct": float(util.max()) if util.size else 0.0,
        "fabric_utilization_mean_pct": float(util.mean()) if util.size else 0.0,
        "fabric_work_parallelism": total_busy / max_busy if max_busy > 0 else 0.0,
    }
    metrics.update(temporal_link_activity_metrics(intervals, fabric_link_count))
    return metrics


def shortest_path(graph: dict[str, list[str]], src: str, dst: str) -> list[str]:
    if src == dst:
        return [src]
    queue: deque[str] = deque([src])
    parent: dict[str, str | None] = {src: None}
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            if neighbor == dst:
                path = [dst]
                current = dst
                while parent[current] is not None:
                    current = parent[current] or src
                    path.append(current)
                return list(reversed(path))
            queue.append(neighbor)
    return []


def deterministic_route_profile(csv_path: str | Path, topology_path: str | Path) -> dict[str, object]:
    key = ((REPO_ROOT / str(csv_path)).as_posix(), (REPO_ROOT / str(topology_path)).as_posix())
    if key in DETERMINISTIC_ROUTING_CACHE:
        return DETERMINISTIC_ROUTING_CACHE[key]

    matrix = communication_matrix(load_matrix(REPO_ROOT / str(csv_path)))
    edges = topology_edges(topology_path)
    _, graph = topology_graph(topology_path)
    demand_by_edge = {edge: 0.0 for edge in edges}
    nonzero_flows: list[tuple[int, int, float, list[str]]] = []
    missing_paths = 0
    n, m = matrix.shape if matrix.size else (0, 0)
    for src in range(n):
        for dst in range(m):
            amount = float(matrix[src, dst])
            if amount <= 0 or src == dst:
                continue
            path = shortest_path(graph, f"node {src}", f"node {dst}")
            if len(path) < 2:
                missing_paths += 1
                continue
            nonzero_flows.append((src, dst, amount, path))
            for hop_src, hop_dst in zip(path, path[1:]):
                demand_by_edge[f"{hop_src}->{hop_dst}"] = demand_by_edge.get(f"{hop_src}->{hop_dst}", 0.0) + amount

    demand = np.asarray(list(demand_by_edge.values()), dtype=np.float64)
    nonzero_edges = [edge for edge, value in demand_by_edge.items() if value > 0]
    hot_count = max(1, int(math.ceil(len(nonzero_edges) * 0.10))) if nonzero_edges else 0
    hot_edges = {
        edge
        for edge, _ in sorted(
            ((edge, demand_by_edge[edge]) for edge in nonzero_edges),
            key=lambda item: (-item[1], item[0]),
        )[:hot_count]
    }
    total_demand = float(demand.sum())
    profile = {
        "demand_by_edge": demand_by_edge,
        "hot_edges": hot_edges,
        "det_route_edge_count": len(edges),
        "det_route_active_edge_count": len(nonzero_edges),
        "det_route_active_edge_ratio": len(nonzero_edges) / len(edges) if edges else 0.0,
        "det_route_missing_flow_paths": missing_paths,
        "det_route_edge_demand_gini": gini(demand),
        "det_route_edge_demand_top10_share": top_share(demand, 0.10),
        "pred_hot_edge_count": hot_count,
        "pred_hot_edge_demand_share": (
            sum(demand_by_edge[edge] for edge in hot_edges) / total_demand if total_demand > 0 else 0.0
        ),
    }
    flow_total = float(sum(amount for _, _, amount, _ in nonzero_flows))
    hot_flow_count = max(1, int(math.ceil(len(nonzero_flows) * 0.10))) if nonzero_flows else 0
    hot_flows = sorted(nonzero_flows, key=lambda item: (-item[2], item[0], item[1]))[:hot_flow_count]
    hot_flow_path_demand_by_edge = {edge: 0.0 for edge in edges}
    hot_flow_path_edges: set[str] = set()
    hot_flow_bytes = 0.0
    for _, _, amount, path in hot_flows:
        hot_flow_bytes += amount
        for hop_src, hop_dst in zip(path, path[1:]):
            edge = f"{hop_src}->{hop_dst}"
            hot_flow_path_edges.add(edge)
            hot_flow_path_demand_by_edge[edge] = hot_flow_path_demand_by_edge.get(edge, 0.0) + amount
    hot_flow_path_demand_total = float(sum(hot_flow_path_demand_by_edge.values()))
    profile.update(
        {
            "hot_flow_path_edges": hot_flow_path_edges,
            "hot_flow_path_demand_by_edge": hot_flow_path_demand_by_edge,
            "hot_flow_count": hot_flow_count,
            "hot_flow_traffic_share": hot_flow_bytes / flow_total if flow_total > 0 else 0.0,
            "hot_flow_path_edge_count": len(hot_flow_path_edges),
            "hot_flow_path_demand_share": (
                hot_flow_path_demand_total / total_demand if total_demand > 0 else 0.0
            ),
        }
    )
    DETERMINISTIC_ROUTING_CACHE[key] = profile
    return profile


def predicted_hotspot_metrics(
    intervals: dict[str, list[tuple[int, int]]],
    makespan_us: float | None,
    csv_path: str | Path | None,
    topology_path: str | Path | None,
) -> dict[str, object]:
    if not csv_path or not topology_path:
        return {}
    profile = deterministic_route_profile(csv_path, topology_path)
    edge_stats, _ = normalized_edge_stats(intervals, topology_path)
    busy_by_edge = {edge: stats["busy_ns"] for edge, stats in edge_stats.items()}
    count_by_edge = {edge: stats["interval_count"] for edge, stats in edge_stats.items()}
    finish_by_edge = {edge: stats["last_finish_ns"] for edge, stats in edge_stats.items()}
    hot_edges = set(profile["hot_edges"])
    all_edges = set(busy_by_edge)
    cold_edges = all_edges - hot_edges
    total_busy = float(sum(busy_by_edge.values()))
    hot_busy_values = np.asarray([busy_by_edge.get(edge, 0.0) for edge in hot_edges], dtype=np.float64)
    cold_busy_values = np.asarray([busy_by_edge.get(edge, 0.0) for edge in cold_edges], dtype=np.float64)
    demand_by_edge = profile["demand_by_edge"]
    demand_values = np.asarray([demand_by_edge.get(edge, 0.0) for edge in sorted(all_edges)], dtype=np.float64)
    busy_values = np.asarray([busy_by_edge.get(edge, 0.0) for edge in sorted(all_edges)], dtype=np.float64)
    count_values = np.asarray([count_by_edge.get(edge, 0.0) for edge in sorted(all_edges)], dtype=np.float64)
    hot_busy = float(hot_busy_values.sum()) if hot_busy_values.size else 0.0
    cold_busy = float(cold_busy_values.sum()) if cold_busy_values.size else 0.0
    makespan_ns = (makespan_us or 0.0) * 1000.0
    hot_util = hot_busy_values / makespan_ns * 100.0 if makespan_ns > 0 else np.zeros_like(hot_busy_values)
    profile_public = {
        key: value
        for key, value in profile.items()
        if key not in {"demand_by_edge", "hot_edges", "hot_flow_path_edges", "hot_flow_path_demand_by_edge"}
    }
    hot_flow_edges = set(profile["hot_flow_path_edges"])
    hot_flow_cold_edges = all_edges - hot_flow_edges
    hot_flow_busy_values = np.asarray([busy_by_edge.get(edge, 0.0) for edge in hot_flow_edges], dtype=np.float64)
    hot_flow_count_values = np.asarray([count_by_edge.get(edge, 0.0) for edge in hot_flow_edges], dtype=np.float64)
    hot_flow_finish_values = np.asarray(
        [finish_by_edge.get(edge, 0.0) for edge in hot_flow_edges if finish_by_edge.get(edge, 0.0) > 0],
        dtype=np.float64,
    )
    hot_flow_cold_busy_values = np.asarray(
        [busy_by_edge.get(edge, 0.0) for edge in hot_flow_cold_edges],
        dtype=np.float64,
    )
    hot_flow_path_demand_by_edge = profile["hot_flow_path_demand_by_edge"]
    hot_flow_path_demand_values = np.asarray(
        [hot_flow_path_demand_by_edge.get(edge, 0.0) for edge in sorted(all_edges)],
        dtype=np.float64,
    )
    hot_flow_path_total_busy = float(hot_flow_busy_values.sum()) if hot_flow_busy_values.size else 0.0
    hot_flow_cold_total_busy = float(hot_flow_cold_busy_values.sum()) if hot_flow_cold_busy_values.size else 0.0
    profile_public.update(
        {
            "pred_hot_edge_actual_busy_share": hot_busy / total_busy if total_busy > 0 else 0.0,
            "pred_hot_edge_actual_vs_demand_delta": (
                hot_busy / total_busy - float(profile["pred_hot_edge_demand_share"]) if total_busy > 0 else 0.0
            ),
            "det_demand_vs_actual_busy_corr": pearson_corr(demand_values, busy_values),
            "det_demand_vs_actual_interval_corr": pearson_corr(demand_values, count_values),
            "pred_hot_edge_busy_mean_ns": float(hot_busy_values.mean()) if hot_busy_values.size else 0.0,
            "pred_hot_edge_busy_max_ns": float(hot_busy_values.max()) if hot_busy_values.size else 0.0,
            "pred_hot_edge_busy_gini": gini(hot_busy_values),
            "pred_hot_edge_utilization_mean_pct": float(hot_util.mean()) if hot_util.size else 0.0,
            "pred_hot_edge_utilization_max_pct": float(hot_util.max()) if hot_util.size else 0.0,
            "pred_cold_edge_busy_mean_ns": float(cold_busy_values.mean()) if cold_busy_values.size else 0.0,
            "pred_cold_edge_used_ratio": (
                float(np.count_nonzero(cold_busy_values > 0)) / cold_busy_values.size if cold_busy_values.size else 0.0
            ),
            "pred_hot_to_cold_busy_ratio": (
                float(hot_busy_values.mean()) / float(cold_busy_values.mean())
                if hot_busy_values.size and cold_busy_values.size and float(cold_busy_values.mean()) > 0
                else 0.0
            ),
            "pred_hot_edge_busy_total_ns": hot_busy,
            "pred_cold_edge_busy_total_ns": cold_busy,
            "hot_flow_path_actual_busy_share": (
                hot_flow_path_total_busy / total_busy if total_busy > 0 else 0.0
            ),
            "hot_flow_path_actual_vs_demand_delta": (
                hot_flow_path_total_busy / total_busy - float(profile["hot_flow_path_demand_share"])
                if total_busy > 0
                else 0.0
            ),
            "hot_flow_path_busy_mean_ns": (
                float(hot_flow_busy_values.mean()) if hot_flow_busy_values.size else 0.0
            ),
            "hot_flow_path_busy_max_ns": (
                float(hot_flow_busy_values.max()) if hot_flow_busy_values.size else 0.0
            ),
            "hot_flow_path_interval_count_total": (
                float(hot_flow_count_values.sum()) if hot_flow_count_values.size else 0.0
            ),
            "hot_flow_path_interval_count_mean": (
                float(hot_flow_count_values.mean()) if hot_flow_count_values.size else 0.0
            ),
            "hot_flow_path_finish_p95_ns": (
                percentile(hot_flow_finish_values.tolist(), 95) if hot_flow_finish_values.size else 0.0
            ),
            "hot_flow_path_finish_max_ns": (
                float(hot_flow_finish_values.max()) if hot_flow_finish_values.size else 0.0
            ),
            "hot_flow_path_to_other_busy_ratio": (
                float(hot_flow_busy_values.mean()) / float(hot_flow_cold_busy_values.mean())
                if hot_flow_busy_values.size
                and hot_flow_cold_busy_values.size
                and float(hot_flow_cold_busy_values.mean()) > 0
                else 0.0
            ),
            "hot_flow_path_cold_edge_used_ratio": (
                float(np.count_nonzero(hot_flow_cold_busy_values > 0)) / hot_flow_cold_busy_values.size
                if hot_flow_cold_busy_values.size
                else 0.0
            ),
            "hot_flow_path_demand_vs_actual_busy_corr": pearson_corr(hot_flow_path_demand_values, busy_values),
            "hot_flow_path_busy_total_ns": hot_flow_path_total_busy,
            "hot_flow_path_other_busy_total_ns": hot_flow_cold_total_busy,
        }
    )
    return profile_public


def link_balance_metrics(intervals: dict[str, list[tuple[int, int]]], makespan_us: float | None = None) -> dict[str, object]:
    busy_ns = []
    use_counts = []
    for edge_intervals in intervals.values():
        use_counts.append(len(edge_intervals))
        busy_ns.append(sum(max(0, end - start) for start, end in edge_intervals))
    if not busy_ns:
        return {
            "used_links": 0,
            "link_use_count_sum": 0,
            "link_use_count_max": 0,
            "busy_time_max_ns": 0,
            "busy_time_mean_ns": 0,
            "busy_time_cv": 0,
            "busy_time_gini": 0,
            "busy_time_jain": 0,
            "busy_time_top10_share": 0,
            "utilization_max_pct": 0,
            "utilization_mean_pct": 0,
        }
    busy = np.asarray(busy_ns, dtype=np.float64)
    counts = np.asarray(use_counts, dtype=np.float64)
    jain = float((busy.sum() ** 2) / (len(busy) * np.sum(busy * busy))) if np.sum(busy * busy) > 0 else 0.0
    makespan_ns = (makespan_us or 0.0) * 1000.0
    util = busy / makespan_ns * 100.0 if makespan_ns > 0 else np.zeros_like(busy)
    return {
        "used_links": int(len(busy)),
        "link_use_count_sum": int(counts.sum()),
        "link_use_count_mean": float(counts.mean()),
        "link_use_count_max": int(counts.max()),
        "link_use_count_cv": cv(counts),
        "link_use_count_gini": gini(counts),
        "busy_time_max_ns": int(busy.max()),
        "busy_time_mean_ns": float(busy.mean()),
        "busy_time_p95_ns": percentile(busy.tolist(), 95),
        "busy_time_cv": cv(busy),
        "busy_time_gini": gini(busy),
        "busy_time_jain": jain,
        "busy_time_top10_share": top_share(busy, 0.10),
        "utilization_max_pct": float(util.max()) if util.size else 0.0,
        "utilization_mean_pct": float(util.mean()) if util.size else 0.0,
    }


def analyze_existing_network_logs(_: argparse.Namespace) -> None:
    ensure_dirs()
    rows_by_key: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    log_roots = [
        ("evaluation_assets_link", REPO_ROOT / "evaluation_assets" / "raw_logs" / "link"),
        ("evaluation_assets_synthetic", REPO_ROOT / "evaluation_assets" / "raw_logs" / "synthetic"),
        (
            "evaluation_assets_link",
            REPO_ROOT / "rebuttal" / "eval_style_torus_halfringdr_refresh" / "raw_logs" / "link",
        ),
        (
            "evaluation_assets_synthetic",
            REPO_ROOT / "rebuttal" / "eval_style_torus_halfringdr_refresh" / "raw_logs" / "synthetic",
        ),
    ]
    for suite, root in log_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.log")):
            parts = path.relative_to(root).parts
            topology_key = display_topology_key(parts[0] if len(parts) > 0 else "")
            method = parts[1] if len(parts) > 1 else ""
            size_label = parts[2] if suite.endswith("synthetic") and len(parts) > 3 else "256MB"
            sample = path.stem
            makespan, solver = parse_log_metrics(path)
            intervals = parse_link_intervals(path)
            row = {
                "suite": suite,
                "topology_key": topology_key,
                "method": method,
                "size_label": size_label,
                "sample": sample,
                "log_path": rel(path),
                "makespan_us": makespan if makespan is not None else "",
                "solver_time_us": solver if solver is not None else "",
                **link_balance_metrics(intervals, makespan),
            }
            rows_by_key[(suite, topology_key, method, size_label, sample)] = row
    rows = list(rows_by_key.values())
    write_csv(RESULTS_DIR / "network_balance_existing_link_logs.csv", rows)
    write_csv(RESULTS_DIR / "network_balance_existing_link_logs_summary.csv", summarize_network(rows))
    print(json.dumps({"network_log_rows": len(rows)}, indent=2))


def summarize_network(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("suite")),
                str(row.get("topology_key")),
                str(row.get("method")),
                str(row.get("size_label")),
            )
        ].append(row)
    metrics = [
        "makespan_us",
        "used_links",
        "link_use_count_sum",
        "link_use_count_max",
        "busy_time_cv",
        "busy_time_gini",
        "busy_time_jain",
        "busy_time_top10_share",
        "utilization_max_pct",
        "utilization_mean_pct",
    ]
    out: list[dict[str, object]] = []
    for (suite, topology, method, size_label), items in sorted(groups.items()):
        row: dict[str, object] = {
            "suite": suite,
            "topology_key": topology,
            "method": method,
            "size_label": size_label,
            "count": len(items),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in items if item.get(metric, "") != ""]
            row[f"{metric}_mean"] = mean(vals) if vals else 0.0
            row[f"{metric}_p50"] = percentile(vals, 50)
            row[f"{metric}_max"] = max(vals) if vals else 0.0
        out.append(row)
    return out


def runtime_command() -> list[str]:
    binary = REPO_ROOT / "build" / "bin" / "tacos"
    return [str(binary)]


def run_performance_suite(args: argparse.Namespace) -> None:
    ensure_dirs()
    manifest_path = RESULTS_DIR / "generated_trace_variant_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}; run materialize-suite first")
    with manifest_path.open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    if args.case_filter:
        pattern = re.compile(args.case_filter)
        cases = [case for case in cases if pattern.search(case["case_id"]) or pattern.search(case["topology_key"])]
    if args.max_cases:
        cases = cases[: args.max_cases]
    tasks = []
    for case in cases:
        methods = METHODS_64 if int(case["target_devices"]) == 64 else METHODS_32
        for method in methods:
            log_path = LOGS_DIR / "performance" / case["topology_key"] / method / f"{case['case_id']}.log"
            tasks.append((case, method, log_path))
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    print(json.dumps({"tasks": len(tasks), "workers": args.workers}, indent=2))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(run_one_task, case, method, log_path, args.force): (case, method, log_path)
            for case, method, log_path in tasks
        }
        failures: list[dict[str, object]] = []
        for future in as_completed(future_map):
            case, method, _ = future_map[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"failed:exception:{exc}"
            if str(status).startswith("failed"):
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "topology_key": case["topology_key"],
                        "method": method,
                        "status": status,
                    }
                )
            print(f"{status} {case['case_id']} {case['topology_key']} {method}", flush=True)
    if failures:
        write_csv(RESULTS_DIR / "performance_failures.csv", failures)
    elif (RESULTS_DIR / "performance_failures.csv").exists():
        write_csv(RESULTS_DIR / "performance_failures.csv", [])
    parse_performance_logs(args)


def run_one_task(case: dict[str, str], method: str, log_path: Path, force: bool) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and log_path.exists() and log_path.stat().st_size > 0:
        text = log_path.read_text(errors="replace")
        if "[TACOS Solver3] Done!" in text or "[TACOS Baseline] Done!" in text:
            return "skip"
    command = runtime_command() + [
        str(REPO_ROOT / case["topology_json"]),
        str(REPO_ROOT / case["collective_json"]),
        *METHOD_ARGS[method],
    ]
    env = os.environ.copy()
    gcc_runtime = "/hpc2ssd/Edatools/opensource/compile/gcc-12.2.0/lib64"
    env["LD_LIBRARY_PATH"] = f"{gcc_runtime}:{env.get('LD_LIBRARY_PATH', '')}"
    with log_path.open("w") as handle:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False, env=env)
    return "done" if result.returncode == 0 else f"failed:{result.returncode}"


def parse_performance_logs(_: argparse.Namespace) -> None:
    manifest_path = RESULTS_DIR / "generated_trace_variant_manifest.csv"
    if not manifest_path.exists():
        return
    with manifest_path.open(newline="") as handle:
        cases = {(row["case_id"], row["topology_key"]): row for row in csv.DictReader(handle)}
    rows: list[dict[str, object]] = []
    for log_path in sorted((LOGS_DIR / "performance").rglob("*.log")):
        method = log_path.parent.name
        topology_key = log_path.parent.parent.name
        case_id = log_path.stem
        case = cases.get((case_id, topology_key), {})
        if not case:
            continue
        makespan, solver = parse_log_metrics(log_path)
        intervals = parse_link_intervals(log_path)
        error_hint = ""
        if makespan is None:
            for line in log_path.read_text(errors="replace").splitlines()[:20]:
                if line.strip():
                    error_hint = line.strip()
                    break
        row: dict[str, object] = {
            "case_id": case_id,
            "topology_key": topology_key,
            "method": method,
            "status": "success" if makespan is not None else "failed",
            "error_hint": error_hint,
            "log_path": rel(log_path),
            "makespan_us": makespan if makespan is not None else "",
            "solver_time_us": solver if solver is not None else "",
            "bandwidth_gbps_per_rank": (
                (float(case.get("total_tokens", 0)) * TOKEN_BYTES / max(1, int(case.get("target_devices", 1)))) / makespan / 1000.0
                if makespan and case
                else ""
            ),
            **{k: case.get(k, "") for k in ("model", "phase", "source", "selection_label", "variant", "target_devices")},
            **{
                f"input_{k}": case.get(k, "")
                for k in (
                    "bimodal_pressure_index",
                    "bpi_flow_entropy",
                    "bpi_dest_entropy",
                    "bpi_entropy_max",
                    "bpi_lorenz_destmax",
                    "bpi_flow_hhi",
                    "bpi_dest_hhi",
                    "bpi_flow_lorenz80",
                    "bpi_flow_lorenz80_uniform",
                    "flow_gini_all",
                    "flow_top10pct_share",
                    "topk_dest_share",
                )
            },
            **link_balance_metrics(intervals, makespan),
            **fabric_balance_metrics(intervals, makespan, case.get("topology_json", "")),
            **predicted_hotspot_metrics(
                intervals,
                makespan,
                case.get("csv_path", ""),
                case.get("topology_json", ""),
            ),
        }
        rows.append(row)
    write_csv(RESULTS_DIR / "performance_results.csv", rows)
    write_csv(RESULTS_DIR / "performance_results_summary.csv", summarize_performance(rows))
    print(json.dumps({"performance_rows": len(rows)}, indent=2))


def original_probe_source_csv(case_id: str, probe_csv_path: Path) -> Path:
    base_name = case_id.replace("_balanced_probe", "")
    candidate = REPO_ROOT / "evaluation_assets" / "csv" / "synthetic" / "64devices" / f"{base_name}.csv"
    return candidate if candidate.exists() else probe_csv_path


def parse_probe_collective_csv(collective_path: Path) -> Path:
    payload = json.loads(collective_path.read_text())
    csv_path = Path(str(payload.get("v_datasize", "")))
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    return csv_path


def parse_torus_balanced_probe_results(probe_root: Path) -> list[dict[str, object]]:
    topology_json = "evaluation_assets/topologies/synthetic/torus_tpuv4_4x4x4.json"
    rows: list[dict[str, object]] = []
    collectives = sorted((probe_root / "collectives").glob("*.json"))
    for collective_path in collectives:
        case_id = collective_path.stem
        probe_csv_path = parse_probe_collective_csv(collective_path)
        source_csv_path = original_probe_source_csv(case_id, probe_csv_path)
        matrix = load_matrix(probe_csv_path)
        matrix_info = matrix_metrics(matrix, TOP_K_DEFAULT)
        for method in METHODS_64:
            log_path = probe_root / "logs" / f"{case_id}_{method}_stagefixed.log"
            makespan, solver = parse_log_metrics(log_path)
            intervals = parse_link_intervals(log_path)
            row: dict[str, object] = {
                "case_id": case_id,
                "source_csv": rel(source_csv_path),
                "probe_csv": rel(probe_csv_path),
                "method": method,
                "makespan_us": makespan if makespan is not None else "",
                "solver_time_us": solver if solver is not None else "",
                "log_path": rel(log_path),
                **matrix_info,
                **parse_schedule_stats(log_path),
                **link_balance_metrics(intervals, makespan),
                **fabric_balance_metrics(intervals, makespan, topology_json),
                **predicted_hotspot_metrics(intervals, makespan, rel(probe_csv_path), topology_json),
            }
            rows.append(row)
    return rows


def write_torus_balanced_probe_outputs(probe_root: Path, rows: list[dict[str, object]]) -> None:
    write_csv(probe_root / "torus_balanced_64d_256mb_method_results.csv", rows)
    write_csv(probe_root / "torus_balanced_64d_256mb_results.csv", rows)

    by_case: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_case[str(row.get("case_id", ""))][str(row.get("method", ""))] = row

    speedup_rows: list[dict[str, object]] = []
    for case_id, items in sorted(by_case.items()):
        glaive = items.get("glaive")
        halfring = items.get("halfringdr")
        if not glaive or not halfring:
            continue
        glaive_makespan = fnum(glaive, "makespan_us")
        halfring_makespan = fnum(halfring, "makespan_us")
        if glaive_makespan <= 0 or halfring_makespan <= 0:
            continue
        speedup_rows.append(
            {
                "case_id": case_id,
                "bpi": fnum(glaive, "bimodal_pressure_index"),
                "glaive_vs_halfring_speedup": halfring_makespan / glaive_makespan,
                "halfring_vs_glaive_speedup": glaive_makespan / halfring_makespan,
                "glaive_over_halfring_makespan_ratio": glaive_makespan / halfring_makespan,
                "halfring_over_glaive_makespan_ratio": halfring_makespan / glaive_makespan,
                "glaive_makespan_us": glaive_makespan,
                "halfring_makespan_us": halfring_makespan,
                "glaive_fabric_active": fnum(glaive, "fabric_active_link_fraction_mean"),
                "halfring_fabric_active": fnum(halfring, "fabric_active_link_fraction_mean"),
                "glaive_fabric_gini": fnum(glaive, "fabric_busy_time_gini"),
                "halfring_fabric_gini": fnum(halfring, "fabric_busy_time_gini"),
                "halfring_total_rounds": fnum(halfring, "halfring_total_rounds"),
                "halfring_stage_substage_events": fnum(halfring, "halfring_stage_substage_events"),
                "glaive_hot_share": fnum(glaive, "pred_hot_edge_actual_busy_share"),
                "halfring_hot_share": fnum(halfring, "pred_hot_edge_actual_busy_share"),
            }
        )
    write_csv(probe_root / "torus_balanced_64d_256mb_speedup.csv", speedup_rows)


def run_torus_balanced_probe(args: argparse.Namespace) -> None:
    ensure_dirs()
    probe_root = OUT_ROOT / "probes" / "torus_balanced_64d_256mb"
    topology_json = "evaluation_assets/topologies/synthetic/torus_tpuv4_4x4x4.json"
    collectives = sorted((probe_root / "collectives").glob("*.json"))
    if not collectives:
        raise FileNotFoundError(f"Missing probe collectives under {probe_root / 'collectives'}")

    tasks: list[tuple[dict[str, str], str, Path]] = []
    for collective_path in collectives:
        case = {
            "case_id": collective_path.stem,
            "topology_json": topology_json,
            "collective_json": rel(collective_path),
        }
        for method in METHODS_64:
            log_path = probe_root / "logs" / f"{collective_path.stem}_{method}_stagefixed.log"
            tasks.append((case, method, log_path))

    failures: list[dict[str, object]] = []
    print(json.dumps({"probe_tasks": len(tasks), "workers": args.workers}, indent=2))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(run_one_task, case, method, log_path, args.force): (case, method, log_path)
            for case, method, log_path in tasks
        }
        for future in as_completed(future_map):
            case, method, _ = future_map[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"failed:exception:{exc}"
            if str(status).startswith("failed"):
                failures.append({"case_id": case["case_id"], "method": method, "status": status})
            print(f"{status} {case['case_id']} {method}", flush=True)

    rows = parse_torus_balanced_probe_results(probe_root)
    write_torus_balanced_probe_outputs(probe_root, rows)
    if failures:
        write_csv(probe_root / "torus_balanced_64d_256mb_failures.csv", failures)
    elif (probe_root / "torus_balanced_64d_256mb_failures.csv").exists():
        write_csv(probe_root / "torus_balanced_64d_256mb_failures.csv", [])
    print(json.dumps({"probe_rows": len(rows), "failures": len(failures)}, indent=2))


def summarize_performance(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("model")),
                str(row.get("phase")),
                str(row.get("variant")),
                str(row.get("topology_key")),
                str(row.get("method")),
            )
        ].append(row)
    metrics = [
        "makespan_us",
        "solver_time_us",
        "bandwidth_gbps_per_rank",
        "busy_time_gini",
        "busy_time_jain",
        "link_use_count_sum",
        "utilization_mean_pct",
        "fabric_used_link_ratio",
        "fabric_busy_time_gini",
        "fabric_busy_time_jain",
        "fabric_busy_time_top10_share",
        "fabric_active_link_fraction_mean",
        "fabric_work_parallelism",
        "det_route_edge_demand_gini",
        "det_route_edge_demand_top10_share",
        "pred_hot_edge_demand_share",
        "pred_hot_edge_actual_busy_share",
        "pred_hot_edge_actual_vs_demand_delta",
        "det_demand_vs_actual_busy_corr",
        "det_demand_vs_actual_interval_corr",
        "pred_cold_edge_used_ratio",
        "pred_hot_to_cold_busy_ratio",
        "hot_flow_count",
        "hot_flow_traffic_share",
        "hot_flow_path_edge_count",
        "hot_flow_path_demand_share",
        "hot_flow_path_actual_busy_share",
        "hot_flow_path_actual_vs_demand_delta",
        "hot_flow_path_interval_count_total",
        "hot_flow_path_interval_count_mean",
        "hot_flow_path_finish_p95_ns",
        "hot_flow_path_finish_max_ns",
        "hot_flow_path_to_other_busy_ratio",
        "hot_flow_path_cold_edge_used_ratio",
        "hot_flow_path_demand_vs_actual_busy_corr",
    ]
    out: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        model, phase, variant, topology, method = key
        row: dict[str, object] = {
            "model": model,
            "phase": phase,
            "variant": variant,
            "topology_key": topology,
            "method": method,
            "count": len(items),
        }
        for metric in metrics:
            vals = [float(item[metric]) for item in items if item.get(metric, "") != ""]
            row[f"{metric}_mean"] = mean(vals) if vals else 0.0
            row[f"{metric}_p50"] = percentile(vals, 50)
        out.append(row)
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def summarize_speedup_rows(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    metrics = [
        "input_bimodal_pressure_index",
        "speedup_vs_best_baseline",
        "bandwidth_ratio_vs_best_baseline",
        "busy_time_gini_delta_vs_best_baseline",
        "busy_time_jain_delta_vs_best_baseline",
        "fabric_busy_time_gini_delta_vs_best_baseline",
        "fabric_busy_time_jain_delta_vs_best_baseline",
        "fabric_used_link_ratio_delta_vs_best_baseline",
        "fabric_active_link_fraction_delta_vs_best_baseline",
        "pred_hot_edge_actual_busy_share_delta_vs_best_baseline",
        "pred_cold_edge_used_ratio_delta_vs_best_baseline",
        "pred_hot_to_cold_busy_ratio_delta_vs_best_baseline",
        "det_demand_vs_actual_busy_corr_delta_vs_best_baseline",
        "det_demand_vs_actual_interval_corr_delta_vs_best_baseline",
        "hot_flow_path_actual_busy_share_delta_vs_best_baseline",
        "hot_flow_path_interval_count_ratio_vs_best_baseline",
        "hot_flow_path_finish_p95_ratio_vs_best_baseline",
        "hot_flow_path_to_other_busy_ratio_delta_vs_best_baseline",
        "hot_flow_path_cold_edge_used_ratio_delta_vs_best_baseline",
        "hot_flow_path_demand_vs_actual_busy_corr_delta_vs_best_baseline",
        "det_route_edge_demand_gini",
        "pred_hot_edge_demand_share",
        "hot_flow_traffic_share",
        "hot_flow_path_demand_share",
        "link_use_count_ratio_vs_best_baseline",
    ]
    out: list[dict[str, object]] = []
    for key, items in sorted(groups.items()):
        row: dict[str, object] = {name: value for name, value in zip(keys, key)}
        row["count"] = len(items)
        for metric in metrics:
            vals = [fnum(item, metric) for item in items if item.get(metric, "") != ""]
            row[f"{metric}_mean"] = mean(vals) if vals else 0.0
            row[f"{metric}_p50"] = percentile(vals, 50)
            row[f"{metric}_p10"] = percentile(vals, 10)
            row[f"{metric}_p90"] = percentile(vals, 90)
        out.append(row)
    return out


def derive_summaries(_: argparse.Namespace) -> None:
    ensure_dirs()
    perf_rows = read_csv(RESULTS_DIR / "performance_results.csv")
    by_case_topology: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in perf_rows:
        if row.get("status") != "success":
            continue
        by_case_topology[(row["case_id"], row["topology_key"])].append(row)

    speedup_rows: list[dict[str, object]] = []
    for (case_id, topology_key), items in sorted(by_case_topology.items()):
        glaive = next((row for row in items if row["method"] == "glaive"), None)
        baselines = [row for row in items if row["method"] != "glaive"]
        if not glaive or not baselines:
            continue
        best = min(baselines, key=lambda row: fnum(row, "makespan_us", float("inf")))
        best_makespan = fnum(best, "makespan_us")
        glaive_makespan = fnum(glaive, "makespan_us")
        best_bandwidth = fnum(best, "bandwidth_gbps_per_rank")
        glaive_bandwidth = fnum(glaive, "bandwidth_gbps_per_rank")
        row: dict[str, object] = {
            "case_id": case_id,
            "topology_key": topology_key,
            "target_devices": glaive.get("target_devices", ""),
            "model": glaive.get("model", ""),
            "phase": glaive.get("phase", ""),
            "source": glaive.get("source", ""),
            "selection_label": glaive.get("selection_label", ""),
            "variant": glaive.get("variant", ""),
            "input_bimodal_pressure_index": glaive.get("input_bimodal_pressure_index", ""),
            "glaive_makespan_us": glaive_makespan,
            "best_baseline_method": best["method"],
            "best_baseline_makespan_us": best_makespan,
            "speedup_vs_best_baseline": best_makespan / glaive_makespan if glaive_makespan > 0 else "",
            "glaive_bandwidth_gbps_per_rank": glaive_bandwidth,
            "best_baseline_bandwidth_gbps_per_rank": best_bandwidth,
            "bandwidth_ratio_vs_best_baseline": glaive_bandwidth / best_bandwidth if best_bandwidth > 0 else "",
            "glaive_busy_time_gini": glaive.get("busy_time_gini", ""),
            "best_baseline_busy_time_gini": best.get("busy_time_gini", ""),
            "busy_time_gini_delta_vs_best_baseline": fnum(best, "busy_time_gini") - fnum(glaive, "busy_time_gini"),
            "glaive_busy_time_jain": glaive.get("busy_time_jain", ""),
            "best_baseline_busy_time_jain": best.get("busy_time_jain", ""),
            "busy_time_jain_delta_vs_best_baseline": fnum(glaive, "busy_time_jain") - fnum(best, "busy_time_jain"),
            "glaive_fabric_busy_time_gini": glaive.get("fabric_busy_time_gini", ""),
            "best_baseline_fabric_busy_time_gini": best.get("fabric_busy_time_gini", ""),
            "fabric_busy_time_gini_delta_vs_best_baseline": (
                fnum(best, "fabric_busy_time_gini") - fnum(glaive, "fabric_busy_time_gini")
            ),
            "glaive_fabric_busy_time_jain": glaive.get("fabric_busy_time_jain", ""),
            "best_baseline_fabric_busy_time_jain": best.get("fabric_busy_time_jain", ""),
            "fabric_busy_time_jain_delta_vs_best_baseline": (
                fnum(glaive, "fabric_busy_time_jain") - fnum(best, "fabric_busy_time_jain")
            ),
            "glaive_fabric_used_link_ratio": glaive.get("fabric_used_link_ratio", ""),
            "best_baseline_fabric_used_link_ratio": best.get("fabric_used_link_ratio", ""),
            "fabric_used_link_ratio_delta_vs_best_baseline": (
                fnum(glaive, "fabric_used_link_ratio") - fnum(best, "fabric_used_link_ratio")
            ),
            "glaive_fabric_active_link_fraction_mean": glaive.get("fabric_active_link_fraction_mean", ""),
            "best_baseline_fabric_active_link_fraction_mean": best.get("fabric_active_link_fraction_mean", ""),
            "fabric_active_link_fraction_delta_vs_best_baseline": (
                fnum(glaive, "fabric_active_link_fraction_mean")
                - fnum(best, "fabric_active_link_fraction_mean")
            ),
            "glaive_pred_hot_edge_actual_busy_share": glaive.get("pred_hot_edge_actual_busy_share", ""),
            "best_baseline_pred_hot_edge_actual_busy_share": best.get("pred_hot_edge_actual_busy_share", ""),
            "pred_hot_edge_actual_busy_share_delta_vs_best_baseline": (
                fnum(best, "pred_hot_edge_actual_busy_share")
                - fnum(glaive, "pred_hot_edge_actual_busy_share")
            ),
            "glaive_pred_cold_edge_used_ratio": glaive.get("pred_cold_edge_used_ratio", ""),
            "best_baseline_pred_cold_edge_used_ratio": best.get("pred_cold_edge_used_ratio", ""),
            "pred_cold_edge_used_ratio_delta_vs_best_baseline": (
                fnum(glaive, "pred_cold_edge_used_ratio")
                - fnum(best, "pred_cold_edge_used_ratio")
            ),
            "glaive_pred_hot_to_cold_busy_ratio": glaive.get("pred_hot_to_cold_busy_ratio", ""),
            "best_baseline_pred_hot_to_cold_busy_ratio": best.get("pred_hot_to_cold_busy_ratio", ""),
            "pred_hot_to_cold_busy_ratio_delta_vs_best_baseline": (
                fnum(best, "pred_hot_to_cold_busy_ratio") - fnum(glaive, "pred_hot_to_cold_busy_ratio")
            ),
            "glaive_det_demand_vs_actual_busy_corr": glaive.get("det_demand_vs_actual_busy_corr", ""),
            "best_baseline_det_demand_vs_actual_busy_corr": best.get("det_demand_vs_actual_busy_corr", ""),
            "det_demand_vs_actual_busy_corr_delta_vs_best_baseline": (
                fnum(best, "det_demand_vs_actual_busy_corr") - fnum(glaive, "det_demand_vs_actual_busy_corr")
            ),
            "glaive_det_demand_vs_actual_interval_corr": glaive.get("det_demand_vs_actual_interval_corr", ""),
            "best_baseline_det_demand_vs_actual_interval_corr": best.get("det_demand_vs_actual_interval_corr", ""),
            "det_demand_vs_actual_interval_corr_delta_vs_best_baseline": (
                fnum(best, "det_demand_vs_actual_interval_corr")
                - fnum(glaive, "det_demand_vs_actual_interval_corr")
            ),
            "glaive_hot_flow_path_actual_busy_share": glaive.get("hot_flow_path_actual_busy_share", ""),
            "best_baseline_hot_flow_path_actual_busy_share": best.get("hot_flow_path_actual_busy_share", ""),
            "hot_flow_path_actual_busy_share_delta_vs_best_baseline": (
                fnum(best, "hot_flow_path_actual_busy_share") - fnum(glaive, "hot_flow_path_actual_busy_share")
            ),
            "glaive_hot_flow_path_interval_count_total": glaive.get("hot_flow_path_interval_count_total", ""),
            "best_baseline_hot_flow_path_interval_count_total": best.get("hot_flow_path_interval_count_total", ""),
            "hot_flow_path_interval_count_ratio_vs_best_baseline": (
                fnum(glaive, "hot_flow_path_interval_count_total")
                / fnum(best, "hot_flow_path_interval_count_total")
                if fnum(best, "hot_flow_path_interval_count_total") > 0
                else ""
            ),
            "glaive_hot_flow_path_finish_p95_ns": glaive.get("hot_flow_path_finish_p95_ns", ""),
            "best_baseline_hot_flow_path_finish_p95_ns": best.get("hot_flow_path_finish_p95_ns", ""),
            "hot_flow_path_finish_p95_ratio_vs_best_baseline": (
                fnum(best, "hot_flow_path_finish_p95_ns") / fnum(glaive, "hot_flow_path_finish_p95_ns")
                if fnum(glaive, "hot_flow_path_finish_p95_ns") > 0
                else ""
            ),
            "glaive_hot_flow_path_to_other_busy_ratio": glaive.get("hot_flow_path_to_other_busy_ratio", ""),
            "best_baseline_hot_flow_path_to_other_busy_ratio": best.get("hot_flow_path_to_other_busy_ratio", ""),
            "hot_flow_path_to_other_busy_ratio_delta_vs_best_baseline": (
                fnum(best, "hot_flow_path_to_other_busy_ratio")
                - fnum(glaive, "hot_flow_path_to_other_busy_ratio")
            ),
            "glaive_hot_flow_path_cold_edge_used_ratio": glaive.get("hot_flow_path_cold_edge_used_ratio", ""),
            "best_baseline_hot_flow_path_cold_edge_used_ratio": best.get("hot_flow_path_cold_edge_used_ratio", ""),
            "hot_flow_path_cold_edge_used_ratio_delta_vs_best_baseline": (
                fnum(glaive, "hot_flow_path_cold_edge_used_ratio")
                - fnum(best, "hot_flow_path_cold_edge_used_ratio")
            ),
            "glaive_hot_flow_path_demand_vs_actual_busy_corr": glaive.get(
                "hot_flow_path_demand_vs_actual_busy_corr", ""
            ),
            "best_baseline_hot_flow_path_demand_vs_actual_busy_corr": best.get(
                "hot_flow_path_demand_vs_actual_busy_corr", ""
            ),
            "hot_flow_path_demand_vs_actual_busy_corr_delta_vs_best_baseline": (
                fnum(best, "hot_flow_path_demand_vs_actual_busy_corr")
                - fnum(glaive, "hot_flow_path_demand_vs_actual_busy_corr")
            ),
            "det_route_edge_demand_gini": glaive.get("det_route_edge_demand_gini", ""),
            "det_route_edge_demand_top10_share": glaive.get("det_route_edge_demand_top10_share", ""),
            "pred_hot_edge_demand_share": glaive.get("pred_hot_edge_demand_share", ""),
            "hot_flow_traffic_share": glaive.get("hot_flow_traffic_share", ""),
            "hot_flow_path_demand_share": glaive.get("hot_flow_path_demand_share", ""),
            "glaive_link_use_count_sum": glaive.get("link_use_count_sum", ""),
            "best_baseline_link_use_count_sum": best.get("link_use_count_sum", ""),
            "link_use_count_ratio_vs_best_baseline": (
                fnum(glaive, "link_use_count_sum") / fnum(best, "link_use_count_sum")
                if fnum(best, "link_use_count_sum") > 0
                else ""
            ),
        }
        speedup_rows.append(row)

    write_csv(RESULTS_DIR / "performance_speedup_vs_best_baseline.csv", speedup_rows)
    negative_balanced = [
        row
        for row in speedup_rows
        if row.get("variant") == "balanced" and fnum(row, "speedup_vs_best_baseline", 1.0) < 1.0
    ]
    write_csv(
        RESULTS_DIR / "balanced_negative_speedup_cases.csv",
        sorted(
            negative_balanced,
            key=lambda row: (
                fnum(row, "speedup_vs_best_baseline", 1.0),
                str(row.get("case_id", "")),
                str(row.get("topology_key", "")),
            ),
        ),
    )
    write_csv(RESULTS_DIR / "performance_speedup_by_variant.csv", summarize_speedup_rows(speedup_rows, ("variant",)))
    write_csv(
        RESULTS_DIR / "performance_speedup_by_selection_variant.csv",
        summarize_speedup_rows(speedup_rows, ("selection_label", "variant")),
    )
    write_csv(
        RESULTS_DIR / "performance_speedup_by_model_phase_variant.csv",
        summarize_speedup_rows(speedup_rows, ("model", "phase", "variant")),
    )

    network_rows = read_csv(RESULTS_DIR / "network_balance_existing_link_logs_summary.csv")
    link_256 = [
        row
        for row in network_rows
        if row.get("suite") == "evaluation_assets_link" and row.get("size_label") == "256MB"
    ]
    write_csv(RESULTS_DIR / "network_balance_existing_link_256mb_summary.csv", link_256)
    print(
        json.dumps(
            {
                "speedup_rows": len(speedup_rows),
                "speedup_by_variant_rows": len(summarize_speedup_rows(speedup_rows, ("variant",))),
                "network_link_256mb_rows": len(link_256),
            },
            indent=2,
        )
    )


def plot_results(_: argparse.Namespace) -> None:
    ensure_dirs()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )

    method_colors = {
        "glaive": "#2a9d8f",
        "biring": "#7b2cbf",
        "halfringdr": "#7b2cbf",
        "mpibaseline": "#e76f51",
    }
    variant_colors = {
        "real": "#111827",
        "balanced": "#00BFC4",
        "blend_0.25": "#009E73",
        "blend_0.50": "#CC79A7",
        "blend_0.75": "#E69F00",
        "eplb_x2": "#0072B2",
        "worst_top8": "#D55E00",
    }
    topology_markers = {
        "mesh_nebula_8x4": "o",
        "clos_8x4": "s",
        "cm384_16x2": "^",
        "rail_optimized_8x4": "P",
        "torus_tpuv4_4x4x4": "D",
    }
    topology_jitter = {
        "mesh_nebula_8x4": -0.012,
        "clos_8x4": -0.006,
        "cm384_16x2": 0.0,
        "rail_optimized_8x4": 0.012,
        "torus_tpuv4_4x4x4": 0.006,
    }

    figures: list[dict[str, object]] = []

    def save_figure(fig: object, figure_id: str, title: str, description: str, source_csvs: list[Path]) -> None:
        pdf_path = PLOTS_DIR / f"{figure_id}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        figures.append(
            {
                "figure_id": figure_id,
                "title": title,
                "description": description,
                "pdf_path": rel(pdf_path),
                "source_csvs": ";".join(rel(path) for path in source_csvs),
            }
        )

    def clean_label(value: str) -> str:
        return (
            value.replace("deepseekv3", "DeepSeek")
            .replace("olmoe", "OLMoE")
            .replace("qwen", "Qwen")
            .replace("evaluation_synthetic", "synthetic")
            .replace("_", " ")
        )

    real_metric_path = RESULTS_DIR / "trace_metrics_real_full.csv"
    real_rows = [
        row
        for row in read_csv(real_metric_path)
        if fnum(row, "total_tokens") > 0 and row.get("source") == "official"
    ]
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in real_rows:
        groups[(row["model"], row["phase"])].append(fnum(row, "bimodal_pressure_index"))
    ordered_groups = sorted(groups)
    if ordered_groups:
        data = [groups[key] for key in ordered_groups]
        labels = [f"{clean_label(key[0])}\n{key[1]}" for key in ordered_groups]
        fig, ax = plt.subplots(figsize=(10.8, 5.2))
        box = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "linewidth": 1.4})
        for patch, key in zip(box["boxes"], ordered_groups):
            patch.set_facecolor("#a8dadc" if key[1] == "prefill" else "#f1faee")
            patch.set_edgecolor("#1d3557")
        ax.set_ylabel("Bimodal Pressure Index")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_title("BPI Distribution Across Official Nonzero MoE Alltoallv Traces")
        for idx, values in enumerate(data, start=1):
            ax.text(idx, 0.99, f"n={len(values)}", ha="center", va="top", fontsize=8, color="#333333")
        save_figure(
            fig,
            "trace_bpi_distribution",
            "BPI distribution of official traces",
            "Candidate bimodal pressure index distributions for nonzero official traces, grouped by model and phase.",
            [real_metric_path],
        )

    variant_metric_path = RESULTS_DIR / "trace_metrics_reference_variants.csv"
    variant_rows = [row for row in read_csv(variant_metric_path) if fnum(row, "total_tokens") > 0]
    variant_groups: dict[str, list[float]] = defaultdict(list)
    for row in real_rows:
        variant_groups["real"].append(fnum(row, "bimodal_pressure_index"))
    for row in variant_rows:
        variant_groups[row["variant"]].append(fnum(row, "bimodal_pressure_index"))
    variant_order = ["balanced", "blend_0.75", "blend_0.50", "blend_0.25", "eplb_x2", "real", "worst_top8"]
    variant_order = [variant for variant in variant_order if variant in variant_groups]
    if variant_order:
        data = [variant_groups[variant] for variant in variant_order]
        fig, ax = plt.subplots(figsize=(10.6, 5.2))
        box = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "linewidth": 1.4})
        for patch, variant in zip(box["boxes"], variant_order):
            patch.set_facecolor(variant_colors.get(variant, "#bbbbbb"))
            patch.set_alpha(0.82)
            patch.set_edgecolor("#222222")
        ax.set_ylabel("Bimodal Pressure Index")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(1, len(variant_order) + 1))
        ax.set_xticklabels([variant_label(variant) for variant in variant_order], rotation=20, ha="right")
        ax.set_title("BPI Under Real and Reference Traffic Controls")
        save_figure(
            fig,
            "reference_variant_bpi_distribution",
            "BPI distribution of reference variants",
            "Comparison of nonzero real traces and synthetic controls with the same total traffic.",
            [real_metric_path, variant_metric_path],
        )

    speedup_path = RESULTS_DIR / "performance_speedup_vs_best_baseline.csv"
    speedup_rows = [row for row in read_csv(speedup_path) if row.get("speedup_vs_best_baseline", "") != ""]
    if speedup_rows:
        fig, ax = plt.subplots(figsize=(12.6, 6.2))
        for variant in sorted({row["variant"] for row in speedup_rows}):
            rows = [row for row in speedup_rows if row["variant"] == variant]
            for topology in sorted({row["topology_key"] for row in rows}):
                topo_rows = [row for row in rows if row["topology_key"] == topology]
                ax.scatter(
                    [
                        max(
                            0.0,
                            min(1.0, fnum(row, "input_bimodal_pressure_index") + topology_jitter.get(topology, 0.0)),
                        )
                        for row in topo_rows
                    ],
                    [fnum(row, "speedup_vs_best_baseline") for row in topo_rows],
                    s=46,
                    alpha=0.82,
                    marker=topology_markers.get(topology, "o"),
                    color=variant_colors.get(variant, "#666666"),
                    edgecolors="white",
                    linewidths=0.55,
                )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
        negative_rows = [
            row
            for row in speedup_rows
            if fnum(row, "speedup_vs_best_baseline", 1.0) < 1.0
        ]
        if negative_rows:
            ax.scatter(
                [
                    max(
                        0.0,
                        min(
                            1.0,
                            fnum(row, "input_bimodal_pressure_index")
                            + topology_jitter.get(row.get("topology_key", ""), 0.0),
                        ),
                    )
                    for row in negative_rows
                ],
                [fnum(row, "speedup_vs_best_baseline") for row in negative_rows],
                s=118,
                marker="o",
                facecolors="none",
                edgecolors="#B00020",
                linewidths=1.35,
                label="speedup < 1",
                zorder=5,
            )
        ax.set_xlabel("Input Bimodal Pressure Index")
        ax.set_ylabel("Glaive Speedup vs Best Baseline")
        ax.set_title("Glaive Performance Across Bimodality Levels")
        ax.set_xlim(-0.025, 0.9)
        variant_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                    markerfacecolor=variant_colors.get(variant, "#666666"),
                    markeredgecolor="white",
                    markersize=7,
                    label=variant_label(variant),
                )
            for variant in sorted({row["variant"] for row in speedup_rows})
        ]
        topology_legend = [
            Line2D(
                [0],
                [0],
                marker=topology_markers.get(topology, "o"),
                color="#444444",
                markerfacecolor="#444444",
                markersize=7,
                linestyle="none",
                label=topology,
            )
            for topology in sorted({row["topology_key"] for row in speedup_rows})
        ]
        if negative_rows:
            variant_legend.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#B00020",
                    markerfacecolor="none",
                    markersize=8,
                    linestyle="none",
                    label="speedup < 1",
                )
            )
        first_legend = ax.legend(
            handles=variant_legend,
            ncol=1,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=topology_legend,
            ncol=1,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(1.01, 0.0),
            borderaxespad=0.0,
            fontsize=8,
        )
        ax.text(
            0.01,
            0.02,
            "Small x-jitter separates topologies with identical input BPI.",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )
        save_figure(
            fig,
            "speedup_vs_bpi",
            "Speedup versus input BPI",
            "Glaive speedup over the best non-Glaive baseline versus input BPI for each case; small horizontal jitter only separates topology points with identical BPI.",
            [speedup_path],
        )

    speedup_variant_path = RESULTS_DIR / "performance_speedup_by_variant.csv"
    speedup_variant = read_csv(speedup_variant_path)
    if speedup_variant:
        order = [row["variant"] for row in speedup_variant]
        x = np.arange(len(order))
        means = np.asarray([fnum(row, "speedup_vs_best_baseline_mean") for row in speedup_variant])
        p10 = np.asarray([fnum(row, "speedup_vs_best_baseline_p10") for row in speedup_variant])
        p90 = np.asarray([fnum(row, "speedup_vs_best_baseline_p90") for row in speedup_variant])
        fig, ax = plt.subplots(figsize=(10.8, 5.2))
        ax.bar(x, means, color=[variant_colors.get(variant, "#777777") for variant in order], alpha=0.88)
        ax.errorbar(x, means, yerr=[means - p10, p90 - means], fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([variant_label(variant) for variant in order], rotation=22, ha="right")
        ax.set_ylabel("Speedup vs Best Baseline")
        ax.set_title("Aggregate Glaive Speedup by Traffic Variant")
        save_figure(
            fig,
            "speedup_by_variant",
            "Speedup aggregated by variant",
            "Mean, p10, and p90 speedup for real, balanced, blended, EPLB, and worst-case traffic variants.",
            [speedup_variant_path],
        )

        fig, axes = plt.subplots(4, 1, figsize=(11.8, 10.0), sharex=True)
        load_metrics = [
            (
                "fabric_active_link_fraction_delta_vs_best_baseline_mean",
                "Active-link\nfraction delta",
                "#1F77B4",
            ),
            (
                "pred_hot_edge_actual_busy_share_delta_vs_best_baseline_mean",
                "Hot-edge\nshare reduction",
                "#D62728",
            ),
            (
                "hot_flow_path_actual_busy_share_delta_vs_best_baseline_mean",
                "Hot-flow-path\nshare reduction",
                "#E69F00",
            ),
            (
                "det_demand_vs_actual_busy_corr_delta_vs_best_baseline_mean",
                "Deterministic\ncorr. reduction",
                "#2CA02C",
            ),
        ]
        for ax, (metric, ylabel, color) in zip(axes, load_metrics):
            values = [fnum(row, metric) for row in speedup_variant]
            ax.bar(x, values, color=color, alpha=0.84)
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
            ax.set_ylabel(ylabel)
        axes[0].set_title("Glaive Load-Balance Effects vs Best Baseline")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([variant_label(variant) for variant in order], rotation=22, ha="right")
        save_figure(
            fig,
            "load_balance_by_variant",
            "Load-balance effect aggregated by variant",
            "Changes in fabric active-link fraction, predicted hot-link busy share, hot-flow-path busy share, and deterministic-demand correlation relative to the best non-Glaive baseline.",
            [speedup_variant_path],
        )

    network_path = RESULTS_DIR / "network_balance_existing_link_256mb_summary.csv"
    network_rows = read_csv(network_path)
    if network_rows:
        network_rows = sorted(network_rows, key=lambda row: (row["topology_key"], row["method"]))
        x = np.arange(len(network_rows))
        labels = [f"{row['topology_key']}\n{row['method']}" for row in network_rows]
        colors = [method_colors.get(row["method"], "#777777") for row in network_rows]
        fig, axes = plt.subplots(3, 1, figsize=(15.8, 10.2), sharex=True)
        for ax, metric, ylabel in [
            (axes[0], "used_links_mean", "Used Links"),
            (axes[1], "link_use_count_sum_mean", "Link Busy Intervals"),
            (axes[2], "busy_time_gini_mean", "Busy-Time Gini"),
        ]:
            ax.bar(x, [fnum(row, metric) for row in network_rows], color=colors, alpha=0.88)
            ax.set_ylabel(ylabel)
        axes[0].set_title("Existing 256MB Link-Utilization Load Balance")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=62, ha="right", fontsize=8)
        legend_methods = sorted({row["method"] for row in network_rows})
        axes[0].legend(
            [Line2D([0], [0], color=method_colors.get(method, "#777777"), lw=8) for method in legend_methods],
            legend_methods,
            ncol=len(legend_methods),
            frameon=False,
            loc="upper right",
        )
        save_figure(
            fig,
            "network_balance_256mb",
            "Existing 256 MB network load balance",
            "Used links, link busy intervals, and busy-time Gini from existing synthetic 256 MB link-utilization logs.",
            [network_path],
        )

    write_csv(RESULTS_DIR / "figure_manifest.csv", figures)
    print(json.dumps({"figures": len(figures), "manifest": rel(RESULTS_DIR / "figure_manifest.csv")}, indent=2))


PLOT_SCRIPT_IDS = (
    "trace_bpi_distribution",
    "reference_variant_bpi_distribution",
    "speedup_vs_bpi",
    "speedup_by_variant",
    "load_balance_by_variant",
    "network_balance_256mb",
)


def plot_results_standalone(_: argparse.Namespace) -> None:
    ensure_dirs()
    figures: list[dict[str, object]] = []
    for figure_id in PLOT_SCRIPT_IDS:
        script = PLOT_SCRIPT_DIR / f"{figure_id}.py"
        if not script.exists():
            raise FileNotFoundError(f"Missing plot script: {script}")
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        entry = json.loads(payload)
        if not entry.get("skipped"):
            figures.append(entry)
    write_csv(RESULTS_DIR / "figure_manifest.csv", figures)
    print(json.dumps({"figures": len(figures), "manifest": rel(RESULTS_DIR / "figure_manifest.csv")}, indent=2))


def bpi_component_dominance_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if fnum(row, "total_tokens") <= 0:
            continue
        groups[(row.get("model", ""), row.get("phase", ""), row.get("source", ""))].append(row)

    out: list[dict[str, object]] = []
    for (model, phase, source), items in sorted(groups.items()):
        flow_values = [fnum(item, "bpi_flow_lorenz80") for item in items]
        dest_values = [fnum(item, "bpi_dest_entropy") for item in items]
        bpi_values = [fnum(item, "bimodal_pressure_index") for item in items]
        flow_dom = sum(1 for flow, dest in zip(flow_values, dest_values) if flow >= dest)
        dest_dom = sum(1 for flow, dest in zip(flow_values, dest_values) if dest > flow)
        out.append(
            {
                "model": model,
                "phase": phase,
                "source": source,
                "nonzero_count": len(items),
                "bpi_mean": mean(bpi_values) if bpi_values else 0.0,
                "bpi_p50": percentile(bpi_values, 50),
                "flow_lorenz80_mean": mean(flow_values) if flow_values else 0.0,
                "flow_lorenz80_p50": percentile(flow_values, 50),
                "dest_entropy_mean": mean(dest_values) if dest_values else 0.0,
                "dest_entropy_p50": percentile(dest_values, 50),
                "flow_lorenz80_dominant_pct": 100.0 * flow_dom / len(items) if items else 0.0,
                "dest_entropy_dominant_pct": 100.0 * dest_dom / len(items) if items else 0.0,
            }
        )
    return out


def trace_source_inventory(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("model", ""), row.get("phase", ""), row.get("source", ""))].append(row)

    out: list[dict[str, object]] = []
    for (model, phase, source), items in sorted(groups.items()):
        nonzero = [item for item in items if fnum(item, "total_tokens") > 0]
        shapes = sorted(
            {
                f"{int(fnum(item, 'rows'))}x{int(fnum(item, 'cols'))}"
                for item in items
                if fnum(item, "rows") > 0 and fnum(item, "cols") > 0
            }
        )
        target_devices = sorted(
            {
                str(int(fnum(item, "target_devices")))
                for item in items
                if item.get("target_devices", "") not in {"", None}
            },
            key=lambda value: int(value),
        )
        merge_flags = sorted(
            {str(item.get("merge_adjacent_pairs", "")) for item in items if item.get("merge_adjacent_pairs", "") != ""}
        )
        size_labels = sorted(
            {str(item.get("size_label", "")) for item in items if item.get("size_label", "") != ""},
            key=lambda value: {"1MB": 0, "8MB": 1, "64MB": 2, "256MB": 3, "2GB": 4}.get(value, 99),
        )
        root_hint = "input/official_data"
        if source == "raw":
            root_hint = "input/raw_data"
        elif source == "evaluation_synthetic":
            root_hint = "evaluation_assets/manifests/synthetic_cases.json"
        out.append(
            {
                "model": model,
                "phase": phase,
                "source": source,
                "all_rows": len(items),
                "nonzero_rows": len(nonzero),
                "matrix_shapes": ", ".join(shapes),
                "target_devices": ", ".join(target_devices) if target_devices else "expert-level",
                "size_labels": ", ".join(size_labels) if size_labels else "",
                "merge_adjacent_pairs": ", ".join(merge_flags),
                "root": root_hint,
            }
        )
    return out


def gpu_mapping_sensitivity_rows(selected_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_sources: set[tuple[str, str, str]] = set()
    for entry in selected_rows:
        expert_trace = entry.get("source_expert_trace", "")
        if not expert_trace:
            continue
        source_key = (expert_trace, entry.get("size_label", ""), entry.get("sample_index", ""))
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        expert_path = REPO_ROOT / str(expert_trace)
        if not expert_path.exists():
            continue
        scale_factor = int(float(entry.get("scale_factor", 1) or 1))
        expert_matrix = communication_matrix(load_matrix(expert_path)) * scale_factor
        for target_devices in (64, 32, 16, 8):
            resized = resize_matrix(expert_matrix, target_devices)
            if resized is None:
                continue
            resized = communication_matrix(resized)
            metrics = matrix_metrics(resized, TOP_K_DEFAULT)
            rows.append(
                {
                    "trace_id": entry.get("trace_id", ""),
                    "size_label": entry.get("size_label", ""),
                    "sample_index": entry.get("sample_index", ""),
                    "source_expert_trace": expert_trace,
                    "scale_factor": scale_factor,
                    "target_devices": target_devices,
                    "experts_per_gpu": int(expert_matrix.shape[0] // target_devices),
                    "total_tokens": metrics["total_tokens"],
                    "bimodal_pressure_index": metrics["bimodal_pressure_index"],
                    "bpi_flow_lorenz80": metrics["bpi_flow_lorenz80"],
                    "bpi_dest_entropy": metrics["bpi_dest_entropy"],
                    "flow_gini_all": metrics["flow_gini_all"],
                    "flow_top10pct_share": metrics["flow_top10pct_share"],
                    "topk_dest_share": metrics["topk_dest_share"],
                    "nonzero_offdiag": metrics["nonzero_offdiag"],
                    "offdiag_cells": metrics["offdiag_cells"],
                }
            )

    base_by_trace = {
        (row["trace_id"], row["size_label"], row["sample_index"]): fnum(row, "bimodal_pressure_index")
        for row in rows
        if int(row["target_devices"]) == 64
    }
    for row in rows:
        key = (row["trace_id"], row["size_label"], row["sample_index"])
        row["delta_bpi_vs_64_experts"] = fnum(row, "bimodal_pressure_index") - base_by_trace.get(key, 0.0)
    return rows


def summarize_gpu_mapping_sensitivity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["experts_per_gpu"]), int(row["target_devices"]))].append(row)

    out: list[dict[str, object]] = []
    for (experts_per_gpu, target_devices), items in sorted(groups.items()):
        bpi = [fnum(item, "bimodal_pressure_index") for item in items]
        flow = [fnum(item, "bpi_flow_lorenz80") for item in items]
        dest = [fnum(item, "bpi_dest_entropy") for item in items]
        delta = [fnum(item, "delta_bpi_vs_64_experts") for item in items]
        out.append(
            {
                "experts_per_gpu": experts_per_gpu,
                "target_devices": target_devices,
                "count": len(items),
                "BPI mean": mean(bpi) if bpi else 0.0,
                "BPI p50": percentile(bpi, 50),
                "flow-lorenz80 mean": mean(flow) if flow else 0.0,
                "dest-entropy mean": mean(dest) if dest else 0.0,
                "delta vs 64 mean": mean(delta) if delta else 0.0,
                "delta vs 64 min": min(delta) if delta else 0.0,
                "delta vs 64 max": max(delta) if delta else 0.0,
            }
        )
    return out


def torus_probe_method_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("method", "")].append(row)
    out: list[dict[str, object]] = []
    for method, items in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "count": len(items),
                "makespan mean us": mean([fnum(item, "makespan_us") for item in items]),
                "makespan p50 us": percentile([fnum(item, "makespan_us") for item in items], 50),
                "fabric active mean": mean([fnum(item, "fabric_active_link_fraction_mean") for item in items]),
                "fabric Gini mean": mean([fnum(item, "fabric_busy_time_gini") for item in items]),
                "link intervals mean": mean([fnum(item, "link_use_count_sum") for item in items]),
                "halfring rounds mean": mean([fnum(item, "halfring_total_rounds") for item in items]),
                "halfring stage-substage mean": mean([fnum(item, "halfring_stage_substage_events") for item in items]),
            }
        )
    return out


def torus_probe_speedup_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_case[row.get("case_id", "")][row.get("method", "")] = row
    halfring_over_glaive = []
    glaive_over_halfring = []
    for items in by_case.values():
        glaive = items.get("glaive")
        halfring = items.get("halfringdr")
        if not glaive or not halfring:
            continue
        glaive_makespan = fnum(glaive, "makespan_us")
        halfring_makespan = fnum(halfring, "makespan_us")
        if glaive_makespan <= 0 or halfring_makespan <= 0:
            continue
        halfring_over_glaive.append(halfring_makespan / glaive_makespan)
        glaive_over_halfring.append(glaive_makespan / halfring_makespan)
    return {
        "case_count": len(halfring_over_glaive),
        "halfring_over_glaive_mean": mean(halfring_over_glaive) if halfring_over_glaive else 0.0,
        "halfring_over_glaive_p50": percentile(halfring_over_glaive, 50),
        "halfring_over_glaive_min": min(halfring_over_glaive) if halfring_over_glaive else 0.0,
        "halfring_over_glaive_max": max(halfring_over_glaive) if halfring_over_glaive else 0.0,
        "glaive_over_halfring_mean": mean(glaive_over_halfring) if glaive_over_halfring else 0.0,
        "glaive_over_halfring_p50": percentile(glaive_over_halfring, 50),
        "glaive_over_halfring_min": min(glaive_over_halfring) if glaive_over_halfring else 0.0,
        "glaive_over_halfring_max": max(glaive_over_halfring) if glaive_over_halfring else 0.0,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuttal simulation experiment manager.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_metrics = sub.add_parser("metrics")
    p_metrics.add_argument("--limit", type=int, default=0)
    p_metrics.add_argument("--per-group", type=int, default=3)
    p_metrics.add_argument("--no-raw-olmoe", action="store_true")
    p_metrics.set_defaults(func=analyze_traces)

    p_official_eplb = sub.add_parser("official-eplb-metrics")
    p_official_eplb.add_argument("--targets", default="32,64")
    p_official_eplb.add_argument("--limit", type=int, default=0)
    p_official_eplb.set_defaults(func=official_eplb_metrics)

    p_select = sub.add_parser("select-suite")
    p_select.add_argument("--per-group", type=int, default=3)
    p_select.set_defaults(func=select_suite_from_existing)

    p_materialize = sub.add_parser("materialize-suite")
    p_materialize.add_argument("--max-suite", type=int, default=0)
    p_materialize.set_defaults(func=materialize_suite)

    p_network = sub.add_parser("network-existing")
    p_network.set_defaults(func=analyze_existing_network_logs)

    p_run = sub.add_parser("run-suite")
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--max-cases", type=int, default=0)
    p_run.add_argument("--max-tasks", type=int, default=0)
    p_run.add_argument("--case-filter", default="")
    p_run.set_defaults(func=run_performance_suite)

    p_probe = sub.add_parser("run-torus-balanced-probe")
    p_probe.add_argument("--workers", type=int, default=4)
    p_probe.add_argument("--force", action="store_true")
    p_probe.set_defaults(func=run_torus_balanced_probe)

    p_parse = sub.add_parser("parse-suite")
    p_parse.set_defaults(func=parse_performance_logs)

    p_derived = sub.add_parser("derive-summaries")
    p_derived.set_defaults(func=derive_summaries)

    p_plot = sub.add_parser("plot")
    p_plot.set_defaults(func=plot_results_standalone)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
