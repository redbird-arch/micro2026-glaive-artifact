#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_ROOT = REPO_ROOT / "inputs" / "workload" / "DeepSeekV3Proxy"
SYSTEM_ROOT = REPO_ROOT / "inputs" / "system" / "DeepSeekV3Proxy"
NETWORK_ROOT = REPO_ROOT / "inputs" / "network" / "analytical" / "DeepSeekV3Proxy"
EXTERNAL_ROOT = REPO_ROOT / "inputs" / "external_collective" / "DeepSeekV3Proxy"
METADATA_ROOT = REPO_ROOT / "inputs" / "metadata" / "DeepSeekV3Proxy"

PREFILL_RE = re.compile(r"^Prefill_BS1024_Layer(?P<layer>\d+)_0\.csv$")
DECODE_RE = re.compile(r"^Decode_BS4096_Layer(?P<layer>\d+)_0\.csv$")

MODEL_NAME = "deepseek-ai/DeepSeek-V3"
MODEL_LABEL = "DeepSeek-V3-671B"
MODEL_LABEL_PLOT = "DeepSeek-V3 671B"
TRACE_PROXY_LABEL = "OLMoE 64-expert selected traces"
TOTAL_RANKS = 64
DENSE_DATA_PARALLEL_DEGREE = 64
EXPERT_PARALLEL_DEGREE = 64
EXPERT_REPLICA_GROUPS = 1
TOTAL_LAYERS = 61
DENSE_LAYERS = 3
FIRST_MOE_LAYER = DENSE_LAYERS + 1
NUM_TRACE_LAYERS = 16
HIDDEN_SIZE = 7168
NUM_ATTENTION_HEADS = 128
NUM_ROUTED_EXPERTS = 256
LOCAL_EXPERTS_PER_RANK = 4
NUM_EXPERTS_PER_TOKEN = 8
MOE_INTERMEDIATE_SIZE = 2048
DENSE_INTERMEDIATE_SIZE = 18432
SHARED_EXPERTS_PER_LAYER = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
MAX_POSITION_EMBEDDINGS = 163840
BFLOAT16_BYTES = 2
TOKEN_DISPATCH_BYTES = HIDDEN_SIZE * BFLOAT16_BYTES
KV_CACHE_BYTES_PER_TOKEN_LAYER = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * BFLOAT16_BYTES
WORKLOAD_LAYER_DELAY = 10


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    nominal_batch_size: int
    local_batch_size: int
    source_local_batch_size: int
    matcher: re.Pattern[str]


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    label: str
    topology_label: str
    topology_prefix: str
    peak_bf16_tflops: float
    utilization: float
    hbm_capacity_gib: float
    peak_hbm_bandwidth_gbps: float
    hbm_utilization: float
    topologies_per_dim: tuple[str, ...]
    dimension_type: tuple[str, ...]
    units_count: tuple[int, ...]
    links_count: tuple[int, ...]
    link_latency_ns: tuple[int, ...]
    link_bandwidth_gbps: tuple[float, ...]
    link_failure: tuple[int, ...]
    nic_latency_ns: tuple[int, ...]
    router_latency_ns: tuple[int, ...]
    hbm_latency_ns: tuple[int, ...]
    hbm_scale: tuple[int, ...]
    all_to_all_implementation: str

    @property
    def effective_hbm_bandwidth_gbps(self) -> float:
        return self.peak_hbm_bandwidth_gbps * self.hbm_utilization


PHASES = {
    "prefill": PhaseSpec(
        phase="prefill",
        nominal_batch_size=1024,
        local_batch_size=16,
        source_local_batch_size=16,
        matcher=PREFILL_RE,
    ),
    "decode": PhaseSpec(
        phase="decode",
        nominal_batch_size=4096,
        local_batch_size=64,
        source_local_batch_size=64,
        matcher=DECODE_RE,
    ),
}

HARDWARE = {
    "tpuv7": HardwareSpec(
        key="tpuv7",
        label="TPUv7",
        topology_label="8x8 torus",
        topology_prefix="tpuv7_torus_8x8",
        peak_bf16_tflops=2307.0,
        utilization=0.30,
        hbm_capacity_gib=192.0,
        peak_hbm_bandwidth_gbps=7380.0,
        hbm_utilization=0.80,
        topologies_per_dim=("Ring", "Ring"),
        dimension_type=("N", "N"),
        units_count=(8, 8),
        links_count=(2, 2),
        link_latency_ns=(100, 100),
        link_bandwidth_gbps=(1200.0, 1200.0),
        link_failure=(0, 0),
        nic_latency_ns=(0, 0),
        router_latency_ns=(0, 0),
        hbm_latency_ns=(500, 500),
        hbm_scale=(0, 0),
        all_to_all_implementation="ring_ring",
    ),
    "h100": HardwareSpec(
        key="h100",
        label="H100",
        topology_label="8 nodes x 8 GPUs switch",
        topology_prefix="h100_hgx_8node_8gpu_fattree",
        peak_bf16_tflops=1979.0,
        utilization=0.30,
        hbm_capacity_gib=80.0,
        peak_hbm_bandwidth_gbps=3350.0,
        hbm_utilization=0.80,
        topologies_per_dim=("Switch", "Switch"),
        dimension_type=("N", "N"),
        units_count=(8, 8),
        links_count=(18, 1),
        link_latency_ns=(500, 500),
        link_bandwidth_gbps=(50.0, 50.0),
        link_failure=(0, 0),
        nic_latency_ns=(0, 0),
        router_latency_ns=(50, 50),
        hbm_latency_ns=(500, 500),
        hbm_scale=(0, 0),
        all_to_all_implementation="ring_ring",
    ),
}

METHODS_BY_HARDWARE = {
    "tpuv7": ("halfrdr", "mpi", "glaive"),
    "h100": ("biring", "mpi", "glaive"),
}

SOURCE_LINKS = {
    "deepseek_v3_hf": "https://huggingface.co/deepseek-ai/DeepSeek-V3",
    "deepseek_v3_report": "https://arxiv.org/abs/2412.19437",
    "tpuv7_docs": "https://cloud.google.com/blog/products/compute/ironwood-tpu-for-ai-inference",
    "h100_docs": "https://www.nvidia.com/en-us/data-center/h100/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DeepSeek-V3 proxy inference workloads and analytical "
            "Astra-Sim assets using OLMoE expert-routing traces."
        )
    )
    parser.add_argument("--prefill-dir", type=Path, required=True)
    parser.add_argument("--decode-dir", type=Path, required=True)
    parser.add_argument("--external-root", type=Path)
    return parser.parse_args()


def read_square_matrix(path: Path) -> list[list[int]]:
    with path.open(newline="") as handle:
        matrix = [[int(cell) for cell in row] for row in csv.reader(handle) if row]
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError(f"{path} is not a non-empty square matrix")
    return matrix


def average_row_blocks(matrix: list[list[int]]) -> float:
    return sum(sum(row) for row in matrix) / len(matrix)


def effective_flops_per_ns(spec: HardwareSpec) -> float:
    return spec.peak_bf16_tflops * spec.utilization * 1000.0


def flop_time_ns(total_flops: float, spec: HardwareSpec) -> int:
    return max(1, int(round(total_flops / effective_flops_per_ns(spec))))


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_phase_entries(
    phase_spec: PhaseSpec, root: Path
) -> dict[int, dict[str, float | str]]:
    entries: dict[int, dict[str, float | str]] = {}
    for csv_path in sorted(root.glob("*.csv")):
        match = phase_spec.matcher.match(csv_path.name)
        if not match:
            continue
        layer = int(match.group("layer"))
        matrix = read_square_matrix(csv_path)
        avg_blocks = average_row_blocks(matrix)
        avg_tokens = avg_blocks / NUM_EXPERTS_PER_TOKEN
        avg_sequence_length = avg_tokens / phase_spec.local_batch_size
        source_avg_sequence_length = avg_tokens / phase_spec.source_local_batch_size
        entries[layer] = {
            "csv_path": str(csv_path.resolve()),
            "avg_row_blocks": avg_blocks,
            "avg_tokens_per_rank": avg_tokens,
            "avg_sequence_length": avg_sequence_length,
            "source_avg_sequence_length": source_avg_sequence_length,
        }
    if sorted(entries) != list(range(1, NUM_TRACE_LAYERS + 1)):
        raise ValueError(
            f"{phase_spec.phase} input does not provide layers 1..{NUM_TRACE_LAYERS}"
        )
    return entries


def norm_and_residual_flops(tokens: float) -> float:
    return 10.0 * HIDDEN_SIZE * tokens


def attention_linear_flops(tokens: float) -> float:
    return 8.0 * HIDDEN_SIZE * HIDDEN_SIZE * tokens


def prefill_attention_score_flops(tokens: float, sequence_length: float) -> float:
    return (
        4.0 * tokens * sequence_length * HIDDEN_SIZE
        + 5.0 * NUM_ATTENTION_HEADS * tokens * sequence_length
    )


def decode_attention_score_flops(tokens: float, context_length: float) -> float:
    return (
        4.0 * tokens * context_length * HIDDEN_SIZE
        + 5.0 * NUM_ATTENTION_HEADS * tokens * context_length
    )


def router_flops(tokens: float) -> float:
    return 2.0 * HIDDEN_SIZE * NUM_ROUTED_EXPERTS * tokens


def routed_expert_flops(tokens: float) -> float:
    return (
        6.0
        * HIDDEN_SIZE
        * MOE_INTERMEDIATE_SIZE
        * NUM_EXPERTS_PER_TOKEN
        * tokens
    )


def shared_expert_flops(tokens: float) -> float:
    return (
        6.0
        * HIDDEN_SIZE
        * MOE_INTERMEDIATE_SIZE
        * SHARED_EXPERTS_PER_LAYER
        * tokens
    )


def dense_mlp_flops(tokens: float) -> float:
    return 6.0 * HIDDEN_SIZE * DENSE_INTERMEDIATE_SIZE * tokens


def dense_dispatch_combine_flops(
    phase_name: str, tokens: float, sequence_length: float, context_length: float
) -> tuple[float, float]:
    dispatch = attention_linear_flops(tokens)
    if phase_name == "prefill":
        dispatch += prefill_attention_score_flops(tokens, sequence_length)
    else:
        dispatch += decode_attention_score_flops(tokens, context_length)
    dispatch += norm_and_residual_flops(tokens) / 2.0
    combine = dense_mlp_flops(tokens) + (norm_and_residual_flops(tokens) / 2.0)
    return dispatch, combine


def moe_dispatch_combine_flops(
    phase_name: str, tokens: float, sequence_length: float, context_length: float
) -> tuple[float, float]:
    dispatch = attention_linear_flops(tokens)
    if phase_name == "prefill":
        dispatch += prefill_attention_score_flops(tokens, sequence_length)
    else:
        dispatch += decode_attention_score_flops(tokens, context_length)
    dispatch += router_flops(tokens) + (norm_and_residual_flops(tokens) / 2.0)
    combine = (
        routed_expert_flops(tokens)
        + shared_expert_flops(tokens)
        + (norm_and_residual_flops(tokens) / 2.0)
    )
    return dispatch, combine


def attention_param_count() -> int:
    return 4 * HIDDEN_SIZE * HIDDEN_SIZE


def router_param_count() -> int:
    return HIDDEN_SIZE * NUM_ROUTED_EXPERTS


def expert_param_count(intermediate_size: int) -> int:
    return 3 * HIDDEN_SIZE * intermediate_size


def dispatch_memory_bytes(
    phase_name: str,
    tokens: float,
    context_length: float,
    shared_param_count: int,
) -> tuple[int, int]:
    read_bytes = shared_param_count * BFLOAT16_BYTES
    if phase_name == "decode":
        read_bytes += int(round(tokens * context_length * KV_CACHE_BYTES_PER_TOKEN_LAYER))
        write_bytes = int(round(tokens * KV_CACHE_BYTES_PER_TOKEN_LAYER))
    else:
        write_bytes = int(round(tokens * KV_CACHE_BYTES_PER_TOKEN_LAYER))
    return read_bytes, write_bytes


def output_activation_write_bytes(tokens: float) -> int:
    return int(round(tokens * HIDDEN_SIZE * BFLOAT16_BYTES))


def workload_line(
    layer_name: str,
    compute_ns: int,
    comm_type: str,
    comm_bytes: int,
    mem_read_bytes: int,
    mem_write_bytes: int,
) -> str:
    return (
        f"{layer_name}\t-1\t{compute_ns}\t{comm_type}\t{comm_bytes}\t"
        "0\tNONE\t0\t0\tNONE\t0\t"
        f"{WORKLOAD_LAYER_DELAY}\t{mem_read_bytes}\t{mem_write_bytes}"
    )


def generate_workloads(
    phase_entries: dict[str, dict[int, dict[str, float | str]]]
) -> tuple[dict[str, str], dict[str, dict]]:
    workload_files: dict[str, str] = {}
    memory_sanity: dict[str, dict] = {}

    trace_context_by_layer = {
        layer: float(phase_entries["prefill"][layer]["avg_sequence_length"])
        for layer in range(1, NUM_TRACE_LAYERS + 1)
    }
    attention_params = attention_param_count()
    router_params = router_param_count()
    moe_expert_params = expert_param_count(MOE_INTERMEDIATE_SIZE)
    dense_mlp_params = expert_param_count(DENSE_INTERMEDIATE_SIZE)
    dispatch_shared_params = attention_params + router_params
    moe_combine_params = (LOCAL_EXPERTS_PER_RANK + SHARED_EXPERTS_PER_LAYER) * moe_expert_params

    dense_resident_params_total = DENSE_LAYERS * (attention_params + dense_mlp_params)
    moe_resident_params_total = (TOTAL_LAYERS - DENSE_LAYERS) * (
        attention_params + router_params + moe_combine_params
    )
    resident_params_total = dense_resident_params_total + moe_resident_params_total

    for hw_key, hw in HARDWARE.items():
        for phase_name, phase_spec in PHASES.items():
            lines = ["DISTRIBUTED_INFERENCE", str(TOTAL_LAYERS * 2)]
            per_layer_summary: list[dict[str, float | int | str]] = []

            for model_layer in range(1, TOTAL_LAYERS + 1):
                if model_layer <= DENSE_LAYERS:
                    trace_layer = 1
                    entry = phase_entries[phase_name][trace_layer]
                    tokens = float(entry["avg_tokens_per_rank"])
                    sequence_length = max(1.0, float(entry["avg_sequence_length"]))
                    context_length = max(1.0, trace_context_by_layer[trace_layer])
                    dispatch_flops, combine_flops = dense_dispatch_combine_flops(
                        phase_name,
                        tokens,
                        sequence_length,
                        context_length,
                    )
                    dispatch_compute_ns = flop_time_ns(dispatch_flops, hw)
                    combine_compute_ns = flop_time_ns(combine_flops, hw)
                    dispatch_mem_read, dispatch_mem_write = dispatch_memory_bytes(
                        phase_name, tokens, context_length, attention_params
                    )
                    combine_mem_read = dense_mlp_params * BFLOAT16_BYTES
                    combine_mem_write = output_activation_write_bytes(tokens)
                    dispatch_name = f"DeepSeekV3Dense{model_layer:02d}Dispatch"
                    combine_name = f"DeepSeekV3Dense{model_layer:02d}Combine"
                    lines.append(
                        workload_line(
                            dispatch_name,
                            dispatch_compute_ns,
                            "NONE",
                            0,
                            dispatch_mem_read,
                            dispatch_mem_write,
                        )
                    )
                    lines.append(
                        workload_line(
                            combine_name,
                            combine_compute_ns,
                            "NONE",
                            0,
                            combine_mem_read,
                            combine_mem_write,
                        )
                    )
                    per_layer_summary.append(
                        {
                            "model_layer": model_layer,
                            "trace_layer": trace_layer,
                            "layer_type": "dense",
                            "tokens_per_rank": round(tokens, 6),
                            "sequence_length": round(sequence_length, 6),
                            "context_length": round(context_length, 6),
                            "dispatch_compute_ns": dispatch_compute_ns,
                            "combine_compute_ns": combine_compute_ns,
                            "dispatch_memory_read_bytes": dispatch_mem_read,
                            "dispatch_memory_write_bytes": dispatch_mem_write,
                            "combine_memory_read_bytes": combine_mem_read,
                            "combine_memory_write_bytes": combine_mem_write,
                            "dispatch_comm_bytes": 0,
                            "combine_comm_bytes": 0,
                        }
                    )
                    continue

                trace_layer = ((model_layer - FIRST_MOE_LAYER) % NUM_TRACE_LAYERS) + 1
                entry = phase_entries[phase_name][trace_layer]
                tokens = float(entry["avg_tokens_per_rank"])
                sequence_length = max(1.0, float(entry["avg_sequence_length"]))
                context_length = max(1.0, trace_context_by_layer[trace_layer])
                dispatch_flops, combine_flops = moe_dispatch_combine_flops(
                    phase_name,
                    tokens,
                    sequence_length,
                    context_length,
                )
                dispatch_compute_ns = flop_time_ns(dispatch_flops, hw)
                combine_compute_ns = flop_time_ns(combine_flops, hw)
                comm_bytes = max(
                    TOKEN_DISPATCH_BYTES,
                    int(round(float(entry["avg_row_blocks"]) * TOKEN_DISPATCH_BYTES)),
                )
                dispatch_mem_read, dispatch_mem_write = dispatch_memory_bytes(
                    phase_name,
                    tokens,
                    context_length,
                    dispatch_shared_params,
                )
                combine_mem_read = moe_combine_params * BFLOAT16_BYTES
                combine_mem_write = output_activation_write_bytes(tokens)
                dispatch_name = f"DeepSeekV3MoE{model_layer:02d}Dispatch"
                combine_name = f"DeepSeekV3MoE{model_layer:02d}Combine"
                lines.append(
                    workload_line(
                        dispatch_name,
                        dispatch_compute_ns,
                        "ALLTOALLV",
                        comm_bytes,
                        dispatch_mem_read,
                        dispatch_mem_write,
                    )
                )
                lines.append(
                    workload_line(
                        combine_name,
                        combine_compute_ns,
                        "ALLTOALLV",
                        comm_bytes,
                        combine_mem_read,
                        combine_mem_write,
                    )
                )
                per_layer_summary.append(
                    {
                        "model_layer": model_layer,
                        "trace_layer": trace_layer,
                        "layer_type": "moe",
                        "tokens_per_rank": round(tokens, 6),
                        "sequence_length": round(sequence_length, 6),
                        "context_length": round(context_length, 6),
                        "dispatch_compute_ns": dispatch_compute_ns,
                        "combine_compute_ns": combine_compute_ns,
                        "dispatch_memory_read_bytes": dispatch_mem_read,
                        "dispatch_memory_write_bytes": dispatch_mem_write,
                        "combine_memory_read_bytes": combine_mem_read,
                        "combine_memory_write_bytes": combine_mem_write,
                        "dispatch_comm_bytes": comm_bytes,
                        "combine_comm_bytes": comm_bytes,
                    }
                )

            workload_path = WORKLOAD_ROOT / f"{hw.key}_{phase_name}_inference.txt"
            workload_path.parent.mkdir(parents=True, exist_ok=True)
            workload_path.write_text("\n".join(lines) + "\n")
            workload_files[f"{hw.key}_{phase_name}"] = relative_to_repo(workload_path)

            max_context_length = max(
                float(item["context_length"]) for item in per_layer_summary
            )
            kv_cache_gib = (
                phase_spec.local_batch_size
                * max_context_length
                * TOTAL_LAYERS
                * KV_CACHE_BYTES_PER_TOKEN_LAYER
                / (1024.0**3)
            )
            resident_model_gib = resident_params_total * BFLOAT16_BYTES / (1024.0**3)
            memory_sanity[f"{hw.key}_{phase_name}"] = {
                "hardware_hbm_capacity_gib": hw.hbm_capacity_gib,
                "resident_model_gib_per_rank": round(resident_model_gib, 6),
                "kv_cache_gib_per_rank": round(kv_cache_gib, 6),
                "fits_hbm_capacity": (resident_model_gib + kv_cache_gib) < hw.hbm_capacity_gib,
                "layers": per_layer_summary,
            }

    return workload_files, memory_sanity


def generate_networks() -> dict[str, str]:
    network_files: dict[str, str] = {}
    for hw_key, hw in HARDWARE.items():
        payload = {
            "topology-name": "Hierarchical",
            "topologies-per-dim": list(hw.topologies_per_dim),
            "dimension-type": list(hw.dimension_type),
            "dimensions-count": len(hw.units_count),
            "units-count": list(hw.units_count),
            "links-count": list(hw.links_count),
            "link-latency": list(hw.link_latency_ns),
            "link-bandwidth": list(hw.link_bandwidth_gbps),
            "link-failure": list(hw.link_failure),
            "nic-latency": list(hw.nic_latency_ns),
            "router-latency": list(hw.router_latency_ns),
            "hbm-latency": list(hw.hbm_latency_ns),
            "hbm-bandwidth": [hw.effective_hbm_bandwidth_gbps] * len(hw.units_count),
            "hbm-scale": list(hw.hbm_scale),
        }
        path = NETWORK_ROOT / f"{hw.key}.json"
        write_json(path, payload)
        network_files[hw_key] = relative_to_repo(path)
    return network_files


def write_system_file(path: Path, hw: HardwareSpec, external_csv: Path) -> None:
    contents = [
        "scheduling-policy: LIFO",
        "endpoint-delay: 1",
        "active-chunks-per-dimension: 1",
        "preferred-dataset-splits: 1",
        "boost-mode: 0",
        "all-reduce-implementation: ring_ring",
        "all-gather-implementation: ring_ring",
        "reduce-scatter-implementation: ring_ring",
        f"all-to-all-implementation: {hw.all_to_all_implementation}",
        "collective-optimization: baseline",
        "intra-dimension-scheduling: FIFO",
        "inter-dimension-scheduling: baseline",
        f"external-collective-file: {relative_to_repo(external_csv)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(contents) + "\n")


def generate_systems_and_copy_external(
    external_root: Path | None,
) -> tuple[dict[str, str], list[dict[str, object]], list[str]]:
    system_files: dict[str, str] = {}
    scenarios: list[dict[str, object]] = []
    missing: list[str] = []

    if external_root is None:
        return system_files, scenarios, missing

    external_root = external_root.resolve()
    if not external_root.exists():
        raise FileNotFoundError(f"external root does not exist: {external_root}")

    for hw_key, hw in HARDWARE.items():
        for phase_name in PHASES:
            for method in METHODS_BY_HARDWARE[hw_key]:
                source_name = f"{hw.topology_prefix}_{phase_name}_{method}.csv"
                source_csv = external_root / source_name
                if not source_csv.exists():
                    missing.append(source_name)
                    continue

                copied_csv = EXTERNAL_ROOT / source_name
                copied_csv.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_csv, copied_csv)

                system_path = SYSTEM_ROOT / f"{hw_key}_{phase_name}_{method}.txt"
                write_system_file(system_path, hw, copied_csv)
                system_key = f"{hw_key}_{phase_name}_{method}"
                system_files[system_key] = relative_to_repo(system_path)

                scenarios.append(
                    {
                        "run_name": system_key,
                        "hardware": hw_key,
                        "hardware_label": hw.label,
                        "topology_label": hw.topology_label,
                        "phase": phase_name,
                        "method": method,
                        "network": relative_to_repo(NETWORK_ROOT / f"{hw_key}.json"),
                        "system": relative_to_repo(system_path),
                        "workload": relative_to_repo(
                            WORKLOAD_ROOT / f"{hw_key}_{phase_name}_inference.txt"
                        ),
                        "units_count": list(hw.units_count),
                        "num_queues_per_dim": [1] * len(hw.units_count),
                    }
                )

    if not scenarios:
        raise FileNotFoundError(
            f"no external collective CSVs were found under {external_root}"
        )

    return system_files, scenarios, missing


def main() -> None:
    args = parse_args()
    phase_entries = {
        "prefill": load_phase_entries(PHASES["prefill"], args.prefill_dir.resolve()),
        "decode": load_phase_entries(PHASES["decode"], args.decode_dir.resolve()),
    }

    workload_files, memory_sanity = generate_workloads(phase_entries)
    network_files = generate_networks()
    system_files, scenarios, missing_external = generate_systems_and_copy_external(
        args.external_root
    )

    hardware_summary = {}
    for hw_key, hw in HARDWARE.items():
        hardware_summary[hw_key] = {
            "label": hw.label,
            "topology_label": hw.topology_label,
            "peak_bf16_tflops": hw.peak_bf16_tflops,
            "utilization_assumption": hw.utilization,
            "effective_bf16_tflops": round(hw.peak_bf16_tflops * hw.utilization, 6),
            "hbm_capacity_gib": hw.hbm_capacity_gib,
            "peak_hbm_bandwidth_gbps": hw.peak_hbm_bandwidth_gbps,
            "hbm_utilization_assumption": hw.hbm_utilization,
            "effective_hbm_bandwidth_gbps": hw.effective_hbm_bandwidth_gbps,
            "network_file": network_files[hw_key],
        }

    phase_summary = {}
    for phase_name, phase_spec in PHASES.items():
        phase_summary[phase_name] = {
            "nominal_batch_size": phase_spec.nominal_batch_size,
            "local_batch_size": phase_spec.local_batch_size,
            "source_local_batch_size": phase_spec.source_local_batch_size,
            "selected_input_dir": str(
                args.prefill_dir.resolve()
                if phase_name == "prefill"
                else args.decode_dir.resolve()
            ),
            "trace_layers": {
                str(layer): phase_entries[phase_name][layer]
                for layer in range(1, NUM_TRACE_LAYERS + 1)
            },
        }

    metadata = {
        "model": {
            "name": MODEL_LABEL,
            "plot_label": MODEL_LABEL_PLOT,
            "source_model": MODEL_NAME,
            "trace_proxy": TRACE_PROXY_LABEL,
            "parallelism": {
                "total_ranks": TOTAL_RANKS,
                "dense_data_parallel_degree": DENSE_DATA_PARALLEL_DEGREE,
                "expert_parallel_degree": EXPERT_PARALLEL_DEGREE,
                "expert_replica_groups": EXPERT_REPLICA_GROUPS,
                "shared_dense_weights_replicated_on_all_ranks": True,
                "one_full_expert_set_across_all_ranks": True,
                "description": (
                    "Overlapped DP=EP=64 semantics on the same 64 ranks: "
                    "dense/shared layers use per-rank local batches, while "
                    "MoE dispatch/combine uses a single 64-rank alltoallv group."
                ),
            },
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": TOTAL_LAYERS,
            "dense_layers_without_moe": DENSE_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "dense_intermediate_size": DENSE_INTERMEDIATE_SIZE,
            "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "num_routed_experts": NUM_ROUTED_EXPERTS,
            "local_experts_per_rank": LOCAL_EXPERTS_PER_RANK,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOKEN,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
            "dispatch_payload_bytes": TOKEN_DISPATCH_BYTES,
            "kv_cache_bytes_per_token_layer": KV_CACHE_BYTES_PER_TOKEN_LAYER,
        },
        "sources": SOURCE_LINKS,
        "selection": {
            "parallelism_semantics": (
                "The same 64 ranks overlap dense-layer data parallelism and "
                "expert parallelism. Each rank processes its own local batch "
                "for dense/shared layers, and the expert layers use the full "
                "64-rank alltoallv instead of a multicast-style replicated input."
            ),
            "prefill": {
                "nominal_batch_size": PHASES["prefill"].nominal_batch_size,
                "local_batch_size": PHASES["prefill"].local_batch_size,
            },
            "decode": {
                "nominal_batch_size": PHASES["decode"].nominal_batch_size,
                "local_batch_size": PHASES["decode"].local_batch_size,
                "decode_context_assumption": (
                    "Per-layer decode attention context length reuses the "
                    "matching repeated prefill trace layer average sequence length."
                ),
            },
        },
        "hardware": hardware_summary,
        "phases": phase_summary,
        "memory_sanity": memory_sanity,
        "generated_files": {
            "workloads": workload_files,
            "networks": network_files,
            "systems": system_files,
        },
        "external_collective_root": (
            str(args.external_root.resolve()) if args.external_root else None
        ),
        "missing_external_collectives": missing_external,
    }

    metadata_path = METADATA_ROOT / "deepseekv3_proxy_selected.json"
    write_json(metadata_path, metadata)

    experiments = {
        "model": MODEL_LABEL,
        "scenarios": scenarios,
        "summary_metadata": relative_to_repo(metadata_path),
    }
    experiments_path = METADATA_ROOT / "deepseekv3_proxy_experiments.json"
    write_json(experiments_path, experiments)

    print(f"Wrote {metadata_path}")
    print(f"Wrote {experiments_path}")
    print(
        f"Generated {len(workload_files)} workload files and "
        f"{len(network_files)} network files"
    )
    if system_files:
        print(f"Generated {len(system_files)} system files from external collective CSVs")
    if missing_external:
        print(f"Missing {len(missing_external)} external collective CSVs")


if __name__ == "__main__":
    main()
