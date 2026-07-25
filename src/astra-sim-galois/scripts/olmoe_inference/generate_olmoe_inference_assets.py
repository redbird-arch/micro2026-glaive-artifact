#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_ROOT = REPO_ROOT / "inputs" / "workload" / "OLMoE"
SYSTEM_ROOT = REPO_ROOT / "inputs" / "system" / "OLMoE"
NETWORK_ROOT = REPO_ROOT / "inputs" / "network" / "analytical" / "OLMoE"
EXTERNAL_ROOT = REPO_ROOT / "inputs" / "external_collective" / "OLMoE"
METADATA_ROOT = REPO_ROOT / "inputs" / "metadata" / "OLMoE"

PREFILL_RE = re.compile(r"^Prefill_BS1024_Layer(?P<layer>\d+)_0\.csv$")
DECODE_RE = re.compile(r"^Decode_BS4096_Layer(?P<layer>\d+)_0\.csv$")

MODEL_NAME = "allenai/OLMoE-1B-7B-0924"
MODEL_LABEL = "OLMoE-7B"
NUM_HIDDEN_LAYERS = 16
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1024
NUM_ATTENTION_HEADS = 16
NUM_LOCAL_EXPERTS = 64
NUM_EXPERTS_PER_TOKEN = 8
MAX_POSITION_EMBEDDINGS = 4096
TOKEN_DISPATCH_BYTES = 4096
BFLOAT16_BYTES = 2
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
    hbm_bandwidth_gbps: float
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


PHASES = {
    "prefill": PhaseSpec(
        phase="prefill",
        nominal_batch_size=1024,
        local_batch_size=32,
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
        utilization=0.20,
        hbm_capacity_gib=192.0,
        hbm_bandwidth_gbps=7380.0,
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
        utilization=0.20,
        hbm_capacity_gib=80.0,
        hbm_bandwidth_gbps=3350.0,
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
    "olmoe_config": "https://huggingface.co/allenai/OLMoE-1B-7B-0924/raw/main/config.json",
    "tpuv7_docs": "https://docs.cloud.google.com/tpu/docs/tpu7x",
    "h100_docs": "https://www.nvidia.com/en-us/data-center/h100/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate OLMoE-7B inference workloads, analytical network/system "
            "inputs, and metadata for Astra-Sim-Galois."
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


def attention_linear_flops(tokens: float) -> float:
    return 8.0 * HIDDEN_SIZE * HIDDEN_SIZE * tokens


def prefill_attention_flops(tokens: float, sequence_length: float) -> float:
    return (
        4.0 * tokens * sequence_length * HIDDEN_SIZE
        + 5.0 * NUM_ATTENTION_HEADS * tokens * sequence_length
    )


def decode_attention_flops(tokens: float, context_length: float) -> float:
    return (
        4.0 * tokens * context_length * HIDDEN_SIZE
        + 5.0 * NUM_ATTENTION_HEADS * tokens * context_length
    )


def router_flops(tokens: float) -> float:
    return 2.0 * HIDDEN_SIZE * NUM_LOCAL_EXPERTS * tokens


def expert_mlp_flops(tokens: float) -> float:
    # OLMoE uses a gated MLP path per selected expert: gate/up/down projections.
    return (
        6.0
        * HIDDEN_SIZE
        * INTERMEDIATE_SIZE
        * NUM_EXPERTS_PER_TOKEN
        * tokens
    )


def norm_and_residual_flops(tokens: float) -> float:
    return 10.0 * HIDDEN_SIZE * tokens


def split_block_flops(
    phase_name: str,
    tokens: float,
    seq_len: float,
    context_len: float,
) -> tuple[float, float]:
    attention_flops = attention_linear_flops(tokens)
    if phase_name == "prefill":
        attention_flops += prefill_attention_flops(tokens, seq_len)
    else:
        attention_flops += decode_attention_flops(tokens, context_len)

    norm_flops = norm_and_residual_flops(tokens)
    dispatch_flops = attention_flops + router_flops(tokens) + (norm_flops / 2.0)
    combine_flops = expert_mlp_flops(tokens) + (norm_flops / 2.0)
    return dispatch_flops, combine_flops


def effective_flops_per_ns(spec: HardwareSpec) -> float:
    return spec.peak_bf16_tflops * spec.utilization * 1000.0


def flop_time_ns(total_flops: float, spec: HardwareSpec) -> int:
    return max(1, int(round(total_flops / effective_flops_per_ns(spec))))


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load_phase_entries(phase_spec: PhaseSpec, root: Path) -> dict[int, dict[str, float | str]]:
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
    if sorted(entries) != list(range(1, NUM_HIDDEN_LAYERS + 1)):
        raise ValueError(
            f"{phase_spec.phase} input does not provide layers 1..{NUM_HIDDEN_LAYERS}"
        )
    return entries


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def workload_line(layer_name: str, compute_ns: int, comm_bytes: int) -> str:
    return (
        f"{layer_name}\t-1\t{compute_ns}\tALLTOALLV\t{comm_bytes}\t"
        "0\tNONE\t0\t0\tNONE\t0\t"
        f"{WORKLOAD_LAYER_DELAY}"
    )


def generate_workloads(
    phase_entries: dict[str, dict[int, dict[str, float | str]]]
) -> tuple[dict[str, str], dict[str, dict]]:
    workload_files: dict[str, str] = {}
    memory_sanity: dict[str, dict] = {}

    for hw_key, hw in HARDWARE.items():
        for phase_name, phase_spec in PHASES.items():
            lines = ["DISTRIBUTED_INFERENCE", str(NUM_HIDDEN_LAYERS * 2)]
            per_layer_summary: list[dict[str, float | int]] = []

            for layer in range(1, NUM_HIDDEN_LAYERS + 1):
                entry = phase_entries[phase_name][layer]
                tokens = float(entry["avg_tokens_per_rank"])
                seq_len = max(1.0, float(entry["avg_sequence_length"]))
                context_len = seq_len
                if phase_name == "decode":
                    context_len = max(
                        1.0,
                        float(phase_entries["prefill"][layer]["avg_sequence_length"]),
                    )

                dispatch_flops, combine_flops = split_block_flops(
                    phase_name,
                    tokens,
                    seq_len,
                    context_len,
                )
                dispatch_compute_ns = flop_time_ns(dispatch_flops, hw)
                combine_compute_ns = flop_time_ns(combine_flops, hw)
                total_flops = dispatch_flops + combine_flops
                comm_bytes = max(
                    TOKEN_DISPATCH_BYTES,
                    int(round(float(entry["avg_row_blocks"]) * TOKEN_DISPATCH_BYTES)),
                )
                lines.append(workload_line(f"OLMoEBlock{layer:02d}Dispatch", dispatch_compute_ns, comm_bytes))
                lines.append(workload_line(f"OLMoEBlock{layer:02d}Combine", combine_compute_ns, comm_bytes))

                kv_cache_bytes = (
                    2.0
                    * NUM_HIDDEN_LAYERS
                    * phase_spec.local_batch_size
                    * context_len
                    * HIDDEN_SIZE
                    * BFLOAT16_BYTES
                )
                hidden_state_bytes = tokens * HIDDEN_SIZE * BFLOAT16_BYTES
                per_layer_summary.append(
                    {
                        "layer": layer,
                        "dispatch_compute_ns": dispatch_compute_ns,
                        "combine_compute_ns": combine_compute_ns,
                        "total_compute_ns": dispatch_compute_ns + combine_compute_ns,
                        "dispatch_comm_bytes": comm_bytes,
                        "combine_comm_bytes": comm_bytes,
                        "avg_tokens_per_rank": round(tokens, 6),
                        "avg_sequence_length": round(seq_len, 6),
                        "source_avg_sequence_length": round(
                            float(entry["source_avg_sequence_length"]),
                            6,
                        ),
                        "decode_context_length": round(context_len, 6),
                        "dispatch_flops": round(dispatch_flops, 3),
                        "combine_flops": round(combine_flops, 3),
                        "total_flops": round(total_flops, 3),
                        "approx_kv_cache_gib": round(
                            kv_cache_bytes / (1024.0**3), 6
                        ),
                        "approx_hidden_state_mib": round(
                            hidden_state_bytes / (1024.0**2), 6
                        ),
                    }
                )

            workload_path = WORKLOAD_ROOT / f"{hw.key}_{phase_name}_inference.txt"
            workload_path.write_text("\n".join(lines) + "\n")
            workload_files[f"{hw.key}_{phase_name}"] = relative_to_repo(workload_path)

            max_kv = max(item["approx_kv_cache_gib"] for item in per_layer_summary)
            max_hidden = max(
                item["approx_hidden_state_mib"] for item in per_layer_summary
            )
            memory_sanity[f"{hw.key}_{phase_name}"] = {
                "hardware_hbm_capacity_gib": hw.hbm_capacity_gib,
                "max_layer_kv_cache_gib": round(max_kv, 6),
                "max_layer_hidden_state_mib": round(max_hidden, 6),
                "fits_hbm_capacity": max_kv < hw.hbm_capacity_gib,
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
            "hbm-bandwidth": [hw.hbm_bandwidth_gbps] * len(hw.units_count),
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
            "hbm_bandwidth_gbps": hw.hbm_bandwidth_gbps,
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
            "layers": {
                str(layer): phase_entries[phase_name][layer]
                for layer in range(1, NUM_HIDDEN_LAYERS + 1)
            },
        }

    metadata = {
        "model": {
            "name": MODEL_LABEL,
            "source_model": MODEL_NAME,
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "num_attention_heads": NUM_ATTENTION_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_local_experts": NUM_LOCAL_EXPERTS,
            "num_experts_per_tok": NUM_EXPERTS_PER_TOKEN,
            "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        },
        "sources": SOURCE_LINKS,
        "selection": {
            "token_dispatch_bytes": TOKEN_DISPATCH_BYTES,
            "prefill": {
                "nominal_batch_size": PHASES["prefill"].nominal_batch_size,
                "local_batch_size": PHASES["prefill"].local_batch_size,
                "source_local_batch_size": PHASES["prefill"].source_local_batch_size,
                "sample_index": 0,
            },
            "decode": {
                "nominal_batch_size": PHASES["decode"].nominal_batch_size,
                "local_batch_size": PHASES["decode"].local_batch_size,
                "source_local_batch_size": PHASES["decode"].source_local_batch_size,
                "sample_index": 0,
                "decode_context_assumption": (
                    "Per-layer decode attention context length is reused from the "
                    "matching prefill layer average sequence length."
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

    metadata_path = METADATA_ROOT / "olmoe_inference_selected.json"
    write_json(metadata_path, metadata)

    experiments = {
        "model": MODEL_LABEL,
        "scenarios": scenarios,
        "summary_metadata": relative_to_repo(metadata_path),
    }
    experiments_path = METADATA_ROOT / "olmoe_inference_experiments.json"
    write_json(experiments_path, experiments)

    print(f"Wrote {metadata_path}")
    print(f"Wrote {experiments_path}")
    print(f"Generated {len(workload_files)} workload files and {len(network_files)} network files")
    if system_files:
        print(f"Generated {len(system_files)} system files from external collective CSVs")
    if missing_external:
        print(f"Missing {len(missing_external)} external collective CSVs")


if __name__ == "__main__":
    main()
