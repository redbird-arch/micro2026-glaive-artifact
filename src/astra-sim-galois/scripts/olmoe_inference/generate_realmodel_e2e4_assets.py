#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODING_ROOT = REPO_ROOT.parent
ASSET_ROOT = Path(
    os.environ.get("GLAIVE_ASTRA_ASSET_ROOT", REPO_ROOT / "inputs")
).resolve()
WORKLOAD_ROOT = ASSET_ROOT / "workload" / "RealModelE2E4"
SYSTEM_ROOT = ASSET_ROOT / "system" / "RealModelE2E4"
NETWORK_ROOT = ASSET_ROOT / "network" / "analytical" / "RealModelE2E4"
EXTERNAL_ROOT = ASSET_ROOT / "external_collective" / "RealModelE2E4"
METADATA_ROOT = ASSET_ROOT / "metadata" / "RealModelE2E4"

BF16_BYTES = 2
WORKLOAD_LAYER_DELAY = 10


@dataclass(frozen=True)
class HardwareSpec:
    key: str
    label: str
    topology_label: str
    topology_name: str
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


HARDWARE = {
    "tpuv7": HardwareSpec(
        key="tpuv7",
        label="TPUv7",
        topology_label="8x8 torus",
        topology_name="tpuv7_torus_8x8",
        peak_bf16_tflops=2307.0,
        utilization=0.35,
        hbm_capacity_gib=192.0,
        peak_hbm_bandwidth_gbps=7380.0,
        hbm_utilization=0.90,
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
        topology_name="h100_hgx_8node_8gpu_fattree",
        peak_bf16_tflops=1979.0,
        utilization=0.35,
        hbm_capacity_gib=80.0,
        peak_hbm_bandwidth_gbps=3350.0,
        hbm_utilization=0.90,
        topologies_per_dim=("Switch", "Switch"),
        dimension_type=("N", "N"),
        units_count=(8, 8),
        links_count=(18, 1),
        link_latency_ns=(500, 500),
        # The advertised 900 GB/s NVLink bandwidth is bidirectional aggregate.
        # ASTRA-Sim models directed links, so each of the 18 local links uses
        # 25 GB/s. The scale-out link remains 50 GB/s.
        link_bandwidth_gbps=(25.0, 50.0),
        link_failure=(0, 0),
        nic_latency_ns=(0, 0),
        router_latency_ns=(50, 50),
        hbm_latency_ns=(500, 500),
        hbm_scale=(0, 0),
        all_to_all_implementation="ring_ring",
    ),
    "tpuv4": HardwareSpec(
        key="tpuv4",
        label="TPUv4",
        topology_label="4x4 torus",
        topology_name="tpuv4_torus_4x4",
        peak_bf16_tflops=275.0,
        utilization=0.35,
        hbm_capacity_gib=32.0,
        peak_hbm_bandwidth_gbps=1200.0,
        hbm_utilization=0.90,
        topologies_per_dim=("Ring", "Ring"),
        dimension_type=("N", "N"),
        units_count=(4, 4),
        links_count=(2, 2),
        link_latency_ns=(500, 500),
        link_bandwidth_gbps=(56.0, 56.0),
        link_failure=(0, 0),
        nic_latency_ns=(0, 0),
        router_latency_ns=(0, 0),
        hbm_latency_ns=(500, 500),
        hbm_scale=(0, 0),
        all_to_all_implementation="ring_ring",
    ),
    "a100": HardwareSpec(
        key="a100",
        label="A100",
        topology_label="2 nodes x 8 GPUs switch",
        topology_name="a100_2node_8gpu_fattree",
        peak_bf16_tflops=312.0,
        utilization=0.35,
        hbm_capacity_gib=80.0,
        peak_hbm_bandwidth_gbps=1935.0,
        hbm_utilization=0.90,
        topologies_per_dim=("Switch", "Switch"),
        dimension_type=("N", "N"),
        units_count=(8, 2),
        links_count=(18, 1),
        link_latency_ns=(500, 500),
        link_bandwidth_gbps=(300.0, 25.0),
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
    "tpuv4": ("halfrdr", "mpi", "glaive"),
    "a100": ("biring", "mpi", "glaive"),
}

TOPOLOGY_TO_HARDWARE = {
    "tpuv7_torus_8x8": "tpuv7",
    "h100_hgx_8node_8gpu_fattree": "h100",
    "tpuv4_torus_4x4": "tpuv4",
    "a100_2node_8gpu_fattree": "a100",
}

MANIFEST_DEFAULTS = {
    "deepseekv32_small": {
        "manifest": CODING_ROOT
        / "collective-synthesizer"
        / "input"
        / "generated"
        / "realmodel_e2e4"
        / "deepseekv32_small"
        / "manifest.json",
        "external_root": CODING_ROOT
        / "collective-synthesizer"
        / "results"
        / "realmodel_e2e4"
        / "deepseekv32_small"
        / "astra_external",
    },
    "deepseekv32_large": {
        "manifest": CODING_ROOT
        / "collective-synthesizer"
        / "input"
        / "generated"
        / "realmodel_e2e4"
        / "deepseekv32_large"
        / "manifest.json",
        "external_root": CODING_ROOT
        / "collective-synthesizer"
        / "results"
        / "realmodel_e2e4"
        / "deepseekv32_large"
        / "astra_external",
    },
    "qwen3_30b_a3b_small": {
        "manifest": CODING_ROOT
        / "collective-synthesizer"
        / "input"
        / "generated"
        / "realmodel_e2e4"
        / "qwen3_30b_a3b_small"
        / "manifest.json",
        "external_root": CODING_ROOT
        / "collective-synthesizer"
        / "results"
        / "realmodel_e2e4"
        / "qwen3_30b_a3b_small"
        / "astra_external",
    },
    "qwen3_30b_a3b_large": {
        "manifest": CODING_ROOT
        / "collective-synthesizer"
        / "input"
        / "generated"
        / "realmodel_e2e4"
        / "qwen3_30b_a3b_large"
        / "manifest.json",
        "external_root": CODING_ROOT
        / "collective-synthesizer"
        / "results"
        / "realmodel_e2e4"
        / "qwen3_30b_a3b_large"
        / "astra_external",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Astra-Sim analytical assets for the Figure 12 real-model "
            "DeepSeek-V3.2 and Qwen3-30B-A3B experiments."
        )
    )
    parser.add_argument(
        "--manifests-json",
        type=Path,
        help="Optional JSON file that overrides the default Collective-Synthesizer manifest/external roots.",
    )
    return parser.parse_args()


def rel_to_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def effective_flops_per_ns(hw: HardwareSpec) -> float:
    return hw.peak_bf16_tflops * hw.utilization * 1000.0


def flop_time_ns(total_flops: float, hw: HardwareSpec) -> int:
    return max(1, int(round(total_flops / effective_flops_per_ns(hw))))


def norm_and_residual_flops(hidden_size: int, tokens: float) -> float:
    return 10.0 * hidden_size * tokens


def attention_linear_flops(hidden_size: int, tokens: float) -> float:
    return 8.0 * hidden_size * hidden_size * tokens


def attention_score_flops(
    hidden_size: int,
    num_attention_heads: int,
    tokens: float,
    context_length: float,
) -> float:
    return (
        4.0 * tokens * context_length * hidden_size
        + 5.0 * num_attention_heads * tokens * context_length
    )


def router_flops(hidden_size: int, num_routed_experts: int, tokens: float) -> float:
    return 2.0 * hidden_size * num_routed_experts * tokens


def expert_flops(hidden_size: int, intermediate_size: int, active_experts: int, tokens: float) -> float:
    return 6.0 * hidden_size * intermediate_size * active_experts * tokens


def attention_param_count(hidden_size: int) -> int:
    return 4 * hidden_size * hidden_size


def router_param_count(hidden_size: int, num_routed_experts: int) -> int:
    return hidden_size * num_routed_experts


def expert_param_count(hidden_size: int, intermediate_size: int) -> int:
    return 3 * hidden_size * intermediate_size


def output_activation_write_bytes(hidden_size: int, tokens: float) -> int:
    return int(round(tokens * hidden_size * BF16_BYTES))


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


def load_manifest_payloads(path_overrides: Path | None) -> dict[str, dict]:
    if path_overrides is None:
        manifest_map = {
            key: {
                "manifest": str(spec["manifest"]),
                "external_root": str(spec["external_root"]),
            }
            for key, spec in MANIFEST_DEFAULTS.items()
        }
    else:
        manifest_map = json.loads(path_overrides.read_text())

    payloads: dict[str, dict] = {}
    for scenario_key, spec in manifest_map.items():
        manifest_path = Path(spec["manifest"]).resolve()
        external_root = Path(spec["external_root"]).resolve()
        payload = json.loads(manifest_path.read_text())
        payload["_manifest_path"] = str(manifest_path)
        payload["_external_root"] = str(external_root)
        payloads[scenario_key] = payload
    return payloads


def build_phase_lookup(payload: dict) -> dict[str, dict[int, dict]]:
    lookup: dict[str, dict[int, dict]] = {"prefill": {}, "decode": {}}
    for case in payload["cases"]:
        lookup[str(case["phase"])][int(case["layer"])] = case
    return lookup


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
        network_files[hw_key] = rel_to_repo(path)
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
        f"external-collective-file: {rel_to_repo(external_csv)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(contents) + "\n")


def generate_workloads_for_scenario(
    scenario_key: str,
    payload: dict,
    network_files: dict[str, str],
) -> tuple[dict[str, str], dict[str, dict], list[dict]]:
    model = payload["model"]
    config = payload["config"]
    phase_lookup = build_phase_lookup(payload)
    hardware_keys = [TOPOLOGY_TO_HARDWARE[name] for name in payload["topologies"].keys()]

    attention_params = attention_param_count(int(model["hidden_size"]))
    router_params = router_param_count(
        int(model["hidden_size"]),
        int(model["num_routed_experts"]),
    )
    dense_mlp_params = expert_param_count(
        int(model["hidden_size"]),
        int(model["dense_intermediate_size"]),
    )
    moe_expert_params = expert_param_count(
        int(model["hidden_size"]),
        int(model["moe_intermediate_size"]),
    )
    kv_cache_bytes_per_token_layer = int(model.get("kv_cache_bytes_per_token_layer", 0))
    if kv_cache_bytes_per_token_layer == 0:
        if model["key"] == "deepseekv32":
            kv_cache_bytes_per_token_layer = 1152
        else:
            kv_cache_bytes_per_token_layer = 1024

    trace_context_by_layer = {
        layer: float(case["stats"]["avg_sequence_length"])
        for layer, case in phase_lookup["prefill"].items()
    }
    first_moe_layer = int(model["first_moe_layer"])
    first_context = trace_context_by_layer[min(trace_context_by_layer)]

    workload_files: dict[str, str] = {}
    memory_sanity: dict[str, dict] = {}
    scenarios: list[dict] = []

    dense_layers = int(model["dense_layers_without_moe"])
    moe_layers = int(model["total_layers"]) - dense_layers
    local_experts_per_rank = int(model["experts_per_rank"])
    shared_experts_per_layer = int(model["shared_experts_per_layer"])
    resident_params_total = (
        dense_layers * (attention_params + dense_mlp_params)
        + moe_layers
        * (
            attention_params
            + router_params
            + (local_experts_per_rank + shared_experts_per_layer) * moe_expert_params
        )
    )

    for hw_key in hardware_keys:
        hw = HARDWARE[hw_key]
        for phase in ("prefill", "decode"):
            lines = ["DISTRIBUTED_INFERENCE", str(int(model["total_layers"]) * 2)]
            layer_summaries = []
            phase_cases = phase_lookup[phase]
            first_phase_case = phase_cases[min(phase_cases)]

            for model_layer in range(1, int(model["total_layers"]) + 1):
                if model_layer < first_moe_layer:
                    trace_case = first_phase_case
                    tokens = float(trace_case["stats"]["avg_tokens_per_rank"])
                    context_length = first_context
                    if phase == "prefill":
                        sequence_length = context_length
                    else:
                        sequence_length = 1.0
                    dispatch_flops = (
                        attention_linear_flops(int(model["hidden_size"]), tokens)
                        + attention_score_flops(
                            int(model["hidden_size"]),
                            int(model["num_attention_heads"]),
                            tokens,
                            context_length if phase == "decode" else sequence_length,
                        )
                        + (norm_and_residual_flops(int(model["hidden_size"]), tokens) / 2.0)
                    )
                    combine_flops = (
                        expert_flops(
                            int(model["hidden_size"]),
                            int(model["dense_intermediate_size"]),
                            1,
                            tokens,
                        )
                        + (norm_and_residual_flops(int(model["hidden_size"]), tokens) / 2.0)
                    )
                    dispatch_compute_ns = flop_time_ns(dispatch_flops, hw)
                    combine_compute_ns = flop_time_ns(combine_flops, hw)
                    dispatch_mem_read = attention_params * BF16_BYTES
                    if phase == "decode":
                        dispatch_mem_read += int(round(tokens * context_length * kv_cache_bytes_per_token_layer))
                    dispatch_mem_write = int(round(tokens * kv_cache_bytes_per_token_layer))
                    combine_mem_read = dense_mlp_params * BF16_BYTES
                    combine_mem_write = output_activation_write_bytes(int(model["hidden_size"]), tokens)
                    dispatch_name = f"{model['plot_label'].replace('-', '').replace('.', '').replace(' ', '')}Dense{model_layer:02d}Dispatch"
                    combine_name = f"{model['plot_label'].replace('-', '').replace('.', '').replace(' ', '')}Dense{model_layer:02d}Combine"
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
                    layer_summaries.append(
                        {
                            "model_layer": model_layer,
                            "layer_type": "dense",
                            "tokens_per_rank": tokens,
                            "sequence_length": sequence_length,
                            "context_length": context_length,
                            "dispatch_comm_bytes": 0,
                            "combine_comm_bytes": 0,
                        }
                    )
                    continue

                case = phase_cases[model_layer]
                tokens = float(case["stats"]["avg_tokens_per_rank"])
                sequence_length = max(1.0, float(case["stats"]["avg_sequence_length"]))
                context_length = max(1.0, trace_context_by_layer[model_layer])
                dispatch_flops = (
                    attention_linear_flops(int(model["hidden_size"]), tokens)
                    + attention_score_flops(
                        int(model["hidden_size"]),
                        int(model["num_attention_heads"]),
                        tokens,
                        context_length if phase == "decode" else sequence_length,
                    )
                    + router_flops(
                        int(model["hidden_size"]),
                        int(model["num_routed_experts"]),
                        tokens,
                    )
                    + (norm_and_residual_flops(int(model["hidden_size"]), tokens) / 2.0)
                )
                combine_flops = (
                    expert_flops(
                        int(model["hidden_size"]),
                        int(model["moe_intermediate_size"]),
                        int(model["num_experts_per_tok"]),
                        tokens,
                    )
                    + expert_flops(
                        int(model["hidden_size"]),
                        int(model["moe_intermediate_size"]),
                        int(model["shared_experts_per_layer"]),
                        tokens,
                    )
                    + (norm_and_residual_flops(int(model["hidden_size"]), tokens) / 2.0)
                )
                dispatch_compute_ns = flop_time_ns(dispatch_flops, hw)
                combine_compute_ns = flop_time_ns(combine_flops, hw)
                comm_bytes = max(
                    int(model["dispatch_payload_bytes"]),
                    int(round(float(case["stats"]["avg_row_blocks"]) * int(model["dispatch_payload_bytes"]))),
                )
                dispatch_mem_read = (attention_params + router_params) * BF16_BYTES
                if phase == "decode":
                    dispatch_mem_read += int(round(tokens * context_length * kv_cache_bytes_per_token_layer))
                dispatch_mem_write = int(round(tokens * kv_cache_bytes_per_token_layer))
                combine_mem_read = (
                    (local_experts_per_rank + shared_experts_per_layer)
                    * moe_expert_params
                    * BF16_BYTES
                )
                combine_mem_write = output_activation_write_bytes(int(model["hidden_size"]), tokens)
                lines.append(
                    workload_line(
                        str(case["astra_dispatch_layer_name"]),
                        dispatch_compute_ns,
                        "ALLTOALLV",
                        comm_bytes,
                        dispatch_mem_read,
                        dispatch_mem_write,
                    )
                )
                lines.append(
                    workload_line(
                        str(case["astra_combine_layer_name"]),
                        combine_compute_ns,
                        "ALLTOALLV",
                        comm_bytes,
                        combine_mem_read,
                        combine_mem_write,
                    )
                )
                layer_summaries.append(
                    {
                        "model_layer": model_layer,
                        "layer_type": "moe",
                        "tokens_per_rank": tokens,
                        "sequence_length": sequence_length,
                        "context_length": context_length,
                        "dispatch_comm_bytes": comm_bytes,
                        "combine_comm_bytes": comm_bytes,
                    }
                )

            workload_path = WORKLOAD_ROOT / f"{scenario_key}_{hw_key}_{phase}_inference.txt"
            workload_path.parent.mkdir(parents=True, exist_ok=True)
            workload_path.write_text("\n".join(lines) + "\n")
            workload_files[f"{scenario_key}_{hw_key}_{phase}"] = rel_to_repo(workload_path)

            max_context = max(item["context_length"] for item in layer_summaries)
            local_batch = int(config["local_batch_sizes"][phase])
            kv_cache_gib = (
                local_batch
                * max_context
                * int(model["total_layers"])
                * kv_cache_bytes_per_token_layer
                / (1024.0**3)
            )
            resident_model_gib = resident_params_total * BF16_BYTES / (1024.0**3)
            memory_sanity[f"{scenario_key}_{hw_key}_{phase}"] = {
                "hardware_hbm_capacity_gib": hw.hbm_capacity_gib,
                "resident_model_gib_per_rank": round(resident_model_gib, 6),
                "kv_cache_gib_per_rank": round(kv_cache_gib, 6),
                "fits_hbm_capacity": (resident_model_gib + kv_cache_gib) < hw.hbm_capacity_gib,
                "layers": layer_summaries,
            }

            external_src_root = Path(payload["_external_root"])
            copied_root = EXTERNAL_ROOT / scenario_key
            copied_root.mkdir(parents=True, exist_ok=True)
            for method in METHODS_BY_HARDWARE[hw_key]:
                source_csv = external_src_root / f"{hw.topology_name}_{phase}_{method}.csv"
                if not source_csv.exists():
                    raise FileNotFoundError(f"missing external collective CSV: {source_csv}")
                copied_csv = copied_root / source_csv.name
                if source_csv.resolve() != copied_csv.resolve():
                    shutil.copy2(source_csv, copied_csv)

                system_path = SYSTEM_ROOT / f"{scenario_key}_{hw_key}_{phase}_{method}.txt"
                write_system_file(system_path, hw, copied_csv)
                scenarios.append(
                    {
                        "run_name": f"{scenario_key}_{hw_key}_{phase}_{method}",
                        "model_key": str(model["key"]),
                        "model_label": str(model["plot_label"]),
                        "config_key": str(config["key"]),
                        "config_label": str(config["label"]),
                        "hardware": hw_key,
                        "hardware_label": hw.label,
                        "topology_label": hw.topology_label,
                        "phase": phase,
                        "method": method,
                        "network": network_files[hw_key],
                        "system": rel_to_repo(system_path),
                        "workload": rel_to_repo(workload_path),
                        "units_count": list(hw.units_count),
                        "num_queues_per_dim": [1] * len(hw.units_count),
                    }
                )

    return workload_files, memory_sanity, scenarios


def main() -> None:
    args = parse_args()
    payloads = load_manifest_payloads(args.manifests_json)
    network_files = generate_networks()

    all_workloads: dict[str, str] = {}
    all_memory: dict[str, dict] = {}
    all_scenarios: list[dict] = []
    selection: dict[str, dict] = {}

    for scenario_key, payload in payloads.items():
        workload_files, memory_sanity, scenarios = generate_workloads_for_scenario(
            scenario_key,
            payload,
            network_files,
        )
        all_workloads.update(workload_files)
        all_memory.update(memory_sanity)
        all_scenarios.extend(scenarios)
        selection[scenario_key] = {
            "manifest_path": payload["_manifest_path"],
            "external_root": payload["_external_root"],
            "model": payload["model"],
            "config": payload["config"],
        }

    metadata = {
        "selection": selection,
        "hardware": {
            hw_key: {
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
            for hw_key, hw in HARDWARE.items()
        },
        "memory_sanity": all_memory,
        "generated_files": {
            "workloads": all_workloads,
            "networks": network_files,
        },
    }
    metadata_path = METADATA_ROOT / "realmodel_e2e4_selected.json"
    write_json(metadata_path, metadata)

    experiments = {
        "experiment": "realmodel_e2e4",
        "scenarios": all_scenarios,
        "summary_metadata": rel_to_repo(metadata_path),
    }
    experiments_path = METADATA_ROOT / "realmodel_e2e4_experiments.json"
    write_json(experiments_path, experiments)

    print(f"Wrote {metadata_path}")
    print(f"Wrote {experiments_path}")


if __name__ == "__main__":
    main()
