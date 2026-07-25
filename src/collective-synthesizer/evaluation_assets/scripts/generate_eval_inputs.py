#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TOKEN_BYTES = 4 * 1024
TARGET_SAMPLES = [0, 1, 2, 3]
TARGET_SIZE_BYTES = {
    "1MB": 1 * 1024 * 1024,
    "8MB": 8 * 1024 * 1024,
    "64MB": 64 * 1024 * 1024,
    "256MB": 256 * 1024 * 1024,
    "2GB": 2 * 1024 * 1024 * 1024,
}
SYNTHETIC_CASE_SPECS = {
    64: {
        "1MB": {"raw_bs": 32, "merge_pairs": False, "scale": 1},
        "8MB": {"raw_bs": 256, "merge_pairs": False, "scale": 1},
        "64MB": {"raw_bs": 512, "merge_pairs": False, "scale": 4},
        "256MB": {"raw_bs": 512, "merge_pairs": False, "scale": 16},
        "2GB": {"raw_bs": 512, "merge_pairs": False, "scale": 128},
    },
    32: {
        "1MB": {"raw_bs": 16, "merge_pairs": True, "scale": 1},
        "8MB": {"raw_bs": 128, "merge_pairs": True, "scale": 1},
        "64MB": {"raw_bs": 512, "merge_pairs": True, "scale": 2},
        "256MB": {"raw_bs": 512, "merge_pairs": True, "scale": 8},
        "2GB": {"raw_bs": 512, "merge_pairs": True, "scale": 64},
    },
}
SYNTHETIC_TOPOLOGIES = {
    "torus_tpuv4_4x4x4": {
        "topology": "torus",
        "dimension": 3,
        "shape": [4, 4, 4],
        "bandwidth": [56, 56, 56],
        "latency": [500, 500, 500],
    },
    "mesh_nebula_8x4": {
        "topology": "mesh",
        "dimension": 2,
        "shape": [8, 4],
        "bandwidth": [50, 50],
        "latency": [500, 500],
    },
    "cm384_16x2_eval": {
        "topology": "cm384",
        "dimension": 2,
        "shape": [16, 2],
        "switch-shape": [1, 1],
        "link-count": [1, 1],
        "direct-bandwidth": 270,
        "direct-latency": 200,
        "bandwidth": [196, 196],
        "latency": [200, 200],
    },
    "rail_optimized_8x4_eval": {
        "topology": "rail-optimized",
        "dimension": 2,
        "shape": [8, 4],
        "switch-dimension": 2,
        "switch-shape": [1, 8],
        "link-count": [1, 1],
        "bandwidth": [300, 25],
        "latency": [500, 500],
    },
    "fattree_8x4_eval": {
        "topology": "fat-tree",
        "dimension": 2,
        "shape": [8, 4],
        "switch-dimension": 2,
        "switch-shape": [1, 1],
        "link-count": [1, 1],
        "bandwidth": [300, 25],
        "latency": [500, 500],
    },
}
SCALABILITY_CASE_SPECS = {
    "mesh": [
        {"point": 8, "shape": [8, 8]},
        {"point": 12, "shape": [12, 12]},
        {"point": 16, "shape": [16, 16]},
        {"point": 24, "shape": [24, 24]},
        {"point": 32, "shape": [32, 32]},
        {"point": 48, "shape": [48, 48]},
        {"point": 64, "shape": [64, 64]},
        {"point": 80, "shape": [80, 80]},
        {"point": 96, "shape": [96, 96]},
    ],
    "torus": [
        {"point": "4x4x4", "shape": [4, 4, 4]},
        {"point": "6x6x6", "shape": [6, 6, 6]},
        {"point": "8x8x8", "shape": [8, 8, 8]},
        {"point": "10x10x10", "shape": [10, 10, 10]},
        {"point": "12x12x12", "shape": [12, 12, 12]},
        {"point": "16x16x16", "shape": [16, 16, 16]},
        {"point": "20x20x20", "shape": [20, 20, 20]},
    ],
    "fullmesh": [
        {"point": 4, "shape": [8, 4]},
        {"point": 8, "shape": [8, 8]},
        {"point": 16, "shape": [8, 16]},
        {"point": 32, "shape": [8, 32]},
        {"point": 64, "shape": [8, 64]},
        {"point": 128, "shape": [8, 128]},
        {"point": 256, "shape": [8, 256]},
        {"point": 512, "shape": [8, 512]},
        {"point": 1024, "shape": [8, 1024]},
    ],
    "fat-tree": [
        {"point": 4, "shape": [8, 4]},
        {"point": 8, "shape": [8, 8]},
        {"point": 16, "shape": [8, 16]},
        {"point": 32, "shape": [8, 32]},
        {"point": 64, "shape": [8, 64]},
        {"point": 128, "shape": [8, 128]},
        {"point": 256, "shape": [8, 256]},
        {"point": 512, "shape": [8, 512]},
        {"point": 1024, "shape": [8, 1024]},
    ],
    "cm384": [
        {"point": 4, "shape": [16, 4]},
        {"point": 8, "shape": [16, 8]},
        {"point": 16, "shape": [16, 16]},
        {"point": 32, "shape": [16, 32]},
        {"point": 64, "shape": [16, 64]},
        {"point": 128, "shape": [16, 128]},
        {"point": 256, "shape": [16, 256]},
        {"point": 512, "shape": [16, 512]},
    ],
}
SCALABILITY_WORKLOAD_SOURCE_CSV = "input/Decode_BS64_Layer8_56.csv"
SCALABILITY_WORKLOAD_TAG = "decode_bs64_layer8_56_tiled"
SCALABILITY_BLOCK_BYTES = 4 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation inputs for Collective-Synthesizer.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def load_token_matrix(path: Path) -> list[list[int]]:
    with path.open(newline="") as handle:
        return [[int(cell) for cell in row] for row in csv.reader(handle) if row]


def merge_adjacent_pairs(matrix: list[list[int]]) -> list[list[int]]:
    if len(matrix) % 2 != 0 or len(matrix[0]) % 2 != 0:
        raise ValueError("merge_adjacent_pairs expects even matrix dimensions")
    merged: list[list[int]] = []
    for row in range(0, len(matrix), 2):
        merged_row: list[int] = []
        for col in range(0, len(matrix[row]), 2):
            merged_row.append(
                matrix[row][col]
                + matrix[row][col + 1]
                + matrix[row + 1][col]
                + matrix[row + 1][col + 1]
            )
        merged.append(merged_row)
    return merged


def scale_matrix(matrix: list[list[int]], scale: int) -> list[list[int]]:
    return [[value * scale for value in row] for row in matrix]


def write_csv(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def relative_repo_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def collect_matrix_stats(matrix: list[list[int]]) -> dict[str, object]:
    row_sums_tokens = [sum(row) for row in matrix]
    total_tokens = sum(row_sums_tokens)
    row_sums_bytes = [value * TOKEN_BYTES for value in row_sums_tokens]
    return {
        "matrix_rows": len(matrix),
        "matrix_cols": len(matrix[0]) if matrix else 0,
        "row_sums_tokens": row_sums_tokens,
        "row_sums_bytes": row_sums_bytes,
        "row_tokens_min": min(row_sums_tokens),
        "row_tokens_max": max(row_sums_tokens),
        "row_bytes_min": min(row_sums_bytes),
        "row_bytes_max": max(row_sums_bytes),
        "avg_row_tokens": total_tokens / len(row_sums_tokens),
        "avg_row_bytes": (total_tokens * TOKEN_BYTES) / len(row_sums_tokens),
        "total_tokens": total_tokens,
        "total_bytes": total_tokens * TOKEN_BYTES,
    }


def generate_synthetic_topologies(repo_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for topo_key, payload in SYNTHETIC_TOPOLOGIES.items():
        path = repo_root / "evaluation_assets" / "topologies" / "synthetic" / f"{topo_key}.json"
        write_json(path, payload)
        records.append(
            {
                "topology_key": topo_key,
                "topology_type": payload["topology"],
                "shape": payload["shape"],
                "target_devices": math.prod(int(v) for v in payload["shape"]),
                "topology_json": relative_repo_path(path, repo_root),
            }
        )
    manifest_path = repo_root / "evaluation_assets" / "manifests" / "synthetic_topologies.json"
    write_json(manifest_path, {"topologies": records})
    return records


def generate_synthetic_collectives(repo_root: Path) -> list[dict[str, object]]:
    raw_root = repo_root / "input" / "raw_data" / "olmoe_inf" / "OLMoE_Inference"
    records: list[dict[str, object]] = []

    for target_devices, size_specs in SYNTHETIC_CASE_SPECS.items():
        for size_label, spec in size_specs.items():
            for sample_idx in TARGET_SAMPLES:
                source_path = raw_root / f"Decode_BS{spec['raw_bs']}_Layer1_{sample_idx}.csv"
                matrix = load_token_matrix(source_path)
                transform_steps = ["layer1"]
                if spec["merge_pairs"]:
                    matrix = merge_adjacent_pairs(matrix)
                    transform_steps.append("merge_adjacent_pairs")
                matrix = scale_matrix(matrix, spec["scale"])
                if spec["scale"] != 1:
                    transform_steps.append(f"scale_x{spec['scale']}")

                case_id = f"layer1_group{sample_idx}_{target_devices}devices_{size_label}"
                csv_path = (
                    repo_root
                    / "evaluation_assets"
                    / "csv"
                    / "synthetic"
                    / f"{target_devices}devices"
                    / f"{case_id}.csv"
                )
                collective_path = (
                    repo_root
                    / "evaluation_assets"
                    / "collectives"
                    / "synthetic"
                    / f"{target_devices}devices"
                    / f"{case_id}.json"
                )
                write_csv(csv_path, matrix)
                write_json(
                    collective_path,
                    {
                        "collective": "alltoallv",
                        "v_datasize": relative_repo_path(csv_path, repo_root),
                        "chunkfactor": 1,
                    },
                )

                stats = collect_matrix_stats(matrix)
                records.append(
                    {
                        "case_id": case_id,
                        "sample_index": sample_idx,
                        "layer": 1,
                        "target_devices": target_devices,
                        "size_label": size_label,
                        "nominal_bytes_per_rank": TARGET_SIZE_BYTES[size_label],
                        "raw_batch_size": spec["raw_bs"],
                        "merge_adjacent_pairs": spec["merge_pairs"],
                        "scale_factor": spec["scale"],
                        "transform_steps": transform_steps,
                        "source_csv": relative_repo_path(source_path, repo_root),
                        "generated_csv": relative_repo_path(csv_path, repo_root),
                        "collective_json": relative_repo_path(collective_path, repo_root),
                        **stats,
                    }
                )

    manifest_path = repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.json"
    write_json(
        manifest_path,
        {
            "token_bytes": TOKEN_BYTES,
            "size_labels": TARGET_SIZE_BYTES,
            "cases": records,
        },
    )

    summary_csv = repo_root / "evaluation_assets" / "manifests" / "synthetic_cases.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "sample_index",
        "layer",
        "target_devices",
        "size_label",
        "nominal_bytes_per_rank",
        "raw_batch_size",
        "merge_adjacent_pairs",
        "scale_factor",
        "source_csv",
        "generated_csv",
        "collective_json",
        "matrix_rows",
        "matrix_cols",
        "avg_row_bytes",
        "row_bytes_min",
        "row_bytes_max",
        "total_bytes",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row[field] for field in fieldnames})

    return records


def write_scalability_topology(
    repo_root: Path, topology_type: str, case_spec: dict[str, object]
) -> tuple[str, str, str, int]:
    point = case_spec["point"]
    shape = [int(v) for v in case_spec["shape"]]
    if topology_type == "mesh":
        payload = {
            "topology": "mesh",
            "dimension": 2,
            "shape": shape,
            "bandwidth": [50, 50],
            "latency": [500, 500],
        }
        topo_key = f"mesh_{shape[0]}x{shape[1]}"
    elif topology_type == "torus":
        payload = {
            "topology": "torus",
            "dimension": 3,
            "shape": shape,
            "bandwidth": [56, 56, 56],
            "latency": [500, 500, 500],
        }
        topo_key = f"torus_{shape[0]}x{shape[1]}x{shape[2]}"
    elif topology_type == "fullmesh":
        payload = {
            "topology": "fullmesh",
            "dimension": 2,
            "shape": shape,
            "bandwidth": [300, 300],
            "latency": [500, 500],
        }
        topo_key = f"fullmesh_{shape[0]}x{shape[1]}"
    elif topology_type == "fat-tree":
        payload = {
            "topology": "fat-tree",
            "dimension": 2,
            "shape": shape,
            "switch-dimension": 2,
            "switch-shape": [1, 1],
            "link-count": [1, 1],
            "bandwidth": [300, 25],
            "latency": [500, 500],
        }
        topo_key = f"fattree_{shape[0]}x{shape[1]}"
    elif topology_type == "cm384":
        payload = {
            "topology": "cm384",
            "dimension": 2,
            "shape": shape,
            "switch-shape": [1, 1],
            "link-count": [1, 1],
            "direct-bandwidth": 270,
            "direct-latency": 200,
            "bandwidth": [196, 196],
            "latency": [200, 200],
        }
        topo_key = f"cm384_{shape[0]}x{shape[1]}"
    else:
        raise ValueError(f"Unsupported scalability topology: {topology_type}")

    topology_path = repo_root / "evaluation_assets" / "topologies" / "scalability" / f"{topo_key}.json"
    write_json(topology_path, payload)

    collective_path = (
        repo_root
        / "evaluation_assets"
        / "collectives"
        / "scalability"
        / f"{topo_key}_{SCALABILITY_WORKLOAD_TAG}.json"
    )
    write_json(
        collective_path,
        {
            "collective": "alltoallv",
            "block_bytes": SCALABILITY_BLOCK_BYTES,
            "synthetic_v_datasize": {
                "pattern": "tiled-csv",
                "base_csv": SCALABILITY_WORKLOAD_SOURCE_CSV,
            },
        },
    )
    log_stem = f"{topo_key}_{SCALABILITY_WORKLOAD_TAG}"

    return (
        relative_repo_path(topology_path, repo_root),
        relative_repo_path(collective_path, repo_root),
        log_stem,
        int(shape[0] * shape[1] * (shape[2] if len(shape) == 3 else 1)),
    )


def generate_scalability_cases(repo_root: Path) -> list[dict[str, object]]:
    source_matrix = load_token_matrix(repo_root / SCALABILITY_WORKLOAD_SOURCE_CSV)
    source_rows = len(source_matrix)
    source_cols = len(source_matrix[0]) if source_matrix else 0
    source_nonzeros = sum(1 for row in source_matrix for value in row if value > 0)
    source_total = source_rows * source_cols
    source_sparsity = 1.0 - (source_nonzeros / source_total if source_total else 0.0)

    records: list[dict[str, object]] = []
    for topology_type, case_specs in SCALABILITY_CASE_SPECS.items():
        for case_spec in case_specs:
            topology_json, collective_json, log_stem, npus_count = write_scalability_topology(
                repo_root, topology_type, case_spec
            )
            records.append(
                {
                    "case_id": log_stem,
                    "topology_type": topology_type,
                    "point": case_spec["point"],
                    "npus_count": npus_count,
                    "topology_json": topology_json,
                    "collective_json": collective_json,
                    "log_stem": log_stem,
                    "workload_mode": "synthetic_v_datasize.tiled-csv",
                    "workload_tag": SCALABILITY_WORKLOAD_TAG,
                    "workload_source_csv": SCALABILITY_WORKLOAD_SOURCE_CSV,
                    "block_bytes": SCALABILITY_BLOCK_BYTES,
                    "source_matrix_rows": source_rows,
                    "source_matrix_cols": source_cols,
                    "source_matrix_nonzeros": source_nonzeros,
                    "source_matrix_sparsity": source_sparsity,
                }
            )

    manifest_path = repo_root / "evaluation_assets" / "manifests" / "scalability_cases.json"
    write_json(
        manifest_path,
        {
            "workload_mode": "synthetic_v_datasize.tiled-csv",
            "workload_tag": SCALABILITY_WORKLOAD_TAG,
            "workload_source_csv": SCALABILITY_WORKLOAD_SOURCE_CSV,
            "block_bytes": SCALABILITY_BLOCK_BYTES,
            "source_matrix_rows": source_rows,
            "source_matrix_cols": source_cols,
            "source_matrix_nonzeros": source_nonzeros,
            "source_matrix_sparsity": source_sparsity,
            "cases": records,
        },
    )
    return records


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    generate_synthetic_topologies(repo_root)
    synthetic_cases = generate_synthetic_collectives(repo_root)
    scalability_cases = generate_scalability_cases(repo_root)

    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "synthetic_case_count": len(synthetic_cases),
                "scalability_case_count": len(scalability_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
