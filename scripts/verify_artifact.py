#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REQUIRED = [
    "scripts/setup_environment.sh",
    "scripts/run_simulation_direct.sh",
    "scripts/run_cpu_artifact_direct.sh",
    "scripts/collect_target_figures.py",
    "scripts/run_end2end_direct.sh",
    "scripts/prepare_end2end_external_inputs.py",
    "src/collective-synthesizer/utils/run_olmoe64_selected_batch.py",
    "src/collective-synthesizer/input/official_data",
    "src/collective-synthesizer/input/raw_data/olmoe_inf",
    "src/collective-synthesizer/evaluation_assets/manifests",
    "src/astra-sim-galois/scripts/olmoe_inference/run_olmoe_inference_analytical.py",
    "src/astra-sim-galois/scripts/olmoe_inference/generate_realmodel_e2e4_assets.py",
    "src/astra-sim-galois/Pictures/RealModel_Alltoallv_EndToEnd.py",
    "src/astra-sim-galois/inputs/paper_reference/RealModelE2E4",
]
FORBIDDEN_SOURCE_NAMES = {
    "summary.json",
    "summary.csv",
    "samples.csv",
    "benchmark.log",
    "EndToEnd.csv",
}
FORBIDDEN_SOURCE_SUFFIXES = {
    ".pdf",
    ".png",
    ".log",
}
RUN_REQUIRED_BY_PHASE = {
    "target": ["figures/target_manifest.json"],
}


def assert_exists(path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.stat().st_size == 0:
        raise RuntimeError("empty file: {}".format(path))


def is_ignored_source_path(path):
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & {".git", ".venv", "__pycache__", "runs", "figures"}:
        return True
    rel_text = rel.as_posix()
    generated_build_roots = (
        "src/collective-synthesizer/build/",
        "src/astra-sim-galois/build/astra_analytical/build/",
        "src/astra-sim-galois/build/astra_analytical/result/",
    )
    if rel_text.startswith(generated_build_roots):
        return True
    # nlohmann_json is vendored as a complete upstream source tree.  Its
    # documentation and fuzz-test images are not Glaive result artifacts.
    if "nlohmann_json" in parts and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}:
        return True
    return False


def check_source_tree():
    for rel in SOURCE_REQUIRED:
        assert_exists(ROOT / rel)

    paper_reference = (
        ROOT
        / "src/astra-sim-galois/inputs/paper_reference/RealModelE2E4"
    )
    scenarios = (
        "deepseekv32_small",
        "deepseekv32_large",
        "qwen3_30b_a3b_small",
        "qwen3_30b_a3b_large",
    )
    reference_count = 0
    for scenario in scenarios:
        scenario_files = sorted((paper_reference / scenario).glob("*.csv"))
        if len(scenario_files) != 12:
            raise RuntimeError(
                "expected 12 Figure 12 paper-reference inputs for {}, found {}".format(
                    scenario, len(scenario_files)
                )
            )
        reference_count += len(scenario_files)
    if reference_count != 48:
        raise RuntimeError(
            "expected 48 Figure 12 paper-reference inputs, found {}".format(
                reference_count
            )
        )

    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or is_ignored_source_path(path):
            continue
        if path.name == ".DS_Store" or path.name.startswith("core."):
            offenders.append(path)
            continue
        if ".bak" in path.name or path.suffix in {".pkl", ".npy", ".lp"}:
            offenders.append(path)
            continue
        if path.name in FORBIDDEN_SOURCE_NAMES:
            offenders.append(path)
            continue
        if path.name.startswith("submit_") and path.suffix == ".json":
            offenders.append(path)
            continue
        if path.name.endswith("dimension_utilization.csv"):
            offenders.append(path)
            continue
        if path.suffix in FORBIDDEN_SOURCE_SUFFIXES:
            offenders.append(path)
    if offenders:
        formatted = "\n".join("  {}".format(path.relative_to(ROOT)) for path in offenders[:50])
        raise RuntimeError("pre-generated result/image files found:\n{}".format(formatted))


def check_packaging_cleanliness():
    offenders = []
    for directory_name in (".venv", "runs"):
        directory = ROOT / directory_name
        if directory.exists() and any(path.is_file() for path in directory.rglob("*")):
            offenders.append(directory_name)
    if offenders:
        raise RuntimeError(
            "generated directories must be removed before packaging: {}".format(
                ", ".join(offenders)
            )
        )


def check_run_dir(run_dir, phases):
    for phase in phases:
        if phase not in RUN_REQUIRED_BY_PHASE:
            raise ValueError("unknown phase: {}".format(phase))
        for rel in RUN_REQUIRED_BY_PHASE[phase]:
            assert_exists(run_dir / rel)
    manifest_path = run_dir / "figures" / "target_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete", False):
            raise RuntimeError("target_manifest.json reports missing target figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, help="run root produced by run_all.sh")
    parser.add_argument(
        "--phases",
        default="target",
        help="comma-separated phases to verify under --run-dir",
    )
    parser.add_argument(
        "--check-packaging-clean",
        action="store_true",
        help="also require .venv/ and runs/ to contain no generated files",
    )
    args = parser.parse_args()

    check_source_tree()
    if args.check_packaging_clean:
        check_packaging_cleanliness()
    if args.run_dir:
        phases = [item.strip() for item in args.phases.split(",") if item.strip()]
        check_run_dir(args.run_dir.resolve(), phases)
        print("Artifact source and run outputs verified: {}".format(args.run_dir))
    else:
        print(
            "Artifact source tree verified: required inputs are present; "
            "generated environment and run directories were ignored."
        )


if __name__ == "__main__":
    main()
