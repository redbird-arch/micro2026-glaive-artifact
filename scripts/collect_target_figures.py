#!/usr/bin/env python3
"""Collect generated target panels into one stable, paper-numbered directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TARGETS = {
    "Figure06_Synthetic_Experiment": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Synthetic_Experiment.pdf",
    "Figure07_Link_Utilization": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Link_Utilization.pdf",
    "Figure08_Scalability_Study": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Scalability_Study.pdf",
    "Figure09a_BPI_Distribution": "simulation/simulation_exp/plots/reference_variant_bpi_distribution.pdf",
    "Figure09b_Speedup_vs_BPI": "simulation/simulation_exp/plots/speedup_vs_bpi.pdf",
    "Figure10a_Hot_Flow_Cap": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Sensitivity_Breakdown.pdf",
    "Figure10b_Solver_Time_Sensitivity": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Sensitivity_Solver_Time.pdf",
    "Figure10c_Path_Score_Weights": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Weight_Sensitivity_Breakdown.pdf",
    "Figure11a_Torus_JLU": "simulation/simulation_exp/plots/torus_jain_adjusted_link_utilization.pdf",
    "Figure11b_Torus_Sweep_Breakdown": "simulation/evaluation_assets/plots/Speed_Torus_Breakdown.pdf",
    "Figure12_EndToEnd": "end2end/figures/Figure12_Re_RealModel_Alltoallv_EndToEnd.pdf",
    "Figure16a_1Node_Overhead": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Breakeven_8GPU.pdf",
    "Figure16b_2Node_Overhead": "simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Breakeven_16GPU.pdf",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    missing = []
    for name, relative in TARGETS.items():
        source = run_dir / relative
        record = {"figure": name, "source": relative, "status": "missing"}
        if source.is_file():
            destination = output_dir / f"{name}{source.suffix}"
            shutil.copy2(source, destination)
            try:
                output_path = destination.relative_to(run_dir)
            except ValueError:
                output_path = destination
            record.update({"status": "ok", "output": str(output_path)})
        else:
            missing.append(name)
        manifest.append(record)

    report = {
        "artifact": "Glaive AE",
        "run_dir": str(run_dir),
        "targets": manifest,
        "missing": missing,
        "complete": not missing,
    }
    (output_dir / "target_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit("Missing target figures: " + ", ".join(missing))


if __name__ == "__main__":
    main()
