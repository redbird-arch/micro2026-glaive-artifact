# Target Figure Mapping

The top-level `run_all.sh` reproduces Figures 6--12 and 16. It does not invoke
the paper's multi-GPU workflows for Figures 13--15.

All stable outputs are collected below
`runs/ae_<timestamp>/figures/`.

| Paper figure | Stable output | Generated source relative to the run directory | Entry point |
|---|---|---|---|
| 6 | `Figure06_Synthetic_Experiment.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Synthetic_Experiment.pdf` | `scripts/run_simulation_direct.sh` |
| 7 | `Figure07_Link_Utilization.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Link_Utilization.pdf` | `scripts/run_simulation_direct.sh` |
| 8 | `Figure08_Scalability_Study.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Scalability_Study.pdf` | `scripts/run_simulation_direct.sh` |
| 9(a) | `Figure09a_BPI_Distribution.pdf` | `simulation/evaluation_assets/trace_studies/plots/reference_variant_bpi_distribution.pdf` | `scripts/run_simulation_direct.sh` |
| 9(b) | `Figure09b_Speedup_vs_BPI.pdf` | `simulation/evaluation_assets/trace_studies/plots/speedup_vs_bpi.pdf` | `scripts/run_simulation_direct.sh` |
| 10(a) | `Figure10a_Hot_Flow_Cap.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Sensitivity_Breakdown.pdf` | `scripts/run_simulation_direct.sh` |
| 10(b) | `Figure10b_Solver_Time_Sensitivity.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Sensitivity_Solver_Time.pdf` | `scripts/run_simulation_direct.sh` |
| 10(c) | `Figure10c_Path_Score_Weights.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Weight_Sensitivity_Breakdown.pdf` | `scripts/run_simulation_direct.sh` |
| 11(a) | `Figure11a_Torus_JLU.pdf` | `simulation/evaluation_assets/trace_studies/plots/torus_jain_adjusted_link_utilization.pdf` | `scripts/run_simulation_direct.sh` |
| 11(b) | `Figure11b_Torus_Sweep_Breakdown.pdf` | `simulation/evaluation_assets/plots/Speed_Torus_Breakdown.pdf` | `scripts/run_simulation_direct.sh` |
| 12 | `Figure12_EndToEnd.pdf` | `end2end/figures/Figure12_Re_RealModel_Alltoallv_EndToEnd.pdf` | `scripts/run_end2end_direct.sh` |
| 16(a) | `Figure16a_1Node_Overhead.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Breakeven_8GPU.pdf` | `scripts/run_simulation_direct.sh` |
| 16(b) | `Figure16b_2Node_Overhead.pdf` | `simulation/evaluation_assets/collective-synthesizer/evaluation_assets/plots/Speed_Breakeven_16GPU.pdf` | `scripts/run_simulation_direct.sh` |

Figures 9--11 and 16 contain multiple paper panels, so the collector retains
each panel as a separate PDF and records it independently in
`target_manifest.json`.

`Scalability_Study_Standard.pdf` is an auxiliary scalability plot and is not
collected as Figure 8.
