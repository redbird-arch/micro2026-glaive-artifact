# Glaive: Cleaving Bandwidth and Latency Scheduling for Efficient Non-uniform All-to-All Collective Communication

## Overview

This repository contains the source code, input traces, topology and model
configurations, experiment drivers, parsers, and plotting scripts required to
reproduce Figures 6--12 and 16 of the Glaive paper.

The artifact is available from:

- GitHub: `git@github.com:redbird-arch/micro2026-glaive-artifact.git`
- Zenodo: [https://doi.org/10.5281/zenodo.21547687](https://doi.org/10.5281/zenodo.21547687)

No completed experiment logs or final paper figures are included. Each run
creates a new directory under `runs/` and produces every target figure as a
PDF.

## Implemented Methods

- **Glaive standard** is the complete topology-aware synthesis path used for
  the main performance and scalability experiments.
- **Glaive Speed** separates hot and regular traffic, applies Thrust path
  selection to hot flows, schedules the remaining traffic with Sweep, and
  exposes the event-level breakdown used by Figures 10 and 11(b).
- **MPICH-style baseline** implements pairwise-exchange or XOR-round
  scheduling according to the topology and collective size.
- **BiRing** implements bidirectional ring scheduling for Mesh, Clos,
  FullMesh, and CM384 cases.
- **HalfR+DR** implements HalfRing with dimension rotation for torus cases.
- **ASTRA-Sim analytical backend** evaluates communication, memory, and
  computation timing for the four Figure 12 model and hardware scenarios.

## Directory Structure

```text
Glaive_AE/
├── run_all.sh
│   └── one-command setup, execution, collection, and validation
├── requirements.txt
│   └── pinned CMake and Python analysis dependencies
├── scripts/
│   ├── setup_environment.sh
│   │   └── native-tool checks and isolated Python environment setup
│   ├── build_all.sh
│   │   └── Glaive and ASTRA-Sim build workflow
│   ├── run_cpu_artifact_direct.sh
│   │   └── top-level experiment workflow
│   ├── run_simulation_direct.sh
│   │   └── Figures 6--11 and 16 workflow
│   ├── run_evaluation_assets_direct.sh
│   │   └── synthetic, scalability, sensitivity, and overhead experiments
│   ├── run_end2end_solver_direct.sh
│   │   └── collective generation for the four Figure 12 scenarios
│   ├── run_end2end_direct.sh
│   │   └── Figure 12 solver, ASTRA-Sim, and plotting workflow
│   ├── prepare_end2end_external_inputs.py
│   │   └── preparation of the Figure 12 simulator inputs
│   ├── collect_target_figures.py
│   │   └── paper-numbered PDF collector and completion manifest
│   └── verify_artifact.py
│       └── source-tree and generated-output validation
├── src/
│   ├── collective-synthesizer/
│   │   ├── README.md                  component overview and TACOS provenance
│   │   ├── src/
│   │   │   ├── synthesizer/           profiling, scheduling, and fusion
│   │   │   ├── synthesizer_standard/  standard and Speed scheduling paths
│   │   │   ├── baselines/             MPICH, BiRing, HalfR+DR, and helpers
│   │   │   ├── topology/              Mesh, Torus, Clos, CM384, FullMesh
│   │   │   ├── collective/            collective representations
│   │   │   └── event_queue/           event and makespan accounting
│   │   ├── include/tacos/              public C++ interfaces
│   │   ├── input/
│   │   │   ├── official_data/         MoE routing traces
│   │   │   ├── raw_data/olmoe_inf/    OLMoE inference traces
│   │   │   └── generated/realmodel_e2e4/
│   │   │       └── Figure 12 scenario manifests
│   │   ├── evaluation_assets/
│   │   │   ├── collectives/           checked-in collective inputs
│   │   │   ├── topologies/            evaluation topology descriptions
│   │   │   ├── manifests/             experiment definitions
│   │   │   ├── scripts/               experiment runners, parsers, and plots
│   │   │   └── parsed/figure9_h100_glaive_reference.csv
│   │   │       └── measured timing input used by Figure 16
│   │   └── libs/                       bundled C++ dependencies
│   └── astra-sim-galois/
│       ├── astra-sim/                  ASTRA-Sim core
│       ├── build/astra_analytical/     analytical backend build scripts
│       ├── scripts/olmoe_inference/    Figure 12 generator and runner
│       ├── Pictures/                   Figure 12 plotting script
│       └── inputs/paper_reference/     Figure 12 collective-schedule inputs
└── docs/
    ├── figure_mapping.md               exact figure-to-output mapping
    └── target_reproducibility.md       execution and validation details
```

### Collective-synthesizer implementation

`src/collective-synthesizer/src/main.cpp` provides the command-line entry
point used by every solver experiment. The implementation is divided as
follows:

- `synthesizer/` implements demand profiling, latency/bandwidth scheduling,
  schedule fusion, and event emission for the `complete` and `clean` modes.
- `synthesizer_standard/` implements the `standard` path and the Speed
  hot-flow/regular-flow decomposition used by Figures 6--11 and 16.
- `baselines/` implements Bruck, Spreadout, pairwise exchange, BiRing,
  HalfR+DR, and the analytical baseline variants used in the comparisons.
- `topology/`, `collective/`, and `event_queue/` provide the network models,
  collective demand representations, and timing primitives shared by the
  schedulers.

The `evaluation_assets/scripts/` directory also contains the controlled trace
studies used by Figure 9 and the torus post-processing used by Figure 11.

## Hardware Requirements

The complete workflow requires one x86-64 CPU machine with:

- at least 8 CPU cores;
- at least 64 GiB of memory;
- at least 20 GiB of free storage.

More CPU cores reduce experiment time. No GPU is required to execute the
artifact.

## Software Requirements

The reference environment is Ubuntu 22.04 LTS on x86-64 hardware with:

- glibc 2.17 or newer;
- GCC and G++ 12 or newer;
- GNU Make;
- Python 3.10 or 3.11 with `venv`;
- Bash and `rsync`.

CMake 3.27.9, Matplotlib 3.7.0, NumPy 1.23.5, Pandas 1.5.3, and Pillow 9.5.0
are installed from `requirements.txt` into `.venv/`. The C++ libraries
required by Glaive and the ASTRA-Sim source tree are included in the repository.

For Ubuntu 22.04, the native prerequisites can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y gcc-12 g++-12 make python3.10 python3.10-venv rsync
export CC=gcc-12
export CXX=g++-12
```

Create and verify the isolated environment with:

```bash
bash scripts/setup_environment.sh
```

## One-Command Reproduction

From the repository root:

```bash
bash run_all.sh
```

The default is eight local workers. Change the process-level parallelism when
needed:

```bash
GLAIVE_WORKERS=16 bash run_all.sh
```

The command performs the following steps:

1. verifies the native compiler, glibc, Python, and storage-independent
   software requirements;
2. creates `.venv/` and installs all pinned analysis dependencies;
3. builds Glaive and the ASTRA-Sim analytical backend;
4. runs the experiments for Figures 6--12 and 16 using local worker pools;
5. parses the generated logs and renders PDF figures;
6. collects the 13 paper panels and writes a completion manifest.

The full workflow can take several hours. A custom non-overwriting run
directory can be selected with:

```bash
GLAIVE_RUN_DIR=$PWD/runs/my_run bash run_all.sh
```

## Figure-to-Output Mapping

All final paths below are relative to `runs/ae_<timestamp>/figures/`.

| Paper figure | Final output |
|---|---|
| Figure 6 | `Figure06_Synthetic_Experiment.pdf` |
| Figure 7 | `Figure07_Link_Utilization.pdf` |
| Figure 8 | `Figure08_Scalability_Study.pdf` |
| Figure 9(a) | `Figure09a_BPI_Distribution.pdf` |
| Figure 9(b) | `Figure09b_Speedup_vs_BPI.pdf` |
| Figure 10(a) | `Figure10a_Hot_Flow_Cap.pdf` |
| Figure 10(b) | `Figure10b_Solver_Time_Sensitivity.pdf` |
| Figure 10(c) | `Figure10c_Path_Score_Weights.pdf` |
| Figure 11(a) | `Figure11a_Torus_JLU.pdf` |
| Figure 11(b) | `Figure11b_Torus_Sweep_Breakdown.pdf` |
| Figure 12 | `Figure12_EndToEnd.pdf` |
| Figure 16(a) | `Figure16a_1Node_Overhead.pdf` |
| Figure 16(b) | `Figure16b_2Node_Overhead.pdf` |

The authoritative completion signal is:

```text
runs/ae_<timestamp>/figures/target_manifest.json
```

A successful run reports `"complete": true` and 13 targets with
`"status": "ok"`.

## Generated Output Structure

```text
runs/ae_<timestamp>/
├── metadata.txt
├── simulation/
│   ├── evaluation_assets/
│   │   ├── collective-synthesizer/evaluation_assets/
│   │   │   ├── raw_logs/             solver logs
│   │   │   ├── parsed/               CSV and JSON summaries
│   │   │   └── plots/                Figures 6--8, 10, 11(b), and 16
│   │   └── trace_studies/
│   │       ├── generated/            generated trace variants and inputs
│   │       ├── logs/                 performance-suite logs
│   │       ├── results/              derived trace and performance metrics
│   │       └── plots/                Figures 9 and 11(a)
├── end2end/
│   ├── collective_results/           generated collective schedules
│   ├── paper_external_collective/    materialized simulator inputs
│   ├── astra_assets/                 generated ASTRA-Sim configurations
│   ├── astra_results/                per-case logs and timing tables
│   └── figures/                      Figure 12 plotting output
└── figures/
    ├── Figure06_*.pdf ... Figure16b_*.pdf
    └── target_manifest.json
```

## Measured Input Used by Figure 16

Figure 16(a,b) combines synthesis times generated by the current run with the
measured H100 timing table at:

```text
src/collective-synthesizer/evaluation_assets/parsed/figure9_h100_glaive_reference.csv
```

The workflow does not launch a GPU benchmark. Figures 6--12 are generated by
the CPU solver and analytical simulator.

## Verification

Verify a clean source tree before running:

```bash
.venv/bin/python scripts/verify_artifact.py
```

Verify a completed run:

```bash
.venv/bin/python scripts/verify_artifact.py \
  --run-dir runs/ae_<timestamp> \
  --phases target
```

See `docs/figure_mapping.md` for the generated source of every panel and
`docs/target_reproducibility.md` for stage-level validation.
