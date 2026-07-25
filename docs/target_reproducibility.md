# Glaive Artifact Reproduction Workflow

This artifact reproduces Figures 6--12 and 16. Figures 13--15 require
multi-GPU execution and are not invoked.

## Environment

The reference environment is Ubuntu 22.04 LTS on an x86-64 machine with at
least 8 CPU cores, 64 GiB of memory, and 20 GiB of free storage. The required
software is:

- glibc 2.17 or newer;
- GCC and G++ 12 or newer;
- GNU Make;
- Python 3.10 or 3.11 with `venv`;
- Bash and `rsync`.

The setup script creates `.venv/` and installs CMake 3.27.9, Matplotlib 3.7.0,
NumPy 1.23.5, Pandas 1.5.3, and Pillow 9.5.0:

```bash
bash scripts/setup_environment.sh
```

## Complete Execution

Run all target experiments from the repository root:

```bash
bash run_all.sh
```

The default worker count is eight. It controls the local solver and
ASTRA-Sim worker pools:

```bash
GLAIVE_WORKERS=16 bash run_all.sh
```

The workflow creates `runs/ae_<timestamp>/` and never writes generated
experiment results into the source directories.

## Execution Stages

1. `scripts/build_all.sh` builds the Glaive solver and ASTRA-Sim analytical
   backend.
2. `scripts/run_evaluation_assets_direct.sh` executes the synthetic,
   link-utilization, scalability, sensitivity, and synthesis-overhead cases
   for Figures 6--8, 10, 11(b), and 16.
3. `src/collective-synthesizer/evaluation_assets/scripts/run_trace_studies.py`
   generates the trace variants and performance summaries for Figures 9 and
   11(a).
4. `scripts/run_end2end_direct.sh` generates the collective schedules, runs
   the 48 analytical ASTRA-Sim cases, and plots Figure 12.
5. `scripts/collect_target_figures.py` copies all 13 panels to the stable
   paper-numbered output directory.

## Generated Data

The solver and evaluation-assets stage writes:

```text
simulation/evaluation_assets/collective-synthesizer/evaluation_assets/
├── raw_logs/
├── parsed/
└── plots/
```

The trace-variant stage writes:

```text
simulation/evaluation_assets/trace_studies/
├── generated/
├── logs/
├── results/
└── plots/
```

The Figure 12 stage writes:

```text
end2end/
├── collective_results/
├── paper_external_collective/
├── astra_assets/
├── astra_results/
└── figures/
```

The final PDFs and completion manifest are stored in:

```text
runs/ae_<timestamp>/figures/
```

## Figure 16 Input

Figure 16(a,b) uses the fixed measured H100 timing table:

```text
src/collective-synthesizer/evaluation_assets/parsed/figure9_h100_glaive_reference.csv
```

The current run generates the corresponding synthesis times and combines
them with this table. No GPU benchmark is launched.

## Validation

Before execution:

```bash
.venv/bin/python scripts/verify_artifact.py
```

After execution:

```bash
.venv/bin/python scripts/verify_artifact.py \
  --run-dir runs/ae_<timestamp> \
  --phases target
```

The target manifest must contain `"complete": true`, an empty `"missing"`
list, and 13 targets with `"status": "ok"`.

The Figure 12 summary can be checked independently:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

run = Path("runs/ae_<timestamp>")
summary = json.loads(
    (run / "end2end/astra_results/realmodel_e2e4_analytical/summary.json").read_text()
)
assert len(summary["results"]) == 48
assert all(row["status"] == "ok" for row in summary["results"])
print("Figure 12 validation passed")
PY
```
