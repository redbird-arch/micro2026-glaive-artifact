#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${GLAIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/ae_$STAMP}"
WORKERS="${GLAIVE_WORKERS:-8}"

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLAIVE_WORKERS must be a positive integer" >&2
  exit 2
fi

export GLAIVE_EVAL_MAX_WORKERS="${GLAIVE_EVAL_MAX_WORKERS:-$WORKERS}"
export GLAIVE_SIM_WORKERS="${GLAIVE_SIM_WORKERS:-$WORKERS}"
export GLAIVE_ASTRA_WORKERS="${GLAIVE_ASTRA_WORKERS:-$WORKERS}"
export PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

echo "Glaive artifact: Figures 6-12 and 16"
echo "Parallel workers: $WORKERS"
echo "Figure 16 uses the bundled measured H100 timing table; no GPU benchmark is launched."
echo "Run directory: $RUN_DIR"

bash "$ROOT/scripts/setup_environment.sh"
"$PYTHON_BIN" "$ROOT/scripts/verify_artifact.py"
GLAIVE_RUN_DIR="$RUN_DIR" bash "$ROOT/scripts/run_cpu_artifact_direct.sh"

echo "Completed run: $RUN_DIR"
echo "Experiment data: $RUN_DIR/simulation and $RUN_DIR/end2end"
echo "Final figures: $RUN_DIR/figures"
echo "Completion manifest: $RUN_DIR/figures/target_manifest.json"
