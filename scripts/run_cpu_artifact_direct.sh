#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/ae_direct_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
mkdir -p "$RUN_DIR"

{
  echo "artifact=glaive-ae"
  echo "figure16_fixed_hardware_input=src/collective-synthesizer/evaluation_assets/parsed/figure9_h100_glaive_reference.csv"
  echo "date=$(date -Iseconds)"
  "$PYTHON_BIN" --version
  gcc --version | head -n 1
  cmake --version | head -n 1
} > "$RUN_DIR/metadata.txt"

echo "[1/3] Figure 6-11 and Figure 16: CPU solver/simulation experiments"
GLAIVE_RUN_DIR="$RUN_DIR/simulation" \
  bash "$ROOT/scripts/run_simulation_direct.sh"

echo "[2/3] Figure 12: ASTRA-Sim end-to-end simulation"
GLAIVE_RUN_DIR="$RUN_DIR/end2end" \
  bash "$ROOT/scripts/run_end2end_direct.sh"

echo "[3/3] Collecting paper-numbered outputs"
"$PYTHON_BIN" "$ROOT/scripts/collect_target_figures.py" \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/figures"

echo "Completed: $RUN_DIR"
