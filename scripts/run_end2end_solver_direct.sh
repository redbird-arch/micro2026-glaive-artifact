#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${GLAIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/end2end_solver_$STAMP}"
CS_ROOT="$ROOT/src/collective-synthesizer"
OUT_ROOT="${GLAIVE_END2END_SOLVER_OUT:-$RUN_DIR/collective_results}"
SCENARIOS="${GLAIVE_END2END_SCENARIOS:-}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

mkdir -p "$OUT_ROOT"
if [ "${GLAIVE_SKIP_BUILD:-0}" != "1" ]; then
  GLAIVE_BUILD_SOLVER=1 GLAIVE_BUILD_ASTRA=0 bash "$ROOT/scripts/build_all.sh"
fi

scenario_filter=",$SCENARIOS,"
for scenario_dir in "$CS_ROOT"/input/generated/realmodel_e2e4/*; do
  [ -d "$scenario_dir" ] || continue
  scenario="$(basename "$scenario_dir")"
  if [ -n "$SCENARIOS" ] && [[ "$scenario_filter" != *",$scenario,"* ]]; then
    continue
  fi
  echo "[end2end-solver] $scenario"
  "$PYTHON_BIN" "$CS_ROOT/utils/run_olmoe64_selected_batch.py" \
    --manifest "$scenario_dir/manifest.json" \
    --collective-root "$CS_ROOT" \
    --solver-bin "$CS_ROOT/build/bin/tacos" \
    --output-dir "$OUT_ROOT/$scenario"
done

echo "$OUT_ROOT"
