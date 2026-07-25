#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="${GLAIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/end2end_$STAMP}"
CS_ROOT="$ROOT/src/collective-synthesizer"
ASTRA_ROOT="$ROOT/src/astra-sim-galois"
COLLECTIVE_OUT="${GLAIVE_END2END_SOLVER_OUT:-$RUN_DIR/collective_results}"
ASTRA_OUT="$RUN_DIR/astra_results/realmodel_e2e4_analytical"
ASTRA_ASSET_ROOT="$RUN_DIR/astra_assets"
PLOT_OUT="$RUN_DIR/figures"
OVERRIDE_JSON="$RUN_DIR/realmodel_e2e4_manifest_overrides.json"
PROVENANCE_JSON="$RUN_DIR/realmodel_e2e4_external_input_provenance.json"
MATERIALIZED_EXTERNAL="$RUN_DIR/paper_external_collective"
PAPER_REFERENCE="$ASTRA_ROOT/inputs/paper_reference/RealModelE2E4"
INPUT_MODE="${GLAIVE_END2END_INPUT_MODE:-paper}"
SCENARIOS="${GLAIVE_END2END_SCENARIOS:-}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

mkdir -p "$RUN_DIR" "$PLOT_OUT"

if [ "${GLAIVE_END2END_SKIP_SOLVER:-0}" != "1" ]; then
  GLAIVE_RUN_DIR="$RUN_DIR" GLAIVE_END2END_SOLVER_OUT="$COLLECTIVE_OUT" \
    bash "$ROOT/scripts/run_end2end_solver_direct.sh"
fi

if [ "${GLAIVE_SKIP_BUILD:-0}" != "1" ]; then
  GLAIVE_BUILD_SOLVER=0 GLAIVE_BUILD_ASTRA=1 bash "$ROOT/scripts/build_all.sh"
fi

if [ -n "$SCENARIOS" ]; then
  echo "GLAIVE_END2END_SCENARIOS is not supported by the paper-input Figure 12 flow." >&2
  echo "Run all four scenarios to preserve the accepted-paper panel." >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT/scripts/prepare_end2end_external_inputs.py" \
  --mode "$INPUT_MODE" \
  --reference-root "$PAPER_REFERENCE" \
  --fresh-root "$COLLECTIVE_OUT" \
  --manifest-root "$CS_ROOT/input/generated/realmodel_e2e4" \
  --output-root "$MATERIALIZED_EXTERNAL" \
  --override-json "$OVERRIDE_JSON" \
  --report-json "$PROVENANCE_JSON"

cd "$ASTRA_ROOT"
GLAIVE_ASTRA_ASSET_ROOT="$ASTRA_ASSET_ROOT" \
  "$PYTHON_BIN" scripts/olmoe_inference/generate_realmodel_e2e4_assets.py \
  --manifests-json "$OVERRIDE_JSON"
"$PYTHON_BIN" scripts/olmoe_inference/run_olmoe_inference_analytical.py \
  --manifest "$ASTRA_ASSET_ROOT/metadata/RealModelE2E4/realmodel_e2e4_experiments.json" \
  --output-root "$ASTRA_OUT" \
  --workers "${GLAIVE_ASTRA_WORKERS:-${GLAIVE_WORKERS:-8}}"
GLAIVE_END2END_SUMMARY="$ASTRA_OUT/summary.json" \
GLAIVE_END2END_PLOT_DIR="$PLOT_OUT" \
  "$PYTHON_BIN" Pictures/RealModel_Alltoallv_EndToEnd.py

cp "$PLOT_OUT/RealModel_Alltoallv_EndToEnd.pdf" "$PLOT_OUT/Figure12_Re_RealModel_Alltoallv_EndToEnd.pdf"
echo "$RUN_DIR"
