#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="${GLAIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/simulation_$STAMP}"
CS_ROOT="$ROOT/src/collective-synthesizer"
EVAL_OUT="$RUN_DIR/evaluation_assets"
TRACE_OUT="$EVAL_OUT/trace_studies"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

mkdir -p "$RUN_DIR"
export GLAIVE_TRACE_STUDIES_OUT_ROOT="$TRACE_OUT"
export GLAIVE_EVAL_ASSETS_OUT_ROOT="$EVAL_OUT"

if [ "${GLAIVE_SKIP_BUILD:-0}" != "1" ]; then
  GLAIVE_BUILD_SOLVER=1 GLAIVE_BUILD_ASTRA=0 bash "$ROOT/scripts/build_all.sh"
fi

{
  echo "ROOT=$ROOT"
  echo "RUN_DIR=$RUN_DIR"
  echo "TRACE_OUT=$TRACE_OUT"
  echo "HOST=$(hostname)"
  echo "DATE=$(date -Iseconds)"
  "$PYTHON_BIN" --version
} > "$RUN_DIR/metadata.txt"

cd "$CS_ROOT"
TRACE_STUDIES="evaluation_assets/scripts/run_trace_studies.py"

if [ "${GLAIVE_RUN_EVAL_ASSETS:-1}" != "0" ]; then
  GLAIVE_RUN_DIR="$EVAL_OUT" GLAIVE_SKIP_BUILD=1 \
    bash "$ROOT/scripts/run_evaluation_assets_direct.sh"
fi

if [ ! -s "$TRACE_OUT/results/trace_metrics_real_full.csv" ]; then
  "$PYTHON_BIN" "$TRACE_STUDIES" metrics --per-group "${GLAIVE_SIM_PER_GROUP:-3}"
else
  echo "[simulation] reusing existing trace metrics under $TRACE_OUT/results"
fi
if [ ! -s "$TRACE_OUT/results/trace_metrics_official_eplb.csv" ]; then
  "$PYTHON_BIN" "$TRACE_STUDIES" official-eplb-metrics --targets "${GLAIVE_SIM_EPLB_TARGETS:-32,64}"
else
  echo "[simulation] reusing existing official EPLB metrics"
fi
if [ ! -s "$TRACE_OUT/results/generated_trace_variant_manifest.csv" ]; then
  "$PYTHON_BIN" "$TRACE_STUDIES" materialize-suite --max-suite "${GLAIVE_SIM_MAX_SUITE:-0}"
else
  echo "[simulation] reusing existing materialized suite"
fi
if [ ! -s "$TRACE_OUT/results/performance_results.csv" ]; then
  "$PYTHON_BIN" "$TRACE_STUDIES" run-suite \
    --workers "${GLAIVE_SIM_WORKERS:-8}" \
    --max-cases "${GLAIVE_SIM_MAX_CASES:-0}" \
    --max-tasks "${GLAIVE_SIM_MAX_TASKS:-0}" \
    ${GLAIVE_SIM_FORCE:+--force}
  "$PYTHON_BIN" "$TRACE_STUDIES" parse-suite
  "$PYTHON_BIN" "$TRACE_STUDIES" derive-summaries
  "$PYTHON_BIN" "$TRACE_STUDIES" plot

  if [ "${GLAIVE_SIM_MAX_SUITE:-0}" = "0" ] && \
     [ "${GLAIVE_SIM_MAX_CASES:-0}" = "0" ] && \
     [ "${GLAIVE_SIM_MAX_TASKS:-0}" = "0" ]; then
    "$PYTHON_BIN" evaluation_assets/scripts/run_torus_link_metrics.py ${GLAIVE_SIM_FORCE:+--force}
    "$PYTHON_BIN" evaluation_assets/scripts/torus_jain_adjusted_link_utilization.py
    "$PYTHON_BIN" evaluation_assets/scripts/run_torus_speed_breakdown.py ${GLAIVE_SIM_FORCE:+--force}
  else
    echo "[simulation] skipping full torus deep-dive post-processing because GLAIVE_SIM_MAX_* limits are set"
  fi
else
  echo "[simulation] reusing parsed suite and simulation plots"
fi

echo "$RUN_DIR"
