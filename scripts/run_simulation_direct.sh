#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="${GLAIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/simulation_$STAMP}"
CS_ROOT="$ROOT/src/collective-synthesizer"
SIM_OUT="$RUN_DIR/simulation_exp"
EVAL_OUT="$RUN_DIR/evaluation_assets"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"

mkdir -p "$RUN_DIR"
export GLAIVE_SIM_EXP_OUT_ROOT="$SIM_OUT"
export GLAIVE_EVAL_ASSETS_OUT_ROOT="$EVAL_OUT"

if [ "${GLAIVE_SKIP_BUILD:-0}" != "1" ]; then
  GLAIVE_BUILD_SOLVER=1 GLAIVE_BUILD_ASTRA=0 bash "$ROOT/scripts/build_all.sh"
fi

{
  echo "ROOT=$ROOT"
  echo "RUN_DIR=$RUN_DIR"
  echo "SIM_OUT=$SIM_OUT"
  echo "HOST=$(hostname)"
  echo "DATE=$(date -Iseconds)"
  "$PYTHON_BIN" --version
} > "$RUN_DIR/metadata.txt"

cd "$CS_ROOT"
SIM_EXP="rebuttal/simulation_exp/scripts/sim_exp.py"

if [ "${GLAIVE_RUN_EVAL_ASSETS:-1}" != "0" ]; then
  GLAIVE_RUN_DIR="$EVAL_OUT" GLAIVE_SKIP_BUILD=1 \
    bash "$ROOT/scripts/run_evaluation_assets_direct.sh"
fi

if [ ! -s "$SIM_OUT/results/trace_metrics_real_full.csv" ]; then
  "$PYTHON_BIN" "$SIM_EXP" metrics --per-group "${GLAIVE_SIM_PER_GROUP:-3}"
else
  echo "[simulation] reusing existing trace metrics under $SIM_OUT/results"
fi
if [ ! -s "$SIM_OUT/results/trace_metrics_official_eplb.csv" ]; then
  "$PYTHON_BIN" "$SIM_EXP" official-eplb-metrics --targets "${GLAIVE_SIM_EPLB_TARGETS:-32,64}"
else
  echo "[simulation] reusing existing official EPLB metrics"
fi
if [ ! -s "$SIM_OUT/results/generated_trace_variant_manifest.csv" ]; then
  "$PYTHON_BIN" "$SIM_EXP" materialize-suite --max-suite "${GLAIVE_SIM_MAX_SUITE:-0}"
else
  echo "[simulation] reusing existing materialized suite"
fi
if [ ! -s "$SIM_OUT/results/performance_results.csv" ]; then
  "$PYTHON_BIN" "$SIM_EXP" run-suite \
    --workers "${GLAIVE_SIM_WORKERS:-8}" \
    --max-cases "${GLAIVE_SIM_MAX_CASES:-0}" \
    --max-tasks "${GLAIVE_SIM_MAX_TASKS:-0}" \
    ${GLAIVE_SIM_FORCE:+--force}
  "$PYTHON_BIN" "$SIM_EXP" parse-suite
  "$PYTHON_BIN" "$SIM_EXP" derive-summaries
  "$PYTHON_BIN" "$SIM_EXP" plot

  if [ "${GLAIVE_SIM_MAX_SUITE:-0}" = "0" ] && \
     [ "${GLAIVE_SIM_MAX_CASES:-0}" = "0" ] && \
     [ "${GLAIVE_SIM_MAX_TASKS:-0}" = "0" ]; then
    "$PYTHON_BIN" rebuttal/simulation_exp/scripts/rerun_torus_link_hotedge_metrics.py ${GLAIVE_SIM_FORCE:+--force}
    "$PYTHON_BIN" rebuttal/simulation_exp/plots/torus_jain_adjusted_link_utilization.py
    "$PYTHON_BIN" rebuttal/simulation_exp/scripts/rerun_torus_speed_breakdown.py ${GLAIVE_SIM_FORCE:+--force}
  else
    echo "[simulation] skipping full torus deep-dive post-processing because GLAIVE_SIM_MAX_* limits are set"
  fi
else
  echo "[simulation] reusing parsed suite and simulation plots"
fi

echo "$RUN_DIR"
