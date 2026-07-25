#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${GLAIVE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${GLAIVE_RUN_DIR:-$ROOT/runs/evaluation_assets_$STAMP}"
CS_ROOT="$ROOT/src/collective-synthesizer"
WORK_ROOT="$RUN_DIR/collective-synthesizer"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_WORKERS="${GLAIVE_EVAL_MAX_WORKERS:-${GLAIVE_WORKERS:-8}}"
STAGE="${GLAIVE_EVAL_STAGE:-all}"
TASK_FILTER_ARGS=()
if [ -n "${GLAIVE_EVAL_TASK_REGEX:-}" ]; then
  TASK_FILTER_ARGS+=(--task-name-regex "$GLAIVE_EVAL_TASK_REGEX")
fi
if [ -n "${GLAIVE_EVAL_MAX_TASKS:-}" ]; then
  TASK_FILTER_ARGS+=(--max-tasks "$GLAIVE_EVAL_MAX_TASKS")
fi

mkdir -p "$RUN_DIR" "$WORK_ROOT"

if [ "${GLAIVE_SKIP_BUILD:-0}" != "1" ]; then
  GLAIVE_BUILD_SOLVER=1 GLAIVE_BUILD_ASTRA=0 bash "$ROOT/scripts/build_all.sh"
fi
if [ -f "$CS_ROOT/build/glaive_solver_env.sh" ]; then
  # shellcheck disable=SC1091
  . "$CS_ROOT/build/glaive_solver_env.sh"
fi

rsync -a --delete \
  --exclude raw_logs \
  --exclude parsed \
  --exclude plots \
  "$CS_ROOT/evaluation_assets/" "$WORK_ROOT/evaluation_assets/"

# Figure 16 combines newly generated synthesis times with a checked-in H100
# measurement table. Copy only this fixed input while keeping all generated
# parsed outputs isolated in WORK_ROOT.
mkdir -p "$WORK_ROOT/evaluation_assets/parsed"
if [ -f "$CS_ROOT/evaluation_assets/parsed/figure9_h100_glaive_reference.csv" ]; then
  cp "$CS_ROOT/evaluation_assets/parsed/figure9_h100_glaive_reference.csv" \
    "$WORK_ROOT/evaluation_assets/parsed/figure9_h100_glaive_reference.csv"
fi

ln -sfn "$CS_ROOT/build" "$WORK_ROOT/build"
ln -sfn "$CS_ROOT/input" "$WORK_ROOT/input"
ln -sfn "$CS_ROOT/src" "$WORK_ROOT/src"
ln -sfn "$CS_ROOT/tacos.sh" "$WORK_ROOT/tacos.sh"
ln -sfn "$CS_ROOT/CMakeLists.txt" "$WORK_ROOT/CMakeLists.txt"

EVAL_ARGS=(
  --repo-root "$WORK_ROOT"
  --stage "$STAGE"
  --max-workers "$MAX_WORKERS"
)
if [ "${#TASK_FILTER_ARGS[@]}" -gt 0 ]; then
  EVAL_ARGS+=("${TASK_FILTER_ARGS[@]}")
fi
if [ -n "${GLAIVE_EVAL_FORCE:-}" ]; then
  EVAL_ARGS+=(--force)
fi
"$PYTHON_BIN" "$WORK_ROOT/evaluation_assets/scripts/run_evaluation.py" "${EVAL_ARGS[@]}"

"$PYTHON_BIN" "$WORK_ROOT/evaluation_assets/scripts/parse_eval_results.py" \
  --repo-root "$WORK_ROOT" \
  --stage "$STAGE"
"$PYTHON_BIN" "$WORK_ROOT/evaluation_assets/scripts/plot_evaluation.py" \
  --repo-root "$WORK_ROOT"

if [ "${GLAIVE_EVAL_RUN_SPEED:-1}" != "0" ]; then
  "$PYTHON_BIN" "$WORK_ROOT/evaluation_assets/scripts/run_speed_opt3.py" \
    --repo-root "$WORK_ROOT" \
    --stage "${GLAIVE_EVAL_SPEED_OPT3_STAGE:-figure9}" \
    --max-workers "$MAX_WORKERS" \
    ${GLAIVE_EVAL_FORCE:+--force}
fi

if [ "${GLAIVE_EVAL_RUN_SENSITIVITY:-1}" != "0" ]; then
  "$PYTHON_BIN" "$WORK_ROOT/evaluation_assets/scripts/run_speed_sensitivity.py" \
    --repo-root "$WORK_ROOT" \
    --max-workers "$MAX_WORKERS" \
    ${GLAIVE_EVAL_FORCE:+--force}
  "$PYTHON_BIN" "$WORK_ROOT/evaluation_assets/scripts/run_speed_weight_sensitivity.py" \
    --repo-root "$WORK_ROOT" \
    --max-workers "$MAX_WORKERS" \
    ${GLAIVE_EVAL_FORCE:+--force}
fi

echo "$RUN_DIR"
