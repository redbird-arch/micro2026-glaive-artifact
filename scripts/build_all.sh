#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/.venv/bin:$PATH"
CS_ROOT="$ROOT/src/collective-synthesizer"
ASTRA_ROOT="$ROOT/src/astra-sim-galois"

BUILD_SOLVER="${GLAIVE_BUILD_SOLVER:-1}"
BUILD_ASTRA="${GLAIVE_BUILD_ASTRA:-1}"
BUILD_JOBS="${GLAIVE_BUILD_JOBS:-${GLAIVE_WORKERS:-8}}"
CC_BIN="${CC:-gcc}"
CXX_BIN="${CXX:-g++}"

for command_name in "$CC_BIN" "$CXX_BIN" cmake make; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "[build] ERROR: missing $command_name; run scripts/setup_environment.sh" >&2
    exit 1
  fi
done
gcc_major="$("$CC_BIN" -dumpfullversion -dumpversion | cut -d. -f1)"
if [ "$gcc_major" -lt 12 ]; then
  echo "[build] ERROR: GCC 12 or newer is required" >&2
  exit 1
fi
cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
if [ "$(printf '%s\n' 3.26 "$cmake_version" | sort -V | head -n 1)" != "3.26" ]; then
  echo "[build] ERROR: CMake 3.26 or newer is required" >&2
  exit 1
fi
export CC="$(command -v "$CC_BIN")"
export CXX="$(command -v "$CXX_BIN")"
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"

# Invalidate an ASTRA CMake cache created with the system GCC 4.8.
ASTRA_BUILD_CACHE="$ASTRA_ROOT/build/astra_analytical/build/CMakeCache.txt"
if [ -f "$ASTRA_BUILD_CACHE" ] && grep -q '/usr/bin/\(cc\|c++\)' "$ASTRA_BUILD_CACHE"; then
  bash "$ASTRA_ROOT/build/astra_analytical/build.sh" -l
fi

if [ "$BUILD_SOLVER" = "1" ]; then
  echo "[build] Collective-Synthesizer"
  bash "$CS_ROOT/tacos.sh" configure
  bash "$CS_ROOT/tacos.sh" build
  LIBSTDCPP="$(${CXX:-c++} -print-file-name=libstdc++.so 2>/dev/null || true)"
  if [ -n "$LIBSTDCPP" ] && [ -f "$LIBSTDCPP" ]; then
    LIBSTDCPP_DIR="$(cd "$(dirname "$LIBSTDCPP")" && pwd)"
    {
      printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$LIBSTDCPP_DIR"
      printf 'export GLAIVE_SOLVER_LIBSTDCPP_DIR=%q\n' "$LIBSTDCPP_DIR"
    } > "$CS_ROOT/build/glaive_solver_env.sh"
  fi
fi

if [ "$BUILD_ASTRA" = "1" ]; then
  echo "[build] ASTRA-Sim analytical backend"
  ASTRA_BIN="$ASTRA_ROOT/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
  set +e
  CC="$CC" CXX="$CXX" bash "$ASTRA_ROOT/build/astra_analytical/build.sh" -c
  ASTRA_STATUS=$?
  set -e
  if [ "$ASTRA_STATUS" -ne 0 ]; then
    if [ -x "$ASTRA_BIN" ]; then
      echo "[build] WARNING: ASTRA rebuild failed; reusing existing $ASTRA_BIN" >&2
    else
      echo "[build] ERROR: ASTRA rebuild failed and no existing AnalyticalAstra binary is available" >&2
      exit "$ASTRA_STATUS"
    fi
  fi
fi

echo "[build] done"
