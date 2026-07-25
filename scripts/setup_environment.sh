#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then
  CHECK_ONLY=1
elif [ "$#" -ne 0 ]; then
  echo "Usage: $0 [--check-only]" >&2
  exit 2
fi

CC_BIN="${CC:-gcc}"
CXX_BIN="${CXX:-g++}"

for command_name in "$CC_BIN" "$CXX_BIN" make python3 rsync getconf; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

gcc_major="$("$CC_BIN" -dumpfullversion -dumpversion | cut -d. -f1)"
if [ "$gcc_major" -lt 12 ]; then
  echo "GCC 12 or newer is required; found $("$CC_BIN" --version | head -n 1)" >&2
  exit 1
fi

glibc_version="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
if [ "$(printf '%s\n' 2.17 "$glibc_version" | sort -V | head -n 1)" != "2.17" ]; then
  echo "glibc 2.17 or newer is required; found $glibc_version" >&2
  exit 1
fi

python3 - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    raise SystemExit(
        "Python 3.10 or 3.11 is required by the pinned packages; found "
        + ".".join(str(item) for item in sys.version_info[:3])
    )
PY

if [ "$CHECK_ONLY" -eq 0 ]; then
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    python3 -m venv "$ROOT/.venv"
  fi
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi

PYTHON_BIN="$ROOT/.venv/bin/python"
CMAKE_BIN="$ROOT/.venv/bin/cmake"
if [ "$CHECK_ONLY" -eq 1 ] && [ ! -x "$PYTHON_BIN" ]; then
  echo "Python environment is missing; run scripts/setup_environment.sh" >&2
  exit 1
fi
if [ ! -x "$CMAKE_BIN" ]; then
  echo "CMake environment is missing; run scripts/setup_environment.sh" >&2
  exit 1
fi
cmake_version="$("$CMAKE_BIN" --version | awk 'NR == 1 {print $3}')"
if [ "$(printf '%s\n' 3.26 "$cmake_version" | sort -V | head -n 1)" != "3.26" ]; then
  echo "CMake 3.26 or newer is required; found $cmake_version" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import matplotlib
import numpy
import pandas
import PIL

expected = {
    "matplotlib": "3.7.0",
    "numpy": "1.23.5",
    "pandas": "1.5.3",
    "pillow": "9.5.0",
}
actual = {
    "matplotlib": matplotlib.__version__,
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pillow": PIL.__version__,
}
if actual != expected:
    details = ", ".join(
        f"{name}={actual[name]} (expected {expected[name]})" for name in expected
    )
    raise SystemExit(
        "Python dependency mismatch: " + details
        + ". Run scripts/setup_environment.sh without --check-only."
    )
print(f"Python dependencies: matplotlib={matplotlib.__version__}, "
      f"numpy={numpy.__version__}, pandas={pandas.__version__}, "
      f"pillow={PIL.__version__}")
PY

echo "Compiler: $("$CC_BIN" --version | head -n 1)"
echo "CMake: $("$CMAKE_BIN" --version | head -n 1)"
echo "Make: $(make --version | head -n 1)"
echo "glibc: $glibc_version"
echo "Python environment: $PYTHON_BIN"
echo "Environment is ready."
