#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_all_benchmarks.sh [--output-dir DIR] [--latest-dir DIR] [--skip-validate]

Run the Raptors vs NumPy benchmark matrix (SIMD forced/disabled, threaded/single-threaded),
capture axis-0 diagnostics, and emit CSV/plot summaries with NumPy configuration metadata.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${ROOT_DIR}/benchmarks/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${RESULT_ROOT}/${STAMP}"
LATEST_DIR="${RESULT_ROOT}/latest"
SKIP_VALIDATE=0

VALIDATE_SLACK="${VALIDATE_SLACK:-0.05}"
VALIDATE_ABS_MS="${VALIDATE_ABS_MS:-0.05}"
SIMD_REPEATS="${SIMD_REPEATS:-21}"
SIMD_WARMUP="${SIMD_WARMUP:-2}"
AXIS0_WARMUP="${AXIS0_WARMUP:-1}"
AXIS0_REPEATS="${AXIS0_REPEATS:-5}"
SUITE_WARMUP="${SUITE_WARMUP:-1}"
SUITE_REPEATS="${SUITE_REPEATS:-7}"
THREAD_POOL="${RAPTORS_THREADS:-10}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --latest-dir)
      LATEST_DIR="$2"
      shift 2
      ;;
    --skip-validate)
      SKIP_VALIDATE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "${OUT_DIR}")"
mkdir -p "$(dirname "${LATEST_DIR}")"
OUT_DIR="$(cd "$(dirname "${OUT_DIR}")" && pwd)/$(basename "${OUT_DIR}")"
LATEST_DIR="$(cd "$(dirname "${LATEST_DIR}")" && pwd)/$(basename "${LATEST_DIR}")"
mkdir -p "${OUT_DIR}"

export PIP_USE_TRUSTSTORE="${PIP_USE_TRUSTSTORE:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

VENV_DIR="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
MATURIN_BIN="${MATURIN_BIN:-${VENV_DIR}/bin/maturin}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "error: unable to locate a Python interpreter (set PYTHON_BIN)" >&2
  exit 1
fi

TEMP_WHEEL_DIR=""
cleanup() {
  if [[ -n "${TEMP_WHEEL_DIR}" && -d "${TEMP_WHEEL_DIR}" ]]; then
    rm -rf "${TEMP_WHEEL_DIR}"
  fi
}
trap cleanup EXIT

if [[ "${SKIP_MATURIN:-0}" != "1" ]]; then
  if [[ ! -x "${MATURIN_BIN}" ]]; then
    MATURIN_BIN="$(command -v maturin || true)"
  fi
  if [[ -z "${MATURIN_BIN}" ]]; then
    echo "error: unable to locate maturin (set MATURIN_BIN or export SKIP_MATURIN=1)" >&2
    exit 1
  fi
  echo "== maturin build =="
  TEMP_WHEEL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/raptors-bench.XXXXXX")"
  (
    cd "${ROOT_DIR}/rust"
    "${MATURIN_BIN}" build --release -o "${TEMP_WHEEL_DIR}"
  )
  WHEEL="$(ls -1t "${TEMP_WHEEL_DIR}"/raptors-*.whl | head -n1)"
  if [[ -z "${WHEEL}" ]]; then
    echo "error: maturin build did not produce a wheel" >&2
    exit 1
  fi
  EXTRACT_DIR="${TEMP_WHEEL_DIR}/extracted"
  TARGET_DIR="${ROOT_DIR}/python/raptors"
  "${PYTHON_BIN}" - "${WHEEL}" "${EXTRACT_DIR}" "${TARGET_DIR}" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel = Path(sys.argv[1])
extract_root = Path(sys.argv[2])
target_dir = Path(sys.argv[3])
package_root = extract_root / "raptors"
if package_root.exists():
    import shutil
    shutil.rmtree(package_root)
with zipfile.ZipFile(wheel, "r") as zf:
    members = [name for name in zf.namelist() if name.startswith("raptors/")]
    if not members:
        raise SystemExit("wheel did not contain the raptors package")
    for member in members:
        zf.extract(member, extract_root)
extensions = [
    path for path in package_root.glob("*")
    if path.suffix.lower() in {".so", ".pyd", ".dll"}
]
if not extensions:
    raise SystemExit("wheel extraction did not produce an extension module")
extension = extensions[0]
target_name = extension.name
if not target_name.startswith("_raptors"):
    parts = target_name.split(".", 1)
    if len(parts) == 2:
        target_name = "_raptors." + parts[1]
    else:
        target_name = "_raptors_" + target_name
renamed = package_root / target_name
extension.rename(renamed)
target_dir.mkdir(parents=True, exist_ok=True)
for existing in target_dir.glob("_raptors*.so"):
    existing.unlink()
for existing in target_dir.glob("_raptors*.pyd"):
    existing.unlink()
for existing in target_dir.glob("_raptors*.dll"):
    existing.unlink()
import shutil
shutil.copy2(renamed, target_dir / renamed.name)
PY
fi
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${ROOT_DIR}/python"
else
  export PYTHONPATH="${ROOT_DIR}/python:${PYTHONPATH}"
fi
export RAPTORS_BLAS="${RAPTORS_BLAS:-auto}"

AXIS0_LOG="${RESULT_ROOT}/axis0_history.jsonl"

with_numpy_threads() {
  local threads="$1"
  shift
  env \
    OMP_NUM_THREADS="${threads}" \
    MKL_NUM_THREADS="${threads}" \
    OPENBLAS_NUM_THREADS="${threads}" \
    NUMEXPR_NUM_THREADS="${threads}" \
    VECLIB_MAXIMUM_THREADS="${threads}" \
    "$@"
}

run_axis0() {
  local label="$1"
  local threads="$2"
  local env_mode="$3"
  local simd_mode="$4"
  local wrapper_cmd
  if [[ "${env_mode}" == "clamp" ]]; then
    wrapper_cmd=(with_numpy_threads 1)
  else
    wrapper_cmd=(with_numpy_threads "${threads}")
  fi
  echo
  echo "== axis0_suite (${label}, simd=${simd_mode}, threads=${threads}) =="
  "${wrapper_cmd[@]}" \
    RAPTORS_THREADS="${threads}" \
    "${PYTHON_BIN}" \
    "${ROOT_DIR}/benchmarks/run_axis0_suite.py" \
    --simd-mode "${simd_mode}" \
    --warmup "${AXIS0_WARMUP}" \
    --repeats "${AXIS0_REPEATS}" \
    --output-json "${OUT_DIR}/axis0_${label}.json" \
    --append-log "${AXIS0_LOG}"
}

run_suite() {
  local label="$1"
  local simd_mode="$2"
  local threads="$3"
  local env_mode="$4"
  local layout="${5:-contiguous}"
  local wrapper_cmd
  if [[ "${env_mode}" == "clamp" ]]; then
    wrapper_cmd=(with_numpy_threads 1)
  else
    wrapper_cmd=(with_numpy_threads "${threads}")
  fi
  echo
  echo "== compare_numpy_raptors (${label}, simd=${simd_mode}, threads=${threads}) =="
  "${wrapper_cmd[@]}" \
    RAPTORS_THREADS="${threads}" \
    "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/compare_numpy_raptors.py" \
    --suite 2d \
    --simd-mode "${simd_mode}" \
    --warmup "${SUITE_WARMUP}" \
    --repeats "${SUITE_REPEATS}" \
    --layout "${layout}" \
    --output-json "${OUT_DIR}/suite_${label}.json"
}

echo "Benchmark artefacts will be stored under ${OUT_DIR}"

run_axis0 "threads_force" "${THREAD_POOL}" "env" "force"
run_axis0 "threads_disable" "${THREAD_POOL}" "env" "disable"
run_axis0 "single_force" 1 "clamp" "force"
run_axis0 "single_disable" 1 "clamp" "disable"

run_suite "threads_force" "force" "${THREAD_POOL}" "env"
run_suite "threads_disable" "disable" "${THREAD_POOL}" "env"
run_suite "threads_force_transpose" "force" "${THREAD_POOL}" "env" "transpose"
run_suite "threads_disable_transpose" "disable" "${THREAD_POOL}" "env" "transpose"
run_suite "single_force" "force" 1 "clamp"
run_suite "single_disable" "disable" 1 "clamp"
run_suite "single_force_transpose" "force" 1 "clamp" "transpose"
run_suite "single_disable_transpose" "disable" 1 "clamp" "transpose"

echo
echo "== summarizing results =="
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_benchmarks.py" \
  "${OUT_DIR}/suite_threads_force.json" \
  "${OUT_DIR}/suite_threads_disable.json" \
  "${OUT_DIR}/suite_threads_force_transpose.json" \
  "${OUT_DIR}/suite_threads_disable_transpose.json" \
  "${OUT_DIR}/suite_single_force.json" \
  "${OUT_DIR}/suite_single_disable.json" \
  "${OUT_DIR}/suite_single_force_transpose.json" \
  "${OUT_DIR}/suite_single_disable_transpose.json" \
  "${OUT_DIR}/axis0_threads_force.json" \
  "${OUT_DIR}/axis0_threads_disable.json" \
  "${OUT_DIR}/axis0_single_force.json" \
  "${OUT_DIR}/axis0_single_disable.json" \
  --output-csv "${OUT_DIR}/nightly_summary.csv" \
  --plot "${OUT_DIR}/nightly_slowdowns.svg" \
  --sub-one

if [[ "${SKIP_VALIDATE}" -ne 1 ]]; then
  echo
  echo "== validating vs baselines =="
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/validate_benchmark_results.py" \
    --results-dir "${OUT_DIR}" \
    --baseline-dir "${ROOT_DIR}/benchmarks/baselines" \
    --slack "${VALIDATE_SLACK}" \
    --absolute-slack-ms "${VALIDATE_ABS_MS}" \
    --allow-missing-baseline
fi

cat > "${OUT_DIR}/index.html" <<EOF
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Raptors Nightly Benchmarks</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background-color: #f8f9fa; }
      h1 { margin-bottom: 0.5rem; }
      ul { line-height: 1.6; }
      .chart { margin-top: 2rem; }
      code { background-color: #e9ecef; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }
    </style>
  </head>
  <body>
    <h1>Nightly Benchmark Summary</h1>
    <p>Generated on $(date -u +"%Y-%m-%d %H:%M UTC").</p>
    <ul>
      <li><a href="suite_threads_force.json">threads SIMD forced</a> (<a href="suite_threads_force.metadata.json">metadata</a>)</li>
      <li><a href="suite_threads_disable.json">threads SIMD disabled</a> (<a href="suite_threads_disable.metadata.json">metadata</a>)</li>
      <li><a href="suite_threads_force_transpose.json">threads transpose forced</a> (<a href="suite_threads_force_transpose.metadata.json">metadata</a>)</li>
      <li><a href="suite_threads_disable_transpose.json">threads transpose disabled</a> (<a href="suite_threads_disable_transpose.metadata.json">metadata</a>)</li>
      <li><a href="suite_single_force.json">single-thread SIMD forced</a> (<a href="suite_single_force.metadata.json">metadata</a>)</li>
      <li><a href="suite_single_disable.json">single-thread SIMD disabled</a> (<a href="suite_single_disable.metadata.json">metadata</a>)</li>
      <li><a href="suite_single_force_transpose.json">single-thread transpose forced</a> (<a href="suite_single_force_transpose.metadata.json">metadata</a>)</li>
      <li><a href="suite_single_disable_transpose.json">single-thread transpose disabled</a> (<a href="suite_single_disable_transpose.metadata.json">metadata</a>)</li>
      <li><a href="axis0_threads_force.json">axis-0 threaded (force)</a></li>
      <li><a href="axis0_threads_disable.json">axis-0 threaded (disable)</a></li>
      <li><a href="axis0_single_force.json">axis-0 single-thread (force)</a></li>
      <li><a href="axis0_single_disable.json">axis-0 single-thread (disable)</a></li>
      <li><a href="nightly_summary.csv">slowdown CSV summary</a></li>
      <li><a href="nightly_slowdowns.svg">slowest operations chart</a></li>
    </ul>
    <div class="chart">
      <h2>Slowest Entries (&lt;1.0×)</h2>
      <img src="nightly_slowdowns.svg" alt="Slowdown chart" />
    </div>
  </body>
</html>
EOF

cat > "${OUT_DIR}/environment.json" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "thread_pool": ${THREAD_POOL},
  "validate_slack": ${VALIDATE_SLACK},
  "validate_absolute_ms": ${VALIDATE_ABS_MS},
  "simd_repeats": ${SIMD_REPEATS},
  "simd_warmup": ${SIMD_WARMUP},
  "axis0_warmup": ${AXIS0_WARMUP},
  "axis0_repeats": ${AXIS0_REPEATS},
  "suite_warmup": ${SUITE_WARMUP},
  "suite_repeats": ${SUITE_REPEATS}
}
EOF

if [[ -d "${LATEST_DIR}" ]]; then
  rm -rf "${LATEST_DIR}"
fi
cp -R "${OUT_DIR}" "${LATEST_DIR}"

echo
echo "Benchmark runs complete. Results stored under ${OUT_DIR}"

