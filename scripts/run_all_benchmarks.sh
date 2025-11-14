#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_all_benchmarks.sh [--output-dir DIR] [--latest-dir DIR] [--skip-validate] [--quick] [--full]

Run the Raptors vs NumPy benchmark matrix using ASV (Airspeed Velocity).
By default runs a quick smoke test; use --full for complete benchmarks.

Options:
  --quick      Run quick benchmarks (single sample, fast validation)
  --full       Run full benchmarks (multiple samples, slower but more accurate)
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
THREAD_POOL="${RAPTORS_THREADS:-10}"
ASV_MODE="quick"
ASV_STEPS="${ASV_STEPS:-1}"

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
    --quick)
      ASV_MODE="quick"
      ASV_STEPS=1
      shift
      ;;
    --full)
      ASV_MODE="full"
      ASV_STEPS="${ASV_STEPS:-3}"
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

# Install or update the module in the venv for ASV to find it
if [[ "${SKIP_MATURIN:-0}" != "1" ]]; then
  echo "== installing raptors module in venv =="
  (
    cd "${ROOT_DIR}/rust"
    # Ensure we use the venv's Python by setting PATH to venv/bin first
    PATH="$(dirname "${PYTHON_BIN}"):${PATH}" "${PYTHON_BIN}" -m maturin develop --release 2>&1 | grep -v "^Compiling\|^Finished\|^Running\|^    Blocking" || true
  )
fi

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${ROOT_DIR}/python"
else
  export PYTHONPATH="${ROOT_DIR}/python:${PYTHONPATH}"
fi
export RAPTORS_BLAS="${RAPTORS_BLAS:-auto}"

BENCHMARKS_DIR="${ROOT_DIR}/benchmarks"
ASV_BIN="${VENV_DIR}/bin/asv"

# Check if asv is installed
if [[ ! -x "${ASV_BIN}" ]]; then
  ASV_BIN="$(command -v asv || true)"
  if [[ -z "${ASV_BIN}" ]]; then
    echo "error: asv not found. Installing..." >&2
    "${PYTHON_BIN}" -m pip install asv >&2
    ASV_BIN="${VENV_DIR}/bin/asv"
    if [[ ! -x "${ASV_BIN}" ]]; then
      echo "error: failed to install or locate asv" >&2
      exit 1
    fi
  fi
fi

# Ensure ASV machine info exists (must be run from benchmarks directory)
if ! (cd "${BENCHMARKS_DIR}" && "${ASV_BIN}" machine --yes >/dev/null 2>&1); then
  echo "== setting up ASV machine information =="
  (
    cd "${BENCHMARKS_DIR}"
    "${ASV_BIN}" machine --yes
  )
fi

with_numpy_threads() {
  local threads="$1"
  shift
  env \
    OMP_NUM_THREADS="${threads}" \
    MKL_NUM_THREADS="${threads}" \
    OPENBLAS_NUM_THREADS="${threads}" \
    NUMEXPR_NUM_THREADS="${threads}" \
    VECLIB_MAXIMUM_THREADS="${threads}" \
    PYTHONPATH="${PYTHONPATH}" \
    "$@"
}

run_asv() {
  local label="$1"
  local threads="$2"
  local simd_mode="$3"
  local asv_args=()
  
  if [[ "${ASV_MODE}" == "quick" ]]; then
    asv_args+=("--quick")
  else
    asv_args+=("--steps" "${ASV_STEPS}")
  fi
  
  local wrapper_cmd
  if [[ "${threads}" == "1" ]]; then
    wrapper_cmd=(with_numpy_threads 1)
  else
    wrapper_cmd=(with_numpy_threads "${threads}")
  fi
  
  echo
  echo "== ASV benchmarks (${label}, simd=${simd_mode}, threads=${threads}) =="
  
  (
    cd "${BENCHMARKS_DIR}"
    # Ensure PYTHONPATH is set for ASV discovery and execution (needed even if module is installed)
    # Use absolute path to ensure it's available to ASV's subprocesses
    # Note: ROOT_DIR should be available in this subshell since it's defined before the function
    local python_path="${ROOT_DIR}/python"
    if [[ -n "${PYTHONPATH:-}" ]]; then
      python_path="${PYTHONPATH}:${python_path}"
    fi
    # Debug: verify PYTHONPATH is set (remove in production)
    if [[ "${DEBUG:-0}" == "1" ]]; then
      echo "DEBUG: PYTHONPATH=${python_path}" >&2
      echo "DEBUG: ROOT_DIR=${ROOT_DIR}" >&2
      echo "DEBUG: Testing import: $("${PYTHON_BIN}" -c "import sys; sys.path.insert(0, '${python_path}'); import raptors; print('OK')" 2>&1)" >&2
    fi
    # Set all environment variables inline to ensure they're passed to ASV and its subprocesses
    env \
      OMP_NUM_THREADS="${threads}" \
      MKL_NUM_THREADS="${threads}" \
      OPENBLAS_NUM_THREADS="${threads}" \
      NUMEXPR_NUM_THREADS="${threads}" \
      VECLIB_MAXIMUM_THREADS="${threads}" \
      RAPTORS_THREADS="${threads}" \
      RAPTORS_SIMD="${simd_mode}" \
      PYTHONPATH="${python_path}" \
      "${ASV_BIN}" run \
      --python="${PYTHON_BIN}" \
      "${asv_args[@]}" \
      --show-stderr \
      2>&1 | tee "${OUT_DIR}/asv_${label}.log"
  )
  
  # Export results summary
  echo
  echo "== exporting ASV results summary (${label}) =="
  (
    cd "${BENCHMARKS_DIR}"
    "${ASV_BIN}" show \
      > "${OUT_DIR}/asv_${label}.txt" 2>&1 || echo "Note: ASV show may have warnings" > "${OUT_DIR}/asv_${label}.txt"
  )
}

echo "Benchmark artefacts will be stored under ${OUT_DIR}"
echo "ASV mode: ${ASV_MODE} (steps: ${ASV_STEPS})"

# Run ASV benchmarks with different configurations
# Note: ASV benchmarks internally test both NumPy and Raptors across various
# configurations (SIMD modes, thread counts, layouts) as defined in the suite files
run_asv "threads_force" "${THREAD_POOL}" "force"
run_asv "threads_disable" "${THREAD_POOL}" "disable"
run_asv "single_force" 1 "force"
run_asv "single_disable" 1 "disable"

echo
echo "== publishing ASV HTML results =="
(
  cd "${BENCHMARKS_DIR}"
  "${ASV_BIN}" publish --html-dir "${OUT_DIR}/html" 2>&1 || echo "Note: ASV publish may have warnings"
)

echo
echo "== generating summary report =="
# Extract key metrics from ASV results
cat > "${OUT_DIR}/asv_summary.txt" <<EOF
ASV Benchmark Summary
=====================

Mode: ${ASV_MODE}
Steps: ${ASV_STEPS}
Thread Pool: ${THREAD_POOL}
Timestamp: $(date -u +"%Y-%m-%d %H:%M UTC")

Results:
- ASV results database: ${BENCHMARKS_DIR}/.asv/results/
- HTML report: ${OUT_DIR}/html/index.html
- Raw logs: ${OUT_DIR}/asv_*.log
- Summary reports: ${OUT_DIR}/asv_*.txt

View results:
  cd ${BENCHMARKS_DIR}
  ${ASV_BIN} show
  ${ASV_BIN} compare benchmarks/baselines/asv/

Compare with baselines:
  ${ASV_BIN} compare benchmarks/baselines/asv/ HEAD
EOF
cat "${OUT_DIR}/asv_summary.txt"

if [[ "${SKIP_VALIDATE}" -ne 1 ]]; then
  echo
  echo "== comparing with ASV baselines =="
  if [[ -d "${BENCHMARKS_DIR}/baselines/asv" ]]; then
    (
      cd "${BENCHMARKS_DIR}"
      "${ASV_BIN}" compare benchmarks/baselines/asv/ HEAD \
        2>&1 | tee "${OUT_DIR}/asv_comparison.txt" || echo "Note: Comparison may show differences"
    )
  else
    echo "No ASV baselines found at ${BENCHMARKS_DIR}/baselines/asv"
    echo "Skipping baseline comparison. Run 'asv run --steps 3 && rsync -a .asv/results/ baselines/asv/' to create baselines."
  fi
fi

cat > "${OUT_DIR}/index.html" <<EOF
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Raptors ASV Benchmark Results</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; background-color: #f8f9fa; }
      h1 { margin-bottom: 0.5rem; }
      h2 { margin-top: 2rem; margin-bottom: 1rem; }
      ul { line-height: 1.6; }
      code { background-color: #e9ecef; padding: 0.1rem 0.3rem; border-radius: 0.2rem; }
      .info { background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    </style>
  </head>
  <body>
    <h1>ASV Benchmark Results</h1>
    <p>Generated on $(date -u +"%Y-%m-%d %H:%M UTC").</p>
    
    <div class="info">
      <strong>Configuration:</strong><br/>
      Mode: ${ASV_MODE} (steps: ${ASV_STEPS})<br/>
      Thread Pool: ${THREAD_POOL}<br/>
      Python: ${PYTHON_BIN}
    </div>
    
    <h2>Results</h2>
    <ul>
      <li><a href="html/index.html">Full ASV HTML Report</a> (interactive charts and comparisons)</li>
      <li><a href="asv_summary.txt">Summary Report</a></li>
      <li><a href="asv_comparison.txt">Baseline Comparison</a> (if available)</li>
      <li><a href="asv_threads_force.txt">Threads SIMD forced (Summary)</a></li>
      <li><a href="asv_threads_disable.txt">Threads SIMD disabled (Summary)</a></li>
      <li><a href="asv_single_force.txt">Single-thread SIMD forced (Summary)</a></li>
      <li><a href="asv_single_disable.txt">Single-thread SIMD disabled (Summary)</a></li>
      <li><a href="asv_threads_force.log">Threads SIMD forced (Log)</a></li>
      <li><a href="asv_threads_disable.log">Threads SIMD disabled (Log)</a></li>
      <li><a href="asv_single_force.log">Single-thread SIMD forced (Log)</a></li>
      <li><a href="asv_single_disable.log">Single-thread SIMD disabled (Log)</a></li>
    </ul>
    
    <h2>ASV Commands</h2>
    <p>To view results interactively:</p>
    <pre><code>cd ${BENCHMARKS_DIR}
${ASV_BIN} show
${ASV_BIN} compare benchmarks/baselines/asv/ HEAD</code></pre>
    
    <h2>Note</h2>
    <p>The ASV suite includes comprehensive benchmarks across:</p>
    <ul>
      <li><strong>Broadcast operations</strong> (same shape, row, column broadcasts)</li>
      <li><strong>Reduction operations</strong> (sum, mean, axis reductions)</li>
      <li><strong>Scale operations</strong> (elementwise multiplication)</li>
      <li>Multiple configurations: shapes (512², 1024², 2048²), dtypes (float32, float64), layouts (contiguous, transpose), SIMD modes, and thread counts</li>
    </ul>
  </body>
</html>
EOF

cat > "${OUT_DIR}/environment.json" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "thread_pool": ${THREAD_POOL},
  "validate_slack": ${VALIDATE_SLACK},
  "validate_absolute_ms": ${VALIDATE_ABS_MS},
  "asv_mode": "${ASV_MODE}",
  "asv_steps": ${ASV_STEPS},
  "python": "${PYTHON_BIN}",
  "benchmark_suite": "asv"
}
EOF

if [[ -d "${LATEST_DIR}" ]]; then
  rm -rf "${LATEST_DIR}"
fi
cp -R "${OUT_DIR}" "${LATEST_DIR}"

echo
echo "Benchmark runs complete. Results stored under ${OUT_DIR}"

