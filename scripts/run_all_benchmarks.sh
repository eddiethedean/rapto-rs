#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="benchmarks/results/latest"
VALIDATE_SLACK="0.05"
SIMD_REPEATS="11"
SIMD_WARMUP="2"
AXIS0_OUTPUT="$OUTPUT_DIR/axis0_suite.json"

usage() {
  cat <<'EOF'
Usage: run_all_benchmarks.sh [--output-dir DIR]

Runs the nightly benchmark suite, validating results against the checked-in
baselines and emitting summaries (CSV + optional plots) into OUTPUT_DIR.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
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

mkdir -p "$OUTPUT_DIR"

# Sanitize potentially noisy environment variables for reproducibility.
unset PYTHONPATH
export OMP_NUM_THREADS=1
export SKIP_MATURIN=1
export RAPTORS_BLAS="${RAPTORS_BLAS:-auto}"
export RAPTORS_BLAS_SCALE="${RAPTORS_BLAS_SCALE:-0}"

run_benchmark() {
  local label="$1"
  shift
  echo ">>> Running $label"
  python scripts/compare_numpy_raptors.py "$@"
}

RA32="$OUTPUT_DIR/float32_simd.json"
RA64="$OUTPUT_DIR/float64_simd.json"
RA32_SCALAR="$OUTPUT_DIR/float32_scalar.json"
RA64_SCALAR="$OUTPUT_DIR/float64_scalar.json"

# SIMD benchmarks (threads pinned to reduce variance)
ORIGINAL_THREADS="${RAPTORS_THREADS:-}"
export RAPTORS_THREADS="${ORIGINAL_THREADS:-10}"

run_benchmark "float64 SIMD baseline" \
  --suite 2d \
  --simd-mode force \
  --warmup "$SIMD_WARMUP" \
  --repeats "$SIMD_REPEATS" \
  --layout contiguous \
  --output-json "$RA64" \
  --validate-json benchmarks/baselines/2d_float64.json \
  --validate-slack "$VALIDATE_SLACK"

run_benchmark "float32 SIMD baseline" \
  --suite 2d \
  --simd-mode force \
  --warmup "$SIMD_WARMUP" \
  --repeats "$SIMD_REPEATS" \
  --layout contiguous \
  --output-json "$RA32" \
  --validate-json benchmarks/baselines/2d_float32.json \
  --validate-slack "$VALIDATE_SLACK"

# Scalar passes (allow pool to size naturally)
if [[ -n "${ORIGINAL_THREADS}" ]]; then
  export RAPTORS_THREADS="$ORIGINAL_THREADS"
else
  export RAPTORS_THREADS=10
fi

run_benchmark "float64 scalar baseline" \
  --suite 2d \
  --simd-mode disable \
  --warmup "$SIMD_WARMUP" \
  --repeats "$SIMD_REPEATS" \
  --layout contiguous \
  --output-json "$RA64_SCALAR" \
  --validate-json benchmarks/baselines/2d_float64_scalar.json \
  --validate-slack "$VALIDATE_SLACK"

run_benchmark "float32 scalar baseline" \
  --suite 2d \
  --simd-mode disable \
  --warmup "$SIMD_WARMUP" \
  --repeats "$SIMD_REPEATS" \
  --layout contiguous \
  --output-json "$RA32_SCALAR" \
  --validate-json benchmarks/baselines/2d_float32_scalar.json \
  --validate-slack "$VALIDATE_SLACK"

# Axis-0 suite (captures stride-aware tiling and BLAS results).
echo ">>> Running axis-0 suite (SIMD forced)"
  export RAPTORS_THREADS="${ORIGINAL_THREADS:-10}"
python benchmarks/run_axis0_suite.py \
  --simd-mode force \
  --warmup 1 \
  --repeats 7 \
  --output-json "$AXIS0_OUTPUT"

# Summaries (CSV + plots if matplotlib is present).
echo ">>> Generating nightly summary"
python scripts/summarize_benchmarks.py \
  "$RA64" "$RA32" "$RA64_SCALAR" "$RA32_SCALAR" \
  --output-csv "$OUTPUT_DIR/nightly_summary.csv" \
  --plot "$OUTPUT_DIR/nightly_slowdowns.svg" \
  --sub-one

cat > "$OUTPUT_DIR/index.html" <<EOF
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
    </style>
  </head>
  <body>
    <h1>Nightly Benchmark Summary</h1>
    <p>Generated on $(date -u +"%Y-%m-%d %H:%M UTC").</p>
    <ul>
      <li><a href="float64_simd.json">float64 SIMD results</a></li>
      <li><a href="float32_simd.json">float32 SIMD results</a></li>
      <li><a href="float64_scalar.json">float64 scalar results</a></li>
      <li><a href="float32_scalar.json">float32 scalar results</a></li>
      <li><a href="axis0_suite.json">axis-0 suite</a></li>
      <li><a href="nightly_summary.csv">slowdown CSV summary</a></li>
    </ul>
    <div class="chart">
      <h2>Slowest Entries (&lt;1.0×)</h2>
      <img src="nightly_slowdowns.svg" alt="Slowdown chart" />
    </div>
  </body>
</html>
EOF

echo "Nightly benchmarks complete. Results stored under $OUTPUT_DIR"
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MATURIN_BIN="${MATURIN_BIN:-${ROOT_DIR}/.venv/bin/maturin}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ ! -x "${MATURIN_BIN}" ]]; then
  MATURIN_BIN="$(command -v maturin || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "error: unable to locate a Python interpreter (set PYTHON_BIN)" >&2
  exit 1
fi

if [[ "${SKIP_MATURIN:-0}" != "1" ]]; then
  if [[ -z "${MATURIN_BIN}" ]]; then
    echo "error: unable to locate maturin (set MATURIN_BIN or SKIP_MATURIN=1)" >&2
    exit 1
  fi

  echo "== maturin develop =="
  (
    cd "${ROOT_DIR}/rust"
    "${MATURIN_BIN}" develop
  )
fi

export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

RESULT_ROOT="${ROOT_DIR}/benchmarks/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${RESULT_ROOT}/${STAMP}"
mkdir -p "${OUT_DIR}"
AXIS0_LOG="${RESULT_ROOT}/axis0_history.jsonl"

THREAD_POOL="${RAPTORS_THREADS:-8}"
AXIS0_WARMUP="${AXIS0_WARMUP:-1}"
AXIS0_REPEATS="${AXIS0_REPEATS:-5}"
SUITE_WARMUP="${SUITE_WARMUP:-1}"
SUITE_REPEATS="${SUITE_REPEATS:-7}"

clamp_numpy_env() {
  env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    "$@"
}

run_axis0() {
  local label="$1"
  shift
  echo
  echo "== axis0_suite (${label}) =="
  "$@" "${PYTHON_BIN}" \
    "${ROOT_DIR}/benchmarks/run_axis0_suite.py" \
    --simd-mode force \
    --warmup "${AXIS0_WARMUP}" \
    --repeats "${AXIS0_REPEATS}" \
    --output-json "${OUT_DIR}/axis0_${label}.json" \
    --append-log "${AXIS0_LOG}"
}

run_suite() {
  local label="$1"
  local simd_mode="$2"
  shift 2
  echo
  echo "== compare_numpy_raptors (${label}, simd=${simd_mode}) =="
  "$@" "${PYTHON_BIN}" \
    "${ROOT_DIR}/scripts/compare_numpy_raptors.py" \
    --suite 2d \
    --simd-mode "${simd_mode}" \
    --warmup "${SUITE_WARMUP}" \
    --repeats "${SUITE_REPEATS}" \
    --output-json "${OUT_DIR}/suite_${label}_${simd_mode}.json"
}

echo "Benchmark artefacts will be stored under ${OUT_DIR}"

# Threaded runs (default thread pool)
run_axis0 "threads" env \
  RAPTORS_THREADS="${THREAD_POOL}"

for mode in force disable; do
  run_suite "threads" "${mode}" env \
    RAPTORS_THREADS="${THREAD_POOL}"
done

# Single-thread runs (clamp both Raptors and NumPy)
run_axis0 "single_thread" clamp_numpy_env \
  RAPTORS_THREADS=1

for mode in force disable; do
  run_suite "single_thread" "${mode}" clamp_numpy_env \
    RAPTORS_THREADS=1
done

echo
echo "Benchmark runs complete. Results written to ${OUT_DIR}"

