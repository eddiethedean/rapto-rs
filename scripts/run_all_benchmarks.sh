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
    --output-json "${OUT_DIR}/axis0_${label}.json"
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

