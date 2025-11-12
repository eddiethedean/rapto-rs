#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV="${ROOT_DIR}/.venv"

find_host_python() {
  if [[ -n "${HOST_PYTHON:-}" ]]; then
    echo "${HOST_PYTHON}"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "$(command -v python3)"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    echo "$(command -v python)"
    return
  fi
  echo "error: unable to locate a Python interpreter (set HOST_PYTHON)" >&2
  exit 1
}

ensure_virtualenv() {
  local venv_path="$1"
  local host_python
  host_python="$(find_host_python)"
  if [[ ! -d "${venv_path}" ]]; then
    echo "== Creating shared virtualenv at ${venv_path}"
    "${host_python}" -m venv "${venv_path}"
  fi
  if [[ ! -x "${venv_path}/bin/python" ]]; then
    echo "error: virtualenv at ${venv_path} is missing python" >&2
    exit 1
  fi
}

ensure_packages() {
  local venv_path="$1"
  local pip="${venv_path}/bin/pip"
  local packages=(maturin pytest numpy)
  echo "== Ensuring tooling in ${venv_path}"
  "${pip}" install --upgrade pip >/dev/null
  "${pip}" install --upgrade "${packages[@]}" >/dev/null
}

ensure_virtualenv "${DEFAULT_VENV}"
ensure_packages "${DEFAULT_VENV}"

echo "== Prebuilding Raptors extension (shared venv) =="
"${DEFAULT_VENV}/bin/maturin" develop --release --manifest-path "${ROOT_DIR}/rust/Cargo.toml"

echo
echo "== Running run_all_tests.sh =="
SKIP_MATURIN=1 \
VENV_DIR="${DEFAULT_VENV}" \
PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
PYO3_PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
"${ROOT_DIR}/scripts/run_all_tests.sh"

echo
echo "== Running run_all_benchmarks.sh =="
SKIP_MATURIN=1 \
PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
"${ROOT_DIR}/scripts/run_all_benchmarks.sh"

echo
echo "All tests and benchmarks completed successfully."

