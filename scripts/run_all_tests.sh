#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v pyenv >/dev/null 2>&1; then
    PYTHON_BIN="$(pyenv which python 2>/dev/null || true)"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  fi
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "ERROR: Unable to determine Python interpreter. Set PYTHON_BIN to override." >&2
  exit 1
fi

echo "==> Using Python interpreter: ${PYTHON_BIN}"
export PYO3_PYTHON="${PYO3_PYTHON_BIN:-${PYTHON_BIN}}"

PYTHON_LIBDIR="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
if [[ -n "${PYTHON_LIBDIR}" ]]; then
export DYLD_LIBRARY_PATH="${PYTHON_LIBDIR}:${DYLD_LIBRARY_PATH:-}"
fi
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-undefined -C link-arg=dynamic_lookup"

VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.raptors-test-venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m pip --version >/dev/null 2>&1; then
  echo "ERROR: pip is unavailable inside ${VENV_DIR}. Please recreate the virtual environment." >&2
  exit 1
fi

REQUIRED_PACKAGES=(maturin pytest numpy)
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
  if ! env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m pip show "${package}" >/dev/null 2>&1; then
    MISSING_PACKAGES+=("${package}")
  fi
done

if [[ "${#MISSING_PACKAGES[@]}" -gt 0 ]]; then
  echo "==> Installing tooling (${MISSING_PACKAGES[*]}) into ${VENV_DIR}"
  env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m pip install --upgrade pip "${MISSING_PACKAGES[@]}"
fi

if [[ "${SKIP_MATURIN:-0}" != "1" ]]; then
  echo "==> Installing Python extension (maturin develop)"
  env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m maturin develop --release --manifest-path "${SCRIPT_DIR}/rust/Cargo.toml"
else
  echo "==> Skipping maturin develop (SKIP_MATURIN=1)"
fi

echo "==> Running cargo test (Rust unit + integration)"
env -u CONDA_PREFIX \
  PYO3_PYTHON="${PYO3_PYTHON}" \
  VIRTUAL_ENV="${VENV_DIR}" \
  cargo test --manifest-path "${SCRIPT_DIR}/rust/Cargo.toml" \
             --no-default-features \
             --features test-suite

echo "==> Running pytest suite"
env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m pytest "$@"

