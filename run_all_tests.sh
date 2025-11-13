#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PIP_USE_TRUSTSTORE="${PIP_USE_TRUSTSTORE:-0}"

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

if [[ -z "${VENV_DIR:-}" ]]; then
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    VENV_DIR="${SCRIPT_DIR}/.venv"
    SKIP_TOOLS_INSTALL="${SKIP_TOOLS_INSTALL:-1}"
  else
    VENV_DIR="${SCRIPT_DIR}/.raptors-test-venv"
  fi
fi
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
  if [[ "${SKIP_TOOLS_INSTALL:-0}" == "1" ]]; then
    echo "WARNING: Missing packages (${MISSING_PACKAGES[*]}) in ${VENV_DIR}, but SKIP_TOOLS_INSTALL=1; continuing without installation."
  else
    echo "==> Installing tooling (${MISSING_PACKAGES[*]}) into ${VENV_DIR}"
    env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m pip install --upgrade pip "${MISSING_PACKAGES[@]}"
  fi
fi

echo "==> Installing Python extension (maturin develop)"
TEMP_WHEEL_DIR="$(mktemp -d "${SCRIPT_DIR}/dist/run-tests-wheels.XXXXXX")"
cleanup() {
  rm -rf "${TEMP_WHEEL_DIR}"
}
trap cleanup EXIT

env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" "${VENV_PYTHON}" -m maturin build --release --manifest-path "${SCRIPT_DIR}/rust/Cargo.toml" -o "${TEMP_WHEEL_DIR}"

LATEST_WHEEL="$(ls -1t "${TEMP_WHEEL_DIR}"/raptors-*.whl | head -n1)"
if [[ -z "${LATEST_WHEEL}" ]]; then
  echo "ERROR: maturin build did not produce a wheel." >&2
  exit 1
fi

EXT_DIR="${SCRIPT_DIR}/python/raptors"
rm -f "${EXT_DIR}"/_raptors*.so "${EXT_DIR}"/_raptors*.pyd "${EXT_DIR}"/_raptors*.dll

"${PYTHON_BIN}" - "${LATEST_WHEEL}" "${TEMP_WHEEL_DIR}" "${EXT_DIR}" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
package_dir = Path(sys.argv[3])
package_path = Path(package_dir)

with zipfile.ZipFile(wheel, "r") as zf:
    members = [
        name for name in zf.namelist()
        if name.startswith("raptors/") and ("_raptors" in name) and (
            name.endswith(".so") or name.endswith(".pyd") or name.endswith(".dll")
        )
    ]
    if not members:
        raise SystemExit("wheel did not contain compiled extension")
    member = members[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    package_path.mkdir(parents=True, exist_ok=True)
    zf.extract(member, target_dir)
    extracted = target_dir / member
    extracted.rename(package_path / extracted.name)
PY

echo "==> Running cargo test (Rust unit + integration)"
env -u CONDA_PREFIX \
  PYO3_PYTHON="${PYO3_PYTHON}" \
  VIRTUAL_ENV="${VENV_DIR}" \
  cargo test --manifest-path "${SCRIPT_DIR}/rust/Cargo.toml" \
             --no-default-features \
             --features test-suite

echo "==> Running pytest suite"
env -u CONDA_PREFIX VIRTUAL_ENV="${VENV_DIR}" \
  PYTHONPATH="${SCRIPT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VENV_PYTHON}" -m pytest "$@"

