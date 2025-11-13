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
  if [[ "${SKIP_TOOLS_INSTALL:-0}" == "1" ]]; then
    echo "== Skipping tooling installation (SKIP_TOOLS_INSTALL=1)"
    return
  fi
  echo "== Ensuring tooling in ${venv_path}"
  "${pip}" install --upgrade pip >/dev/null
  "${pip}" install --upgrade "${packages[@]}" >/dev/null
}

ensure_virtualenv "${DEFAULT_VENV}"
ensure_packages "${DEFAULT_VENV}"

build_extension() {
  local venv_path="$1"
  if [[ "${SKIP_MATURIN:-0}" == "1" ]]; then
    echo "== Skipping extension build (SKIP_MATURIN=1)"
    return
  fi
  local maturin_bin="${MATURIN_BIN:-${venv_path}/bin/maturin}"
  if [[ ! -x "${maturin_bin}" ]]; then
    maturin_bin="$(command -v maturin || true)"
  fi
  if [[ -z "${maturin_bin}" ]]; then
    echo "error: unable to locate maturin (set MATURIN_BIN or export SKIP_MATURIN=1)" >&2
    exit 1
  fi
  echo "== Building Python extension wheel (shared venv) =="
  local temp_wheel_dir="${ROOT_DIR}/dist/run-all-checks-tmp"
  mkdir -p "${temp_wheel_dir}"
  (
    cd "${ROOT_DIR}/rust"
    cargo build --release >/dev/null
    "${maturin_bin}" build --release -o "${temp_wheel_dir}"
  )
  local wheel
  wheel="$(ls -1t "${temp_wheel_dir}"/raptors-*.whl | head -n1)"
  if [[ -z "${wheel}" ]]; then
    echo "error: maturin build did not produce a wheel" >&2
    exit 1
  fi
  local ext_dir="${ROOT_DIR}/python/raptors"
  rm -f "${ext_dir}"/_raptors*.so "${ext_dir}"/_raptors*.pyd "${ext_dir}"/_raptors*.dll
  "${DEFAULT_VENV}/bin/python" - "${wheel}" "${temp_wheel_dir}" "${ext_dir}" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel = Path(sys.argv[1])
tmp_dir = Path(sys.argv[2])
package_dir = Path(sys.argv[3])
package_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(wheel, "r") as zf:
    members = [
        name for name in zf.namelist()
        if name.startswith("raptors/")
        and (name.endswith(".so") or name.endswith(".pyd") or name.endswith(".dll"))
    ]
    if not members:
        raise SystemExit("wheel did not contain compiled extension module")
    member = members[0]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zf.extract(member, tmp_dir)
    extracted = tmp_dir / member
    target_name = extracted.name
    if not target_name.startswith("_raptors"):
        parts = target_name.split(".", 1)
        if len(parts) == 2:
            target_name = "_raptors." + parts[1]
        else:
            target_name = "_raptors_" + target_name
    extracted.rename(package_dir / target_name)
PY
  rm -rf "${temp_wheel_dir}"
}

build_extension "${DEFAULT_VENV}"

export PYTHONPATH="${ROOT_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"

echo
echo "== Running run_all_tests.sh =="
SKIP_MATURIN=1 \
VENV_DIR="${DEFAULT_VENV}" \
PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
PYO3_PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
"${ROOT_DIR}/scripts/run_all_tests.sh"

echo
echo "== Running run_all_benchmarks.sh =="
RUN_ALL_CHECKS_SKIP_VALIDATE="${RUN_ALL_CHECKS_SKIP_VALIDATE:-0}"
declare -a BENCHMARK_ARGS=()
if [[ "${RUN_ALL_CHECKS_SKIP_VALIDATE}" == "1" ]]; then
  BENCHMARK_ARGS=(--skip-validate)
  echo "== NOTE: Passing --skip-validate to run_all_benchmarks.sh (RUN_ALL_CHECKS_SKIP_VALIDATE=1)"
fi
if (( ${#BENCHMARK_ARGS[@]} )); then
  SKIP_MATURIN=1 PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
    "${ROOT_DIR}/scripts/run_all_benchmarks.sh" "${BENCHMARK_ARGS[@]}"
else
  SKIP_MATURIN=1 PYTHON_BIN="${DEFAULT_VENV}/bin/python" \
    "${ROOT_DIR}/scripts/run_all_benchmarks.sh"
fi

echo
echo "All tests and benchmarks completed successfully."

