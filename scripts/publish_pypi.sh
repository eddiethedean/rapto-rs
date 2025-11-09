#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPLOAD_DIR="$REPO_ROOT/dist/upload"
ARTIFACT_DIR="$REPO_ROOT/dist/artifacts"
WORKFLOW_FILE=".github/workflows/build-wheels.yml"

function info() {
  echo "[publish] $*"
}

function error() {
  echo "[publish] ERROR: $*" >&2
}

function ensure_clean_repo() {
  if [[ -n "$(git status --porcelain)" ]]; then
    error "Git working tree is dirty. Commit or stash changes before publishing."
    exit 1
  fi
}

function wait_for_workflow() {
  local run_id
  run_id="$(gh run list --workflow "$WORKFLOW_FILE" --limit 1 --json databaseId,status --jq '.[0]')"
  if [[ -z "$run_id" ]]; then
    error "No workflow runs found for $WORKFLOW_FILE."
    exit 1
  fi

  local status
  run_id="$(gh run list --workflow "$WORKFLOW_FILE" --limit 1 --json databaseId --jq '.[0].databaseId')"
  status="$(gh run view "$run_id" --json status --jq '.status')"

  info "Watching workflow run $run_id (current status: $status)"

  while [[ "$status" == "queued" || "$status" == "in_progress" ]]; do
    sleep 20
    status="$(gh run view "$run_id" --json status --jq '.status')"
    info "Workflow status: $status"
  done

  local conclusion
  conclusion="$(gh run view "$run_id" --json conclusion --jq '.conclusion')"

  if [[ "$conclusion" != "success" ]]; then
    error "Workflow run $run_id finished with conclusion '$conclusion'."
    exit 1
  fi

  echo "$run_id"
}

function download_artifacts() {
  local run_id="$1"
  rm -rf "$ARTIFACT_DIR" "$UPLOAD_DIR"
  mkdir -p "$ARTIFACT_DIR" "$UPLOAD_DIR"

  info "Downloading artifacts from run $run_id"
  gh run download "$run_id" --dir "$ARTIFACT_DIR"

  info "Collecting wheel files"
  find "$ARTIFACT_DIR" -name '*.whl' -exec cp {} "$UPLOAD_DIR" \;
}

function build_local_distributions() {
  info "Building local wheel and sdist with maturin"
  maturin build --release --out "$UPLOAD_DIR"
}

function require_pypi_token() {
  if [[ -z "${MATURIN_PYPI_TOKEN:-}" ]]; then
    if [[ -f "$HOME/.pypirc" ]]; then
      info "Using credentials from ~/.pypirc"
    else
      error "MATURIN_PYPI_TOKEN is not set and ~/.pypirc not found."
      exit 1
    fi
  else
    info "MATURIN_PYPI_TOKEN detected"
  fi
}

function upload_distributions() {
  info "Uploading $(ls "$UPLOAD_DIR" | wc -l | tr -d ' ') files with twine"
  python -m twine upload --skip-existing "$UPLOAD_DIR"/*
}

function main() {
  ensure_clean_repo
  require_pypi_token

  local run_id
  run_id="$(wait_for_workflow)"

  download_artifacts "$run_id"
  build_local_distributions
  upload_distributions

  info "Publish complete. View release at https://pypi.org/project/raptors/"
}

main "$@"

