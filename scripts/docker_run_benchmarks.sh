#!/usr/bin/env bash
set -euo pipefail

# Run benchmarks in Docker container

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-benchmarks/docker_results/$(date +%Y%m%d-%H%M%S)}"

echo "Running benchmarks in Docker..."
echo "Output directory: $OUTPUT_DIR"

# Determine which Docker Compose command is available.
# Prefer the standalone docker-compose if present, otherwise fall back to
# the Docker CLI plugin (`docker compose`).
if command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker-compose"
elif command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker compose"
else
  echo "Error: Neither 'docker-compose' nor 'docker compose' is available on PATH." >&2
  echo "Please install Docker Compose or Docker Desktop, then re-run this script." >&2
  exit 1
fi

# First, ensure Raptors is built in the container
# Set CARGO_TARGET_DIR to use the volume-mounted target directory
# Create output directory first
mkdir -p "$OUTPUT_DIR"

$DOCKER_COMPOSE_CMD -f docker-compose.bench.yml run --rm bench bash -c "
  export CARGO_TARGET_DIR=/workspace/src/rust/target && \
  export PYTHONPATH=/workspace/.venv/lib/python3.11/site-packages:\$PYTHONPATH && \
  export VIRTUAL_ENV=/workspace/.venv && \
  export PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/openblas-pthread/pkgconfig:/usr/lib/x86_64-linux-gnu/openblas-pthread/pkgconfig:\${PKG_CONFIG_PATH:-} && \
  export RAPTORS_DEBUG_AXIS0=1 && \
  cd /workspace/src && \
  /workspace/.venv/bin/maturin develop --release --features openblas && \
  mkdir -p '$OUTPUT_DIR' && \
  /workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
    --suite 2d \
    --warmup 3 \
    --repeats 30 \
    --output-json '$OUTPUT_DIR/results.json' 2>&1 | tee '$OUTPUT_DIR/debug.log'
"

echo "Benchmarks completed. Results saved to: $OUTPUT_DIR"

