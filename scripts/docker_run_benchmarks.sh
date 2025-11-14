#!/usr/bin/env bash
set -euo pipefail

# Run benchmarks in Docker container

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-benchmarks/docker_results/$(date +%Y%m%d-%H%M%S)}"

echo "Running benchmarks in Docker..."
echo "Output directory: $OUTPUT_DIR"

# First, ensure Raptors is built in the container
# Set CARGO_TARGET_DIR to use the volume-mounted target directory
# Create output directory first
mkdir -p "$OUTPUT_DIR"

docker-compose -f docker-compose.bench.yml run --rm bench bash -c "
  export CARGO_TARGET_DIR=/workspace/src/rust/target && \
  export PYTHONPATH=/workspace/.venv/lib/python3.11/site-packages:\$PYTHONPATH && \
  export VIRTUAL_ENV=/workspace/.venv && \
  cd /workspace/src && \
  /workspace/.venv/bin/maturin develop --release --features openblas && \
  mkdir -p '$OUTPUT_DIR' && \
  /workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
    --suite 2d \
    --warmup 3 \
    --repeats 30 \
    --output-json '$OUTPUT_DIR/results.json'
"

echo "Benchmarks completed. Results saved to: $OUTPUT_DIR"

