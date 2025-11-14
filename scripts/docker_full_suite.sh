#!/usr/bin/env bash
set -euo pipefail

# Run full benchmark suite in Docker
OUTPUT_DIR="${1:-benchmarks/docker_results/$(date +%Y%m%d-%H%M%S)}"

echo "Running full benchmark suite in Docker..."
echo "Output directory: $OUTPUT_DIR"

docker-compose -f docker-compose.bench.yml run --rm bench \
    /workspace/.venv/bin/python /workspace/scripts/compare_numpy_raptors.py \
    --suite 2d \
    --warmup 3 \
    --repeats 30 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Full suite completed. Results saved to: $OUTPUT_DIR"
echo "Review JSON files to identify any remaining laggards."

