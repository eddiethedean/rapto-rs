#!/usr/bin/env bash
set -euo pipefail

# Validate a specific fix by running benchmarks and comparing results
# Usage: docker_validate_fix.sh <operation> <shape> <dtype>

OPERATION="${1:-scale}"
SHAPE="${2:-512x512}"
DTYPE="${3:-float32}"

echo "Validating fix for $OPERATION @ $SHAPE $DTYPE"

# Run benchmark in Docker
docker-compose -f docker-compose.bench.yml run --rm bench \
    /workspace/.venv/bin/python /workspace/scripts/compare_numpy_raptors.py \
    --shape "$SHAPE" \
    --dtype "$DTYPE" \
    --operations "$OPERATION" \
    --warmup 3 \
    --repeats 30

echo ""
echo "Validation complete. Check speedup values above (should be >= 1.0x)"

