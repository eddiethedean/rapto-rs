#!/usr/bin/env bash
set -euo pipefail

# Compare NumPy and Raptors profiles
# Usage: compare_profiles.sh <operation> <shape> <dtype>

OPERATION="${1:-scale}"
SHAPE="${2:-512x512}"
DTYPE="${3:-float32}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="${ROOT_DIR}/docs/profiles"

numpy_perf="${PROFILES_DIR}/${OPERATION}_${SHAPE//x/_}_${DTYPE}_numpy.perf.txt"
raptors_perf="${PROFILES_DIR}/${OPERATION}_${SHAPE//x/_}_${DTYPE}_raptors.perf.txt"

if [ ! -f "$numpy_perf" ] || [ ! -f "$raptors_perf" ]; then
    echo "Error: Profile files not found. Run profile_operation.sh first."
    exit 1
fi

echo "Comparing profiles for $OPERATION @ $SHAPE $DTYPE"
echo "================================================"
echo ""
echo "NumPy Top Functions:"
echo "-------------------"
head -50 "$numpy_perf" | grep -E "^\s+[0-9]+\.\d+%" | head -20 || true

echo ""
echo "Raptors Top Functions:"
echo "---------------------"
head -50 "$raptors_perf" | grep -E "^\s+[0-9]+\.\d+%" | head -20 || true

echo ""
echo "Flamegraphs available:"
echo "  NumPy: ${PROFILES_DIR}/${OPERATION}_${SHAPE//x/_}_${DTYPE}_numpy.flamegraph.svg"
echo "  Raptors: ${PROFILES_DIR}/${OPERATION}_${SHAPE//x/_}_${DTYPE}_raptors.flamegraph.svg"

