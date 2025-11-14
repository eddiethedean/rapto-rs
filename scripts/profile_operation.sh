#!/usr/bin/env bash
set -euo pipefail

# Profile a specific operation using perf and py-spy
# Usage: profile_operation.sh <operation> <shape> <dtype> [numpy|raptors|both]

OPERATION="${1:-scale}"
SHAPE="${2:-512x512}"
DTYPE="${3:-float32}"
BACKEND="${4:-both}"  # numpy, raptors, or both

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="${ROOT_DIR}/docs/profiles"
VENV="${ROOT_DIR}/.venv/bin"
PYTHON="${VENV}/python"

mkdir -p "$PROFILES_DIR"

echo "Profiling $OPERATION @ $SHAPE $DTYPE (backend: $BACKEND)"

# Function to profile with perf
profile_perf() {
    local backend=$1
    local output_file=$2
    local iterations=200
    
    echo "Profiling $backend with perf..."
    perf record -F 99 -g -o "$output_file.perf.data" -- \
        "$PYTHON" /workspace/scripts/profile_single.py \
        "$OPERATION" "$SHAPE" "$DTYPE" "$backend" "$iterations" \
        > /dev/null 2>&1 || {
        echo "  Warning: perf record failed (may need --privileged in Docker)"
    }
    
    # Generate flamegraph if perf data exists
    if [ -f "$output_file.perf.data" ]; then
        perf script -i "$output_file.perf.data" 2>/dev/null | \
            "$PYTHON" -m flamegraph.pl > "$output_file.flamegraph.svg" 2>/dev/null || \
            echo "  Note: flamegraph generation failed, but perf data is available"
        
        # Generate report
        perf report -i "$output_file.perf.data" --stdio > "$output_file.perf.txt" 2>&1 || true
        
        echo "  Perf data: $output_file.perf.data"
        echo "  Flamegraph: $output_file.flamegraph.svg"
        echo "  Report: $output_file.perf.txt"
    fi
}

# Function to profile with py-spy
profile_pyspy() {
    local backend=$1
    local output_file=$2
    local iterations=200
    
    echo "Profiling $backend with py-spy..."
    "$VENV/py-spy" record \
        --rate 100 \
        --output "$output_file.pyspy.svg" \
        --format flamegraph \
        --subprocesses \
        -- \
        "$PYTHON" /workspace/scripts/profile_single.py \
        "$OPERATION" "$SHAPE" "$DTYPE" "$backend" "$iterations" \
        > /dev/null 2>&1 || {
        echo "  Warning: py-spy record failed"
    }
    
    if [ -f "$output_file.pyspy.svg" ]; then
        echo "  Py-spy flamegraph: $output_file.pyspy.svg"
    fi
}

if [ "$BACKEND" = "numpy" ] || [ "$BACKEND" = "both" ]; then
    numpy_output="${PROFILES_DIR}/${OPERATION}_${SHAPE//x/_}_${DTYPE}_numpy"
    profile_perf "numpy" "$numpy_output"
    profile_pyspy "numpy" "$numpy_output"
fi

if [ "$BACKEND" = "raptors" ] || [ "$BACKEND" = "both" ]; then
    raptors_output="${PROFILES_DIR}/${OPERATION}_${SHAPE//x/_}_${DTYPE}_raptors"
    profile_perf "raptors" "$raptors_output"
    profile_pyspy "raptors" "$raptors_output"
fi

echo "Profiling complete. Results saved to: $PROFILES_DIR"

