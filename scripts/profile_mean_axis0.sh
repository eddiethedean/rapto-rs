#!/usr/bin/env bash
set -euo pipefail

# Profile mean_axis0 operations with perf stat to collect detailed metrics
# Usage: profile_mean_axis0.sh <shape> <dtype> [numpy|raptors|both]
# Example: profile_mean_axis0.sh 1024x1024 float64 both

SHAPE="${1:-1024x1024}"
DTYPE="${2:-float64}"
BACKEND="${3:-both}"  # numpy, raptors, or both

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="${ROOT_DIR}/benchmarks/profiles"
VENV="${ROOT_DIR}/.venv/bin"
PYTHON="${VENV}/python"

mkdir -p "$PROFILES_DIR"

echo "Profiling mean_axis0 @ $SHAPE $DTYPE (backend: $BACKEND)"

# Function to profile with perf stat
profile_perf_stat() {
    local backend=$1
    local output_file=$2
    local iterations=100
    
    echo "Profiling $backend with perf stat..."
    
    # Create a simple Python script to run mean_axis0
    cat > /tmp/profile_mean_axis0_${backend}.py <<EOF
import sys
import numpy as np
import time

shape_str = "$SHAPE"
dtype_str = "$DTYPE"
backend = "$backend"
iterations = $iterations

# Parse shape
rows, cols = map(int, shape_str.split('x'))
dtype = np.float32 if dtype_str == 'float32' else np.float64

# Create test data
data = np.random.randn(rows, cols).astype(dtype)

if backend == 'numpy':
    # Warmup
    for _ in range(3):
        _ = np.mean(data, axis=0)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = np.mean(data, axis=0)
    elapsed = time.perf_counter() - start
    print(f"NumPy: {elapsed/iterations*1000:.3f}ms per iteration")
    print(f"Result shape: {result.shape}, sum: {result.sum():.6f}")
    
elif backend == 'raptors':
    import raptors
    
    # Warmup
    arr = raptors.array(data)
    for _ in range(3):
        _ = arr.mean(axis=0)
    
    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = arr.mean(axis=0)
    elapsed = time.perf_counter() - start
    print(f"Raptors: {elapsed/iterations*1000:.3f}ms per iteration")
    print(f"Result shape: {result.shape}, sum: {result.to_list()[0]:.6f}")
EOF

    # Run perf stat with detailed metrics
    perf stat -e \
        cycles,instructions,cache-references,cache-misses,\
        L1-dcache-loads,L1-dcache-load-misses,\
        L1-dcache-stores,L1-dcache-store-misses,\
        LLC-loads,LLC-load-misses,\
        branch-instructions,branch-misses,\
        stalled-cycles-frontend,stalled-cycles-backend \
        -o "$output_file.perf_stat.txt" \
        "$PYTHON" /tmp/profile_mean_axis0_${backend}.py 2>&1 | tee "$output_file.output.txt" || {
        echo "  Warning: perf stat failed (may need --privileged in Docker or native Linux)"
        echo "  Running without perf stat..."
        "$PYTHON" /tmp/profile_mean_axis0_${backend}.py > "$output_file.output.txt" 2>&1
    }
    
    if [ -f "$output_file.perf_stat.txt" ]; then
        echo "  Perf stat results: $output_file.perf_stat.txt"
    fi
    echo "  Output: $output_file.output.txt"
}

if [ "$BACKEND" = "numpy" ] || [ "$BACKEND" = "both" ]; then
    numpy_output="${PROFILES_DIR}/mean_axis0_${SHAPE//x/_}_${DTYPE}_numpy"
    profile_perf_stat "numpy" "$numpy_output"
fi

if [ "$BACKEND" = "raptors" ] || [ "$BACKEND" = "both" ]; then
    raptors_output="${PROFILES_DIR}/mean_axis0_${SHAPE//x/_}_${DTYPE}_raptors"
    profile_perf_stat "raptors" "$raptors_output"
fi

echo "Profiling complete. Results saved to: $PROFILES_DIR"

