#!/bin/bash
# Profile performance comparison between NumPy and Raptors

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT_DIR"

echo "=" | tee -a /tmp/raptors_profile.log
echo "Performance Profiling: NumPy vs Raptors" | tee -a /tmp/raptors_profile.log
echo "=" | tee -a /tmp/raptors_profile.log
echo ""

# Check available profiling tools
echo "Checking available profiling tools..." | tee -a /tmp/raptors_profile.log
echo ""

HAS_PYSPY=false
HAS_DTRACE=false
HAS_INSTRUMENTS=false

if command -v py-spy &> /dev/null; then
    HAS_PYSPY=true
    echo "✓ py-spy found" | tee -a /tmp/raptors_profile.log
else
    echo "✗ py-spy not found (install with: pip install py-spy)" | tee -a /tmp/raptors_profile.log
fi

if command -v dtrace &> /dev/null; then
    HAS_DTRACE=true
    echo "✓ dtrace found" | tee -a /tmp/raptors_profile.log
else
    echo "✗ dtrace not found" | tee -a /tmp/raptors_profile.log
fi

if command -v instruments &> /dev/null; then
    HAS_INSTRUMENTS=true
    echo "✓ instruments found" | tee -a /tmp/raptors_profile.log
else
    echo "✗ instruments not found (install Xcode Command Line Tools)" | tee -a /tmp/raptors_profile.log
fi

echo ""

# Run benchmark with timing to identify which parts are slow
echo "Running detailed benchmark..." | tee -a /tmp/raptors_profile.log
echo ""

PYTHONPATH="$ROOT_DIR/python" "$PYTHON_BIN" scripts/compare_numpy_raptors.py \
    --suite 2d \
    --dtype float32 \
    --operations scale \
    --warmup 10 \
    --repeats 50 \
    2>&1 | tee -a /tmp/raptors_profile.log

echo ""
echo "Profile data saved to /tmp/raptors_profile.log" | tee -a /tmp/raptors_profile.log
echo ""

if [ "$HAS_PYSPY" = true ]; then
    echo "To profile with py-spy (requires sudo):"
    echo "  sudo py-spy record --rate 250 --output /tmp/raptors_profile.svg -- python3 scripts/profile_scale_comparison.py"
    echo ""
fi

if [ "$HAS_DTRACE" = true ]; then
    echo "To trace Accelerate calls with dtrace (requires sudo):"
    echo "  sudo dtrace -n 'pid\$target:libAccelerate:*:entry { @[probefunc] = count(); }' -p \$(pgrep -f 'python.*profile_scale_comparison')"
    echo ""
fi

if [ "$HAS_INSTRUMENTS" = true ]; then
    echo "To profile with Instruments:"
    echo "  1. Open Instruments (Xcode)"
    echo "  2. Choose 'Time Profiler' template"
    echo "  3. Set target: python3 scripts/profile_scale_comparison.py"
    echo "  4. Set environment: PYTHONPATH=$ROOT_DIR/python"
    echo ""
fi

