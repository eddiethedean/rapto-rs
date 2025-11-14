#!/bin/bash
# Script to trace NumPy's Accelerate/vDSP function calls using dtruss

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create a simple Python script that runs NumPy scale operations
cat > /tmp/trace_numpy_scale.py << 'PYEOF'
import numpy as np
import time

# Create test array
arr = np.random.randn(2048, 2048).astype(np.float32)
factor = 2.5

# Warmup
for _ in range(5):
    _ = arr * factor

print("Running NumPy scale operations...")
# Run operations that can be traced
for _ in range(100):
    result = arr * factor
    time.sleep(0.001)  # Small delay to allow tracing

print("Done")
PYEOF

echo "To trace NumPy's Accelerate calls:"
echo "1. Run: sudo dtruss -f -p \$(pgrep -f 'python.*trace_numpy_scale') | grep -E 'vDSP|Accelerate'"
echo "2. In another terminal, run: $PYTHON_BIN /tmp/trace_numpy_scale.py"
echo ""
echo "Or use dtrace with a custom script (see function_trace_analysis.md)"

