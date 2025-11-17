#!/usr/bin/env bash
set -euo pipefail

# Extract NEON assembly from NumPy and Raptors libraries
# Usage: extract_assembly.sh [function_pattern]
# Example: extract_assembly.sh mean_axis0

FUNCTION_PATTERN="${1:-mean_axis0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_DIR="${ROOT_DIR}/benchmarks/profiles"
VENV="${ROOT_DIR}/.venv/bin"

mkdir -p "$PROFILES_DIR"

echo "Extracting assembly for pattern: $FUNCTION_PATTERN"

# Find NumPy library
NUMPY_LIB=$(find "$VENV/../lib/python3.11/site-packages/numpy" -name "*multiarray*.so" 2>/dev/null | head -1)
if [ -z "$NUMPY_LIB" ]; then
    echo "Warning: NumPy library not found, trying alternative paths..."
    NUMPY_LIB=$(python3 -c "import numpy; import os; print(os.path.join(os.path.dirname(numpy.__file__), '_core', '_multiarray_umath.cpython-*.so'))" 2>/dev/null | head -1)
fi

# Find Raptors library
RAPTORS_LIB=$(find "$ROOT_DIR" -name "*raptors*.so" -o -name "*_raptors*.so" 2>/dev/null | head -1)
if [ -z "$RAPTORS_LIB" ]; then
    echo "Warning: Raptors library not found, trying alternative paths..."
    RAPTORS_LIB=$(find "$ROOT_DIR/rust/target" -name "*raptors*.so" 2>/dev/null | head -1)
fi

echo "NumPy library: ${NUMPY_LIB:-NOT FOUND}"
echo "Raptors library: ${RAPTORS_LIB:-NOT FOUND}"

# Function to extract assembly
extract_asm() {
    local lib_path=$1
    local output_file=$2
    local pattern=$3
    
    if [ ! -f "$lib_path" ]; then
        echo "  Library not found: $lib_path"
        return 1
    fi
    
    echo "  Extracting assembly from: $lib_path"
    
    # Use objdump to disassemble
    if command -v objdump >/dev/null 2>&1; then
        objdump -d -C "$lib_path" | \
            grep -A 50 -i "$pattern" > "$output_file.asm" 2>&1 || true
        
        # Extract only NEON instructions
        grep -E "(vld|vst|vadd|vdup|vld1|vst1|vaddq|vdupq|fmla|fadd)" "$output_file.asm" > "$output_file.neon" 2>&1 || true
        
        echo "    Full assembly: $output_file.asm"
        echo "    NEON instructions: $output_file.neon"
    else
        echo "  Warning: objdump not found, trying alternative..."
        
        # Try using readelf + objdump alternative
        if command -v aarch64-linux-gnu-objdump >/dev/null 2>&1; then
            aarch64-linux-gnu-objdump -d -C "$lib_path" | \
                grep -A 50 -i "$pattern" > "$output_file.asm" 2>&1 || true
            grep -E "(vld|vst|vadd|vdup|vld1|vst1|vaddq|vdupq|fmla|fadd)" "$output_file.asm" > "$output_file.neon" 2>&1 || true
            echo "    Full assembly: $output_file.asm"
            echo "    NEON instructions: $output_file.neon"
        else
            echo "  Error: No objdump found. Please install binutils."
            return 1
        fi
    fi
}

if [ -n "${NUMPY_LIB:-}" ] && [ -f "$NUMPY_LIB" ]; then
    numpy_output="${PROFILES_DIR}/numpy_${FUNCTION_PATTERN}"
    extract_asm "$NUMPY_LIB" "$numpy_output" "$FUNCTION_PATTERN"
fi

if [ -n "${RAPTORS_LIB:-}" ] && [ -f "$RAPTORS_LIB" ]; then
    raptors_output="${PROFILES_DIR}/raptors_${FUNCTION_PATTERN}"
    extract_asm "$RAPTORS_LIB" "$raptors_output" "$FUNCTION_PATTERN"
fi

echo "Assembly extraction complete. Results saved to: $PROFILES_DIR"

