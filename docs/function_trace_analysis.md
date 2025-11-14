# Function Call Trace Analysis

## Overview

This document analyzes function calls made by NumPy and Raptors to understand their dispatch paths and identify optimization opportunities.

## Tools Used

### dtruss

`dtruss` is a macOS tool that traces system calls and library calls. To use it:

```bash
# Trace NumPy function calls
PYTHONPATH=python python3 scripts/trace_function_calls.py --operation numpy --iterations 10 &
sudo dtruss -f -p $! | grep -E 'vDSP|Accelerate|BLAS'
```

### dtrace

`dtrace` is a more powerful tracing tool that allows custom scripts:

```dtrace
#!/usr/sbin/dtrace -s

pid$target:libAccelerate:*:entry
{
    @[ustack()] = count();
}

pid$target:libAccelerate:*:return
{
    @[probefunc] = count();
}
```

Usage:
```bash
sudo dtrace -s trace_accelerate.d -p $(pgrep -f 'python.*trace')
```

## NumPy Function Calls

### Expected Functions

NumPy on macOS likely calls:
- `vDSP_vsmul` - Vector-scalar multiply (our implementation uses this too)
- `vDSP_vmul` - Vector-vector multiply
- Other Accelerate framework functions

### Actual Calls

(To be updated with tracing results)

Based on analysis:
- NumPy links to Accelerate framework ✓
- NumPy uses Accelerate for BLAS/LAPACK operations
- NumPy may use different Accelerate functions than we expect

## Raptors Function Calls

### Current Implementation

- `scale_same_shape_f32` - Our SIMD kernel
- `accelerate_vsmul_f32` - Wrapper around `vDSP_vsmul`
- `parallel_scale_f32` - Parallel implementation

### Call Path Analysis

(To be updated with tracing results)

## Findings

### NumPy Dispatch Path

1. **Python ufunc** → C ufunc wrapper
2. **C wrapper** → Accelerate framework
3. **Accelerate** → Optimized assembly

### Raptors Dispatch Path

1. **Python method** → Rust FFI
2. **Rust dispatch** → SIMD/Accelerate/Scalar
3. **Implementation** → Optimized kernel

### Key Differences

1. **NumPy** uses ufunc system (universal functions)
2. **Raptors** uses direct dispatch per operation
3. **NumPy** may have additional runtime optimizations
4. **Raptors** has explicit backend selection

## Optimization Opportunities

### Based on Tracing Results

1. **Function call overhead** - Reduce indirection if present
2. **Dispatch path** - Optimize hot path
3. **Accelerate usage** - Match NumPy's function calls exactly

## Next Steps

1. Run actual tracing to see function calls
2. Compare NumPy vs Raptors call patterns
3. Identify any unnecessary indirection
4. Optimize dispatch logic based on findings

## Limitations

- Tracing requires root/sudo privileges
- May have performance overhead
- Need to filter noise from actual calls
- Some functions may be inlined or optimized away

## References

- macOS `dtruss` man page
- DTrace User Guide
- Accelerate Framework Reference

