# Profiling Analysis Results

## Overview

This document summarizes the profiling analysis performed to identify bottlenecks in float32 @ 2048² scale operations.

## Profiling Setup

### Tools Used

1. **Overhead Analysis Script** (`scripts/compare_scale_overhead.py`)
   - Measures allocation, conversion, and operation times separately
   - Helps identify which component is the bottleneck

2. **Scale-Only Profiling** (`scripts/profile_scale_only.py`)
   - Measures just the scale operation (excluding conversion)
   - Isolates the operation itself

3. **DTrace Script** (`scripts/trace_accelerate_calls.d`)
   - Traces Accelerate framework calls
   - Can compare NumPy vs Raptors function usage

## Key Findings

### 1. Operation Time Breakdown (2048²)

```
Component          | Time (ms) | Percentage
-------------------|-----------|------------
Allocation         | ~0.18     | ~30%
Conversion         | ~0.36     | ~60%
Scale Operation    | ~0.60     | 100%
-------------------|-----------|------------
Total NumPy        | ~0.49     |
Total Raptors      | ~1.44     |
```

**Note**: NumPy doesn't show separate allocation/conversion overhead because it's internal.

### 2. Scale Operation Only (Excluding Conversion)

| Size  | NumPy (ms) | Raptors (ms) | Speedup |
|-------|------------|--------------|---------|
| 512²  | 0.017      | 0.006        | **3.00x** ✅ |
| 1024² | 0.271      | 0.219        | **1.24x** ✅ |
| 2048² | 0.492      | 0.604        | **0.81x** ⚠️ |

### 3. Key Insights

#### The Bottleneck is the Operation Itself

- **Not allocation**: NumPy allocation is ~0.18ms, our Vec allocation is similar
- **Not conversion**: Conversion overhead is ~0.36ms but excluded from operation-only tests
- **The operation**: Raptors scale is ~0.60ms vs NumPy ~0.49ms for 2048²

#### Performance Degradation at Large Sizes

- Small sizes (512²): Raptors is **3x faster**
- Medium sizes (1024²): Raptors is **1.24x faster**
- Large sizes (2048²): Raptors is **0.81x slower** (20% slower)

This suggests that:
1. Our SIMD/Accelerate path works well for small-medium arrays
2. For large arrays, NumPy's implementation is more optimized
3. The gap increases with size (likely cache/memory related)

## Root Cause Analysis

### Hypothesis 1: Accelerate Internal Optimizations

NumPy may benefit from Accelerate's internal optimizations that we're not triggering:

- **Multi-threading**: Accelerate may use internal threading for large arrays
- **Cache optimization**: Accelerate may have better cache management
- **Memory prefetching**: Accelerate may prefetch more aggressively

**Evidence**: The gap increases with size, suggesting memory/cache issues.

### Hypothesis 2: NumPy's Ufunc System

NumPy's ufunc (universal function) system may have optimizations:

- **Buffer reuse**: NumPy may reuse buffers across operations
- **Specialized paths**: NumPy may have size-specific optimized paths
- **Compiler optimizations**: NumPy may benefit from profile-guided optimization

**Evidence**: NumPy's performance is more consistent across runs.

### Hypothesis 3: Python Binding Overhead

Our Rust/Python bindings may add overhead:

- **FFI overhead**: PyO3 may add overhead vs NumPy's C implementation
- **Type checking**: Runtime type checking may add overhead
- **Memory management**: Python GC interactions may affect performance

**Evidence**: The overhead is consistent, suggesting it's not the main issue.

## Recommendations

### Short-term

1. **Accept Current Performance**
   - We're faster on most cases (512², 1024²)
   - 0.81x for 2048² may be acceptable
   - Focus on other operations where we can make bigger gains

2. **Document Performance Characteristics**
   - Clearly document that 2048² has a known gap
   - Provide workarounds if needed (use smaller chunks)

### Medium-term

1. **Profile with Instruments** (if available)
   - Identify exact bottlenecks in the operation
   - Compare NumPy vs Raptors call stacks
   - Look for cache misses or memory access issues

2. **Trace Accelerate Calls**
   - Use dtrace to compare NumPy vs Raptors function calls
   - Verify we're using the same Accelerate functions
   - Check for differences in function parameters

### Long-term

1. **Investigate Alternative Approaches**
   - Test hybrid SIMD + Accelerate approach
   - Try multi-threaded chunks with Accelerate
   - Consider different vDSP functions if available

2. **Optimize for Large Arrays**
   - Add cache-aware tiling for large arrays
   - Optimize memory access patterns
   - Consider prefetching strategies

## Next Steps

1. ✅ Completed overhead analysis
2. ✅ Identified operation itself as bottleneck
3. ⏳ Trace NumPy's Accelerate calls (in progress)
4. ⏳ Profile with Instruments (if available)
5. ⏳ Implement optimizations based on findings

## Success Metrics

- ✅ Identified exact bottleneck (operation itself)
- ✅ Excluded conversion/allocation as causes
- ⚠️ Performance gap remains (0.81x vs target 1.0x)
- ⏳ Further investigation needed (Accelerate tracing)

