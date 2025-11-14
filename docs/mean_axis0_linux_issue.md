# mean_axis0 Linux Performance Issue

## Problem

`mean_axis0` operations are severely slow on Linux (ARM64), with speedups of 0.02x-0.29x compared to NumPy.

## Root Cause Analysis

### Code Path for 2048² float64

1. **BLAS Path** (lines 4442-4454): Tries `blas::current_backend().dgemv_axis0_sum()`
   - Likely not available or not working on Linux
   - Returns `None` if unavailable

2. **SIMD Path** (lines 4458-4469): Tries `simd::reduce_axis0_columns_f64()`
   - Likely returning `None` on Linux/ARM64
   - NEON implementation may not be working correctly

3. **Parallel Path** (lines 4473-4486): Tries `reduce_axis0_parallel_matrix_f64()`
   - Also slow (0.06x)
   - May be using slow matrixmultiply path

4. **Sequential Fallback** (lines 4489-4510): Uses `simd::add_assign_inplace_f64()`
   - Very slow (0.06x)
   - SIMD add_assign may not be optimized for ARM64

## Investigation Needed

1. **Check if BLAS is enabled**: Verify `blas::axis0_enabled()` returns true on Linux
2. **Check SIMD kernel**: Verify `simd::reduce_axis0_columns_f64()` works on ARM64
3. **Profile the slow path**: Use `perf` to identify the bottleneck
4. **Compare with macOS**: Check if the same code path works on macOS

## Potential Solutions

1. **Fix SIMD kernel**: Ensure `simd::reduce_axis0_columns_f64()` works on ARM64/NEON
2. **Enable BLAS**: Ensure OpenBLAS is available and working for axis0 operations
3. **Optimize parallel path**: Fix `reduce_axis0_parallel_matrix_f64()` to use faster algorithm
4. **Platform-specific optimization**: Use different code path for Linux vs macOS

## Current Status

- **Attempted fixes**: Added parallel path fallback, optimized SIMD accumulation
- **Result**: Still slow (0.06x-0.07x)
- **Conclusion**: Requires deeper investigation of SIMD/BLAS availability and correctness on Linux

## Next Steps

1. Profile with `perf` to identify exact bottleneck
2. Check if BLAS is enabled: `blas::axis0_enabled()`
3. Check if SIMD kernel works: `simd::reduce_axis0_columns_f64()`
4. Compare with working macOS implementation
5. Consider platform-specific code paths

