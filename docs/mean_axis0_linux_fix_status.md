# mean_axis0 Linux Fix Status

## Problem
`mean_axis0` operations are severely slow on Linux (ARM64), with speedups of 0.02x-0.29x compared to NumPy.

## Root Cause Analysis

### Current Code Path for 2048² float64

1. **BLAS Path** (lines 4442-4454): Tries `blas::axis0_enabled()` and `blas::current_backend().dgemv_axis0_sum()`
   - **Status**: Not available on Linux (requires `openblas` feature)
   - **Impact**: BLAS path is skipped

2. **SIMD Path** (lines 4458-4469): Tries `simd::reduce_axis0_columns_f64()`
   - **Status**: Should work - NEON implementation exists (lines 2082-2206 in simd/mod.rs)
   - **Issue**: May be returning `None` or not being called correctly
   - **NEON Implementation**: Has `#[target_feature(enable = "neon")]` and looks optimized

3. **Parallel SIMD Path** (lines 4471-4520): Uses parallel SIMD accumulation
   - **Status**: Recently optimized to avoid `matrixmultiply::dgemm`
   - **Issue**: Still slow (0.02x-0.03x), suggesting SIMD add_assign may not be working

4. **Sequential Fallback** (lines 4523-4544): Uses `simd::add_assign_inplace_f64()`
   - **Status**: NEON implementation exists (lines 2724-2738 in simd/mod.rs)
   - **Issue**: Performance suggests this path is being used but is slow

## Diagnostic Results

- **SIMD Enabled**: Yes (`simd_enabled()` returns `true`)
- **Backend Usage**: Not recorded (suggests metrics aren't being tracked for axis0 operations)
- **Performance**: 0.02x-0.03x speedup for 2048² float64
- **System**: Linux ARM64 (aarch64), Docker container

## Changes Made

1. **Updated 2048² parallel path** (rust/src/lib.rs lines 4471-4520):
   - Replaced slow `matrixmultiply::dgemm` with optimized parallel SIMD accumulation
   - Uses `simd::add_assign_inplace_f64()` for each chunk
   - Still not fast enough

2. **Created diagnostic script** (`scripts/check_blas_simd.py`):
   - Checks SIMD/BLAS availability
   - Tests operations to identify which backend is used

3. **Updated profiling script** (`scripts/profile_single.py`):
   - Added support for `mean_axis0` operation

## Next Steps

1. **Investigate why SIMD kernel returns None**:
   - Add logging/debug output to `simd::reduce_axis0_columns_f64()`
   - Verify NEON functions are being called correctly
   - Check if `#[target_feature(enable = "neon")]` is causing issues on aarch64

2. **Enable BLAS on Linux**:
   - Add `openblas` feature to default features or enable it in Docker build
   - Verify OpenBLAS is available in the environment

3. **Optimize NEON implementation**:
   - Review NEON `reduce_axis0_columns_f64()` implementation
   - Ensure proper vectorization and prefetching
   - Consider platform-specific optimizations

4. **Alternative approach**:
   - Use NumPy's implementation as a fallback for Linux until SIMD is fixed
   - Or implement a simpler, faster scalar path optimized for ARM64

## Current Status

- **Investigation**: Partially complete
- **Fixes Applied**: Parallel SIMD path optimization
- **Performance**: Still critical (0.02x-0.03x)
- **Priority**: HIGH - This is blocking significant performance improvements

