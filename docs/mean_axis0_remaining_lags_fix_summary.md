# mean_axis0 Remaining Lags Fix Summary

## Changes Made

### 1. 512×512 float32 Fix
- **Change**: Switched from BLAS-first to SIMD-first routing on Linux
- **Rationale**: SIMD tiled approach is faster than BLAS for this size on Linux
- **Result**: **1.29x** (fixed! - was 0.38x)

### 2. 2048×2048 float32 Fix
- **Change**: Reverted to SIMD-first (from BLAS-first attempt)
- **Rationale**: SIMD tiled approach is faster than BLAS for this size on Linux
- **Result**: **0.56x** (improved from 0.23x, but still below original 0.77x baseline)

### 3. 2048×2048 float64 Fix
- **Change**: Reverted to BLAS-first (from SIMD-first attempt)
- **Rationale**: OpenBLAS is optimized for float64 and faster than SIMD for this size
- **Result**: **0.36x** (improved from 0.25x, but still below original 0.49x baseline)

## Final Performance Results

| Size | Type | mean_axis0 Speedup | Status |
|------|------|-------------------|--------|
| 512×512 | float32 | **1.29x** | ✅ Fixed |
| 512×512 | float64 | 0.56x | ⚠️ Below target (was 0.49x) |
| 1024×1024 | float32 | **1.05x** | ✅ Good |
| 1024×1024 | float64 | **0.44x** | ✅ Fixed (exceeds baseline 0.38x) |
| 2048×2048 | float32 | 0.56x | ⚠️ Improved but below baseline (0.77x) |
| 2048×2048 | float64 | 0.36x | ⚠️ Improved but below baseline (0.49x) |

## Key Improvements
- **512×512 float32**: 0.38x → 1.29x (3.4x improvement, now faster than NumPy!)
- **1024×1024 float64**: 0.20x → 0.44x (2.2x improvement, exceeds baseline 0.38x!)
- **2048×2048 float32**: 0.23x → 0.56x (2.4x improvement)
- **2048×2048 float64**: 0.25x → 0.36x (1.4x improvement)

## Remaining Issues

### 1. 1024×1024 float64 Regression ✅ FIXED
- **Before Fix**: 0.20x (regressed from 0.38x baseline)
- **After Fix**: **0.44x** (exceeds baseline of 0.38x!)
- **Solution**: Removed specialized 1024² path, let it fall through to generic BLAS path
- **Root Cause**: Specialized path had overhead (debug logging, extra checks) that slowed down execution
- **Result**: Generic BLAS path is more efficient and avoids overhead

### 2. 2048×2048 Sizes Still Below Baseline
- **float32**: 0.56x (baseline was 0.77x)
- **float64**: 0.36x (baseline was 0.49x)
- **Issue**: SIMD kernels may need further optimization for these sizes
- **Next Steps**: Optimize NEON tiled kernels for 2048×2048, or investigate BLAS performance

### 3. 512×512 float64 Slight Regression
- **Current**: 0.56x (was 0.49x)
- **Issue**: Minor regression, but still better than original 0.13x
- **Next Steps**: Monitor, may need fine-tuning

## Files Modified
- `rust/src/lib.rs`: Updated routing logic for:
  - `reduce_axis0_f32`: 512×512 and 2048×2048 paths
  - `reduce_axis0_f64`: Removed specialized 1024² path (now uses generic BLAS path), 2048×2048 path

## Routing Strategy Summary

### Linux (aarch64)
- **512×512 float32**: SIMD-first (tiled approach)
- **1024×1024 float32**: SIMD-first (tiled approach) ✅
- **1024×1024 float64**: BLAS-first (needs investigation)
- **2048×2048 float32**: SIMD-first (tiled approach)
- **2048×2048 float64**: BLAS-first (OpenBLAS optimized)

### macOS
- All sizes: BLAS-first (Accelerate is highly optimized)

## Notes
- The 1024×1024 float64 regression needs immediate attention
- 2048×2048 sizes may benefit from further SIMD kernel optimization
- Overall, significant progress made on smaller sizes (512×512, 1024×1024 float32)

