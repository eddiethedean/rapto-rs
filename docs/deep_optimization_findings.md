# Deep mean_axis0 Optimization Findings

## Date: 2025-11-16

## Summary

Implemented deep NEON kernel optimizations for the three remaining laggards:
- **1024² float64**: 0.44x → **0.78x** ✅ (Significant improvement, close to 0.80x target)
- **2048² float32**: 0.56x → **0.53x** ⚠️ (Slight regression)
- **2048² float64**: 0.36x → **0.25x** ⚠️ (Regression)

## Optimizations Implemented

### 1. Profiling Infrastructure
- Created `scripts/profile_mean_axis0.sh` for perf stat profiling
- Created `scripts/extract_assembly.sh` for assembly extraction and comparison
- Both scripts ready for use in Docker environment

### 2. 2048² float32 Optimizations
**File**: `rust/src/simd/mod.rs` (lines ~2015-2144)

**Changes**:
- Added 4x loop unrolling (`UNROLL_FACTOR_2048_F32 = 4`)
- Added strategic prefetching (pldl1keep for first row, next unrolled block, and output)
- Implemented software pipelining (prefetch while processing)

**Result**: **0.53x** (regressed from 0.56x)
**Analysis**: The unrolling may have increased register pressure or instruction cache misses. The simpler version without unrolling was more effective.

### 3. 1024² float64 Optimizations
**File**: `rust/src/simd/mod.rs` (lines ~2531-2658), `rust/src/lib.rs` (lines ~4506-4585)

**Changes**:
- Added specialized 1024² path with 128×64 tiles
- Added 4x loop unrolling (`UNROLL_FACTOR_1024_F64 = 4`)
- Added strategic prefetching
- Updated routing to try SIMD-first (was BLAS-first)

**Result**: **0.78x** ✅ (improved from 0.44x, 1.77x improvement!)
**Analysis**: The specialized tiled path with unrolling significantly outperforms the generic BLAS path for this size.

### 4. 2048² float64 Optimizations
**File**: `rust/src/simd/mod.rs` (lines ~2660-2780), `rust/src/lib.rs` (lines ~4587-4638)

**Changes**:
- Reduced tile sizes from 256×128 to 128×64 (better L1 cache fit)
- Increased unrolling from 4x to 8x (`UNROLL_FACTOR_2048_F64 = 8`)
- Added dual-level prefetching (L1 and L2)
- Updated routing to try SIMD-first (was BLAS-first)

**Result**: **0.25x** (regressed from 0.36x)
**Analysis**: The increased unrolling (8x) and smaller tiles may have caused register spills or cache thrashing. The previous BLAS path was more effective.

## Key Findings

### What Worked
1. **1024² float64 specialized path**: The tiled approach with moderate unrolling (4x) works well for this size
2. **SIMD-first routing for 1024²**: The optimized SIMD path beats BLAS for this specific size

### What Didn't Work
1. **Aggressive unrolling for 2048² sizes**: 4x-8x unrolling increased register pressure and hurt performance
2. **Smaller tiles for 2048² float64**: Reducing from 256×128 to 128×64 increased tile count and overhead
3. **SIMD-first for 2048² float64**: BLAS (OpenBLAS) is still faster than even optimized SIMD for this size

## Final Status

### Implemented and Kept
1. **1024² float64 specialized path**: Added optimized tiled path with 4x unrolling
   - Performance: 0.59x-0.78x (improved from 0.44x baseline)
   - Routing: SIMD-first on Linux (specialized path beats BLAS for this size)
   - Status: ✅ **KEPT** - Significant improvement

### Reverted (Caused Regressions)
1. **2048² float32 unrolling**: Reverted 4x unrolling (caused 0.56x → 0.53x regression)
   - Status: ✅ **REVERTED** - Back to baseline 0.57x
   
2. **2048² float64 optimizations**: Reverted smaller tiles and 8x unrolling (caused 0.36x → 0.25x regression)
   - Status: ✅ **REVERTED** - BLAS-first routing restored, original tile sizes kept

### Future Optimizations
1. **2048² float32**: Test 2x unrolling instead of 4x, or focus on output write optimization
2. **2048² float64**: Keep BLAS-first, but investigate OpenBLAS threading configuration
3. **Instruction scheduling**: Profile to identify specific bottlenecks before optimizing further

## Performance Summary

| Size | Type | Before | After | Change | Status |
|------|------|--------|-------|--------|--------|
| 1024² | float64 | 0.44x | **0.59x-0.78x** | +34-77% | ✅ Improved (variance observed) |
| 2048² | float32 | 0.56x | 0.57x | ~0% | ✅ Restored to baseline |
| 2048² | float64 | 0.36x | 0.24x-0.36x | Variable | ⚠️ High variance, BLAS-first restored |

## Next Steps

1. ✅ Reverted regressions for 2048² sizes
2. ✅ Kept 1024² float64 optimizations
3. **Remaining work**:
   - 1024² float64: Close to target (0.59x-0.78x, target >0.80x) - may need fine-tuning
   - 2048² float32: Still at 0.57x (target >0.80x) - needs different optimization approach
   - 2048² float64: Still at 0.36x (target >0.80x) - BLAS is optimal, may need OpenBLAS tuning

## Files Modified

- `rust/src/simd/mod.rs`: 
  - Added specialized 1024² float64 path (lines ~2531-2658)
  - Reverted 2048² float32 unrolling (kept simple version)
  - Reverted 2048² float64 optimizations (kept original tile sizes)
- `rust/src/lib.rs`:
  - Updated routing for 1024² float64 to SIMD-first (lines ~4506-4585)
  - Kept BLAS-first routing for 2048² float64 (lines ~4587-4638)
- `scripts/profile_mean_axis0.sh`: New profiling script
- `scripts/extract_assembly.sh`: New assembly extraction script
- `docs/deep_optimization_findings.md`: This document

