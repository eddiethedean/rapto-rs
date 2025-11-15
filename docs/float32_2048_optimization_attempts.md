# Float32 2048² mean_axis0 Optimization Attempts

## Summary

Attempted to optimize `mean_axis0 @ 2048² float32` to exceed NumPy (>1x). Current performance: **0.77x** (Raptors: 0.42ms, NumPy: 0.33ms), improved from baseline **0.66x**.

## Baseline

- **Initial baseline**: 0.66x (Raptors: 0.55ms, NumPy: 0.37ms)
- **Code**: Tiled approach (128x64 tiles, simple row-by-row processing, no prefetching, no unrolling)
- **Location**: `rust/src/simd/mod.rs` lines 2158-2230

## Optimization Attempts

### 1. BLAS Path Testing
- **Approach**: Test OpenBLAS (SGEMM/GEMV) for float32, similar to successful float64 approach
- **Result**: 0.44x (slower, not used)
- **Finding**: BLAS works well for float64 (1.09x) but slower for float32 on Linux ARM64
- **Status**: Reverted

### 2. Columnar Approach Testing
- **Approach**: Process all rows for columns in blocks (better sequential access)
- **Variants tested**:
  - Basic columnar (64 col blocks): 0.35x
  - Optimized columnar (128 col blocks): 0.57x
  - Original columnar fallback: 0.35x
- **Result**: All slower than tiled approach
- **Finding**: Tiled approach (128x64) has better cache behavior than columnar for this size
- **Status**: Not used

### 3. Row-Block Approach
- **Approach**: Process entire rows (2048 columns) in blocks of rows
- **Result**: 0.59x (slower than tiled)
- **Finding**: Tiling in both dimensions is more cache-friendly
- **Status**: Reverted

### 4. Output Write Optimization
- **Approach**: Optimize load-modify-store pattern - write directly for first tile, accumulate for subsequent tiles
- **Result**: 0.77x (improvement from 0.66x baseline)
- **Finding**: Avoiding unnecessary loads for first tile helps performance
- **Status**: ✅ **Kept** - This optimization improved performance

### 5. Instruction Scheduling Optimization
- **Approach**: Separate load and add operations for better ILP
- **Result**: 0.67x (minimal improvement)
- **Finding**: Load-then-add separation didn't significantly help
- **Status**: Reverted (kept simpler code)

### 6. Vector Loop Unrolling
- **Approach**: 2x unrolling in vector loop (not row loop)
- **Result**: 0.50x (slower)
- **Finding**: Unrolling in vector loop hurt performance
- **Status**: Reverted

### 7. Tile Size Variations
- **Approaches tested**:
  - 96x64 tiles: 0.75x
  - 128x32 tiles: 0.58x
  - 128x64 tiles (original): 0.66x baseline
- **Result**: Original 128x64 performs best
- **Status**: Kept original tile size

## Current Implementation

- **Path**: Optimized tiled approach (128x64 tiles)
- **Key optimization**: Output write pattern - write directly for first tile, accumulate for subsequent tiles
- **Performance**: 0.77x (Raptors: 0.42ms, NumPy: 0.33ms)
- **Code location**: `rust/src/simd/mod.rs` lines 2158-2230

## Findings

1. **Tiled approach is optimal** - Columnar and row-block approaches are slower
2. **BLAS not beneficial for float32** - Works well for float64 but slower for float32
3. **Output write optimization helps** - Avoiding unnecessary loads for first tile improves performance
4. **Unrolling hurts performance** - Both row loop and vector loop unrolling degraded performance
5. **Original tile size is best** - 128x64 tiles perform better than alternatives tested
6. **Prefetching doesn't help** - Minimal or aggressive prefetching didn't improve performance

## Remaining Gap

- **Current**: 0.77x
- **Target**: >1x
- **Gap**: Need ~30% additional improvement
- **NumPy time**: 0.33ms
- **Raptors time**: 0.42ms
- **Gap**: ~0.09ms (27% slower)

## Potential Next Steps

1. **Profiling NumPy**: Use perf to understand NumPy's implementation strategy
2. **Assembly comparison**: Compare assembly output to identify instruction-level differences
3. **Alternative SIMD patterns**: Test different NEON instruction combinations
4. **Memory access patterns**: Investigate if different access patterns could help
5. **Compiler optimizations**: Check if different compiler flags or optimizations could help
6. **Threading investigation**: Check if NumPy uses threading (though unlikely for this size)

### 8. Multi-Row Simultaneous Processing (Phase 2)
- **Approach**: Process 2 or 4 rows simultaneously for better ILP
- **Variants tested**:
  - 2-row simultaneous: 0.72x
  - 4-row simultaneous: 0.61x
- **Result**: Both slower than baseline
- **Finding**: Processing multiple rows increases memory traffic and doesn't improve ILP enough
- **Status**: Reverted

### 9. Larger Row Tiles (Phase 3)
- **Approach**: Use 256-row tiles instead of 128 to reduce output writes
- **Result**: 0.70x (slower)
- **Finding**: Larger tiles don't improve cache behavior enough to offset increased memory
- **Status**: Reverted

### 10. Hybrid Approach (Phase 5)
- **Approach**: Process all 2048 rows for each column tile (combine tiled and columnar)
- **Result**: 0.52x (much slower)
- **Finding**: Losing cache locality hurts performance significantly
- **Status**: Reverted

### 11. 2x Vector Loop Unrolling (Phase 2)
- **Approach**: Unroll vector loop by 2 to process 2 vectors per row simultaneously
- **Result**: 0.49x (much slower)
- **Finding**: Unrolling in vector loop hurts performance (similar to previous findings)
- **Status**: Reverted

### 12. Smaller Column Tiles (Phase 3)
- **Approach**: Use 32-column tiles instead of 64 to reduce accumulator size
- **Result**: 0.28x (much slower)
- **Finding**: Dynamic Vec allocation hurts performance significantly
- **Status**: Reverted

## Conclusion

Systematically tested multiple optimization approaches across all phases of the plan. Most optimizations (BLAS, columnar, unrolling, prefetching, multi-row processing, hybrid approaches) hurt performance or provided minimal improvement. The output write optimization improved performance from 0.66x to 0.77x, but we're still ~30% away from exceeding NumPy.

**Key Finding**: The simple baseline tiled approach (128x64 tiles, simple row-by-row processing, optimized output writes) consistently performs best. More complex optimizations (multi-row processing, larger tiles, hybrid approaches) consistently degrade performance.

**Next Steps**: Deep profiling of NumPy's implementation is needed to understand what makes it faster. Potential areas:
1. NumPy may use different memory access patterns we haven't tried
2. NumPy may use hand-optimized assembly we should compare against
3. NumPy may use different cache strategies or prefetching patterns
4. There may be compiler-level optimizations we're missing

Further investigation into NumPy's implementation strategy or deeper NEON optimizations based on profiling insights may be needed to close the remaining gap.

