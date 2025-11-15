# Optimization Attempts Summary - mean_axis0 2048² float32

## Goal

Close the 30% performance gap between Raptors (0.73x NumPy) and NumPy for mean_axis0 @ 2048² float32.

## Hypothesis

The expensive read-modify-write pattern on the output buffer (16 writes per column block with 128-row tiles) was causing the performance gap.

## Tested Approaches

### 1. Pure Columnar Approach
**Implementation**: Process ALL 2048 rows for each 64-column block before writing output.

**Result**: **1.98x slower** than NumPy (much worse!)

**Analysis**: 
- Eliminated read-modify-write pattern entirely
- But poor cache locality - by the time we process all rows, first rows evicted from cache
- Sequential column access across entire matrix = many cache misses

### 2. Larger Row Tiles (512 rows)
**Implementation**: Increase row tile size from 128 to 512 rows, reducing output writes from 16 to 4 per column block.

**Result**: **1.93x slower** than NumPy (much worse!)

**Analysis**:
- Reduced output write frequency
- But 512 rows * 64 cols * 4 bytes = 128KB per tile (too large for L1 cache)
- Causes cache thrashing and eviction

### 3. Baseline 128×64 Tiles
**Implementation**: Current tiled approach with 128-row × 64-column tiles.

**Result**: **0.73x NumPy** (best performance!)

**Analysis**:
- 128 rows * 64 cols * 4 bytes = 32KB per tile (perfect fit for L1 cache)
- Good cache locality - processes tiles that fit in cache
- 16 writes per column block, but cache-friendly access pattern is worth it
- Compiler optimizes to 16-column parallelism automatically

## Key Insights

1. **Cache locality is more important than write frequency**: The tiled approach with more writes (16) is faster than columnar with fewer writes (1), because cache-friendly access patterns matter more.

2. **Tile size must fit in cache**: 128×64 tiles (32KB) fit perfectly in L1 cache. Larger tiles (512 rows = 128KB) cause cache misses.

3. **Current approach is already well-optimized**: The 128×64 tiled approach with compiler-driven 16-column parallelism is near-optimal for cache behavior.

## Remaining Gap Analysis

**Current Status**: Raptors 0.73x NumPy (~0.46ms vs ~0.34ms)

**The 30% gap likely comes from**:

1. **Compiler flags**: NumPy may use different optimization flags or architecture-specific optimizations
2. **Algorithm differences**: NumPy may use a fundamentally different algorithm (iterator-based vs tiled)
3. **Memory alignment**: NumPy may have better memory alignment guarantees
4. **Hand-tuned assembly**: NumPy may use hand-tuned assembly for this specific operation
5. **Different SIMD strategy**: NumPy may use different NEON instruction patterns

## Conclusion

The current tiled approach (128×64) is already optimal for our implementation strategy. The read-modify-write pattern on the output buffer is NOT the bottleneck - cache locality is.

To close the remaining 30% gap, we need to investigate:
- NumPy's exact compiler flags and build configuration
- NumPy's actual implementation algorithm
- Potential hand-tuned assembly or specialized code paths

## Recommendations

1. **Accept current performance**: 0.73x is competitive and many optimizations have been tested
2. **Investigate NumPy's build flags**: Compare compiler flags and optimization levels
3. **Consider NumPy's approach**: Investigate if NumPy uses iterator-based algorithms or other strategies
4. **Focus on other operations**: This operation may be near optimal already

