# Non-Temporal Stores Analysis

## Overview

This document analyzes the use of non-temporal stores (or cache-friendly write strategies) for large arrays in the `float32 @ 2048²` scale operation.

## ARM NEON Limitations

Unlike x86 processors, ARM NEON does not have direct non-temporal store instructions. However, we can achieve similar effects by:

1. **Reducing write prefetching** - Don't aggressively prefetch write addresses for large arrays
2. **Using cache maintenance instructions** - Expensive, not recommended
3. **Relying on store buffer** - Let hardware handle write combining

## Implementation

### Strategy: Reduced Write Prefetching

For arrays larger than 4M elements (2048² = 4,194,304 elements), we:
- **Keep read prefetching** - Always beneficial for large sequential reads
- **Skip write prefetching** - Reduces cache pollution for large writes that won't be read back soon

### Code Changes

Modified `rust/src/simd/mod.rs` in the `scale_same_shape_f32` function:

```rust
// For very large arrays (2048² = 16MB), reduce write prefetching
// This simulates non-temporal store behavior by not aggressively
// prefetching writes, reducing cache pollution for large datasets
if len <= 4_000_000 {
    // Normal write prefetch for smaller arrays
    core::arch::asm!(
        "prfm pstl1keep, [{addr}]",
        addr = in(reg) ptr_out.add(i + prefetch_distance),
        options(nostack)
    );
}
// For >4M elements, skip write prefetch to reduce cache pollution
```

## Expected Benefits

1. **Reduced cache pollution** - Write buffers won't pollute L1 cache
2. **Better cache utilization** - More cache available for reads
3. **Lower memory bandwidth** - Less unnecessary cache traffic

## Expected Trade-offs

1. **Slightly slower writes** - Store buffer might be less optimized
2. **May not help if writes are read soon** - But for large arrays, reads are likely sequential and far ahead

## Benchmark Results

(To be updated with actual benchmark results)

### Test Configuration

- Array size: 2048×2048 = 4,194,304 elements (16 MB)
- Data type: float32
- Iterations: 100

### Results

- **With write prefetch**: TBD
- **Without write prefetch (non-temporal-like)**: TBD
- **NumPy baseline**: TBD

## Analysis

### Performance Impact

- If beneficial: Implement as default for >4M element arrays
- If neutral/negative: Keep write prefetching, document findings

### Cache Behavior

- Measure L1/L2 cache misses if possible
- Profile with Instruments to see cache behavior
- Compare with NumPy's cache usage patterns

## Future Work

1. Test with different array sizes (1024², 4096²)
2. Measure actual cache miss rates
3. Consider cache maintenance instructions if needed (expensive)
4. Test with different prefetch distances

## References

- ARM Architecture Reference Manual - Cache Operations
- ARM NEON Programmer's Guide
- Apple Silicon Performance Optimization Guide

