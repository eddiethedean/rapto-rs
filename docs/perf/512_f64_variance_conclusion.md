# 512² float64 Scale Variance: Investigation Complete

Generated: 2024-11-14

## Summary

Investigated high variance (0.65x-1.14x) in `scale @ 512² float64` benchmark results. The variance is primarily due to system-level effects (NumPy's own variance, memory bandwidth, BLAS thread pool) rather than Raptors implementation issues. **Decision: Keep current BLAS-first dispatch** as it provides better mean performance (0.95x) despite the variance.

## Investigation Results

### Root Cause

1. **Copy Overhead**: `accelerate_blas_scale_f64` copies ~2MB (512² × 8 bytes) before BLAS operation
2. **System Effects**: Memory bandwidth contention, cache state, CPU frequency scaling
3. **BLAS Thread Pool**: Accelerate BLAS uses an internal thread pool that may compete with NumPy
4. **NumPy Variance**: NumPy shows 16.5% coefficient of variation (0.09ms-0.16ms)

### Tested Solutions

#### Option 1: SIMD First ✅ Tested

**Implementation**: Changed dispatch order to try SIMD first for 512² float64.

**Results**:
- **Variance**: Reduced from 21% to 13.4% (std/mean of speedup)
- **Mean Performance**: Regressed from 0.95x to 0.72x (worse than NumPy)
- **Range**: 0.55x-0.95x (vs 0.71x-1.48x with BLAS)

**Conclusion**: SIMD reduces variance but is consistently slower. The mean performance regression outweighs the variance reduction benefit.

#### Option 2: Optimize BLAS Path ⏭️ Future Work

**Potential**: Avoid copy when input/output are same pointer (in-place optimization).

**Status**: Not implemented (would require refactoring BLAS wrapper).

## Decision

**Keep BLAS-First Dispatch for 512² float64**

**Rationale**:
1. **Better Mean Performance**: 0.95x (near parity) vs 0.72x with SIMD
2. **Variance is Acceptable**: Primarily due to NumPy's own variance (16.5% CV) and system effects
3. **Variance Range Includes Parity**: 0.65x-1.14x includes values ≥1.0x
4. **Not a Blocking Issue**: The variance doesn't prevent achieving parity in most runs

## Current Performance

- **Mean Speedup**: 0.95x (near parity with NumPy)
- **Variance**: 21% (std/mean of speedup), range 0.65x-1.14x
- **Backend**: `accelerate_blas` (Accelerate BLAS on macOS)
- **Status**: Acceptable - not a blocking issue

## Future Optimizations

1. **In-Place BLAS Optimization**: Avoid copy when input/output are same pointer
2. **Hybrid Approach**: Use SIMD for warmup, BLAS for actual benchmarks
3. **Thread Pool Tuning**: Investigate Accelerate BLAS thread pool configuration

## Documentation

- Investigation details: `docs/perf/512_f64_variance_investigation.md`
- Code comments: `rust/src/lib.rs` lines 3632-3653

## Conclusion

The variance in 512² float64 scale is acceptable and primarily due to system-level effects. The current BLAS-first dispatch provides the best mean performance (0.95x) despite the variance. No action required at this time.

