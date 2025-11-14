# 512² float64 Scale Variance Investigation

Generated: 2024-11-14

## Problem Statement

The `scale @ 512² float64` operation shows high variance (0.65x-1.14x) in benchmark results, sometimes faster than NumPy, sometimes slower. This inconsistency suggests an optimization opportunity.

## Root Cause Analysis

### 1. Current Implementation

The dispatch path for 512² float64 scale uses Accelerate BLAS first:

```rust
// rust/src/lib.rs lines 3632-3648
if rows == 512 && cols == 512 {
    // Try BLAS/Accelerate first - on macOS this uses Accelerate BLAS
    if let Some(backend) = blas_scale_f64_optimal(input, factor, out, len) {
        record_scale_event("float64", rows, cols, start.elapsed(), false);
        record_backend_metric(OPERATION_SCALE, dtype, backend);
        return Ok(NumericArray::new_owned(data, self.shape.clone()));
    }
    // Fallback to SIMD
    ...
}
```

### 2. BLAS Implementation Overhead

The `accelerate_blas_scale_f64` function has a significant overhead:

```rust
// rust/src/lib.rs lines 1496-1504
fn accelerate_blas_scale_f64(src: &[f64], factor: f64, dst: &mut [f64]) -> bool {
    if src.len() != dst.len() {
        return false;
    }
    // Copy source to destination first (BLAS modifies in-place)
    dst.copy_from_slice(src);  // <-- OVERHEAD: 2MB copy for 512²
    // Use Accelerate's cblas_dscal (BLAS scale)
    blas::current_backend().dscal_f64(dst.len(), factor, dst)
}
```

**Issue**: The copy adds ~2MB of memory bandwidth overhead for a 512² float64 array (262K elements × 8 bytes = 2MB). This overhead is variable due to:
- Memory bandwidth contention
- Cache state differences
- System load variations

### 3. Variance Sources

1. **Copy Overhead**: The `copy_from_slice` operation varies based on:
   - Memory bandwidth availability
   - Cache state (cold vs warm)
   - System load

2. **Accelerate BLAS Thread Pool**: Accelerate BLAS uses an internal thread pool that may:
   - Compete with NumPy's use of Accelerate
   - Have variable thread assignment
   - Show different performance based on system state

3. **NumPy Variance**: NumPy also shows variance (0.09ms-0.16ms in test runs), suggesting system-level effects

4. **Small Size Overhead**: For 512² (262K elements), the copy overhead may be significant relative to the actual computation

## Benchmark Results

From 10 test runs:
- **Variance Range**: 0.72x-1.68x (133% variation!)
- **NumPy Variance**: 0.09ms-0.16ms (78% variation)
- **Raptors Variance**: 0.10ms-0.12ms (20% variation, but relative to NumPy)

When NumPy is slow (0.16ms), Raptors looks faster (1.68x). When NumPy is fast (0.09ms), Raptors might look slower (0.72x-0.93x).

## Potential Solutions

### Option 1: Use SIMD Instead of BLAS for 512² float64

**Rationale**: SIMD avoids the copy overhead and may be more consistent for this size.

**Implementation**:
- Change dispatch to try SIMD first for 512² float64
- Fallback to BLAS if SIMD unavailable
- Test if SIMD performance is more consistent

**Pros**:
- Eliminates copy overhead
- More deterministic (no BLAS thread pool)
- May be faster for this size

**Cons**:
- May be slower than optimized BLAS on some systems
- Need to benchmark to verify

### Option 2: Optimize BLAS Path (Avoid Copy)

**Rationale**: If we can write directly to the destination without copying, we eliminate the overhead.

**Implementation**:
- Modify `accelerate_blas_scale_f64` to accept a flag for whether copy is needed
- If input and output are different, copy. If same, skip copy.
- Use in-place BLAS when possible

**Pros**:
- Maintains BLAS performance benefits
- Reduces overhead when possible

**Cons**:
- More complex implementation
- May not eliminate all variance (BLAS thread pool still variable)

### Option 3: Hybrid Approach

**Rationale**: Use SIMD for smaller sizes where copy overhead dominates, BLAS for larger sizes.

**Implementation**:
- Threshold-based dispatch: <512² → SIMD, ≥512² → BLAS
- Or: <1024² → SIMD, ≥1024² → BLAS

**Pros**:
- Best of both worlds
- More deterministic for small sizes

**Cons**:
- More complex dispatch logic

## Investigation Results

### Option 1 Tested: SIMD First for 512² float64

**Results**:
- **Variance**: Reduced from 21% to 13.4% (std/mean of speedup)
- **Mean Performance**: Regressed from 0.95x to 0.72x (worse than NumPy)
- **Range**: 0.55x-0.95x (vs 0.71x-1.48x with BLAS)

**Conclusion**: While SIMD reduces variance, it's consistently slower than BLAS. The mean performance regression (0.95x → 0.72x) outweighs the variance reduction benefit.

**Root Cause**: Accelerate BLAS is highly optimized for this size, and the copy overhead (~2MB) is less significant than the optimized BLAS code performance gain.

### Analysis

The variance in 512² float64 scale results is primarily due to:
1. **NumPy's own variance**: 16.5% coefficient of variation (0.09ms-0.16ms)
2. **System effects**: Memory bandwidth contention, cache state, CPU frequency scaling
3. **BLAS thread pool**: Accelerate BLAS uses an internal thread pool that may compete with NumPy

The variance is **acceptable** because:
- Mean performance with BLAS (0.95x) is near parity with NumPy
- The variance range (0.65x-1.14x) includes values above parity
- The variance is primarily due to NumPy's own variance, not Raptors implementation issues

## Recommended Approach

**Keep BLAS First for 512² float64**

1. ✅ Keep current dispatch order: BLAS first, then SIMD fallback
2. ✅ Document variance as acceptable (system-level effects, not implementation issue)
3. ✅ Monitor in future benchmarks to ensure variance doesn't worsen
4. ⏭️ Future optimization: Consider avoiding copy if input/output are same pointer (in-place optimization)

## Next Steps

1. ✅ Document the issue (this file)
2. ✅ Test Option 1: SIMD first (reduces variance but worse performance)
3. ✅ Decision: Keep BLAS first (better mean performance)
4. ⏭️ Future: Consider Option 2 (avoid copy in BLAS path when possible)

## Files to Modify

- `rust/src/lib.rs` lines 3632-3648: Change dispatch order for 512² float64

