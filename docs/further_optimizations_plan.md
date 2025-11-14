# Plan: Further Optimizations for Remaining Performance Gaps

## Current Status After Fixes

After implementing the initial fixes, we have:
- **31/36 operations (86.1%) faster than NumPy**
- **5 operations still below 1.0×** (all very close)

### Remaining Gaps

1. **float64 (512, 512) scale**: 0.74× (gap: +0.040ms) - Improved from 0.69×
2. **float64 (1024, 1024) mean**: 0.93× (gap: +0.010ms) - Improved from 0.73×
3. **float64 (2048, 2048) mean_axis0**: 0.96× (gap: +0.023ms) - Improved from 0.92×
4. **float32 (2048, 2048) scale**: 0.95× (gap: +0.018ms) - New issue
5. **float32 (1024, 1024) scale**: 0.99× (gap: +0.002ms) - Essentially at parity

## Root Cause Analysis

### Issue 1: float64 (512, 512) scale (0.74×)

**Current Implementation**:
- Tries SIMD first for exactly 512²
- Falls back to BLAS (requires `out.copy_from_slice(input)` copy)
- Falls back to scalar

**Problem**: 
- SIMD path might be slower than expected for 512² (262K elements)
- The `copy_from_slice` overhead before BLAS is significant
- NEON kernel uses 4x unrolling (32 elements at a time), which should be good

**Potential Solutions**:
1. **Skip BLAS entirely for 512²** - If SIMD is available, use it directly without BLAS fallback
2. **Optimize NEON kernel for 512²** - Add specific handling for this size
3. **Use in-place BLAS if possible** - Avoid the copy overhead
4. **Increase unrolling in NEON kernel** - Try 6x or 8x unrolling for better throughput

### Issue 2: float64 (1024, 1024) mean (0.93×)

**Current Implementation**:
- Uses sequential path with 16384 element chunks (128KB, L2 cache)
- Uses 8 accumulators
- Processes chunks sequentially with SIMD reduction

**Problem**:
- Gap is only 0.010ms - very close
- Sequential path might have overhead that parallel could overcome
- Or sequential SIMD reduction might need further optimization

**Potential Solutions**:
1. **Allow parallel path for mean at 1M elements** - Despite overhead, parallel might be faster
2. **Optimize sequential SIMD reduction** - Further tune chunk processing
3. **Try different chunk size** - Maybe 32768 (256KB) would be better
4. **Pipeline chunk processing** - Overlap computation and memory access

### Issue 3: float64 (2048, 2048) mean_axis0 (0.96×)

**Current Implementation**:
- BLAS path should be taken (threshold allows 2048²)
- Falls back to SIMD column reduction if BLAS fails

**Problem**:
- Very close (0.96×) - gap is only 0.023ms
- BLAS might be slower than expected, or SIMD fallback is being used

**Potential Solutions**:
1. **Verify BLAS is actually being used** - Add logging or check backend
2. **Optimize SIMD column reduction prefetch** - Fine-tune prefetch distance
3. **Add specific optimization for 2048 rows** - Custom kernel variant

### Issue 4: float32 (2048, 2048) scale (0.95×)

**Current Implementation**:
- Fast path tries SIMD first, then Accelerate, then BLAS, then scalar
- Should bypass parallel path

**Problem**:
- New issue that appeared - fast path might not be optimal
- Gap is small (0.018ms)

**Potential Solutions**:
1. **Verify fast path is being taken** - Check that `medium_large_square` condition is met
2. **Optimize SIMD kernel for 2048²** - Add specific handling
3. **Try Accelerate first** - Maybe Accelerate is faster than SIMD for this size

### Issue 5: float32 (1024, 1024) scale (0.99×)

**Analysis**:
- Essentially at parity (0.99×, gap: +0.002ms)
- Likely just measurement variance
- Probably doesn't need optimization

## Implementation Plan

### Phase 1: Optimize float64 (512, 512) scale

**Priority**: High (largest gap: 0.040ms)

**Changes**:
1. **Skip BLAS for 512² if SIMD is available** - Avoid copy overhead
2. **Ensure SIMD path is always used** - Don't fall back to BLAS unless SIMD fails
3. **Consider increasing NEON unrolling** - From 4x to 6x or 8x for better throughput

**File**: `rust/src/lib.rs` (around line 3322)

```rust
// For exactly 512², prioritize SIMD and skip BLAS to avoid copy overhead
if rows == 512 && cols == 512 {
    if simd::scale_same_shape_f64(input, factor, out) {
        record_backend_metric(OPERATION_SCALE, dtype, "simd");
        record_scale_event("float64", rows, cols, start.elapsed(), false);
        return Ok(NumericArray::new_owned(data, self.shape.clone()));
    }
    // If SIMD fails, fall back to scalar (skip BLAS to avoid copy overhead)
    scale_block_scalar_f64(input, factor, out);
    record_backend_metric(OPERATION_SCALE, dtype, "scalar");
    record_scale_event("float64", rows, cols, start.elapsed(), false);
    return Ok(NumericArray::new_owned(data, self.shape.clone()));
}
```

### Phase 2: Optimize float64 (1024, 1024) mean

**Priority**: High (very close: 0.010ms gap)

**Changes**:
1. **Try allowing parallel path for mean at 1M elements** - Despite overhead, might be faster
2. **Or optimize sequential path further** - Try larger chunks or better pipelining

**File**: `rust/src/reduce/tiled.rs` (around line 86)

```rust
// For mean at 1M elements, try parallel path (might be faster despite overhead)
// For sum, keep sequential (already optimized)
let prefer_sequential = match op {
    GlobalOp::Sum => elements >= 1 << 20 && elements < DIRECT_PARALLEL_MIN_ELEMENTS,
    GlobalOp::Mean => elements > 1 << 20 && elements < DIRECT_PARALLEL_MIN_ELEMENTS, // Allow parallel for mean at 1M
};
```

### Phase 3: Fine-tune float64 (2048, 2048) mean_axis0

**Priority**: Medium (very close: 0.023ms gap)

**Changes**:
1. **Add comment clarifying BLAS preference**
2. **Verify BLAS path is being taken** - Check threshold logic
3. **Consider optimizing SIMD fallback** - Fine-tune prefetch if BLAS isn't used

**File**: `rust/src/lib.rs` (already done - just verify)

### Phase 4: Fix float32 (2048, 2048) scale

**Priority**: Medium (small gap: 0.018ms)

**Changes**:
1. **Verify fast path is being taken** - Check `medium_large_square` condition
2. **Try Accelerate first for 2048²** - Maybe it's faster than SIMD

**File**: `rust/src/lib.rs` (around line 3544)

```rust
if medium_large_square {
    // For 2048², try Accelerate first (might be faster than SIMD)
    if rows == 2048 && cols == 2048 {
        if accelerate_vsmul_f32(input, factor_f32, out) {
            record_scale_event(dtype, rows, cols, start.elapsed(), false);
            record_backend_metric(OPERATION_SCALE, dtype, "accelerate");
            return Ok(NumericArray::new_owned(data, self.shape.clone()));
        }
    }
    // Try SIMD first (often faster than Accelerate for 1024²)
    if simd_enabled && simd::scale_same_shape_f32(input, factor_f32, out) {
        // ... rest of code
    }
}
```

## Validation

1. Run benchmarks after each phase:
   ```bash
   PYTHONPATH=python python scripts/compare_numpy_raptors.py --suite 2d \
     --simd-mode force --warmup 3 --repeats 30 \
     --output-json benchmarks/results/further_optimizations.json
   ```

2. Verify improvements:
   - float64 (512, 512) scale: Target ≥ 1.0×
   - float64 (1024, 1024) mean: Target ≥ 1.0×
   - float64 (2048, 2048) mean_axis0: Target ≥ 1.0×
   - float32 (2048, 2048) scale: Target ≥ 1.0×

3. Ensure no regressions on other operations

## Success Criteria

- All 5 operations ≥ 1.0× NumPy speed
- Overall: 36/36 operations (100%) faster than NumPy
- No regressions on existing fast paths

