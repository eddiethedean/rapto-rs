# Plan: Investigate and Fix 1024² Float64 Regressions

## Current Status

After recent optimizations, **32/36 operations (88.9%) are faster than NumPy**. However, 3 regressions appeared in 1024² float64 operations:

1. **float64 (1024, 1024) sum**: 0.80× (was 2.94× in previous run, but that may have been an outlier)
2. **float64 (1024, 1024) mean**: 0.89× (was 1.00×)
3. **float64 (1024, 1024) mean_axis1**: 0.87× (was 1.73×)

## Root Cause Analysis

### Issue: 1024² Float64 Global Reductions (sum/mean)

**Current Path**: `reduce_full_f64` in `rust/src/reduce/tiled.rs`:
- For 1M elements (1024²), uses sequential path with `prefer_sequential = true`
- Sequential path uses 16384 element chunks (128KB) and 8 accumulators
- `recommended_accumulators(1 << 20, 8)` returns 7, but we're using 8

**Problem**: 
- Chunk size of 16384 (128KB) might be too large for optimal cache utilization
- Using 8 accumulators instead of 7 might cause register pressure
- The previous 2.94× might have been an outlier, but 0.80× is definitely slower than NumPy

**Root Causes**:
1. **Chunk size too large**: 16384 elements (128KB) might not fit well in L2 cache
2. **Accumulator count too high**: Using 8 instead of 7 (recommended) might cause register pressure
3. **Cache misses**: Large chunks might cause more cache misses

**Solution**:
1. **Reduce chunk size for 1M elements**: Use 8192 (64KB, L1 cache) instead of 16384 (128KB)
2. **Use recommended accumulator count**: Use 7 instead of hardcoded 8 for 1M elements
3. **Or try parallel path**: Despite overhead, parallel might be faster for 1M elements

### Issue: 1024² Float64 mean_axis1

**Current Path**: `reduce_axis1_f64` in `rust/src/lib.rs`:
- Sequential path processes rows in chunks of 8
- Uses `reduce_row_simd_f64` with 6 accumulators for 1024-element rows

**Problem**:
- Sequential path might not be optimal
- Row processing might need optimization

**Solution**:
1. **Verify parallel path**: Check if parallel should be enabled for 1024² mean_axis1
2. **Optimize row processing**: Ensure optimal accumulator count and chunking

## Implementation Plan

### Phase 1: Fix 1024² Float64 Global Reductions (sum/mean)

**Files to modify**: `rust/src/reduce/tiled.rs`

1. **Optimize chunk size and accumulator count for 1M elements** (around line 226):
   - Change chunk size from 16384 to 8192 (64KB, better L1 cache fit)
   - Use `recommended_accumulators(1 << 20, 8)` which returns 7 instead of hardcoded 8
   - This should improve cache utilization and reduce register pressure

### Phase 2: Fix 1024² Float64 mean_axis1

**Files to modify**: `rust/src/lib.rs`

1. **Verify parallel path for mean_axis1** (around line 4546):
   - Check if parallel should be enabled for 1024²
   - Ensure optimal row processing

## Specific Code Changes

### Change 1: Optimize 1M element sequential path

**File**: `rust/src/reduce/tiled.rs` (around line 226)

```rust
} else if data.len() >= 1 << 20 {
    // Optimize for 1M elements: use optimal chunk size and accumulator count
    // For exactly 1M elements (1024²), use 8192 element chunks (64KB, L1 cache)
    // and recommended accumulator count (7) for optimal performance
    let chunk_size = if data.len() >= 1 << 21 {
        // For 2M+ elements, use 16384 element chunks (128KB, L2 cache)
        1 << 14
    } else {
        // For exactly 1M elements, use 8192 element chunks (64KB, better L1 cache fit)
        // This improves cache utilization compared to 16384 element chunks
        1 << 13
    };
    let acc_count = if data.len() == 1 << 20 {
        // For exactly 1M elements, use recommended accumulator count (7)
        recommended_accumulators(1 << 20, 8)
    } else {
        // For 2M+ elements, use 8 accumulators
        8
    };
    let mut sum = 0.0;
    for chunk in data.chunks(chunk_size) {
        sum += simd::reduce_sum_f64(chunk, acc_count)
            .unwrap_or_else(|| chunk.iter().copied().sum());
    }
    sum
}
```

### Change 2: Verify mean_axis1 path

**File**: `rust/src/lib.rs` (around line 4546)

- Check if parallel path should be enabled for 1024²
- Verify row processing is optimal

## Validation

1. Run benchmarks:
   ```bash
   PYTHONPATH=python python scripts/compare_numpy_raptors.py --suite 2d \
     --simd-mode force --warmup 3 --repeats 30 \
     --output-json benchmarks/results/regression_fixes.json
   ```

2. Verify all 3 operations ≥ 1.0× NumPy

3. Ensure no new regressions

## Success Criteria

- All 3 regressed operations ≥ 1.0× NumPy speed
- Overall: 35-36/36 operations (97-100%) faster than NumPy
- No new regressions on other operations

