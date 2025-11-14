# Benchmark Results Summary

## Overall Performance

**26 out of 30 operations (87%) are faster than NumPy**
- Average speedup for faster operations: **2.58x**
- Average slowdown for slower operations: **0.81x**

## Operations Still Slower Than NumPy

### 1. ⚠️ scale @ 2048² float32: 0.62x (Priority Issue)
- **NumPy**: 0.33ms
- **Raptors**: 0.53ms
- **Root Cause**: Default thread pool not available when `RAPTORS_THREADS` is not set
- **Solution**: Use Rayon's global thread pool as default, or ensure `parallel_scale_f32` works without explicit pool
- **Impact**: This is the largest gap remaining

### 2. ⚠️ scale @ 512² float64: 0.75x
- **NumPy**: 0.10ms
- **Raptors**: 0.14ms
- **Root Cause**: Float64 SIMD optimization needed for small arrays
- **Solution**: Optimize float64 SIMD kernel or dispatch logic for 512²
- **Impact**: Small gap, lower priority

### 3. scale @ 1024² float32: 0.90x (Close to Parity)
- **NumPy**: 0.20ms
- **Raptors**: 0.22ms
- **Root Cause**: Minor optimization needed
- **Solution**: Fine-tune dispatch or SIMD kernel for this size
- **Impact**: Very close, acceptable

### 4. mean_axis0 @ 2048² float64: 0.97x (Essentially at Parity)
- **NumPy**: 0.56ms
- **Raptors**: 0.58ms
- **Root Cause**: Within measurement noise
- **Solution**: None needed
- **Impact**: Negligible

## Top Performers

### Extremely Fast (6x+)
- **mean_axis0 @ 512² float64**: 6.51x
- **mean_axis0 @ 1024² float64**: 6.34x
- **mean_axis0 @ 1024² float32**: 6.89x

### Very Fast (3x+)
- **mean @ 512² float32**: 3.37x
- **mean_axis1 @ 2048² float32**: 2.90x
- **mean_axis1 @ 1024² float32**: 3.11x

### Fast (1.5x-3x)
- **mean @ 2048² float32**: 2.46x
- **mean @ 1024² float32**: 2.61x
- **mean_axis0 @ 2048² float32**: 2.62x
- **broadcast_add @ 2048² float32**: 1.91x
- **broadcast_add @ 1024² float64**: 2.25x

## Key Findings

1. **Mean operations are consistently faster** (2-6x speedup)
2. **Broadcast add operations are faster** (1.17x-2.25x speedup)
3. **Scale operations are mixed**: Fast for float32 small/medium, slower for float64 small and float32 large
4. **Thread pool configuration is critical**: Operations that benefit from parallelism need `RAPTORS_THREADS` set or default pool

## Recommendations

### Immediate Priority
1. **Fix default thread pool** for 2048² float32 scale operation
   - Use Rayon's global pool when `RAPTORS_THREADS` not set
   - Or make `parallel_scale_f32` work with global pool

### Medium Priority
2. **Optimize float64 scale for 512²**
   - Review SIMD kernel or dispatch logic
   - Consider size-specific optimizations

### Low Priority
3. **Fine-tune 1024² float32 scale**
   - Minor optimization to reach parity
   - Currently acceptable at 0.90x

## Test Configuration

- Warmup: 20 iterations
- Repeats: 100 iterations
- Shapes: 512², 1024², 2048²
- Dtypes: float32, float64
- Threads: Default (no `RAPTORS_THREADS` set)

