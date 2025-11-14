# Hand-Tuned Assembly Analysis

## Overview

This document describes the hand-tuned assembly implementation for the `float32 @ 2048²` scale operation, using manual instruction scheduling and register allocation to maximize performance.

## Implementation

### Function: `scale_same_shape_f32_handtuned`

Located in `rust/src/simd/mod.rs`, this function provides a hand-tuned version specifically for 2048² arrays.

### Key Optimizations

1. **16× Unrolling** (64 elements per iteration)
   - Matches the optimized SIMD kernel
   - Processes 64 elements per loop iteration

2. **Aggressive Read Prefetching**
   - Prefetch distance: 28 vectors (3.5KB ahead)
   - L1 cache prefetch (`pldl1keep`)
   - Only prefetch reads (no write prefetch for large arrays)

3. **Manual Instruction Scheduling**
   - Deep interleaving of loads, multiplies, and stores
   - Pattern: Load → Load → Load → Multiply → Store → ...
   - Minimizes data dependencies
   - Maximizes instruction-level parallelism

4. **Register Allocation**
   - Uses NEON registers efficiently (v0-v15+)
   - Factor broadcast to vector register and reused
   - Minimizes register pressure

### Dispatch Logic

The hand-tuned version is called specifically for 2048² arrays after Accelerate is tried:

```rust
if simd_enabled && rows == 2048 && cols == 2048 {
    unsafe {
        simd::neon::scale_same_shape_f32_handtuned(input, factor_f32, out);
    }
    record_backend_metric(OPERATION_SCALE, dtype, "simd_handtuned");
    return Ok(NumericArray::new_owned(data, self.shape.clone()));
}
```

## Performance Characteristics

### Expected Benefits

1. **Optimized Instruction Scheduling**
   - Manual scheduling may be better than compiler-generated code
   - Can minimize pipeline stalls
   - Better register pressure management

2. **Specialized for 2048²**
   - No runtime size checks in hot loop
   - Optimized for this specific case
   - Reduced dispatch overhead

3. **Reduced Write Prefetching**
   - For large arrays, skips write prefetch
   - Reduces cache pollution
   - Simulates non-temporal store behavior

### Trade-offs

1. **Code Duplication**
   - Separate function for 2048²
   - Similar to regular SIMD kernel but specialized

2. **Maintenance**
   - Manual assembly requires careful maintenance
   - May need updates for different architectures

## Benchmark Results

(To be updated with actual benchmark results)

### Test Configuration

- Array size: 2048×2048 = 4,194,304 elements (16 MB)
- Data type: float32
- Iterations: 100

### Results

- **Hand-tuned assembly**: TBD
- **Regular SIMD kernel**: TBD
- **NumPy baseline**: TBD

## Comparison with Regular SIMD Kernel

The hand-tuned version is similar to the regular SIMD kernel but:
- Specialized for 2048² (no runtime size checks in hot loop)
- Manual instruction scheduling (may be more optimal)
- Same unrolling factor (16×)
- Same prefetch distance (28 vectors)

## Future Improvements

1. **Use More Registers**
   - Could potentially use all 32 NEON registers
   - Increase unrolling even further
   - Process more elements per iteration

2. **Inline Assembly**
   - Consider using `core::arch::asm!` for complete control
   - Exact instruction ordering
   - Register allocation control

3. **External Assembly File**
   - Write `.s` file for complete control
   - Use `build.rs` to compile and link
   - Maximum optimization flexibility

## References

- ARM NEON Programmer's Guide
- ARM Architecture Reference Manual
- Rust inline assembly guide

