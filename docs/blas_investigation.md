# BLAS Investigation for mean_axis0 Float32 2048²

## Hypothesis

NumPy might be faster because it uses OpenBLAS more effectively than our SIMD implementation.

## Test Results

### Test: Enable BLAS for 2048² Float32

**Configuration**:
- `RAPTORS_BLAS_AXIS0=1` (enable BLAS for axis0)
- OpenBLAS single-threaded (`OPENBLAS_NUM_THREADS=1`)

**Results**:
- **Raptors (BLAS)**: 1.286 ms (3.89x slower than NumPy)
- **NumPy**: 0.330 ms
- **Ratio**: 3.89x slower

### Comparison with SIMD

- **Raptors (SIMD)**: ~0.46ms (0.73x NumPy) ✅
- **Raptors (BLAS)**: ~1.29ms (3.89x NumPy) ❌

**Conclusion**: BLAS is significantly slower than SIMD for float32 2048² on Linux/aarch64.

## NumPy Behavior

**Key Finding**: NumPy's `mean(axis=0)` and `sum(axis=0)` have essentially the same performance:
- `np.sum(axis=0)`: 0.324 ms
- `np.mean(axis=0)`: 0.330 ms
- **Difference**: ~0.005ms (just division overhead)

This suggests NumPy is NOT using BLAS for mean_axis0, or using a different highly optimized path.

## Analysis

### Why BLAS is Slower

1. **Overhead**: BLAS calls have overhead (function calls, parameter marshalling)
2. **Memory allocation**: Creating a vector of ones (4096 floats = 16KB)
3. **Sub-optimal for this case**: `cblas_sgemv` is designed for general matrix-vector operations, not specifically optimized for summing along axis 0
4. **Library overhead**: OpenBLAS may have additional overhead for small operations

### Why NumPy Might Be Faster

1. **Direct SIMD implementation**: NumPy likely has a direct, highly optimized SIMD implementation for mean/sum operations
2. **Better compiler optimizations**: NumPy may use different compiler flags or optimization strategies
3. **Specialized code paths**: NumPy may have specialized paths for common operations like mean_axis0
4. **Better memory access patterns**: NumPy's iterator-based approach may have better cache behavior

## Conclusion

**BLAS is NOT the answer** for float32 2048² mean_axis0 on Linux/aarch64:
- BLAS path is 3.89x slower than NumPy
- SIMD path is 0.73x NumPy (much better)

**Next Steps**:
1. ✅ BLAS tested - not faster
2. ⏭️ Investigate NumPy's actual implementation (may not be BLAS)
3. ⏭️ Compare compiler flags and optimization strategies
4. ⏭️ Consider accepting 0.73x as competitive performance

## Implementation Note

The code was modified to try BLAS first for 2048², but this was reverted because:
- BLAS is slower than SIMD
- Current SIMD path is already well-optimized (0.73x NumPy)

The current implementation correctly uses SIMD for Linux and BLAS for macOS (where Accelerate is faster).

