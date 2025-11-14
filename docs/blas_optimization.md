# BLAS/Accelerate Optimization Strategy

## Overview

This document describes the cross-platform BLAS/Accelerate optimization strategy implemented in Raptors to match NumPy's performance by leveraging optimized linear algebra libraries.

## NumPy's Cross-Platform Strategy

### BLAS Backend Usage

- **macOS**: Accelerate (Apple's framework with hand-tuned assembly)
- **Linux**: OpenBLAS (default via pip) or MKL (if available via conda)
- **Windows**: OpenBLAS (default via pip) or MKL (if available via conda)

### NumPy's Detection Order

NumPy detects and uses the best available BLAS library at build/runtime:
1. MKL (if available)
2. Accelerate (macOS only)
3. OpenBLAS
4. BLIS
5. Reference BLAS (fallback)

## Raptors Implementation

### Current BLAS/Accelerate Integration

#### macOS (Accelerate Framework)

**Available Functions:**
- `vDSP_vsmul` - Vector-scalar multiply (vDSP)
- `vDSP_vadd` - Vector-vector add (vDSP)
- `vDSP_vsadd` - Vector-scalar add (vDSP)
- `cblas_sscal` - BLAS scale for float32 (BLAS)
- `cblas_dscal` - BLAS scale for float64 (BLAS)

**Wrapper Functions:**
- `accelerate_vsmul_f32` - vDSP vector-scalar multiply
- `accelerate_vadd_f32` - vDSP vector-vector add
- `accelerate_vsadd_f32` - vDSP vector-scalar add
- `accelerate_blas_scale_f32` - BLAS scale (copy mode)
- `accelerate_blas_scale_f64` - BLAS scale (copy mode)
- `accelerate_blas_scale_inplace_f32` - BLAS scale (in-place)
- `accelerate_blas_scale_inplace_f64` - BLAS scale (in-place)

#### Linux/Windows (OpenBLAS)

**Available Functions (when `openblas` feature enabled):**
- `cblas_sscal` - BLAS scale for float32
- `cblas_dscal` - BLAS scale for float64

**Wrapper Functions:**
- `openblas_scale_f32` - OpenBLAS scale (copy mode)
- `openblas_scale_f64` - OpenBLAS scale (copy mode)

### Unified Dispatch Functions

**`blas_scale_f32_optimal`** - Chooses best BLAS function per platform:
- **macOS**: 
  - Large arrays (>1M elements): Prefer Accelerate BLAS (`cblas_sscal`)
  - Medium/small arrays (≤1M elements): Prefer Accelerate vDSP (`vDSP_vsmul`)
  - Falls back to the other if first choice fails
- **Linux/Windows**: 
  - Uses OpenBLAS `cblas_sscal` if available
  - Falls back to SIMD if OpenBLAS unavailable

**`blas_scale_f64_optimal`** - Similar logic for float64:
- **macOS**: Uses Accelerate BLAS (`cblas_dscal`)
- **Linux/Windows**: Uses OpenBLAS `cblas_dscal` if available

### Dispatch Priority (New)

#### Scale Operations

**macOS:**
1. BLAS/Accelerate (via `blas_scale_f32_optimal` - chooses BLAS or vDSP based on size)
2. SIMD (if BLAS/Accelerate unavailable)
3. Parallel SIMD (if large enough)
4. Scalar (fallback)

**Linux/Windows:**
1. OpenBLAS (if available via `blas_scale_f32_optimal`)
2. SIMD (if OpenBLAS unavailable)
3. Parallel SIMD (if large enough)
4. Scalar (fallback)

#### Add Operations

**macOS:**
1. Accelerate (`vDSP_vadd`)
2. SIMD (if Accelerate unavailable)
3. Parallel SIMD (if large enough)
4. Scalar (fallback)

**Linux/Windows:**
1. SIMD (primary path, no OpenBLAS equivalent for vDSP_vadd)
2. Parallel SIMD (if large enough)
3. Scalar (fallback)

### Size-Based Optimization

**Small matrices (≤512²):**
- Direct BLAS/Accelerate path
- Skip parallel overhead
- Use `blas_scale_f32_optimal` for best function selection

**Medium matrices (512²-1024²):**
- Try BLAS/Accelerate first
- Fall back to SIMD
- Skip parallel for smaller sizes

**Large matrices (≥2048²):**
- Try BLAS/Accelerate first (primary path)
- Fall back to single-threaded SIMD (optimized kernel)
- Parallel as backup only

### Function Selection Logic

**For scale (float32) - macOS:**
- Array size > 1M elements: Use Accelerate BLAS (`cblas_sscal`)
- Array size ≤ 1M elements: Use Accelerate vDSP (`vDSP_vsmul`)
- If in-place allowed: Use `cblas_sscal` in-place mode
- Fallback to SIMD if Accelerate unavailable

**For scale (float32) - Linux/Windows:**
- Array size > 1M elements: Use OpenBLAS `cblas_sscal` (if available)
- Array size ≤ 1M elements: Use SIMD (lower overhead)
- If in-place allowed: Use OpenBLAS `cblas_sscal` in-place mode (if available)
- Fallback to SIMD if OpenBLAS unavailable

**For scale (float64):**
- **macOS**: Uses Accelerate BLAS (`cblas_dscal`)
- **Linux/Windows**: Uses OpenBLAS `cblas_dscal` (if available)

## Performance Characteristics

### Expected Performance

- **macOS**: Should match or exceed NumPy's performance using Accelerate
- **Linux/Windows**: Should match NumPy's performance when OpenBLAS is available
- **Fallback**: SIMD implementation provides competitive performance when BLAS unavailable

### Optimization Techniques

1. **Aggressive Inlining**: All Accelerate/BLAS wrapper functions use `#[inline(always)]` to minimize function call overhead
2. **Size-Based Selection**: Adaptive function selection based on array size
3. **Platform-Specific Paths**: Optimized dispatch per platform
4. **In-Place Operations**: Support for in-place modification to reduce memory allocation

## Usage

### Environment Variables

- `RAPTORS_SIMD={0,1}` - Control SIMD usage (default: auto-detect)
- `RAPTORS_THREADS=<N>` - Control thread pool size
- `RAPTORS_BLAS_SCALE={0,1}` - Override BLAS usage for scale operations
- `RAPTORS_BLAS_OPS=scale,axis0` - Specify which operations should use BLAS

### Python API

```python
import raptors

# Scale operation (uses BLAS/Accelerate automatically when available)
arr = raptors.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32')
result = arr * 2.5  # Uses BLAS/Accelerate on macOS, OpenBLAS on Linux/Windows

# In-place scale (when implemented)
# arr.scale_inplace(2.5)  # Modifies array in-place using BLAS
```

## Benchmarking

Use `scripts/benchmark_blas_options.py` to compare:
- BLAS vs vDSP performance on macOS
- OpenBLAS vs SIMD performance on Linux/Windows
- In-place vs copy operations
- Different array sizes (512², 1024², 2048²)

## Future Improvements

1. **In-Place API**: Complete Python API for in-place operations
2. **More Operations**: Extend BLAS/Accelerate usage to more operations
3. **Dynamic Selection**: Runtime benchmarking to choose optimal function
4. **Cross-Platform Testing**: Validate performance on all platforms

## References

- [NumPy BLAS/LAPACK Building](https://numpy.org/doc/stable/building/blas_lapack.html)
- [Apple Accelerate Framework](https://developer.apple.com/documentation/accelerate)
- [OpenBLAS](https://www.openblas.net/)

