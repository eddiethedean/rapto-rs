# NumPy Optimization of 2D Array Operations on Linux

## 1. Matrix Multiply (`A @ B`)

### Steps

1.  Python dispatches to `np.matmul` / `np.dot`.
2.  NumPy calls BLAS `GEMM` (`dgemm` for float64).
3.  BLAS handles tiling, SIMD microkernels, threading.
4.  Memory layout matters (Fortran order preferred).
5.  Threading controlled via environment variables:
    -   `OPENBLAS_NUM_THREADS`
    -   `MKL_NUM_THREADS`

### Inspect BLAS on Linux

``` bash
python - <<'PY'
import numpy as np
np.show_config()
PY

ldd $(python -c "import numpy as np, os; print(np.core.multiarray.__file__)") | egrep -i 'blas|lapack'
```

## 2. Elementwise Ops (e.g. `A + B*2 + np.sin(A)`)

### Steps

1.  Broken into ufunc calls.
2.  Ufunc dispatcher selects best kernel (baseline or SIMD).
3.  SIMD via NEP-38 universal intrinsics (SSE/AVX/AVX2/AVX-512).
4.  Strided or buffered loops depending on array contiguity.
5.  Temporaries created unless `out=` used.

### Optimization Tips

-   Ensure contiguous arrays (`np.ascontiguousarray`).
-   Use `out=` to avoid temporaries.
-   Combine operations explicitly using ufuncs.

## 3. Profiling Checklist

### Microbenchmarks

``` python
import numpy as np
A = np.random.random((1000,1000))
B = np.random.random((1000,1000))

%timeit A @ B
%timeit A + B*2 + np.sin(A)
```

### Check contiguity

``` python
print(A.flags)
```

## 4. Internal Mechanisms Summary

### BLAS Path

-   Matrix multiply uses highly optimized `GEMM`.
-   Threading controlled externally.

### Ufunc Path

-   Loop selection uses dtype + strides.
-   SIMD kernels chosen via CPU feature detection.
-   Buffered loops used when data isn't contiguous.

## 5. Useful Notes

-   `np.show_config()` reveals BLAS backend.
-   `ldd` shows linked shared libraries.
-   Some ufuncs lack SIMD kernels (work ongoing).
