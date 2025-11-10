# Profiling Notes — 2025-11-10

Goal: quantify remaining gaps vs NumPy for 2-D workloads before implementing fused reductions.

## Environment

- Hardware: Apple M3 Pro
- Python: 3.11.13
- Raptors build: commit `$(git rev-parse --short HEAD)`
- SIMD: runtime auto unless otherwise noted
- Threads: default (`RAPTORS_THREADS` unset)

## Float64 Reductions & Broadcasts (SIMD auto)

```
./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 \
    --operations mean mean_axis0 mean_axis1 broadcast_add --warmup 3 --repeats 7
```

| Operation | NumPy (ms) | Raptors (ms) | Speedup |
| --- | ---:| ---:| ---:|
| mean | 0.15 ± 0.00 | 0.37 ± 0.01 | 0.40× |
| mean_axis0 | 0.18 ± 0.06 | 0.12 ± 0.05 | **1.50×** |
| mean_axis1 | 0.15 ± 0.00 | 0.12 ± 0.02 | **1.28×** |
| broadcast_add (row) | 0.70 ± 0.08 | 0.67 ± 0.19 | 1.05× |

## Float64 Scalar Fallback (SIMD disabled)

```
./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 \
    --operations mean mean_axis0 mean_axis1 broadcast_add --simd-mode disable
```

| Operation | NumPy (ms) | Raptors (ms) | Speedup |
| --- | ---:| ---:| ---:|
| mean | 0.46 ± 0.11 | 0.72 ± 0.22 | 0.64× |
| mean_axis0 | 0.15 ± 0.00 | 2.53 ± 1.83 | 0.06× |
| mean_axis1 | 0.40 ± 0.05 | 0.93 ± 0.07 | 0.43× |
| broadcast_add (row) | 0.85 ± 0.30 | 0.62 ± 0.03 | **1.37×** |

## Column Broadcast (SIMD vs Scalar)

Custom micro-benchmark (`1024×1024` * `1024×1`):

| Mode | Mean (ms) | Stdev |
| --- | --- | --- |
| SIMD auto | 0.72 | 0.11 |
| `RAPTORS_SIMD=0` | 1.25 | 0.15 |

## Takeaways

- Row broadcasts stay ahead of NumPy; scalar fallback remains much slower, so SIMD detection is critical.
- Column broadcasts gain ~1.7× from SIMD and now sit close to NumPy on ARM (x86 validation pending).
- Axis reductions improved dramatically: `mean_axis0/1` are now 1.3–1.5× faster than NumPy with the fused SIMD/Rayon kernels, while scalar mode is still several times slower.
- The global `mean` kernel is still behind NumPy; we should investigate cache tiling and wider lanes (AVX-512/SVE) for the full-array sum path.

## Next Steps

1. Capture equivalent traces on x86-64 AVX2/AVX-512 hardware (blocked pending access).
2. Explore wider float64 lanes (AVX-512/SVE) and cache-aware tiling to close the gap on the full-array `mean` kernel.
3. Re-run the suite after further kernel work and promote the JSON outputs to CI thresholds.
