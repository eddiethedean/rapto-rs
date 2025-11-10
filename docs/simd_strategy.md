# SIMD Strategy for Raptors

## Goals

- Provide portable, testable vectorized kernels for the hot array paths (elementwise math, broadcasts, reductions).
- Support x86-64 (AVX/AVX2) and ARM64 (NEON) at runtime with automatic fallback to scalar code on unsupported CPUs.
- Layer in multi-threading for large workloads without sacrificing determinism or Python ergonomics.

## Implementation Overview

1. **SIMD Layer** (`rust/src/simd/mod.rs`)
   - Uses `core::arch` intrinsics behind per-ISA modules (`x86`, `neon`) compiled conditionally.
   - Runtime dispatch relies on `is_x86_feature_detected!("avx2")`; ARM builds always expose NEON. The scalar path stays available for all dtypes.
   - Public helpers cover contiguous add/scale kernels plus row/column scalar broadcasts. Each helper returns `false` when the ISA is unavailable, signalling the caller to fall back to scalar code.

2. **NumericArray Integration**
   - `NumericArray::try_simd_add_*` normalises slices, calls into the SIMD helpers,
     and retains a scalar tail for mixed layouts.
   - `try_parallel` now works with adaptive heuristics (`should_parallelize`) so
     only genuinely large, well-shaped workloads pay the Rayon cost; scalar loops
     remain for small arrays.
   - Column broadcasts now use SIMD row scalars and parallel rows when available.

3. **Threading Controls**
   - `RAPTORS_THREADS=<N>` configures a fixed-size Rayon pool (defaults to hardware parallelism when >1 core). Values ≤1 or build failures fall back to single-threaded execution.
   - SIMD dispatch remains independently controlled via `RAPTORS_SIMD={0,1}` or the default runtime detection.

4. **Benchmark Harness**
   - `scripts/compare_numpy_raptors.py` gained `--suite` presets (e.g. `2d`, `mixed`) and `--output-json` to aid CI dashboards.
   - `--simd-mode {auto,force,disable}` still toggles the vector path for A/B comparisons.

## Current Status (2025-11-10)

Recent measurements on an Apple M3 Pro (Python 3.11, SIMD auto, `RAPTORS_THREADS=8`):

| Shape / Dtype | Operation | NumPy (ms) | Raptors (ms) | Speedup | Notes |
| --- | --- | --- | --- | --- | --- |
| `(1024, 1024)` `float32` | sum | 0.31 ± 0.00 | 0.21 ± 0.01 | **1.48×** | AVX2/NEON lanes engaged; adaptive heuristics keep work single-threaded |
| `(1024, 1024)` `float32` | mean | 0.32 ± 0.01 | 0.21 ± 0.01 | **1.54×** | Scalar fallback handles reductions |
| `(1024, 1024)` `float64` | mean_axis0 | 0.18 ± 0.06 | 0.12 ± 0.05 | **1.50×** | Fused SIMD/Rayon reduction |
| `(1024, 1024)` `float64` | mean_axis1 | 0.15 ± 0.00 | 0.12 ± 0.02 | **1.28×** | Fused SIMD/Rayon reduction |
| `(1024, 1024)` `float64` | broadcast add (row) | 0.70 ± 0.08 | 0.67 ± 0.19 | 1.05× | Row broadcast hits SIMD + tiling |
| `(2048, 2048)` `float64` | scale | 2.80 ± 0.12 | 2.19 ± 0.15 | **1.28×** | Parallel lanes plus vector scalar multiply |

Key observations:

- Broadcasted row/column additions now share the same SIMD helpers as the dense path and outperform NumPy once row tiling kicks in.
— Reduction-heavy operations (`mean_axis0/1`) now beat NumPy when arrays are
  contiguous float64/f32 thanks to the fused reduce kernels; scalar fallback
  still lags when SIMD is disabled.
— Column broadcasts benefit modestly (close to parity) thanks to scalar-vector
  helpers and row-level parallelism.

## Testing & Tooling

- `tests/test_placeholder.py` includes regression coverage for same-shape, row, and column SIMD paths plus scaling parity.
- `tests/test_simd_env.py` validates environment toggles (`RAPTORS_SIMD`).
- New benchmark suites allow CI to pin JSON results for trend tracking or publish comparisons in docs.

## Remaining Gaps & Next Steps

- **Global Reductions**: The all-elements `mean/sum` still trail NumPy; wider
  lanes (AVX-512/SVE) and cache-aware tiling are next targets.
- **Fused Kernels**: Add batched primitives (e.g., `axpy`, dot products) to
  amortize memory traffic and compete with NumPy BLAS-backed routes.
- **Dynamic Tiling**: Tune tile sizes per ISA and array shape; consider
  cache-aware blocking to reduce L1 misses on large matrices.
- **Adaptive Threading Diagnostics**: Surface a Python-side toggle (or context
  manager) for thread pools and expose the actual thread count for
  instrumentation.
- **Cross-Platform Benchmarks**: Capture AVX2/Xeon results to confirm parity with ARM NEON, and publish combined charts in the docs.

The SIMD + threading foundation is in place; the next iterations focus on reduction fusion and polishing the ergonomics around performance instrumentation.
