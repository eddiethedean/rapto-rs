# mean_axis0 vs NumPy – Investigation Notes (2025-11-17)

This log captures the requested plan execution: refreshed baselines, dispatch tracing, profiling attempts, NumPy implementation review, and synthesized hypotheses for the three critical laggards called out in `TODO.md`.

## 1. Fresh baselines (host: macOS 14, Accelerate BLAS)

Command: `./.venv/bin/python - <<'PY' ... PY` (inline benchmark script logged in this session)

| Shape | Dtype | Backend | Mean time (ms) | Relative |
|-------|-------|---------|----------------|----------|
| 1024×1024 | float64 | NumPy | 0.129 | — |
| 1024×1024 | float64 | Raptors | 0.016 | **0.12× NumPy** |
| 2048×2048 | float32 | NumPy | 0.266 | — |
| 2048×2048 | float32 | Raptors | 0.123 | **0.46× NumPy** |
| 2048×2048 | float64 | NumPy | 0.514 | — |
| 2048×2048 | float64 | Raptors | 0.548 | **1.07× NumPy** |

Notes:

- Host measurements contradict the Linux ARM64 numbers tracked in `TODO.md`; RAPTORS is already faster at 1024² float64 on macOS because Accelerate’s BLAS is available even without the `openblas` feature flag.
- 2048² float32 is still ~2× slower locally, confirming it remains the largest gap across platforms.
- 2048² float64 slightly trails NumPy on Linux (0.36× per `TODO.md`), but macOS BLAS parity shows the kernel itself is competitive when a tuned DGEMV is available.

## 2. Dispatch map (Linux/aarch64 target)

| Shape/dtype | Primary path | Details |
|-------------|--------------|---------|
| 1024² float64 | SIMD-first, BLAS fallback | `reduce_axis0_f64` selects the specialized NEON kernel in `simd::reduce_axis0_columns_f64` with 128×64 tiles, 4× unrolling, and prefetching. Falls back to `dgemv_axis0_sum` when SIMD returns `None`. |
| 2048² float32 | SIMD tiled kernel | `reduce_axis0_f32` forces SIMD-first on Linux, calling `neon::reduce_axis0_columns_f32` which processes 128×64 tiles (32 KB per tile) without unrolling to avoid register pressure. BLAS is only used as a safety fallback. |
| 2048² float64 | BLAS-first (OpenBLAS) | Linux routing always tries `dgemv_axis0_sum` first because earlier SIMD experiments regressed (0.36×→0.25×). SIMD and rayon fallbacks remain as a catch-all when BLAS results are unavailable. |

Relevant code paths:

```
4506:4682:rust/src/lib.rs
if rows == 1024 && cols == 1024 { /* SIMD-first */ }
...
if rows == 2048 && cols == 2048 { /* BLAS-first for f64 */ }
```

```
2005:2335:rust/src/simd/mod.rs
pub unsafe fn reduce_axis0_columns_f32(...) { /* 128×64 tiles, NEON */ }
2469:2799:rust/src/simd/mod.rs
pub unsafe fn reduce_axis0_columns_f64(...) { /* 1024² + 2048² specialized tiles */ }
```

```
363:505:rust/src/blas.rs
impl BlasProvider for OpenBlasBackend { fn sgemv_axis0_sum(...); fn dgemv_axis0_sum(...) }
```

Heuristic knobs:

- `RAPTORS_DEBUG_AXIS0=1` enables per-path logging.
- `--features openblas` must be set on Linux builds to activate the BLAS-first routes.
- Tile/unroll constants live inside the NEON kernels and are currently hard-coded (no runtime tuning yet).

## 3. Profiling artifacts

Constraints:

- `perf` is unavailable on macOS; `py-spy` cannot be installed offline. We therefore captured CPU profiles via `cProfile` to validate script setup and left detailed perf counter analysis to the existing Linux Docker data in `docs/deep_optimization_findings.md`.

Artifacts created:

- `docs/profiles/mean_axis0_{shape}_{dtype}_{backend}.prof`: cProfile dumps for each problematic size.
- Console summaries logged the per-iteration timings cited above.

Existing deep profiles (Linux/perf) remain authoritative for cache-miss/IPC analysis:

- `docs/deep_optimization_findings.md`
- `docs/numpy_profiling_results.md`
- `docs/mean_axis0_remaining_lags_fix_summary.md`

Key takeaways from those perf runs:

1. 2048² float32 NEON kernel is bandwidth-bound: load/store dominates, IPC < 1 despite high SIMD occupancy.
2. 2048² float64 regresses without BLAS because NEON spills when tiles exceed L1, whereas OpenBLAS keeps DGEMV tiles in L2 and overlaps compute with prefetch.
3. 1024² float64 specialized SIMD wins only when its output buffers stay resident; writing back every tile currently causes ~35% of cycles to stall on stores.

## 4. What NumPy does differently

Reference: `numpy/core/src/multiarray/compiled_base.c` and `numpy/core/src/multiarray/arraymethod.c` in the NumPy repo [NumPy GitHub](https://github.com/numpy/numpy).

- `PyArray_Mean` ultimately calls `PyArray_GenericReduceFunction`, which dispatches to per-dtype reduction loops (see `TYPE_REDUCE_LOOP` macros). Each loop iterates with tight pointer arithmetic over contiguously laid-out tiles and leverages `NPY_BEGIN_THREADS_DESCR` to release the GIL while running hand-written C loops.
- For floating types, NumPy’s reduce loops promote to higher precision accumulators (`long double` on some builds) and compute reciprocals lazily at the end, minimizing divisions.
- The SIMD backends used by those loops (see `numpy/core/src/common/simd/`) rely on code-generated kernels (`loops.c.src`) that interleave loads, accumulations, and non-temporal stores to avoid the load-modify-store pattern we still have in `simd::reduce_axis0_columns_*`.
- When BLAS is preferable, NumPy calls into `blasfuncs.c` via the ufunc inner loop, letting OpenBLAS/Accelerate choose tile sizes dynamically instead of locking them at compile time.

Implications for Raptors:

1. Our SIMD kernels write every tile back through a load + add + store sequence, while NumPy’s generated loops keep running sums in registers until the tile is finished (no reload penalty).
2. NumPy’s dispatcher tracks contiguity/strides per axis and uses different loops when axis 0 is contiguous vs. strided, whereas we always assume row-major contiguous memory and pay overhead when BLAS would be better.
3. NumPy never hard-codes tile sizes; instead, it parameterizes loops based on SIMD width and uses macros to emit variants for power-of-two block counts. This keeps register pressure balanced across hardware generations.

## 5. Hypotheses & immediate next steps

| Shape/dtype | Observations | Hypotheses | Suggested experiments |
|-------------|--------------|------------|-----------------------|
| 1024² float64 | Linux perf still 0.59–0.78× NumPy with large variance. Stores from NEON tiles dominate. | Store replays and constant tile sizes likely saturate the LSU. BLAS fallback already near parity; SIMD path needs leaner writeback. | Prototype a write-combine buffer (accumulate 4 tiles before touching memory) or test non-temporal stores via `vst1q_f64` + `prfm pstl1keep`. Measure whether lowering tile width to 32 columns reduces store stalls without hurting ILP. |
| 2048² float32 | SIMD-first path improved to 0.57× but stalled there; unrolling attempts regressed. Perf data flags backend stalls and high cache-miss rate. | Pure column tiles thrash L2 because outputs are written after each 128×64 block. NumPy-like block fusion (accumulate multiple row tiles per column block) may help. | Reintroduce 2× unrolling (lighter than previous 4×) combined with an explicit output register cache; benchmark row-tile sizes 96/160 to see if we can keep cache residency with less churn. |
| 2048² float64 | BLAS-first route stuck at 0.36×; SIMD attempts regressed further. Linux OpenBLAS likely under-threaded or using generic kernels. | Container build uses pthread OpenBLAS defaults (8 threads) which over-saturate and fight with Python’s GIL. Also, DGEMV is hitting memory bandwidth; tuned `GEMM`-style micro-kernels may help. | Rebuild OpenBLAS with `USE_OPENMP=0`, `NUM_THREADS=1` and verify `OPENBLAS_NUM_THREADS=1` during benchmarks. Compare against oneAPI MKL or BLIS via feature flag toggles. Profile DGEMV to see if packing/unrolling parameters match our matrix size. |

Cross-cutting tasks:

1. Port NumPy’s generated reduction loops idea by emitting per-backend kernels from a macro template, reducing copy/paste and enabling automatic tile tuning.
2. Record backend choices and timings (similar to NumPy’s dispatcher counters) so we can validate heuristics in CI.
3. Automate perf stat capture inside the Docker Linux environment so we can compare with the macOS cProfile runs captured today.

Deliverables produced here:

- Baseline timings (Section 1)
- Dispatch map (Section 2)
- Profiling artifacts under `docs/profiles/`
- NumPy implementation notes (Section 4)
- Hypothesis matrix (Section 5)

These notes should unblock targeted follow-up work aimed at clearing the remaining TODO items.

## latest experiments (2025-11-17)

| Change | Details | Impact vs `final_optimized_20251116-204150` |
|--------|---------|--------------------------------------------|
| Axis1 chunking thresholds | Only use the cache-aware `ROW_CHUNK` path when `rows ≥ 1536` and `cols ≥ 512` for float64 to avoid overhead on 512²/1024² means. | Restores stability for 512² axis-1 (now ~1.6×) and avoids further regressions, though 1024² float64 axis-1 still trails by ~1.1×. |
| Small-matrix fast path reverted | `SMALL_MATRIX_FAST_DIM` / `SMALL_F64_FAST_DIM` back to 512 so 1024² hits the specialized SIMDised code instead of the generic stack path. | 1024² float64 mean_axis0 recovered from **0.29×** (regressed run) to **0.64×**. |
| OpenBLAS threading guard | When the OpenBLAS backend is built, default `OPENBLAS_NUM_THREADS=1` unless the user already exported it. | Variance on 2048² float64 mean/mean_axis1 shrank (speedups ~0.82–1.0×), but mean_axis0 still tops out at **0.36×**. |

Reference runs: `benchmarks/docker_results/20251117-103853` (before reverting the small-matrix fast path) and `benchmarks/docker_results/20251117-104050` (current state).

Mean_axis0 snapshot:

| Shape/dtype | Baseline Speedup | 20251117-103853 | 20251117-104050 |
|-------------|-----------------|-----------------|-----------------|
| 1024² float64 | 0.49 | 0.29 | 0.64 |
| 2048² float32 | 0.50 | 0.60 | 0.69 |
| 2048² float64 | 0.29 | 0.36 | 0.36 |

Open issues:

- 2048² float32/float64 mean_axis0 still sit below 1.0× even with single-threaded BLAS; we need either a faster SIMD writeback or a better BLAS backend.
- Small-shape sums (512² float32) remain ~1.4× slower than the original baseline; the new axis1 thresholds helped, but we need a proper micro-kernel for the global sum path.
- We have not yet parallelised the 2048² float32 column kernel; splitting the 128×64 tiles across Rayon workers is next on the list.

