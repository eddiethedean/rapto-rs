## Objective

Restore and extend Raptors' 2D matrix performance relative to NumPy by porting key runtime optimizations (SIMD dispatch, stride iteration, BLAS offload, and benchmarking discipline) identified from the NumPy codebase.

## Current State Snapshot (2025-11-12 benchmark run)

- Benchmark artefacts: `benchmarks/results/20251112-183717/`.
- Strengths: Raptors consistently outperforms NumPy on global reductions and axis-1 means (≈2–3× for float32). Single-threaded reductions also lead.
- Gaps:  
  - `scale` (elementwise multiply) lags 8–43% across float32/float64 (worst offenders: 512² float64 single-thread SIMD, 2048² float32 threaded SIMD-off).  
  - `broadcast_add` drops below parity for float64 at 1024² when multi-threaded SIMD is forced.  
  - `mean_axis0` for 2048² float64 hovers at ~0.99× even on the SIMD fast path, hinting at cache/dispatch inefficiency.  
  - SIMD disable mode shows limited degradation in some cases, implying our current SIMD paths are narrow or unused.

## Guiding Principles

1. Maintain ergonomic Python API; focus on backend improvements.
2. Target measurable speedups via reproducible benchmarks.
3. Prefer incremental integration with validation at each layer.

## Phase 1 – Diagnostics & Baseline Hardening

- **Reconfirm BLAS backend usage**: instrument `compare_numpy_raptors.py` to log NumPy’s active BLAS (`numpy.__config__.show()`).  
- **Profiler sweep**: run `perf`, `py-spy`, and Rust `flamegraph` on axis-0 float32 and `scale` kernels (threaded & single-threaded). Capture SIMD-off vs SIMD-forced runs for comparison.  
- **Cache/stride instrumentation**: add counters for contiguous vs non-contiguous paths in Raptors’ kernel dispatch to quantify coverage gaps.
- **Benchmark automation**: export CSV summaries and generate comparative plots for daily tracking.

## Phase 2 – SIMD Dispatch & Runtime Feature Detection

- **Runtime CPUID module**: implement a Rust abstraction mirroring NumPy’s `npy_cpu_dispatch`, caching detected features (NEON, SVE, AVX2, AVX-512).  
- **Multi-version builds**: compile SIMD-specialized kernels via `std::simd` or architecture-specific intrinsics behind feature flags.  
- **Dispatch table**: extend kernel registry to select scalar/SIMD implementations per dtype and stride profile.  
- **Validation**: add microbenchmarks ensuring AVX/NEON variants trigger; compare `--simd-mode=force` vs `disable` deltas.

## Phase 3 – Stride-Aware Iterators & Buffering

- **Iterator redesign**: introduce a lightweight equivalent to NumPy’s `PyArrayIter` that precomputes strides/alignment per axis, enabling contiguous fast paths.  
- **Tiled reductions**: implement cache-blocked axis-0 reduction (e.g., column tiling into L1-sized chunks).  
- **Temporary reuse**: create scoped buffer pools for non-contiguous axes to minimize allocations.  
- **Testing**: expand benchmark coverage to mixed layout matrices (Fortran-order, sliced views) to ensure no regressions.

## Phase 4 – BLAS Integration for Dense Ops

- **Backend abstraction**: design a trait for GEMM/AXPY/ASUM/SCAL operations with pluggable providers (OpenBLAS, Accelerate).  
- **FFI binding**: wire Accelerate (macOS) as default; allow opt-in OpenBLAS via feature flag.  
- **Fallback path**: preserve pure-Rust kernels for portability; confirm correctness parity with BLAS via property tests.  
- **Performance validation**: re-run 2D suite focusing on `scale`/axis-0 to confirm offload improvements.

## Phase 5 – Continuous Benchmarking & Regression Gates

- **CI integration**: add nightly job to run `scripts/run_all_benchmarks.sh` with sanitized environment (set `SKIP_MATURIN`, `RAPTORS_THREADS`).  
- **Trend monitoring**: publish speedup charts (Raptors vs NumPy) to a dashboard or simple HTML report stored in `benchmarks/results/latest/`.  
- **Alerting**: set thresholds (e.g., <1.0× speedup on targeted kernels) to fail CI or send notifications.

## Deliverables & Milestones

- **M1 (Phase 1 complete)**: profiling report + automated benchmark CSVs.  
- **M2**: SIMD dispatch infrastructure with ≥1.2× improvement on axis-0 float32 (2k²).  
- **M3**: Stride iterator & tiling in place; no regression on axis-1; ≥1.5× on axis-0 float32 relative to current baseline.  
- **M4**: BLAS-backed scale kernel achieving parity or better vs NumPy across sizes.  
- **M5**: Nightly benchmarks operational with trend reports committed.

## Open Questions / Dependencies

- Platform parity: how to manage differing SIMD sets (e.g., x86 AVX vs ARM NEON/SVE) in a single build pipeline?  
- Licensing & distribution: confirm BLAS integration complies with project licensing.  
- Testing strategy: need deterministic fixtures for verifying SIMD vs scalar equality within floating tolerances.  
- Resource allocation: assign owners per phase (SIMD specialist, FFI owner, CI engineer).

## Next Steps

## Targeted Remediation Plan for Lagging Kernels

### 1. `scale` (elementwise multiply)

- **Gap summary**: 0.57–0.93× vs NumPy across larger sizes, with worst case in threaded, SIMD-disabled mode.  
- **NumPy reference**: their multiply ufunc uses CPU dispatch templates with vector loops and fallback scalar paths selected via `NPY_CPU_DISPATCH_CURFX` macros, plus per-dtype SIMD kernels (`numpy/core/src/common/simd/loops.h`). [`numpy/core/src/common/simd/loops.h`](https://github.com/numpy/numpy/blob/main/numpy/core/src/common/simd/loops.h)  
- **Plan**:
  - Mirror NumPy’s dispatch layering: generate NEON/ASIMD kernels for contiguous strides and use scalar for fringe elements.  
  - Introduce chunked reduction of scalar tail (process `len % vector_width` via masked loads) to keep SIMD utilization high.  
  - When threads are enabled but SIMD off, call into a BLAS `scal`/`axpy` equivalent via Accelerate or OpenBLAS to regain throughput.  
  - Ensure in-place variant uses cache-friendly write-combine and honors alignment checks before selecting SIMD path.

### 2. `broadcast_add`

- **Gap summary**: ~10% slower at 1024² float64 with SIMD forced on 10 threads.  
- **NumPy reference**: broadcasting loops specialize on “all operands contiguous” vs “one broadcast axis” using templated stride iterators in `loops.c.src`, and they prefetch constant operands into SIMD registers. [`numpy/core/src/umath/loops.c.src`](https://github.com/numpy/numpy/blob/main/numpy/core/src/umath/loops.c.src)  
- **Plan**:
  - Detect common broadcast pattern (rhs row vector) and emit a dedicated kernel that vectorizes the inner loop and caches the broadcast slice.  
  - Add per-thread chunk sizing with cache-aware tiling (e.g., 64-column tiles) to reduce cross-thread cache interference.  
  - Validate SIMD forced vs disabled deltas to ensure dispatch actually swaps implementations.

### 3. `mean_axis0` (float64, 2048²)

- **Gap summary**: currently 0.99× speedup – effectively parity, but below our ≥1.1× target.  
- **NumPy reference**: axis-0 reductions reuse the iterator machinery plus block summation (columns processed in tiles) with `npyv_sum_f64` helpers. [`numpy/core/src/common/simd/vsum.h`](https://github.com/numpy/numpy/blob/main/numpy/core/src/common/simd/vsum.h)  
- **Plan**:
  - Implement a two-stage accumulator: SIMD partial sums into per-thread buffers, followed by scalar finalization (reduces precision loss and cache misses).  
  - Experiment with manual L2 blocking (e.g., operate on 32 columns at a time) and compare against current streaming approach.  
  - Integrate fast-path for contiguous C-order matrices to bypass generic iterator overhead.

### 4. Cross-cutting Actions

- Build microbenchmarks that isolate each kernel (scale, broadcast_add, mean_axis0) for float32/float64 and 512–2048 sizes; wire these into CI threshold checks.  
- Diff profiler traces between Raptors and NumPy to identify instruction mix differences (SIMD width, memory loads).  
- Document SIMD dispatch decisions and heuristic tuning so we can iterate quickly as new regressions appear.

### 5. Decision Checkpoints

- **D1**: After SIMD kernel prototypes, confirm `scale` ≥1.05× across all tested sizes.  
- **D2**: After broadcast tiling work, expect ≥1.1× on problematic float64 size with threads.  
- **D3**: After axis-0 redo, confirm ≥1.1× for float64 2048² and no regressions on float32.

1. Log active NumPy BLAS info during benchmarks and commit instrumentation.  
2. Schedule profiling sessions on target hardware; capture baseline flamegraphs.  
3. Draft SIMD dispatch API design proposal for team review.  
4. Plan stakeholder sync to align on BLAS integration scope and timeline.

