## Objective

Restore and extend Raptors' 2D matrix performance relative to NumPy by porting key runtime optimizations (SIMD dispatch, stride iteration, BLAS offload, and benchmarking discipline) identified from the NumPy codebase.

## Current State Snapshot (2025-11-12 benchmark run)

- Benchmark artefacts: `benchmarks/results/20251112-095127/`.
- Strengths: Raptors consistently outperforms NumPy on global reductions and axis-1 means (≈2–3× for float32). Single-threaded reductions also lead.
- Gaps: Axis-0 float32 reductions regress (~0.76–0.9× at 2k²). `scale` (elementwise multiply) lags by 10–40% in several scenarios. SIMD disable mode shows limited degradation, implying our current SIMD paths are narrow or unused.

## Guiding Principles

1. Maintain ergonomic Python API; focus on backend improvements.
2. Target measurable speedups via reproducible benchmarks.
3. Prefer incremental integration with validation at each layer.

## Phase 1 – Diagnostics & Baseline Hardening

- **Reconfirm BLAS backend usage**: instrument `compare_numpy_raptors.py` to log NumPy’s active BLAS (`numpy.__config__.show()`).  
- **Profiler sweep**: run `perf`, `py-spy`, and Rust `flamegraph` on axis-0 float32 and `scale` kernels (threaded & single-threaded).  
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

1. Log active NumPy BLAS info during benchmarks and commit instrumentation.  
2. Schedule profiling sessions on target hardware; capture baseline flamegraphs.  
3. Draft SIMD dispatch API design proposal for team review.  
4. Plan stakeholder sync to align on BLAS integration scope and timeline.

