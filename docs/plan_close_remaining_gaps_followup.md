# Raptors > NumPy Roadmap

## 1. Refresh Baselines & Envelope
- Re-run `scripts/compare_numpy_raptors.py --suite 2d` for SIMD and scalar targets, capturing fresh results under `benchmarks/results/latest_{simd,scalar}.json`.
- Execute `benchmarks/run_axis0_suite.py` to gather focused axis-0 telemetry.
- Summarize deltas versus NumPy in `docs/perf/2025-11-10-gap-audit.md`, calling out the worst offenders (currently float32 axis-0 reducers and float32 broadcast add at mid/large sizes).

## 2. Float32 Axis-0 Column Kernel
- Implement architecture-specific SIMD column reducers in `rust/src/simd/mod.rs` (AVX2/AVX-512 on x86_64, NEON on aarch64) with tiling sized for L1/L2 residency.
- Integrate kernels into `reduce_axis0_f32`, including shape heuristics and adaptive telemetry emission via `record_axis_event`.
- Extend `benchmarks/run_axis0_suite.py` assertions and refresh regression JSON baselines to cover the new path.

## 3. Broadcast Add Scheduling
- Refine `try_simd_add_column` and `try_simd_add_row` for float32 by experimenting with row/column block sizes, sequential fallbacks, and prefetch depth.
- Feed per-kind telemetry into `broadcast_parallel_policy`, ensuring Rayon spin-up responds to fresh medians.
- Add targeted coverage in `tests/test_reductions.py` and guard with benchmark comparisons.

## 4. Scale Mid-Size Prefetch & Heuristics
- Explore wider unrolling and software prefetch in `simd::scale_same_shape_f32` across x86 and aarch64 backends.
- Enhance `scale_parallel_policy` to model variance (e.g., P95 throughput) and automatically adjust cutovers; expose snapshots through `raptors.threading_info()`.

## 5. Float64 Small-Matrix Fast Paths
- Audit `reduce_axis0_f64` and scale operations for shapes ≤256², adding stack-allocated accumulators and scalar fallbacks where SIMD is overkill.
- Cover the new fast paths with additional cases in `tests/test_reductions.py` and microbenchmarks.

## 6. Tooling & Guardrails
- Extend `scripts/validate_benchmark_results.py` to compare new runs against baselines with tolerance bands.
- Add CI enforcement in `ci/github-actions.yml` to fail on benchmark regressions; document the workflow in `docs/perf_report.md`.
- Publish telemetry snapshot recipes in `docs/perf/2025-11-10-measurement-checklist.md`.

## 7. Final Validation & Landing
- Run `run_all_tests.sh`, full benchmark suites, and collect `raptors.threading_info()` dumps before/after changes.
- Update `docs/perf/2025-11-10-benchmark-analysis.md` with final speedups and prepare release notes / PR summary.

