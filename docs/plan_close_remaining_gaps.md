# Plan: Closing Remaining Performance Gaps vs. NumPy

## Objectives
- Eliminate identified deficits where Raptors remains slower than NumPy on 2-D float workloads.
- Lock in repeatable benchmarks by pinning regression tests and telemetry so adaptive heuristics converge quickly.
- Prepare tooling and validation updates so improvements are automatically guarded in CI.

## Current Gaps (RAPTORS_THREADS=8, SIMD forced)
- **Float32 512²**: `mean_axis0` (0.83×), `broadcast_add` (0.73×), `scale` (0.90×).
- **Float32 1024²**: `mean_axis0` (0.72×), `scale` (0.80×).
- **Float32 2048²**: `mean_axis0` (0.41×) remains the largest deficit; `scale` slightly behind (0.91×) but close.
- **Float64 512²**: `mean` (0.72×) and `scale` (0.97×) trail slightly.
- **Float64 2048²**: `mean_axis0` (0.71×) lags.

## Workstreams

1. **Float32 Axis-0 Reducer Rewrite**
   - Profile current hybrid reducer to isolate cache stalls and gather threading telemetry per tile size.
   - Introduce dedicated column kernels (SIMD + software pipelining) with row blocking tuned for 512²–2048².
   - Update adaptive cutovers to prefer sequential stripes for ≤1024² when parallel throughput does not exceed sequential median.
   - Add regression benchmarks targeting axis0 on 512²/1024²/2048².

2. **Float32 Scale Mid-Size Optimisation**
   - Extend scale SIMD kernels with prefetch/unroll variants (AVX2/NEON) and trial software pipelining for 1K-sized rows.
   - Teach adaptive state to capture per-row variance and select sequential vs. parallel on-the-fly (reuse new telemetry channel).
   - Validate improvements with microbenchmarks (single-row, tiled) before full suite runs.

3. **Float32 Broadcast Add (512²)**
   - Investigate column broadcast scheduling for mid-size matrices; adjust row batching and memory reuse.
   - Consider fallback to sequential SIMD when Rayon startup cost dominates; gate via recorded throughput.
   - Add targeted benchmarks covering `broadcast_add` at 512² and 768² to ensure stability.

4. **Float64 Small Matrix Mean/Scale**
   - Implement more aggressive small-matrix fast paths (SIMD-only loop, no allocation).
  - Capture sequential telemetry for 512² to avoid unnecessary parallel work.
   - Add regression tests ensuring no accuracy regressions from loop unrolling.

5. **Tooling & Automation**
   - Extend benchmark harness to emit structured gap reports (shape/op/dtype/time delta).
   - Update CI guardrails: fail when any tracked case regresses below 1.0× after fixes land.
   - Document new environment knobs and telemetry usage in `docs/perf_report.md`.

## Milestones
1. **Week 1**: Complete axis-0 profiling, prototype sequential SIMD kernel, land telemetry-driven cutover updates.
2. **Week 2**: Finalize float32 scale improvements and broadcast add heuristics; validate with targeted benches.
3. **Week 3**: Address float64 small-matrix paths; refresh documentation and baselines; enable CI gap detection.

## Risks & Mitigations
- **Telemetry Variance**: noisy benchmarks can skew adaptive thresholds. *Mitigation*: median-of-medians aggregation, minimum sample counts before adopting new cutovers.
- **Architecture Divergence**: x86 vs. aarch64 SIMD implementations may diverge. *Mitigation*: add arch-specific microbenches and require parity within guardrails.
- **Regression in Large Matrices**: tuning for mid-size may hurt 2K²+ performance. *Mitigation*: include 2048² checks in the new benchmarking workflow and run `--suite 2d` before merge.

