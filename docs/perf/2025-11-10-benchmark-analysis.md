# 2025-11-10 Benchmark Analysis

## Summary

Fresh runs on Apple silicon (Python 3.11, SIMD forced) show that the outstanding float64 mid-size regressions are resolved and the small float32 axis reducers now clear NumPy. Key deltas (median of `--warmup 5 --repeats 20` micro-runs unless noted):

- 1024² float32 `mean_axis0`: **0.05 ms vs 0.08 ms** (1.56×) – f32-native accumulators + SIMD column folds.
- 2048² float32 `mean_axis0`: **0.21 ms vs 0.25 ms** (1.14×) – parallel column reducers kept.
- 1024² float64 `mean`: **0.10 ms vs 0.13 ms** (1.26×) – tiled reducer no longer bottlenecked.
- 2048² float64 `scale`: **~1.30 ms vs 2.35 ms** (≈1.8×) – wider unroll, prefetch, and row-level Rayon.
- 1024² float32 `broadcast_add`: **0.28 ms vs 0.30 ms** (1.06×) – column broadcasts now go parallel.
- 512² float32 `broadcast_add`: **0.04 ms vs 0.03 ms** (0.85×) – still a remaining gap, though reduced from 0.72×.
- 512² float64 `scale`: **0.15 ms vs 0.14 ms** (0.92×, up from 0.69×) as the tiny sequential fast path kicks in.
- 512² float32 `scale`: **0.02 ms vs 0.02 ms** (~1.2×) after the same tiny fast path.

Scalar fallbacks stay within their baselines; residual gaps are concentrated in the smallest column broadcasts and mid-size float64 scale variance.

## Observations

### Global Reductions (`sum` / `mean`)

- 512² float64 reductions stay in the stack fast path (~0.024 ms) while the tiled reducer now wins by ~25 % at 1024² thanks to smaller direct chunks.
- Global means reuse the same architecture for sum; residual noise at 1024² is now gearing from input variance rather than the reducer.
- Scalar fallbacks remain comfortably below their JSON baselines.

### Axis Reductions (`mean_axis0` / `mean_axis1`)

- float32 axis‑0 avoids f64 promotion entirely; sequential SIMD handles 512² (1.3–1.4×) while column-fold + Rayon handles ≥1024² (1.1–1.6×).
- Telemetry gating guards against over-eager threading—parallel only kicks in when rows or cols exceed the 768/1024 guardrails.
- float64 axis reducers pick up the float32 plumbing without regressions.

### Broadcast & Scale Kernels

- Column broadcasts now choose between sequential SIMD and per-block Rayon. 1024² and 2048² float32 editions are ≥1× (the latter hits ~3×); 512² still trails (~0.85×), so NEON/x86 micro-kernels for the tiniest cases remain on the shortlist.
- Scale kernels gained a dedicated sequential path for matrices ≤512² while keeping the row-parallel SIMD path for large shapes. 512² float64 climbs to ~0.9×, float32 clears ~1.2×, and 2K matrices still benefit from Rayon.

### Adaptive Threading Heuristics

- `scale_parallel_policy` prefers sequential execution until it sees a pair of parallel wins, preventing small-case regressions.
- `broadcast_parallel_policy` now demands at least six sequential samples before vetoing parallelism for small matrices, keeping 512² sequential while letting the 1K/2K cases scale out.
- Latest snapshot lives at `docs/perf/2025-11-10-threading-after-axis-scale.json` for regression triage.

## Next Steps

1. **Mid-size scaling:** Nudge `scale_parallel_policy` further so 1024² float32 clears 1× without giving back the 2K wins; investigate variance sources in the float64 path.
2. **Small column broadcasts:** Micro-tune the NEON kernel (alignment + prefetch) so 512² float32 matches the x86 gains.
3. **Regression guardrails:** Refresh baselines with the new medians, keep the validator in CI, and add a smoke test for the column broadcast parallel path.

These refinements keep the focus on locking in the new wins while trimming the remaining mid-size scaling regression.

