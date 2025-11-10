# 2025-11-10 Benchmark Analysis

## Summary

Fresh benchmark runs on Apple silicon (`/opt/anaconda3` Python 3.12, SIMD forced, `--repeats 7`) show clear improvements for global reductions and column broadcasts, but several workloads remain slower than NumPy. Highlights (NumPy vs. Raptors mean wall time, milliseconds):

| Shape | Dtype     | Operation        | NumPy | Raptors | Δ | Notes |
|-------|-----------|------------------|-------|---------|---|-------|
| 512²  | float64   | `sum`            | 0.03  | 0.04    | +0.01 | Regression smaller but still behind |
| 512²  | float64   | `mean_axis0`     | 0.04  | 0.03    | -0.01 | Now faster than NumPy |
| 1024² | float64   | `sum`            | 0.13  | 0.26    | +0.13 | Direct SIMD path still loses |
| 1024² | float64   | `broadcast_add`  | 1.27  | 0.66    | -0.61 | >1.9× speedup after SIMD tuning |
| 2048² | float64   | `scale`          | 3.23  | 3.59    | +0.36 | SIMD scaling needs further work |
| 1024² | float32   | `mean_axis0`     | 0.08  | 0.13    | +0.05 | Axis reducers for f32 remain costly |
| 2048² | float32   | `mean_axis0`     | 0.29  | 0.42    | +0.13 | Rayon disabled, still slower |
| 2048² | float32   | `scale`          | 0.37  | 0.44    | +0.07 | SIMD scaling lag |

Scalar fallback retains similar gaps (e.g., `sum @ 2048² float32` ≈0.26 ms vs. NumPy 0.20–0.22 ms) but now stays under renewed baseline thresholds.

## Observations

### Global Reductions (`sum` / `mean`)

- A new “direct” SIMD fast path now handles arrays up to 262 k elements (512²), removing the large overhead previously introduced by the tiled reducer. This cuts `sum(512²)` from ~1.3 ms to ~0.04 ms.
- Larger shapes still route through the tiled/parallel path. For 1024² we remain slower than NumPy: the Rayon fan-out and partial-buffer bookkeeping dominate the simple contiguous loop NumPy uses.
- Accumulator counts remain high (5–8 lanes) which hurts cache residency for moderate matrices; we now clamp the rotating buffer to `min(rows, accumulators)` to reduce churn.

### Axis Reductions (`mean_axis0` / `mean_axis1`)

- Parallel execution for axis reducers is now gated by a higher per-axis cutover (e.g., float64 axis-0 requires ≥2048 rows) to avoid Rayon overhead on 512²/1024² shapes.
- Sequential row sums use explicit SIMD accumulators, removing the earlier `iter().sum()` bottleneck. float64 axis reducers now beat NumPy for 512² and 2048².
- float32 axis reducers still lag: `mean_axis0` @ 2048² ≈0.42 ms vs. NumPy 0.29 ms. The remaining cost comes from promoting `f32` values to `f64` and the lack of a dedicated SIMD column kernel.

### Broadcast & Scale Kernels

- Column broadcasts now leverage SIMD inside the Rayon fallback, producing >1.9× speedup for 1024² float64 and eliminating the previous baseline failures. float32 broadcasts are borderline (~1.09× at 2048²) and still worth revisiting.
- Scaling remains our weakest kernel: float64 2048² sits at ~3.6 ms vs. NumPy ~3.2 ms, and float32 2048² is ~0.44 ms vs. 0.37 ms. The SIMD helpers do not parallelise across rows yet, so we rely on a single-core loop.

### Adaptive Threading Heuristics

- Global reducers now record telemetry for the direct/tiled paths, enabling adaptive cutovers for future tuning. Axis reducers still skip telemetry—addressing this is a follow-up.
- Tile descriptors continue to bias wide tiles (e.g., 96×192 on NEON); for 1024² this still produces more column tiles than we need. Future work: make tiling dynamic based on `rows`/`cols`.

## Next Steps

1. **Global reducers:** keep parallel sums for ≥1024² but revisit Rayon's chunk sizes—current tiling still doubles NumPy’s runtime for 1024².
2. **float32 axis reducers:** avoid `f32`→`f64` promotion by keeping partial sums in `f32` when accuracy permits, and add SIMD column kernels for `mean_axis0`.
3. **Scaling kernels:** experiment with row-parallel scaling (Rayon + SIMD) and tune accumulator usage to close the remaining 10–15% gap.
4. **Telemetry & CI:** log axis reducer events so adaptive thresholds converge, and track benchmark JSON artefacts in CI for historical comparison.

These findings guide the algorithmic work planned for Step 2 (tile/accumulator tuning, specialised axis paths) and Step 3 (benchmark reruns and CI tightening).

