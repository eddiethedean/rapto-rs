# Float64/Axis Guardrail Follow-Up Plan

## Objectives
- Restore float32 axis-0 guardrail compliance (>=1.05× at 1024², >=0.65× at 2048²).
- Eliminate remaining sub-1× cases in `scale` benchmarks (float64 + float32, SIMD & scalar).
- Maintain new float64 sum/mean wins while guarding against regressions.

## Current State
- `benchmarks/run_axis0_suite.py` fails for float32 @ 1024² (0.71×) and 2048² (0.41×).
- `compare_numpy_raptors.py` shows `scale` slower than NumPy in several shapes (e.g. float64 512²/1024², float32 1024²/2048² scalar SIMD).
- Float64 global reducer now ~1.2–1.7× faster than NumPy across tested sizes; telemetry reflects chunked SIMD path.

## Action Items
1. **Axis-0 float32 regression**
   - Re-profile `reduce_axis0_parallel_tiled_f32` vs sequential path for 1024² / 2048².
   - Inspect chunk size heuristics (`AXIS0_PAR_MIN_ROWS`, column tiling) and SIMD accumulation.
   - Prototype GEMM fallback or hybrid row/column sum at large widths; compare in microbenchmarks.
2. **Scale operation slowdowns**
   - Audit `NumericArray::scale` dispatch: SIMD path vs fallback vs parallel chunking for float64/float32.
   - Measure impact of `scale_same_shape_*` vector kernels and identify hotspot (likely copy or chunk scheduling).
   - Implement improved tiling/parallel thresholds or fused multiply kernels; re-run suite to confirm >1×.
3. **Regression safety**
   - Add targeted benchmark snippets (1024² & 2048² float32 axis-0, scale) to quick-check script.
   - Capture telemetry snapshots after each tuning round for trend tracking.

## Validation
- Rebuild (`cargo build`, `maturin develop`) before measurement.
- Run `scripts/compare_numpy_raptors.py` (SIMD force/disable) and `benchmarks/run_axis0_suite.py` (SIMD force).
- Update summaries (`scripts/summarize_benchmarks.py`) and ensure all guardrail assertions pass.


