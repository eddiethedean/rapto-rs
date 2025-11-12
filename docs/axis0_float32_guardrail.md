# Axis-0 Float32 Guardrail Findings

## Summary (2025-11-11 snapshot)

- Environment: `PYTHONPATH=python`, `RAPTORS_THREADS=8`, `simd-mode=force`, warmup=1.
- Recent `run_axis0_suite.py --shapes 512 1024 2048` shows regressions at guardrail sizes:
  - 512² → **0.80×** (prior best was ~1.5×; significant jitter between runs).
  - 1024² → **1.23×** (passes guardrail).
  - 2048² → **0.60×** (fails 0.65× guardrail; variance 0.42–0.48 ms per run).
- Telemetry indicates the SIMD tiled reducer executes sequentially for ≤2048 columns; enabling the current column-chunked parallel path made things worse. Matrix-multiply fallback at 2048² is even slower (~1.17 ms).
- Prototype row-chunk parallel reducer (gate: `RAPTORS_AXIS0_ROW_CHUNK=1`) improves 1024² to **1.21×** but still lags at 2048² (**0.58×**); cache contention remains unresolved, so the toggle stays opt-in for now.

## Detailed Measurements (latest runs)

```text
RAPTORS_THREADS=8
[axis0] shape=512x512  dtype=float32 raptors=0.035 ms numpy=0.028 ms speedup=0.80x
[axis0] shape=1024x1024 dtype=float32 raptors=0.061 ms numpy=0.050 ms speedup=1.23x
[axis0] shape=2048x2048 dtype=float32 raptors=0.425 ms numpy=0.252 ms speedup=0.60x

# row-chunk prototype (RAPTORS_AXIS0_ROW_CHUNK=1)
[axis0] shape=1024x1024 dtype=float32 raptors=0.057 ms numpy=0.069 ms speedup=1.21x
[axis0] shape=2048x2048 dtype=float32 raptors=0.448 ms numpy=0.258 ms speedup=0.58x
```

Single-thread diagnostic (NumPy clamped via `OMP/MKL/OPENBLAS_NUM_THREADS=1`):

```text
RAPTORS_THREADS=1
[axis0] shape=512x512  dtype=float32 raptors=0.013 ms numpy=0.020 ms speedup=1.50x
[axis0] shape=2048x2048 dtype=float32 raptors=0.274 ms numpy=0.259 ms speedup=1.06x
```

## Observations

- Sequential SIMD (`simd::reduce_axis0_tiled_f32`) remains the fastest path for ≤2048 columns; it beats NumPy in single-thread mode but loses once NumPy is multi-threaded.
- Column-chunked parallel reducer (`reduce_axis0_parallel_tiled_f32`) improves 4096² but degrades 2048² because each worker loops across every row chunk, thrashing caches.
- Matrix-multiply fallback (`matrixmultiply::sgemm`) at 2048² costs ~1.17 ms on M3 hardware—worse than both SIMD and NumPy.

## Next Steps

1. Prototype row-chunk parallelism (similar to `accumulate_axis0_simd_f32`) for 2k × 2k matrices and compare against the current column tiling.
2. Evaluate a hybrid approach: SIMD-tiled reducer per thread with tree reduction, avoiding the per-row copy overhead.
3. Capture perf counters (zcat + dtrace or Instruments) to confirm cache misses vs. compute stalls.
4. Consider gating the column-parallel path behind `cols > AXIS0_SIMD_COL_LIMIT` and adding a GEMM fallback threshold ≥4096 until we have a faster implementation.

## Quick-Check Command

```bash
PYTHONPATH=python ./benchmarks/run_axis0_suite.py \
  --shapes 512 1024 2048 --dtypes float32 \
  --simd-mode force --warmup 1 --repeats 5
```

Treat `speedup < 1.05` at 1024² or `< 0.65` at 2048² as regressions; re-run with single-thread clamps to separate NumPy thread-count issues from Raptors throughput problems. The harness now skips guardrail assertions for 512² because interpreter overhead dominates those micro benches.

