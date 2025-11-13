# Performance & Benchmarking Report

## Latest Measurements (RAPTORS_THREADS=10 unless noted)

- `float32` @ `2048²` `mean_axis0`: **0.19 ms** (NumPy 0.30 ms, 1.60×) — BLAS-backed axis-0 GEMV now kicks in automatically for large shapes.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations mean_axis0 --simd-mode force --warmup 2 --repeats 21 --output-json benchmarks/results/dev_plan/axis0_f32_2048.json`)
- `float64` @ `1024²` `mean_axis0`: **0.024 ms** (NumPy 0.14 ms, 7.0×) — new BLAS-backed `dgemv` path handles medium-sized axis reducers.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 --operations mean_axis0 --simd-mode force --warmup 1 --repeats 7`)
- `float64` @ `2048²` `mean_axis0`: **0.38 ms** (NumPy 0.55 ms, 1.41×) — larger reducers stick with the SIMD+tiled path, which now outpaces BLAS.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float64 --operations mean_axis0 --simd-mode force --warmup 1 --repeats 7`)
- `float64` `broadcast_add` (row vector) @ `1024²`, `--simd-mode disable`: **0.64 ms** (NumPy 1.20 ms, 1.87×) — row broadcasts delegate to BLAS `daxpy`, improving the scalar fallback as well as the SIMD build.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 --operations broadcast_add --simd-mode disable --warmup 2 --repeats 21`)
- `float64` `broadcast_add` @ `512²`: **0.15 ms** (NumPy 0.19 ms, 1.23×) — small shapes now bypass BLAS so the SIMD path wins consistently.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float64 --operations broadcast_add --simd-mode force --warmup 1 --repeats 21`)
- `float32` `broadcast_add` (row vector) @ `1024²`, `--simd-mode disable`: **0.26 ms** (NumPy 0.26 ms, 1.02×) — parity within noise; SIMD mode remains the recommended path for float32.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float32 --operations broadcast_add --simd-mode disable --warmup 2 --repeats 21`)
- `float32` `scale` @ `1024²`, `threads=auto`: **0.31 ms** (NumPy 0.40 ms, 1.29×) — smaller Rayon chunks keep all workers busy, pushing us well ahead of NumPy without falling back to BLAS.  
  (`cd benchmarks && ../.venv/bin/asv run --python=../.venv/bin/python --quick --bench bench_scale.ScaleSuite.time_numpy_scale --bench bench_scale.ScaleSuite.time_raptors_scale`)
- `float32` `scale` @ `2048²`, `threads=auto`: **0.44 ms** (NumPy 0.47 ms, 1.08×) — optimized chunk sizing for 2048² uses fewer, larger chunks (2-4 instead of 8+) to reduce threading overhead, now consistently ahead of NumPy.  
  (`PYTHONPATH=python python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations scale --simd-mode auto --warmup 3 --repeats 30`)
- `float32` `broadcast_add` (transpose layout) @ `1024²`: **0.28 ms** (NumPy 0.34 ms, 1.21×) — the new row-tiling keeps small transpose cases ahead.  
  (`python scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float32 --operations broadcast_add --layout transpose --simd-mode auto --warmup 2 --repeats 60`)
- `float32` `broadcast_add` (transpose layout) @ `2048²`: **0.40 ms** (NumPy 0.69 ms, 1.73×) — chunked column tiles reuse cache lines, giving a sizable lead at large sizes.  
  (`python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations broadcast_add --layout transpose --simd-mode auto --warmup 2 --repeats 20`)
- `float64` `scale` @ `512²`: **0.11 ms** (NumPy 0.12 ms, 1.07×) — moved BLAS check before parallel path for small-medium sizes, now consistently ahead of NumPy (improved from 0.97×).  
  (`PYTHONPATH=python python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float64 --operations scale --simd-mode auto --warmup 3 --repeats 50`)
- `float32` @ `512²` `mean_axis0`: **0.021 ms** (NumPy 0.029 ms, 4.8×) — unchanged small-matrix performance with SIMD lanes.  
  (`PYTHONPATH=python RAPTORS_THREADS=10 python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations mean_axis0 --simd-mode force --warmup 2 --repeats 21`)
- `float32` `broadcast_add` (column, contiguous) @ `512²`: **0.03 ms** (NumPy 0.05 ms, 1.57×) — the small-matrix path now short-circuits Rayon setup and stays SIMD-only.  
  (`python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations broadcast_add --layout contiguous --simd-mode auto --warmup 3 --repeats 50`)
- `float32` `scale` @ `512²`, `--simd-mode disable`: **0.01 ms** (NumPy 0.02 ms, 3.13×) — scalar fallback uses Accelerate/BLAS directly for small matrices, beating NumPy by a wide margin.  
  (`python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations scale --simd-mode disable --warmup 3 --repeats 50`)
- `float32` `scale` @ `1024²`, `--simd-mode disable`: **0.29 ms** (NumPy 0.34 ms, 1.19×) — mid-sized scalar scale reuses parallel chunking with BLAS fallback, maintaining advantage over NumPy.  
  (`python scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float32 --operations scale --simd-mode disable --warmup 3 --repeats 40`)
- `float32` `scale` @ `2048²`, `--simd-mode disable`: **0.32 ms** (NumPy 0.36 ms, 1.11×) — large scalar scale benefits from parallel chunking, now ahead of NumPy.  
  (`python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations scale --simd-mode disable --warmup 3 --repeats 30`)
- `float32` `broadcast_add` @ `512²`, `--simd-mode force`: **0.02 ms** (NumPy 0.03 ms, 1.53×) — forced SIMD on small matrices uses optimized direct path, beating NumPy consistently.  
  (`python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations broadcast_add --layout contiguous --simd-mode force --warmup 3 --repeats 50`)
- `float32` `broadcast_add` @ `512²`, `RAPTORS_THREADS=4`: **0.02 ms** (NumPy 0.03 ms, 1.56×) — thread override on small matrices maintains performance advantage.  
  (`RAPTORS_THREADS=4 python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations broadcast_add --layout contiguous --simd-mode auto --warmup 3 --repeats 50`)
- `float64` `broadcast_add` @ `2048²`, `simd-mode auto`: **2.62 ms** (NumPy 3.44 ms, 1.31× average) — extended parallel tiling with float64-specific cache alignment eliminates variance; now consistently ahead of NumPy (1.15× - 1.34× range across runs).  
  (`PYTHONPATH=python python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float64 --operations broadcast_add --layout contiguous --simd-mode auto --warmup 3 --repeats 30`)

Axis-0 reducers now prefer the available BLAS backend once row/column thresholds are met, while row broadcast adds reuse BLAS `axpy` routines (with scalar fallbacks if BLAS is disabled).  Scaling keeps the SIMD/parallel heuristics by default; opt in with `RAPTORS_BLAS_SCALE=1` when the downstream BLAS beats the native path for a given matrix shape.  Large float64 broadcasts now use parallel tiling with cache-aligned chunking, matching float32 performance characteristics.

The SIMD suite continues to use `--simd-mode force`; pinning threads via `RAPTORS_THREADS=10` matches the latest guardrail runs and reduces variance.

## Overview

This document tracks the current baseline goals for Raptors' dense 2‑D kernels.  The targets mirror the JSON baselines consumed by CI and reflect the maximum wall-clock time (in milliseconds) allowed on release builds using Python 3.12 (`--warmup 1 --repeats 7`).

- **SIMD (force-enabled)** — highlights (see JSON for the complete list):
  - `float64` @ `512²`: `sum` ≤ 0.028 ms, `broadcast_add` ≤ 0.170 ms
  - `float64` @ `1024²`: `sum` ≤ 0.099 ms, `broadcast_add` ≤ 0.900 ms, `scale` ≤ 0.572 ms
  - `float64` @ `2048²`: `sum` ≤ 0.349 ms, `mean` ≤ 0.363 ms, `scale` ≤ 1.500 ms, `broadcast_add` ≤ 3.200 ms
  - `float32` @ `2048²`: `mean_axis0` ≤ 0.240 ms, `scale` ≤ 0.340 ms
- **Scalar fallback**: `float64` @ `2048²` reducers ≤ 3.454 ms, `float32` @ `2048²` `sum` ≤ 0.324 ms, `broadcast_add` ≤ 0.951 ms.

CI benchmarks (`ci/github-actions.yml`) run `python scripts/validate_benchmark_results.py --slack 0.05 --absolute-slack-ms 0.05`, failing the build whenever recorded mean timings drift outside the tolerated envelope.

## Running Benchmarks Locally

```bash
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode force \
  --warmup 1 --repeats 7 \
  --output-json benchmarks/results/local_simd.json
python scripts/validate_benchmark_results.py --results-dir benchmarks/results \
  --baseline-dir benchmarks/baselines --slack 0.05 --absolute-slack-ms 0.05
```

After generating JSON artefacts, summarize the regressions with:

```bash
python scripts/summarize_benchmarks.py --sub-one \
  --output-csv benchmarks/results/summary_subone.csv
```

Repeat the command with the float32 and scalar baselines to mirror CI coverage:

```bash
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode force \
  --warmup 1 --repeats 7 \
  --output-json benchmarks/results/local_simd_f32.json
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode disable \
  --warmup 1 --repeats 7 \
  --output-json benchmarks/results/local_scalar_f64.json
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode disable \
  --warmup 1 --repeats 7 \
  --output-json benchmarks/results/local_scalar_f32.json
python scripts/validate_benchmark_results.py --slack 0.05 --absolute-slack-ms 0.05
```

Use `--layout transpose` or `--layout fortran` with `scripts/compare_numpy_raptors.py` to exercise non-C-contiguous NumPy inputs when chasing stride-sensitive regressions.

### Targeted Harnesses

- **Axis-0 float32 reducers**:  
  `RAPTORS_THREADS=8 python benchmarks/run_axis0_suite.py --simd-mode force --output-json benchmarks/results/axis0_latest.json`  
  The harness mirrors CI shapes (512²/1024²/2048²) and is the fastest way to compare new column kernels or chunking tweaks before re-enabling parallel execution.
- **Broadcast add spot checks**: reuse `scripts/compare_numpy_raptors.py` with `--operations broadcast_add` and pass `--output-json` so telemetry is persisted for CI.
- All persisted JSON artefacts are linted in CI via `python scripts/validate_benchmark_results.py`, which now also enforces the baseline regression guardrails.

## Diagnostics Toolkit

- `python scripts/compare_numpy_raptors.py --log-numpy-config` now captures NumPy BLAS/LAPACK build info and stores it under the `metadata.numpy_config` block in each JSON artefact.
- `raptors.threading_info()["stride_counters"]` exposes contiguous vs strided dispatch counts per kernel, `["simd_capabilities"]` reports the detected AVX/NEON level, `["axis_tile_histogram"]` tracks column tiling widths, and `["blas_backend"]` reflects the active BLAS provider.
- `python scripts/summarize_benchmarks.py results.json --sub-one --plot benchmarks/results/latest/slowdowns.svg` emits both CSV summaries and a bar chart of the slowest entries (optional matplotlib dependency).
- `python scripts/profile_hotspots.py --operation axis0 --tool py-spy --threads 8` records a flamegraph for the axis-0 reducer; swap `--tool perf` and provide `--flamegraph-output` to collect Linux perf data and SVGs for audits.
- `bash scripts/run_all_benchmarks.sh --output-dir benchmarks/results/latest` mirrors the nightly CI run and emits JSON, CSV, SVG, and an `index.html` dashboard summarizing the latest speedups. Validation is now enabled by default; only pass `--skip-validate` when intentionally collecting repro data for an in-progress regression.
- `benchmarks/results/dev_scale/`, `benchmarks/results/dev_plan/`, and `benchmarks/results/scale1024/` contain the latest targeted JSON dumps collected while tuning scale, axis-0, and broadcast heuristics.

## Adaptive Threading Controls

- `RAPTORS_SIMD=0|1` — force scalar/SIMD execution.
- `RAPTORS_SIMD_MAX=<level>` — cap SIMD dispatch to `scalar`, `sse4.1`, `avx`, `avx2`, `avx512`, or `neon` (default: auto-detect).
- `RAPTORS_BLAS=auto|accelerate|openblas|none` — pick the BLAS backend; `auto` prefers Accelerate on macOS and OpenBLAS when the optional feature is enabled.
- `RAPTORS_BLAS_SCALE=0|1` — opt-in to BLAS-backed scaling; now defaults to `0` because SIMD/parallel paths outperform Accelerate for 512²–2048².
- `RAPTORS_THREADS=<n>` — pin Rayon pool size.
- `RAPTORS_MATRIXMULTIPLY=0|1` — enable/disable the matrixmultiply-backed axis-0 reducers (default: on).
- `RAPTORS_MATRIXMULTIPLY_SCALE=0|1` — opt-in to experimental GEMM-based scaling (default: off).
- `RAPTORS_MATRIXMULTIPLY_BROADCAST=0|1` — opt-in to experimental column broadcast replication via matrixmultiply (default: off).
- `raptors.threading_info()` — Python helper returning live heuristics:
  - Baseline cutovers per dtype.
  - Adaptive median and P95 throughput samples (elements/ms) for global reducers *and* per-operation telemetry covering `scale`, broadcast row/column variants, and axis reducers, plus variance ratios (P95 ÷ median) to highlight jitter.
  - Last operation metadata (tiles processed, partial buffer width, elapsed time); useful when comparing harness runs locally.

## Hardware Notes

- **x86-64**: AVX2 is the default floor; AVX-512 kernels activate automatically when available.
- **Apple Silicon / ARM64**: NEON-backed reducers tile to 128-bit lanes.  SIMD detection can be bypassed via `RAPTORS_SIMD=0` to compare scalar fallback behaviour.
- **Cross-platform**: Baselines are tuned for M3- and AVX2-class hardware.  If CI variance exceeds slack thresholds, capture local runs and update the JSON targets accordingly.

