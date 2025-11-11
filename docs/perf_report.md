# Performance & Benchmarking Report

## Latest Measurements (RAPTORS_THREADS=8, see notes per run)

- `float32` @ `1024²` `mean_axis0`: **0.05 ms** (NumPy 0.08 ms, 1.56×) — sequential SIMD for small, Rayon column fold for ≥1 K rows.  
  (`compare_numpy_raptors.py --warmup 5 --repeats 20`)
- `float32` @ `2048²` `broadcast_add`: **0.35 ms** (NumPy 0.51 ms, 1.46×) — block-parallel column kernel.  
  (`compare_numpy_raptors.py --warmup 2 --repeats 5`)
- `float64` @ `2048²` `scale`: **≈1.3 ms** (NumPy ≈2.4 ms, ~1.8×) — wider unroll + row parallelism.  
  (median across multiple runs; see `benchmarks/results/latest_full_simd.json`)
- `float64` @ `512²` `scale`: **0.15 ms** (NumPy 0.14 ms, ~0.92×) — the new tiny sequential fast path replaces the old 0.69× regression.
- Small gaps remain: `float32` @ `512²` `broadcast_add` ≈0.85× (tiny column path) and `float32` @ `1024²` `scale` still hovers around parity.

The SIMD suite continues to use `--simd-mode force`; pinning threads via `RAPTORS_THREADS=8` is now recommended to limit scheduling variance during regression checks.

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

### Targeted Harnesses

- **Axis-0 float32 reducers**:  
  `RAPTORS_THREADS=8 python benchmarks/run_axis0_suite.py --simd-mode force --output-json benchmarks/results/axis0_latest.json`  
  The harness mirrors CI shapes (512²/1024²/2048²) and is the fastest way to compare new column kernels or chunking tweaks before re-enabling parallel execution.
- **Broadcast add spot checks**: reuse `scripts/compare_numpy_raptors.py` with `--operations broadcast_add` and pass `--output-json` so telemetry is persisted for CI.
- All persisted JSON artefacts are linted in CI via `python scripts/validate_benchmark_results.py`, which now also enforces the baseline regression guardrails.

## Adaptive Threading Controls

- `RAPTORS_SIMD=0|1` — force scalar/SIMD execution.
- `RAPTORS_THREADS=<n>` — pin Rayon pool size.
- `raptors.threading_info()` — Python helper returning live heuristics:
  - Baseline cutovers per dtype.
  - Adaptive median and P95 throughput samples (elements/ms) for global reducers *and* per-operation telemetry covering `scale`, broadcast row/column variants, and axis reducers, plus variance ratios (P95 ÷ median) to highlight jitter.
  - Last operation metadata (tiles processed, partial buffer width, elapsed time); useful when comparing harness runs locally.

## Hardware Notes

- **x86-64**: AVX2 is the default floor; AVX-512 kernels activate automatically when available.
- **Apple Silicon / ARM64**: NEON-backed reducers tile to 128-bit lanes.  SIMD detection can be bypassed via `RAPTORS_SIMD=0` to compare scalar fallback behaviour.
- **Cross-platform**: Baselines are tuned for M3- and AVX2-class hardware.  If CI variance exceeds slack thresholds, capture local runs and update the JSON targets accordingly.

