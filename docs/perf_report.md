# Performance & Benchmarking Report

## Overview

This document tracks the current baseline goals for Raptors' dense 2‑D kernels.  The targets mirror the JSON baselines consumed by CI and reflect the maximum wall-clock time (in milliseconds) allowed on release builds using Python 3.12 (`--warmup 1 --repeats 7`).

- **SIMD (force-enabled)** — highlights:
  - `float64` @ `512²`: `sum` ≤ 0.053 ms, `broadcast_add` ≤ 0.209 ms
  - `float64` @ `1024²`: `sum` ≤ 0.322 ms, `broadcast_add` ≤ 0.827 ms, `scale` ≤ 0.872 ms
  - `float64` @ `2048²`: `sum` ≤ 0.356 ms, `mean` ≤ 0.361 ms, `broadcast_add` ≤ 4.134 ms
  - `float32` baselines (see JSON) remain looser for axis reducers where we trail NumPy (e.g., `mean_axis0` @ `2048²` ≤ 0.527 ms).
- **Scalar fallback**: `float64` @ `2048²` reducers ≤ 3.454 ms, `float32` @ `2048²` `sum` ≤ 0.324 ms, `broadcast_add` ≤ 0.951 ms.

CI benchmarks (`.github/workflows/bench.yml`) now validate against these regenerated baselines with a 0.05 ms slack.

## Running Benchmarks Locally

```bash
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode force \
  --warmup 1 --repeats 7 \
  --output-json results_simd.json \
  --validate-json benchmarks/baselines/2d_float64.json --validate-slack 0.05
```

Repeat the command with the float32 and scalar baselines to mirror CI coverage:

```bash
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode force \
  --warmup 1 --repeats 7 \
  --output-json results_simd_f32.json \
  --validate-json benchmarks/baselines/2d_float32.json --validate-slack 0.05
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode disable \
  --warmup 1 --repeats 7 \
  --output-json results_scalar_f64.json \
  --validate-json benchmarks/baselines/2d_float64_scalar.json --validate-slack 0.05
python scripts/compare_numpy_raptors.py --suite 2d --simd-mode disable \
  --warmup 1 --repeats 7 \
  --output-json results_scalar_f32.json \
  --validate-json benchmarks/baselines/2d_float32_scalar.json --validate-slack 0.05
```

## Adaptive Threading Controls

- `RAPTORS_SIMD=0|1` — force scalar/SIMD execution.
- `RAPTORS_THREADS=<n>` — pin Rayon pool size.
- `raptors.threading_info()` — Python helper returning live heuristics:
  - Baseline cutovers per dtype.
  - Adaptive median throughput samples (elements/ms).
  - Last operation metadata (tiles processed, partial buffer width, elapsed time).

## Hardware Notes

- **x86-64**: AVX2 is the default floor; AVX-512 kernels activate automatically when available.
- **Apple Silicon / ARM64**: NEON-backed reducers tile to 128-bit lanes.  SIMD detection can be bypassed via `RAPTORS_SIMD=0` to compare scalar fallback behaviour.
- **Cross-platform**: Baselines are tuned for M3- and AVX2-class hardware.  If CI variance exceeds slack thresholds, capture local runs and update the JSON targets accordingly.

