# Measurement Checklist — 2-D Float Benchmarks

- Shapes: `(512, 512)`, `(1024, 1024)`, `(2048, 2048)`
- Dtypes: `float32`, `float64`
- Operations: `sum`, `mean`, `mean_axis0`, `mean_axis1`, `broadcast_add`, `scale`
- Environment:
  - `RAPTORS_THREADS=8`
  - `--simd-mode force` plus `--simd-mode disable` for scalar baselines
  - Warmup `1`, repeats `4`
- Artifacts:
  - Axis-0 spot checks → `benchmarks/run_axis0_suite.py --output-json benchmarks/results/axis0_latest.json`
  - SIMD results → `benchmarks/results/latest_simd.json`
  - Scalar results → `benchmarks/results/latest_scalar.json`
  - Update baselines with 5% slack via helper script (`python -m benchmarks.update_baselines` TBD)
- Validation:
  - Re-run `scripts/compare_numpy_raptors.py --suite 2d --validate-json benchmarks/baselines/<file>.json --validate-slack 0.0`
  - Capture summary deltas in worklog before landing changes.


