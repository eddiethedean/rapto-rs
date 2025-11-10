# Raptors

Rust-powered, NumPy-compatible Python array library scaffolding.

> **Current release:** `0.0.2`

## Getting Started

This repository contains the initial skeleton for `raptors`, a Rust-backed
array library exposed to Python. The project is being developed by Odos
Matthews (`odosmatthews@gmail.com`, GitHub: `eddiethedean`).

### Layout

- `rust/`: Rust crate exposing core functionality via `pyo3` (with float64, float32, and int32 array wrappers).
- `python/`: Python package wrapper publishing the `raptors` module.
- `tests/`: pytest-based test suite comparing behavior to NumPy.
- `benches/`: Benchmark harnesses for performance tracking.
- `ci/`: Continuous-integration workflows and scripts.
- `docs/`: Documentation sources (overview, guides, references).

Consult `docs/overview.md` for an expanded project overview and links to the
full plan.

### Installation (local development)

```bash
# Build and install the Rust extension into your active virtualenv.
maturin develop

# Or build a release wheel.
maturin build --release
```

## Quickstart

```python
import raptors

# Construct arrays from Python iterables or helper constructors.
row = raptors.array([1.0, 2.0, 3.0])  # float64 by default
matrix = raptors.array2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
zeros32 = raptors.zeros((2, 3), dtype="float32")
ints = raptors.ones(3, dtype="int32")

# Run elementwise math and reductions across dimensions.
scaled = matrix.add(raptors.array2d([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])).scale(0.5)
total = scaled.sum()
col_sums = scaled.sum_axis(0)
row_means = scaled.mean_axis(1)

# Use dtype-aware broadcasting helpers.
broadcasted = raptors.broadcast_add(matrix, row)
f32_sum = raptors.broadcast_add(zeros32, raptors.array([1.0, 2.0, 3.0], dtype="float32"))

# Slice and index arrays with familiar Python semantics (1-D and 2-D today).
center_block = matrix[0:2, 1:]
last_col = matrix[:, -1]
value = matrix.index_array(1, 2)  # -> 6.0

# Interoperate with NumPy (requires numpy) without leaving Python.
import numpy as np  # type: ignore

numpy_view = raptors.to_numpy(matrix)
assert numpy_view.shape == (2, 3)
assert np.allclose(numpy_view, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# NumPy ↔ Raptors conversions share memory when layouts are compatible.
numpy_view[0, 0] = 42.0
assert matrix.to_list()[0] == 42.0

alias = raptors.from_numpy(numpy_view)
numpy_view[1, 1] = 99.0
assert alias.to_list()[4] == 99.0
```

### Performance Controls

- Check whether the vectorized kernels are active via `raptors.simd_enabled()`.
- Force-disable or force-enable SIMD at import time with `RAPTORS_SIMD=0` or `RAPTORS_SIMD=1`.
- Control the Rayon pool used for large 2-D workloads with `RAPTORS_THREADS=<N>` (values ≤1 fall back to single-threaded execution).
- Contiguous elementwise add/scale and row/column broadcasts now use the SIMD path, and large matrices fan out across threads automatically.
- The benchmarking helper accepts presets and JSON export for repeatable runs:

  ```bash
  # Compare scalar vs SIMD for a single shape
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 --simd-mode auto
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 --simd-mode disable

  # Run the 2-D suite and capture results for dashboards
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --suite 2d --simd-mode auto --output-json results.json

  # Guard against regressions in CI using baseline thresholds
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 \
      --operations mean mean_axis0 mean_axis1 broadcast_add \
      --validate-json benchmarks/baselines/2d_float64.json --validate-slack 1.0
  ```

## Continuous Integration

The `.github/workflows/build-wheels.yml` workflow builds and tests wheels across
Ubuntu, macOS, and Windows for Python 3.9–3.12 using `maturin`. Successful runs
upload ready-to-use wheel artifacts and exercise the Python bindings via
`pytest`.

