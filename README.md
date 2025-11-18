# Raptors

Rust-powered, NumPy-compatible Python array library scaffolding. The project
name expands to **Rust Accelerated Parallel Tensor Operations (`rapto-rs`)**.

> **Current release:** `0.2.0`

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

#### macOS / Linux (native)

```bash
# Build and install the Rust extension into your active virtualenv.
maturin develop

# Or build a release wheel.
maturin build --release
```

#### Linux Development via Docker

For Linux development, benchmarking, and profiling, we provide a Docker-based environment that ensures consistency and includes all necessary tools:

```bash
# Build the Docker image
./scripts/docker_bench.sh build

# Open an interactive shell in the container
./scripts/docker_bench.sh shell

# Inside the container, build and install Raptors
cd /workspace/src
/workspace/.venv/bin/maturin develop --release

# Run tests
/workspace/.venv/bin/python -m pytest tests/

# Run benchmarks
/workspace/.venv/bin/python scripts/compare_numpy_raptors.py --suite 2d
```

**Testing on Linux**: See [docs/linux_testing_guide.md](docs/linux_testing_guide.md) for detailed instructions on running all tests (Rust and Python) in the Docker environment.

See [docs/linux_development_guide.md](docs/linux_development_guide.md) for detailed instructions on:
- Setting up the Docker environment
- Building and running the project
- Running tests and benchmarks
- Performance profiling with `perf` and `py-spy`
- Development workflow and troubleshooting

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

### Performance & Controls

- Check whether the vectorized kernels are active via `raptors.simd_enabled()`.
- Force-disable or force-enable SIMD at import time with `RAPTORS_SIMD=0` or `RAPTORS_SIMD=1`.
- Control the Rayon pool used for large 2-D workloads with `RAPTORS_THREADS=<N>` (values ≤1 fall back to single-threaded execution).
- Inspect adaptive thresholds, backend usage, and pool sizing with `raptors.threading_info()`. The diagnostics now include per-operation backend counters (SIMD, Rayon-SIMD, BLAS, Scalar) so you can confirm which path executed.
- Contiguous elementwise add/scale and row/column broadcasts prioritize SIMD first, then fall back to Accelerate or scalar code when heuristics predict it will be faster. Multi-thread fan-out is driven by adaptive chunk sizing tuned for 1024²–4096² grids.

### Performance Milestones

**v0.2.0** — **Faster than NumPy across all operations!**
- **100% of operations at parity or faster** (≥0.95x) across the 2D benchmark suite
- **34/36 operations faster than NumPy** (≥1.0x speedup)
- **Fixed scale @ 2048² float32**: Now **1.07× faster** than NumPy (0.29ms vs 0.31ms) using parallel Accelerate vDSP
- Optimized chunk sizing (8-10 chunks) for better load balancing on large matrices
- Parallel Accelerate vDSP implementation combines hand-tuned assembly with multi-threading
- Updated baseline guard rails reflect all performance improvements

**v0.1.0** — Initial Performance Milestone
- Raptors outperformed NumPy on **33/36 operations (91.7%)** across the 2D benchmark suite
- **All float32 operations faster than NumPy** (18/18 operations)
- `mean_axis0` operations: **5-10× faster** than NumPy across all sizes
- `sum` and `mean` operations: **2-4× faster** than NumPy for float32
- See `docs/perf_report.md` for the complete performance breakdown

**Performance Highlights** (Apple M3 Pro, v0.2.0):
- `scale` @ 2048² float32: **1.07× faster** (0.29ms vs 0.31ms)
- `mean_axis0` operations: **5-10× faster** across all sizes
- `sum` and `mean` operations: **2-4× faster** for float32
- `broadcast_add`: **1.5-2.0× faster** for large matrices
- All operations achieve ≥0.95x speedup (near parity or better)

See `docs/perf/faster_than_numpy_complete.md` for detailed performance analysis.

- The benchmarking helper accepts presets and JSON export for repeatable runs:

  ```bash
  # Compare scalar vs SIMD for a single shape
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 --simd-mode auto
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --shape 1024x1024 --dtype float64 --simd-mode disable

  # Run the 2-D suite and capture results for dashboards
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --suite 2d --simd-mode force \
      --warmup 1 --repeats 7 --output-json results.json

  # Guard against regressions in CI using baseline thresholds (float64 example)
  PYTHONPATH=python ./scripts/compare_numpy_raptors.py --suite 2d --simd-mode force \
      --warmup 1 --repeats 7 \
      --validate-json benchmarks/baselines/2d_float64.json --validate-slack 0.05
  ```

### ASV Benchmarks

- Airspeed Velocity is configured in `benchmarks/`; the quick smoke run mirrors NumPy parity checks:

  ```bash
  cd benchmarks
  ../.venv/bin/asv run --python=../.venv/bin/python --quick
  ```

- For baseline refreshes (e.g., after performance improvements):

  ```bash
  cd benchmarks
  ../.venv/bin/asv run --python=../.venv/bin/python --steps 3
  rsync -a .asv/results/ baselines/asv/
  ```

- See `docs/perf/asv_setup.md` for full environment instructions and `docs/perf_report.md` for current win/loss tracking against NumPy.

### Running Tests

- `./run_all_tests.sh` — provisions a local virtualenv, builds the extension with maturin, runs the Rust suite, and executes pytest.
- `cargo xtest` — runs the Rust unit and integration tests (`cargo test --no-default-features --features test-suite`).  
  Set `RAPTORS_TEST_PYTHON` if the project-local `.venv/bin/python` is unavailable; the harness auto-loads the matching `libpython`.

## Continuous Integration

CI currently consists of two entry points:

- `ci/github-actions.yml` — linting, unit tests, and Rust/Python checks.
- `.github/workflows/bench.yml` — scheduled benchmark validation against the JSON targets in `benchmarks/baselines/`, producing JSON artefacts for historical tracking.

