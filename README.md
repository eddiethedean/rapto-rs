# Raptors

Rust-powered, NumPy-compatible Python array library scaffolding.

## Getting Started

This repository contains the initial skeleton for `raptors`, a Rust-backed
array library exposed to Python. The project is being developed by Odos
Matthews (`odosmatthews@gmail.com`, GitHub: `eddiethedean`).

### Layout

- `rust/`: Rust crate exposing core functionality via `pyo3`.
- `python/`: Python package wrapper publishing the `raptors` module.
- `tests/`: pytest-based test suite comparing behavior to NumPy.
- `benches/`: Benchmark harnesses for performance tracking.
- `ci/`: Continuous-integration workflows and scripts.
- `docs/`: Documentation sources (overview, guides, references).

Consult `docs/overview.md` for an expanded project overview and links to the
full plan.

## Quickstart

```python
import raptors

# Construct arrays from Python iterables or helper constructors.
a = raptors.array([1.0, 2.0, 3.0])
b = raptors.ones(3)

# Run elementwise math and reductions.
c = a.add(b).scale(2.0)
total = c.sum()
average = c.mean()

# Interoperate with NumPy (requires numpy) without copying data.
import numpy as np

numpy_view = raptors.to_numpy(c)
assert np.allclose(numpy_view, [4.0, 6.0, 8.0])
```

## Continuous Integration

An example GitHub Actions workflow is provided in `ci/github-actions.yml`. It
builds the Rust extension with `maturin`, installs the Python package in
development mode, and runs both `pytest` and `cargo test`.

