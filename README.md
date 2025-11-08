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

## Continuous Integration

An example GitHub Actions workflow is provided in `ci/github-actions.yml`. It
builds the Rust extension with `maturin`, installs the Python package in
development mode, and runs both `pytest` and `cargo test`.

