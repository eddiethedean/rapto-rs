# Raptors Project Overview

Raptors is a Rust-powered, NumPy-compatible array library spearheaded by Odos
Matthews (`odosmatthews@gmail.com`, GitHub: [`eddiethedean`](https://github.com/eddiethedean)).
The project aims to provide safe, high-performance numeric operations with a
familiar Python interface.

## Current Status

- Repository skeleton established with Rust, Python, tests, benches, CI, and
  documentation directories.
- Rust crate initialized via `pyo3`, exporting a placeholder `rustarray_new`
  constructor.
- Python package (`raptors`) imports the Rust extension and exposes a stub
  `array()` function.
- CI workflow template (`ci/github-actions.yml`) demonstrates building the
  extension using `maturin` and running `pytest`/`cargo test`.
- Pytest baseline ensures the placeholder API imports successfully.

## Next Steps

1. Implement real array data structures and operations in Rust.
2. Flesh out Python bindings to mirror NumPy-like ergonomics.
3. Expand tests to validate behavior against NumPy and cover edge cases.
4. Add benchmarks, packaging workflows, and contributor documentation.

## Reference Material

- Full project plan: `../Rust-Powered-NumPy-Replacement-Full-Project-Plan.md`
- Repository layout explained in `README.md`

