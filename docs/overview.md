# Raptors Project Overview

Raptors is a Rust-powered, NumPy-compatible array library spearheaded by Odos
Matthews (`odosmatthews@gmail.com`, GitHub: [`eddiethedean`](https://github.com/eddiethedean)).
The project aims to provide safe, high-performance numeric operations with a
familiar Python interface.

## Current Status

- Repository skeleton established with Rust, Python, tests, benches, CI, and
  documentation directories.
- Rust crate now exports a `RustArray` storing flattened data with explicit
  shape metadata, constructors that accept 1-D or 2-D inputs, elementwise
  arithmetic, and reductions supporting axis-aware operations.
- Python package re-exports the Rust type, adds helpers (`array2d`, generalized
  `zeros`/`ones`), and handles roundtrip conversions with NumPy using the new
  dynamic bindings.
- Test suite covers 1-D and 2-D construction, helpers, arithmetic, axis
  reductions, and roundtrips against NumPy when available.
- CI workflow template (`.github/workflows/build-wheels.yml`) builds platform
  wheels via `maturin`, runs the Python tests, and uploads release artifacts.

## Next Steps

1. Extend the Rust core with additional dtypes and multidimensional support.
2. Broaden Python bindings toward NumPy-compatible indexing and broadcasting.
3. Expand tests to include property-based checks and performance benchmarks.
4. Add benchmarks, packaging workflows, and contributor documentation.

## Reference Material

- Full project plan: `../Rust-Powered-NumPy-Replacement-Full-Project-Plan.md`
- Repository layout explained in `README.md`

