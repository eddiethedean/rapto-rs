# Raptors Project Overview

Raptors is a Rust-powered, NumPy-compatible array library spearheaded by Odos
Matthews (`odosmatthews@gmail.com`, GitHub: [`eddiethedean`](https://github.com/eddiethedean)).
The project aims to provide safe, high-performance numeric operations with a
familiar Python interface.

## Current Status

- Repository skeleton established with Rust, Python, tests, benches, CI, and
  documentation directories.
- Rust crate now exports generic numeric arrays backing `RustArray` (float64),
  `RustArrayF32`, and `RustArrayI32`, including constructors, elementwise
  arithmetic with broadcasting, and axis reductions.
- Python package re-exports the Rust types, adds dtype-aware helpers (`array`,
  `zeros`, `ones`, `broadcast_add`), NumPy-style slicing/indexing helpers, and
  handles roundtrip conversions with NumPy by inspecting array dtype.
- Test suite covers 1-D and 2-D construction, dtype-specific helpers,
  arithmetic, axis reductions, and roundtrips against NumPy when available.
- CI workflow template (`.github/workflows/build-wheels.yml`) builds platform
  wheels via `maturin`, runs the Python tests, and uploads release artifacts.

## Next Steps

1. Extend the Rust core with additional dtypes and multidimensional support.
2. Broaden Python bindings toward advanced indexing modes (boolean masks,
   fancy indexing) and additional broadcasting scenarios.
3. Expand tests to include property-based checks and performance benchmarks.
4. Add benchmarks, packaging workflows, and contributor documentation.

## Reference Material

- Full project plan: `../Rust-Powered-NumPy-Replacement-Full-Project-Plan.md`
- Repository layout explained in `README.md`

