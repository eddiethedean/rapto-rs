# Raptors Project Overview

Raptors is a Rust-powered, NumPy-compatible array library spearheaded by Odos
Matthews (`odosmatthews@gmail.com`, GitHub: [`eddiethedean`](https://github.com/eddiethedean)).
The project aims to provide safe, high-performance numeric operations with a
familiar Python interface.

## Current Status

- Repository skeleton established with Rust, Python, tests, benches, CI, and
  documentation directories.
- Rust crate now exports a `RustArray` type with constructors (`array`, `zeros`,
  `ones`), arithmetic (`add`, `scale`), and reductions (`sum`, `mean`), plus
  NumPy interop helpers.
- Python package re-exports the Rust array type, wraps helpers with Pythonic
  signatures, and offers `to_numpy`/`from_numpy` conversions.
- Test suite covers array construction, helpers, arithmetic, reductions, and
  roundtrips against NumPy when available.
- CI workflow template (`ci/github-actions.yml`) demonstrates building the
  extension using `maturin` and running `pytest`/`cargo test`.

## Next Steps

1. Extend the Rust core with additional dtypes and multidimensional support.
2. Broaden Python bindings toward NumPy-compatible indexing and broadcasting.
3. Expand tests to include property-based checks and performance benchmarks.
4. Add benchmarks, packaging workflows, and contributor documentation.

## Reference Material

- Full project plan: `../Rust-Powered-NumPy-Replacement-Full-Project-Plan.md`
- Repository layout explained in `README.md`

