# Rust-Powered NumPy Replacement — Full Project Plan

> A pragmatic, production-focused plan to build a Rust-backed, NumPy-compatible Python array library. This document outlines goals, scope, architecture, milestones, testing and release practices, community & maintenance strategies, and a step-by-step implementation plan to move from MVP to production-ready library.

# 1. Executive summary

Build a high-performance, Rust-backed Python array library that provides NumPy-compatible core functionality for typical scientific and engineering workloads. The project will prioritize safety, predictable memory semantics (buffer protocol / zero-copy), cross-platform binary packaging, and a compatibility story that enables existing Python code to adopt the library incrementally.

Primary deliverables:
- A Rust core array library with efficient memory layout and high-performance kernels.
- Python bindings exposing a NumPy-like API for the most critical operations (creation, indexing, slicing, elementwise ops, dot product, reductions, dtype conversions, broadcasting basics).
- CI-built manylinux/macos/windows wheels and comprehensive testing comparing outputs against NumPy.
- Documentation, migration guides, benchmarks, and a contributor-friendly repository.

Scope: start focused — 1D/2D arrays, standard numeric dtypes, elementwise ops, dot, broadcasting for common shapes, memory interoperability with NumPy. Expand later to BLAS/LAPACK, FFT, RNG, more dtypes, and broader API parity.

---

# 2. Goals & success criteria

**Primary goals**
- Production-ready for core numeric workloads (safe memory handling, stable wheels, performance parity or improvement for target kernels).
- Drop-in or near-drop-in for a well-defined subset of NumPy used by target users.
- Maintainable, testable codebase with active maintainers and clear release practices.

**Success metrics**
- Test coverage: unit and property tests covering core ops and edge cases.
- Compatibility: behavior matches NumPy for the selected API surface within numerical tolerances.
- Packaging: automated CI that produces and uploads wheels for Linux (manylinux), macOS, and Windows.
- Performance: reproducible benchmarks showing meaningful improvement on target kernels or at least parity.
- Community: at least X external contributors and clear contributor guidelines.

---

# 3. Target audience & use cases

**Primary audience**
- Developers needing faster hot loops in numeric Python with safe memory semantics.
- Data science engineers with critical performance hotspots.
- Organizations seeking Rust’s safety/perf with Python compatibility.

**Primary use cases**
- Numeric kernels (elementwise math, broadcasting, reduction, dot product) in CPU workloads.
- Replacement of specific heavy functions in existing NumPy-based pipelines.
- Interop with Rust-based libraries (polars/arrow) where zero-copy transfer reduces overhead.

**Non-targets (initially)**
- GPU acceleration (no CUDA/ROCm support in MVP).
- Full SciPy feature parity, advanced linear algebra edge cases, or exotic dtypes.

---

# 4. Core technical decisions

**Language & toolchain**
- Core: Rust (stable). Use `ndarray` or a custom array type if needed. Leverage `portable_simd` for vectorization.
- Bindings: `pyo3` for Python interop and `maturin` for building wheels. Consider `rust-numpy` for smooth interoperability with NumPy buffer objects.

**Memory model**
- Row-major contiguous memory (C order) as default; support Fortran order where necessary.
- Implement Python buffer protocol and support zero-copy views into Rust memory.
- Clear ownership rules: arrays own their memory unless explicitly created as views/slices.

**Parallelism**
- Use `rayon` for internal parallelism; release GIL appropriately and document threading semantics.

**Numeric correctness & libraries**
- For BLAS/LAPACK, bind to system libraries (OpenBLAS/BLIS/MKL) for linear algebra operations.
- Use well-tested Rust crates or C libraries for FFTs, RNGs, etc.

**Packaging**
- Use `maturin` for building wheels and CI. Produce manylinux, macOS universal2, and Windows wheels.

---

# 5. API design

- Start with a `rustarray` Python object that mirrors a subset of `numpy.ndarray`. Provide `from_numpy` and `to_numpy` (zero-copy when possible).
- Support dtype enum: `int8/int16/int32/int64`, `uint8/16/32/64`, `float32/64`, `complex64/128`, `bool`.
- Operations to implement initially:
  - Creation: `array()`, `zeros()`, `ones()`, `empty()`, `arange()`, `linspace()`.
  - Basic indexing & slicing with views; fancy indexing later.
  - Elementwise arithmetic, comparisons, reductions (`sum`, `mean`, `min`, `max`), `dot`.
  - Broadcasting for common scenarios.
  - `astype`, contiguous/Fortran checks, `reshape`, `transpose`.

---

# 6. Testing strategy

**Test types**
- Unit tests for Rust core and Python bindings.
- Cross-language property tests comparing outputs vs NumPy.
- Performance regression tests (benchmarks) recorded in CI.
- Fuzz tests for parsing/indexing and memory operations.

**Test infrastructure**
- GitHub Actions matrix for Python versions and platforms.
- Use `pytest` + Rust test harness (`cargo test`) in CI.
- Nightly benchmark runs for long-term monitoring.

**Acceptance tests**
- Run real workloads (e.g., small ML step, numeric transforms) and compare to NumPy.

---

# 7. Security & safety

- Use `cargo-audit` in CI to scan Rust dependencies.
- Python security checks on packaging scripts.
- Responsible disclosure & vulnerability response policy.

---

# 8. CI / CD / Releases

**CI**
- GitHub Actions for unit tests, Python tests, linters, and wheel builds.
- `maturin build --release` on runners to produce platform-specific wheels.
- Automated test import of built wheel before publishing.

**CD**
- Releases: GitHub tag → release notes → `maturin publish`.
- Versioning & stability: semantic versioning; pre-1.0: minor bumps for breaking changes.

---

# 9. Benchmarking & performance validation

**Benchmarks**
- Elementwise arithmetic throughput.
- Memory-bound operations (slicing, contiguous vs non-contiguous).
- BLAS-level operations (dot, gemm) vs NumPy on same hardware.
- Real-world workloads (image kernels, reductions).

**Benchmark tooling**
- `criterion` (Rust), `pytest-benchmark`/`asv` (Python).

**Performance regression policy**
- Fail CI if core kernel regressions exceed threshold.

---

# 10. Packaging & distribution

- Wheels: manylinux, macOS universal2, Windows; publish via PyPI.
- Conda packages (optional) via conda-forge.
- Optimize binary size via LTO and stripping symbols.

---

# 11. Documentation & developer experience

- Docs: quickstart, API reference, migration guide, examples.
- Examples: Jupyter notebooks comparing usage and profiling tips.
- Dev scripts: build-wheel, test, bench; clear CONTRIBUTING.md and PR templates.

---

# 12. Community, governance & maintenance

- Core team or benevolent dictator for decisions.
- Clear CONTRIBUTING.md and issue triage.
- Onboarding guides for new maintainers.
- Sponsor/funding options for sustainability.

---

# 13. Risk analysis & mitigations

- Large API surface → focus subset + bridging.
- Native dependencies (BLAS) → fallback algorithms and documentation.
- Cross-platform packaging → CI builds & integration tests.
- Community adoption → excellent docs, migration guides, focus verticals.

---

# 14. Roadmap & milestones (phased)

**Phase 0 — Project setup**: repo skeleton, CI baseline, MVP API doc.
**Phase 1 — MVP**: `rustarray` with creation, indexing, slicing, elementwise ops, reductions, dot; zero-copy NumPy interop.
**Phase 2 — Compatibility & stability**: broadcasting, reshape/transpose, dtype casts, contiguous checks, test coverage, CI wheels for all platforms.
**Phase 3 — Performance & ecosystem**: SIMD kernels, rayon parallelization, BLAS integration, benchmarks, notebooks, docs.
**Phase 4 — Feature parity expansion**: more dtypes, advanced indexing, FFTs, RNGs, optional SciPy interop.
**Phase 5 — Production readiness**: 1.0 release, hardened tests, performance regression monitoring, maintenance plan.

---

# 15. Repo layout & starter file structure

```
rustynum-rs/
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── rust/
│   ├── Cargo.toml
│   └── src/
├── python/
│   └── examples/
├── tests/
├── benches/
├── ci/
└── docs/
```

---

# 16. Starter implementation checklist (first 20 tasks)

1. Create repo and initialize Cargo + maturin + GitHub Actions.
2. Add CONTRIBUTING.md and CODE_OF_CONDUCT.md.
3. Implement minimal Rust crate exposing a single `array_new` via pyo3.
4. Implement a Python wrapper importing the built wheel.
5. Add CI job building and testing wheel.
6. Implement `from_numpy` and `to_numpy` zero-copy wrapper.
7. Implement elementwise `add` in Rust and expose to Python.
8. Add unit tests for `add` vs NumPy.
9. Implement `dot` for 1D and 2D arrays (single-threaded).
10. Add reductions: `sum`, `mean`.
11. Tests for NaN/Inf and dtype promotion.
12. Add `reshape` and `transpose` (view/copy rules clarified).
13. Add `astype` conversions.
14. Add basic broadcasting.
15. Rust unit tests with `cargo test`.
16. Criterion benchmarks for elementwise and dot.
17. Python benchmark harness and example notebooks.
18. `maturin` publish workflow to PyPI.
19. Initial documentation and quickstart.
20. Label `good-first-issue` in repo.

---

# 17. Maintenance & long-term planning

- Define roadmap for first year post-MVP.
- Maintain CHANGELOG.md and deprecation policy.
- Open calls for contributors; onboarding guides.

---

# 18. Funding & staffing suggestions

- Small core paid team (1–3 engineers) for first year, or
- Sponsorship by organizations adopting the tech, or
- Corporate partner for co-maintenance.

---

# 19. Appendix: recommended libraries & references

- Rust crates: `ndarray`, `rayon`, `ndarray-parallel`, `packed_simd`/`portable_simd`, `blas-src`/`blas-sys`.
- Python interop: `pyo3`, `rust-numpy`, `maturin`.
- Packaging: `maturin`, `cibuildwheel`.
- Benchmarking & testing: `criterion` (Rust), `pytest-benchmark`, `asv` (Python), `cargo-audit`.

---

# 20. Next immediate step

- Create repository skeleton, push initial commit with pyproject.toml, Cargo.toml, README, tiny Rust + Python example with maturin.
- Open initial issue describing MVP API surface and invite contributors.
