# NumPy-Inspired SIMD & Benchmark Strategy

## 1. SIMD & Kernel Architecture

- Borrow the layering that NumPy uses in `numpy/core/src/common/simd/`:
  - Keep per-ISA intrinsics in dedicated modules (`neon.rs`, `avx.rs`, …) behind a trait-based facade.
  - Emit generic loops from a small DSL/macros (NumPy’s `loops.c.src`) to minimise copy/paste and guarantee consistent alignment prefetch choices.
- Proposed Raptors steps:
  - Introduce a `simd::ops` module exposing typed micro-kernels (`mul_add`, `scale`, `axpy`) parametrised on `SimdBackend`.
  - Move the current Neon-specific helpers out of `simd/mod.rs` into `simd/aarch64.rs`, reachable via the backend trait.
  - Add a `SimdDispatchTable` generator macro that mirrors NumPy’s `DispatchTable` but emits per-backend consts with metadata (lane width, required alignment).
  - Extend `parallel_scale_f32` to use tile descriptors exposed by the backend (matching NumPy’s `ROW_TILE_F32`, `COL_TILE_F32`) instead of hard-coded constants.

## 2. BLAS Fallback Policy

- NumPy routes to BLAS via `numpy/core/src/umath/blasfuncs.c` with heuristics in the ufunc loops.
- For Raptors:
  - Define a central `blas::should_use(kind, shape, dtype)` decision helper so callers avoid duplicating threshold logic.
  - Track empirical crossover points in JSON (similar to NumPy’s configuration) and feed them into the heuristic at runtime.
  - Add metrics counters (e.g., `metrics::record_backend_choice`) to measure how often each path triggers.

## 3. ASV Benchmark Suite

- NumPy’s benchmarks live under `benchmarks/` with ASV config (`asv.conf.json`) and per-module benchmark files.
- Proposed structure:

  ```
  benchmarks/
    asv.conf.json
    raptors/
      bench_scale.py
      bench_reductions.py
      bench_broadcast.py
  ```

- Key actions:
  - Add poetry/venv instructions for ASV in `docs/perf/`.
  - Mirror NumPy’s parameter grids: `(dtype, shape, layout, threads, simd-mode)`.
  - Store baseline JSONs in `benchmarks/baselines/asv/`.
  - Create a GitHub workflow `ci/benchmarks.yml` that runs `asv run --quick` nightly and uploads results.

## 4. Deliverables

- `simd/` refactor with backend traits & macro generator.
- New heuristics in `blas.rs` driven by data files.
- ASV suite scaffold + documentation (`docs/perf/asv.md`).
- Metrics instrumentation & dashboards (stretch goal: expose via `threading_info()`).

