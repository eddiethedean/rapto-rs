# Raptors TODO

## Short-Term Enhancements

- [x] Add broadcasting-aware arithmetic for mismatched shapes.
- [x] Support additional numeric dtypes (e.g., `i32`, `f32`) in `RustArray`.
- [x] Implement slicing/indexing utilities on the Python layer.
- [x] Improve NumPy interoperability with zero-copy views where safe.
- [ ] Validate integer reduction rules (e.g., configurable rounding for mean) and document error semantics.

## Testing & Tooling

- [ ] Introduce property-based tests for arithmetic and reductions.
- [ ] Add benchmarks comparing 1-D and 2-D operations to NumPy.

## Documentation & DX

- [ ] Expand usage guides covering broadcasting and advanced indexing.
- [ ] Document contribution workflow, coding standards, and release steps.
- [ ] Add a dtype reference section covering float/int behaviors and conversion caveats.

## Linux Performance Laggards

Performance issues identified on Linux (ARM64) via Docker-based benchmarking. See [docs/docker_linux_benchmarking_summary.md](docs/docker_linux_benchmarking_summary.md) and [docs/performance_fixes.md](docs/performance_fixes.md) for details.

### Critical Priority (< 0.80x)

- [x] **mean_axis0 @ 2048² float64**: 1.02x (improved from 0.62x, now faster than NumPy!) ✅
  - Status: ✅ Optimized with BLAS-first path on Linux (0.62x → 1.02x improvement, exceeds NumPy!)
  - Root cause: BLAS (OpenBLAS) is faster than SIMD for float64 on Linux/ARM64
  - Solution: Use BLAS first for float64, with specialized SIMD fallback
  - Performance: 1.02x (faster than NumPy) using OpenBLAS
  - Code location: `rust/src/lib.rs` lines 4529-4566, `rust/src/blas.rs` lines 429-484
  - Build: Requires `--features openblas` flag

- [ ] **mean_axis0 @ 2048² float32**: 0.77x (improved from 0.66x baseline, target: >1x)
  - Status: ✅ **Optimizations tested and applied** - Improved from 0.66x to 0.77x, still below 1x target
  - Current performance: 0.77x (Raptors: 0.42ms, NumPy: 0.33ms) - improved from baseline 0.66x
  - Actions taken:
    - ✅ Tested BLAS path: 0.44x (slower, not used)
    - ✅ Tested columnar approaches: 0.35x-0.57x (slower than tiled)
    - ✅ Tested row-block approach: 0.59x (slower than tiled)
    - ✅ Optimized output writes: Reduced unnecessary load-modify-store cycles (helped improve to 0.77x)
    - ✅ Tested instruction scheduling optimizations: Minimal improvement
    - ✅ Tested vector loop unrolling: Slower (0.50x), reverted
    - ✅ Kept optimized tiled approach (128x64 tiles, optimized output writes)
  - Performance measurements:
    - Baseline restored: 0.66x
    - After output write optimization: 0.77x (Raptors: 0.42ms, NumPy: 0.33ms)
    - 1024² float32: 0.28x (needs attention - may have regressed)
  - Next steps:
    - Further investigate why NumPy is faster (profiling, assembly comparison)
    - Consider alternative approaches or deeper NEON optimizations
    - Check if threading or other NumPy optimizations are being used
  - Code location: `rust/src/lib.rs` lines 4934-4953, `rust/src/simd/mod.rs` lines 2158-2230 (optimized tiled path)

### High Priority (0.80x - 0.95x)

- [x] **mean_axis0 @ 512² float64**: 0.56x (improved from 0.17x, BLAS path added) ✅
  - Status: ✅ BLAS path added for 512² float64
  - Code location: `rust/src/lib.rs` lines 4411-4429

- [x] **mean_axis0 @ 1024² float32**: 1.58x (improved from 0.29x, now faster than NumPy!) ✅
  - Status: ✅ BLAS path added for 1024² float32
  - Code location: `rust/src/lib.rs` lines 4838-4850

- [x] **broadcast_add @ 1024² float32**: 1.61x (improved from 0.77x, now faster than NumPy!) ✅
  - Status: ✅ NEON kernel optimized with 4x unrolling
  - Code location: `rust/src/simd/mod.rs` lines 2391-2439

- [x] **broadcast_add @ 512² float32**: 2.22x (improved from 0.82x, now faster than NumPy!) ✅
  - Status: ✅ NEON kernel optimized with 4x unrolling
  - Code location: `rust/src/simd/mod.rs` lines 2391-2439

- [x] **scale @ 2048² float64**: 1.17x (faster than NumPy!) ✅
  - Status: Already optimized, performance maintained

- [x] **scale @ 2048² float32**: 0.98x (improved from 0.41x, sequential SIMD on Linux) ✅
  - Status: ✅ Sequential SIMD path on Linux, optimized NEON kernel (0.41x → 0.98x improvement)
  - Code location: `rust/src/lib.rs` lines 3843-3869, `rust/src/simd/mod.rs` lines 2597-2732

### Notes

- **BLAS Integration**: OpenBLAS is available in Docker environment but requires `openblas` feature to be enabled during build. BLAS is now the optimal path for mean_axis0 @ 2048² float64 (1.02x speedup). For float32, SIMD path is optimal.
- **Recent Optimizations (2025-01-XX)**:
  - ✅ BLAS-first dispatch for float64 (1.09x performance)
  - ✅ Baseline tiled code restored for float32 (removed unrolling/prefetching that caused regression)
  - ✅ Specialized 2048x2048 path disabled (caused regression)
  - ⚠️ Unrolling and aggressive prefetching found to hurt performance, removed
  - See [docs/linux_performance_optimization_results.md](docs/linux_performance_optimization_results.md) for details
- **Debug Logging**: Use `RAPTORS_DEBUG_AXIS0=1` to enable debug output for mean_axis0 operations.
- **Profiling Tools**: Use `./scripts/profile_operation.sh` for perf/py-spy profiling.
- **Benchmarking**: Use `./scripts/docker_run_benchmarks.sh` to run full benchmark suite.
- **Build with OpenBLAS**: 
  ```bash
  PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/openblas-pthread/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig \
  maturin develop --release --features openblas
  ```

See [docs/linux_development_guide.md](docs/linux_development_guide.md) for development setup and [docs/docker_benchmarking.md](docs/docker_benchmarking.md) for benchmarking instructions.


