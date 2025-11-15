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

- [ ] **mean_axis0 @ 2048² float64**: 0.62x (improved from 0.34x, SIMD-first on Linux, optimized NEON with tiled approach)
  - Status: ✅ SIMD-first path on Linux, optimized NEON with tiled approach (0.34x → 0.62x improvement)
  - Root cause: SIMD is faster than BLAS on Linux/ARM64, but still below NumPy. Tiled approach provides better cache locality.
  - Next steps: Further optimize tiled NEON kernel or investigate NumPy's approach
  - Code location: `rust/src/lib.rs` lines 4518-4614, `rust/src/simd/mod.rs` lines 2176-2230

- [ ] **mean_axis0 @ 2048² float32**: 0.92x (improved from 0.05x, SIMD-first on Linux, optimized NEON with tiled approach)
  - Status: ✅ SIMD-first path on Linux, optimized NEON with tiled approach (0.05x → 0.92x improvement, very close to parity!)
  - Root cause: Columnar approach had poor cache locality. Tiled approach processes data in cache-friendly blocks.
  - Next steps: Further optimize tiled NEON kernel to cross 1.0x threshold
  - Code location: `rust/src/lib.rs` lines 4915-4981, `rust/src/simd/mod.rs` lines 1935-2103

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

- **BLAS Integration**: OpenBLAS is available in Docker environment but requires `openblas` feature to be enabled during build. Some operations may benefit from BLAS paths on Linux.
- **Debug Logging**: Use `RAPTORS_DEBUG_AXIS0=1` to enable debug output for mean_axis0 operations.
- **Profiling Tools**: Use `./scripts/profile_operation.sh` for perf/py-spy profiling.
- **Benchmarking**: Use `./scripts/docker_run_benchmarks.sh` to run full benchmark suite.

See [docs/linux_development_guide.md](docs/linux_development_guide.md) for development setup and [docs/docker_benchmarking.md](docs/docker_benchmarking.md) for benchmarking instructions.


