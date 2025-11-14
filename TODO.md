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

- [ ] **mean_axis0 @ 2048² float64**: 0.49x (improved from 0.02x with tiled NEON implementation, still critical)
  - Status: Tiled NEON implementation added (0.02x → 0.49x improvement)
  - Root cause: Likely still using slow fallback paths or inefficient NEON code
  - Next steps: Profile with perf/py-spy, compare with working macOS implementation, optimize NEON kernel further
  - Code location: `rust/src/lib.rs` lines 4437-4569, `rust/src/simd/mod.rs` lines 2112-2235

- [ ] **mean_axis0 @ 2048² float32**: 0.04x (SEVERE)
  - Status: Similar to float64, needs investigation
  - Next steps: Profile and apply similar optimizations as float64
  - Code location: Similar to float64 path

- [ ] **mean_axis0 @ 512² float64**: 0.17x (CRITICAL)
  - Status: Needs investigation
  - Next steps: Profile and optimize for smaller sizes

- [ ] **mean_axis0 @ 1024² float32**: 0.29x (CRITICAL)
  - Status: Needs investigation
  - Next steps: Profile and optimize

### High Priority (0.80x - 0.95x)

- [ ] **broadcast_add @ 1024² float32**: 0.77x (23% slower)
  - Status: Sequential SIMD path, NEON vs NumPy's OpenBLAS
  - Analysis: Parallelization overhead too high for this size
  - Next steps: Optimize NEON kernel or consider BLAS path
  - Code location: `rust/src/lib.rs` lines 2645-2729

- [ ] **broadcast_add @ 512² float32**: 0.82x (18% slower)
  - Status: Sequential SIMD path is optimal (parallel overhead too high)
  - Analysis: NEON SIMD vs NumPy's OpenBLAS
  - Next steps: Further NEON kernel tuning may help
  - Code location: `rust/src/lib.rs` lines 2645-2729

- [ ] **scale @ 2048² float64**: 0.88x (12% slower)
  - Status: Well-optimized parallel path
  - Next steps: May benefit from BLAS path or further SIMD optimization

- [ ] **scale @ 2048² float32**: 0.88x (12% slower)
  - Status: Parallel SIMD path
  - Next steps: May benefit from further optimization

- [ ] **mean_axis0 @ 512² float32**: 0.87x (13% slower)
  - Status: Needs investigation
  - Next steps: Profile and optimize

### Notes

- **BLAS Integration**: OpenBLAS is available in Docker environment but requires `openblas` feature to be enabled during build. Some operations may benefit from BLAS paths on Linux.
- **Debug Logging**: Use `RAPTORS_DEBUG_AXIS0=1` to enable debug output for mean_axis0 operations.
- **Profiling Tools**: Use `./scripts/profile_operation.sh` for perf/py-spy profiling.
- **Benchmarking**: Use `./scripts/docker_run_benchmarks.sh` to run full benchmark suite.

See [docs/linux_development_guide.md](docs/linux_development_guide.md) for development setup and [docs/docker_benchmarking.md](docs/docker_benchmarking.md) for benchmarking instructions.


