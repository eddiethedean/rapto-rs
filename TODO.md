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

- [x] **mean_axis0 @ 512² float32**: 1.29x (improved from 0.38x, now faster than NumPy!) ✅
  - Status: ✅ Fixed with SIMD-first routing on Linux (0.38x → 1.29x improvement, exceeds NumPy!)
  - Root cause: SIMD tiled approach is faster than BLAS for this size on Linux/ARM64
  - Solution: Use SIMD-first routing for 512×512 float32
  - Performance: 1.29x (faster than NumPy) using optimized NEON tiled kernel
  - Code location: `rust/src/lib.rs` lines 4838-4850
  - Date: 2025-11-16

- [ ] **mean_axis0 @ 1024² float64**: 0.59x-0.78x (improved from 0.44x, target: >0.80x)
  - Status: ✅ **Significantly improved** - Improved from 0.44x to 0.59x-0.78x (34-77% improvement, close to target)
  - Root cause: Generic BLAS path was slower than optimized SIMD for this specific size
  - Solution: Added specialized 1024² SIMD path with 128×64 tiles and 4x unrolling, SIMD-first routing
  - Current performance: 0.59x-0.78x (varies between runs, best: 0.78x) - significant improvement from 0.44x
  - Next steps: Fine-tune tile sizes or unrolling factor to reach >0.80x consistently
  - Code location: `rust/src/lib.rs` lines 4506-4585, `rust/src/simd/mod.rs` lines 2531-2658
  - Date: 2025-11-16

- [ ] **mean_axis0 @ 2048² float32**: 0.57x (improved from 0.23x, target: >0.80x)
  - Status: ✅ **Optimizations tested** - Improved from 0.23x to 0.57x, still below 0.80x target
  - Current performance: 0.57x (restored to baseline after reverting unrolling that caused regression)
  - Actions taken:
    - ✅ Tested BLAS path: Slower than SIMD, not used
    - ✅ Reverted columnar approaches: Slower than tiled
    - ✅ Tested 4x unrolling: Caused regression (0.56x → 0.53x), reverted
    - ✅ Using SIMD-first routing with simple tiled approach (128×64 tiles, no unrolling)
  - Next steps:
    - Test 2x unrolling instead of 4x (may reduce register pressure)
    - Focus on output write optimization (reduce load-modify-store cycles)
    - Profile to identify specific bottlenecks before further optimization
  - Code location: `rust/src/lib.rs` lines 4934-4953, `rust/src/simd/mod.rs` lines 2015-2093 (tiled path)
  - Date: 2025-11-16

- [ ] **mean_axis0 @ 2048² float64**: 0.36x (improved from 0.25x, target: >0.80x)
  - Status: ✅ **Optimizations tested** - Improved from 0.25x to 0.36x, still below 0.80x target
  - Root cause: BLAS (OpenBLAS) is faster than SIMD for float64, but still below target
  - Solution: Use BLAS-first routing for 2048×2048 float64 (SIMD optimizations tested but caused regression)
  - Current performance: 0.36x (BLAS-first routing restored after SIMD optimizations caused 0.25x regression)
  - Actions taken:
    - ✅ Tested optimized SIMD path (smaller tiles, 8x unrolling): Caused regression (0.36x → 0.25x)
    - ✅ Reverted to BLAS-first routing: Restored to 0.36x baseline
    - ✅ BLAS remains optimal path for this size
  - Next steps:
    - Investigate OpenBLAS configuration or threading settings
    - Profile BLAS path to identify bottlenecks
    - Consider alternative BLAS backends or OpenBLAS build optimizations
  - Code location: `rust/src/lib.rs` lines 4587-4638, `rust/src/blas.rs` lines 429-484
  - Build: Requires `--features openblas` flag
  - Date: 2025-11-16

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

- **BLAS Integration**: OpenBLAS is available in Docker environment but requires `openblas` feature to be enabled during build. BLAS is optimal for float64 on larger sizes (2048²), while SIMD is optimal for float32 on most sizes.
- **Recent Optimizations (2025-11-16)**:
  - ✅ Fixed 512×512 float32: 0.38x → 1.29x (SIMD-first routing)
  - ✅ Improved 1024×1024 float64: 0.44x → 0.59x-0.78x (specialized SIMD path with tiling and unrolling)
  - ✅ Improved 2048×2048 float32: 0.23x → 0.57x (SIMD-first routing, simple tiled approach)
  - ✅ Improved 2048×2048 float64: 0.25x → 0.36x (BLAS-first routing)
  - ✅ Tested deep optimizations: Unrolling and tile size changes for 2048² sizes caused regressions, reverted
  - ✅ Created profiling infrastructure: `scripts/profile_mean_axis0.sh` and `scripts/extract_assembly.sh`
  - See [docs/mean_axis0_remaining_lags_fix_summary.md](docs/mean_axis0_remaining_lags_fix_summary.md) and [docs/deep_optimization_findings.md](docs/deep_optimization_findings.md) for details
- **Debug Logging**: Use `RAPTORS_DEBUG_AXIS0=1` to enable debug output for mean_axis0 operations.
- **Profiling Tools**: 
  - Use `./scripts/profile_operation.sh` for perf/py-spy profiling
  - Use `./scripts/profile_mean_axis0.sh` for detailed perf stat metrics on mean_axis0
  - Use `./scripts/extract_assembly.sh` to extract and compare NEON assembly
- **Benchmarking**: Use `./scripts/docker_run_benchmarks.sh` to run full benchmark suite.
- **Build with OpenBLAS**: 
  ```bash
  PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/openblas-pthread/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig \
  maturin develop --release --features openblas
  ```

See [docs/linux_development_guide.md](docs/linux_development_guide.md) for development setup and [docs/docker_benchmarking.md](docs/docker_benchmarking.md) for benchmarking instructions.


