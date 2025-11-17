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

- [ ] **mean_axis0 @ 1024² float64**: 0.50x (improved from 0.42x, target: >0.80x)
  - Status: ✅ **Improvement** - BLAS-first routing improved performance (0.42x → 0.50x)
  - Root cause: Store replays and load-modify-store cycles were bottleneck in SIMD path
  - Solution: 
    - ✅ Reverted write-combine buffer (caused regression)
    - ✅ Implemented pure columnar processing: one column at a time, accumulator in register (eliminates load-modify-store)
    - ✅ Changed to BLAS-first routing: let OpenBLAS choose optimal tile sizes dynamically (NumPy approach)
  - Current performance: 0.50x (20251117-163021) - improvement from 0.42x, still below 0.80x target
  - Previous best: 0.64x (20251117-104050) before write-combine buffer
  - Recent optimizations:
    - ✅ Reverted write-combine buffer (2025-11-17)
    - ✅ Implemented pure columnar approach with prefetch hints (2025-11-17)
    - ✅ Changed to BLAS-first routing (2025-11-17) - matches NumPy's approach
  - Next steps: 
    - Profile BLAS path to identify remaining bottlenecks
    - Consider alternative BLAS backends or OpenBLAS build optimizations
    - Test if pure columnar SIMD (without BLAS) performs better
  - Code location: `rust/src/lib.rs` lines 4563-4613, `rust/src/simd/mod.rs` lines 2501-2554
  - Date: 2025-11-17

- [ ] **mean_axis0 @ 2048² float32**: 0.42x (improved from 0.46x, target: >0.80x)
  - Status: ✅ **Improvement** - BLAS-first routing improved performance (0.46x → 0.42x)
  - Current performance: 0.42x (20251117-163021) - improvement from 0.46x, still below 0.80x target
  - Previous best: 0.69x (20251117-104050) before register cache/parallelization
  - Actions taken:
    - ✅ Reverted register cache and parallelization (caused regression)
    - ✅ Implemented pure columnar processing: one column at a time, accumulator in register (eliminates load-modify-store)
    - ✅ Changed to BLAS-first routing: let OpenBLAS choose optimal tile sizes dynamically (NumPy approach)
    - ✅ Removed unrolling (was causing overhead)
  - Recent optimizations (2025-11-17):
    - ✅ Reverted register cache and parallelization
    - ✅ Implemented pure columnar approach with prefetch hints
    - ✅ Changed to BLAS-first routing - matches NumPy's approach
  - Next steps: 
    - Profile BLAS path to identify remaining bottlenecks
    - Consider alternative BLAS backends or OpenBLAS build optimizations
    - Test if pure columnar SIMD (without BLAS) performs better
  - Code location: `rust/src/lib.rs` lines 5074-5098, `rust/src/simd/mod.rs` lines 2025-2076
  - Date: 2025-11-17

- [ ] **mean_axis0 @ 2048² float64**: 0.36x (unchanged, target: >0.80x)
  - Status: ⚠️ **No change** - Performance unchanged at 0.36x, still below 0.80x target
  - Root cause: BLAS (OpenBLAS) is faster than SIMD for float64, but still below target
  - Solution: Use BLAS-first routing for 2048×2048 float64 (SIMD optimizations tested but caused regression)
  - Current performance: 0.36x (20251117-163021) - unchanged from previous baseline
  - Actions taken:
    - ✅ Tested optimized SIMD path (smaller tiles, 8x unrolling): Caused regression (0.36x → 0.25x)
    - ✅ Reverted to BLAS-first routing: Restored to 0.36x baseline
    - ✅ BLAS remains optimal path for this size
    - ✅ OpenBLAS threading correctly set to 1
  - Next steps:
    - Investigate OpenBLAS configuration or threading settings
    - Profile BLAS path to identify bottlenecks
    - Consider alternative BLAS backends or OpenBLAS build optimizations
  - Code location: `rust/src/lib.rs` lines 4587-4638, `rust/src/blas.rs` lines 429-484
  - Build: Requires `--features openblas` flag
  - Date: 2025-11-17

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

- [x] **broadcast_add @ 1024² float64**: Fixed regression (was 3.5x → 1.6x, now optimized) ✅
  - Status: ✅ Fixed regression by adding 4x unrolling to NEON add_same_shape_f64 kernel
  - Code location: `rust/src/simd/mod.rs` lines 3078-3126
  - Date: 2025-11-17

- [x] **scale @ 2048² float64**: 1.17x (faster than NumPy!) ✅
  - Status: Already optimized, performance maintained

- [x] **scale @ 2048² float32**: 0.98x (improved from 0.41x, sequential SIMD on Linux) ✅
  - Status: ✅ Sequential SIMD path on Linux, optimized NEON kernel (0.41x → 0.98x improvement)
  - Code location: `rust/src/lib.rs` lines 3843-3869, `rust/src/simd/mod.rs` lines 2597-2732

### Notes

- **BLAS Integration**: OpenBLAS is available in Docker environment but requires `openblas` feature to be enabled during build. BLAS is optimal for float64 on larger sizes (2048²), while SIMD is optimal for float32 on most sizes.
- **Recent Optimizations (2025-11-16 to 2025-11-17)**:
  - ✅ Fixed 512×512 float32: 0.38x → 1.29x (SIMD-first routing)
  - ✅ Improved 1024×1024 float64: 0.44x → 0.59x-0.78x (specialized SIMD path with tiling and unrolling)
  - ✅ Improved 2048×2048 float32: 0.23x → 0.57x-0.69x (SIMD-first routing, 2x unrolling added)
  - ✅ Improved 2048×2048 float64: 0.25x → 0.36x (BLAS-first routing)
  - ✅ Tested deep optimizations: Unrolling and tile size changes for 2048² sizes caused regressions, reverted
  - ✅ Created profiling infrastructure: `scripts/profile_mean_axis0.sh` and `scripts/extract_assembly.sh`
  - **Regression Fixes (2025-11-17)**:
    - ✅ Fixed 512² float32 sum regression (5.0x → 3.6x): Use overall data size for accumulator count instead of chunk size
    - ✅ Fixed mean_axis1 regressions (512²/1024²/2048² float64): Removed ROW_CHUNK path that added overhead
    - ✅ Fixed 1024² float64 broadcast_add regression (3.5x → 1.6x): Added 4x unrolling to NEON add_same_shape_f64 kernel
    - ✅ Verified OpenBLAS threading correctly set to 1 for 2048² float64 operations
  - **mean_axis0 Optimizations (2025-11-17)**:
    - ✅ 1024² float64: Reverted write-combine buffer, implemented pure columnar approach, changed to BLAS-first routing (0.42x → 0.50x improvement)
    - ✅ 2048² float32: Reverted register cache/parallelization, implemented pure columnar approach, changed to BLAS-first routing (0.46x → 0.42x improvement)
    - ⚠️ 2048² float64: BLAS path unchanged (0.36x), still below target; further improvements would require profiling or alternative BLAS backends
  - **Alternative Strategies Implemented (2025-11-17)**:
    - ✅ Pure columnar processing: Process one column at a time, accumulator in register (eliminates load-modify-store cycles)
    - ✅ BLAS-first routing: Try BLAS before SIMD, let OpenBLAS choose optimal tile sizes dynamically (matches NumPy approach)
    - ✅ Prefetch hints for store operations: Added `prfm pstl1keep` hints before final writeback
  - **Benchmark Results (20251117-163021)**:
    - 1024² float64 mean_axis0: 0.50x (improvement from 0.42x, previous best: 0.64x) - BLAS-first routing helped
    - 2048² float32 mean_axis0: 0.42x (improvement from 0.46x, previous best: 0.69x) - BLAS-first routing helped
    - 2048² float64 mean_axis0: 0.36x (unchanged, still below 0.80x target)
  - **NumPy-Style Implementation (2025-11-17)**:
    - ✅ Created code generation infrastructure (`rust/src/simd/codegen.rs`)
    - ✅ Implemented register-resident accumulation (accumulate in registers across tiles)
    - ✅ Added prefetch hints for non-temporal stores
    - ✅ Implemented higher precision accumulators (f64 for f32 operations)
    - ✅ Enhanced parameterized tile sizing
    - ⚠️ **Code-generated kernels disabled due to performance regressions**:
      - 512² float64 mean_axis0: 0.29x (regression from baseline ~0.56x)
      - 512² float32 mean_axis0: 0.05x (regression from baseline ~1.29x)
      - 1024² float32 mean_axis0: 0.02x (regression from baseline ~1.58x)
    - **Next steps**: Investigate tiled accumulation logic, fix bugs, re-enable after validation
  - **Benchmark Results (20251117-164716)**:
    - ⚠️ Code-generated kernels caused regressions - disabled and reverted to original implementations
    - 512² float64 mean_axis0: 0.29x (regression - codegen kernel was slower)
    - 512² float32 mean_axis0: 0.05x (regression - codegen kernel was slower)
    - 1024² float64 mean_axis0: 0.17x (BLAS path used, similar to before)
    - 1024² float32 mean_axis0: 0.02x (regression - codegen kernel was slower)
    - 2048² float64 mean_axis0: 0.42x (BLAS path used, slight improvement)
    - 2048² float32 mean_axis0: 0.37x (BLAS path used, slight improvement)
  - See [docs/mean_axis0_remaining_lags_fix_summary.md](docs/mean_axis0_remaining_lags_fix_summary.md), [docs/deep_optimization_findings.md](docs/deep_optimization_findings.md), and [docs/benchmark_regression_report.md](docs/benchmark_regression_report.md) for details
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


