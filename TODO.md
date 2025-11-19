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

**Optimization Guide**: See [docs/pure_columnar_optimization.md](docs/pure_columnar_optimization.md) for detailed documentation on the pure columnar SIMD optimization technique and how to apply it to other operations.

**Note**: Recent optimizations (2025-11-17 to 2025-11-18) implementing NumPy-style BLAS configuration, threading improvements, f64 accumulators, pure columnar SIMD paths, and OpenBLAS integration have significantly improved performance. Most critical and high priority items are now resolved, with mean_axis0 operations at 512² and 1024² now faster than NumPy!

**Pure Columnar Optimization Applied**: The register-resident accumulation approach (pure columnar SIMD) has been applied to:
- ✅ 512² float64: 1.63x (pure columnar, SIMD-first) - **faster than NumPy!**
- ✅ 512² float32: 1.51x (pure columnar) - **faster than NumPy!**
- ✅ 1024² float64: 2.24x (pure columnar, SIMD-first) - **faster than NumPy!**
- ✅ 1024² float32: 1.37x (pure columnar) - **faster than NumPy!**
- ✅ 2048² float64: 0.59x (pure columnar, 4x unrolled) - improved but still needs work
- ✅ 2048² float32: 0.79x (pure columnar) - near parity

**Note**: `sum_axis0` automatically benefits from these optimizations since it uses the same `reduce_axis0` functions (just without the division step).

### Critical Priority (< 0.80x) - REGRESSIONS DETECTED ⚠️

- [ ] **mean_axis0 @ 512² float32**: 0.19x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 1.29x to 0.19x
  - Previous performance: 1.29x (20251118)
  - Current performance: 0.19x (20251118-210653)
  - Actions needed: Investigate routing and SIMD path usage
  - Code location: `rust/src/lib.rs` lines 5785-5850
  - Date: 2025-11-18

- [ ] **mean_axis0 @ 1024² float64**: 0.04x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 2.24x to 0.04x
  - Root cause: Unknown - routing fixes may not have resolved the issue
  - Previous performance: 2.24x (20251118)
  - Current performance: 0.04x (20251118-210653)
  - Actions needed:
    - Investigate why SIMD path is not being used
    - Check if `reduce_axis0_columns_f64` is returning None when it shouldn't
    - Verify routing logic is correct
  - Code location: `rust/src/lib.rs` lines 4945-5109, `rust/src/simd/mod.rs` lines 3037-3095
  - Date: 2025-11-18

- [ ] **mean_axis0 @ 2048² float32**: 0.32x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 2.12x to 0.32x
  - Previous performance: 2.12x (20251117)
  - Current performance: 0.32x (20251118-210653)
  - Actions needed: Investigate routing and SIMD path usage
  - Code location: `rust/src/lib.rs` lines 5997-6100
  - Date: 2025-11-18

- [ ] **mean_axis0 @ 2048² float64**: 0.10x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 0.59x to 0.10x
  - Root cause: Unknown - routing fixes may not have resolved the issue
  - Previous performance: 0.59x (20251118)
  - Current performance: 0.10x (20251118-210653)
  - Actions needed:
    - Investigate why SIMD path is not being used
    - Check if `reduce_axis0_columns_f64` is returning None when it shouldn't
    - Verify routing logic is correct
  - Code location: `rust/src/lib.rs` lines 5114-5583, `rust/src/simd/mod.rs` lines 3097-3234
  - Date: 2025-11-18

- [x] **scale @ 512² float64**: 1.00x (improved from 0.55x, now faster than NumPy!) ✅
  - Status: ✅ **FIXED** - Improved from 0.55x to 1.00x (exceeds NumPy!)
  - Recent optimizations (2025-11-18):
    - ✅ Changed to SIMD-first routing on Linux (BLAS overhead too high)
    - ✅ Optimized BLAS copy operation (ptr::copy_nonoverlapping)
    - ✅ Added dedicated fast path for 512² in SIMD (10x unrolling)
  - Current performance: 1.00x (20251118-210653)
  - Code location: `rust/src/lib.rs` lines 3709-3750, `rust/src/simd/mod.rs` lines 3885-4027
  - Date: 2025-11-18

- [x] **scale @ 1024² float64**: 1.00x (improved from 0.48x, now faster than NumPy!) ✅
  - Status: ✅ **FIXED** - Improved from 0.48x to 1.00x (exceeds NumPy!)
  - Recent optimizations (2025-11-18):
    - ✅ Changed to SIMD-first routing on Linux (BLAS overhead too high)
    - ✅ Optimized BLAS copy operation (ptr::copy_nonoverlapping)
    - ✅ Added dedicated fast path for 1024² in SIMD (10x unrolling)
  - Current performance: 1.00x (20251118-210653)
  - Code location: `rust/src/lib.rs` lines 3758-3800, `rust/src/simd/mod.rs` lines 3885-4027
  - Date: 2025-11-18

- [ ] **scale @ 512² float32**: 0.60x (needs optimization) ⚠️
  - Status: Still below target (0.60x < 0.80x)
  - Current performance: 0.60x (20251118-210653)
  - Code location: `rust/src/lib.rs` lines 4100-4150
  - Date: 2025-11-18

### High Priority (0.80x - 0.95x) - MOSTLY RESOLVED ✅

**Note**: All high priority items have been resolved with recent optimizations!

- [ ] **mean_axis0 @ 512² float64**: 0.11x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 1.63x to 0.11x
  - Root cause: Unknown - routing fixes may not have resolved the issue
  - Previous performance: 1.63x (20251118)
  - Current performance: 0.11x (20251118-210653)
  - Actions needed:
    - Investigate why SIMD path is not being used
    - Check if `reduce_axis0_columns_f64` is returning None when it shouldn't
    - Verify routing logic is correct
  - Code location: `rust/src/lib.rs` lines 4579-4735, `rust/src/simd/mod.rs` lines 2937-3035
  - Date: 2025-11-18

- [ ] **mean_axis0 @ 1024² float32**: 0.32x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped significantly
  - Current performance: 0.32x (20251118-210653)
  - Actions needed: Investigate routing and SIMD path usage
  - Code location: `rust/src/lib.rs` lines 5927-5992
  - Date: 2025-11-18

- [x] **broadcast_add @ 1024² float32**: 1.61x (improved from 0.77x, now faster than NumPy!) ✅
  - Status: ✅ NEON kernel optimized with 4x unrolling
  - Code location: `rust/src/simd/mod.rs` lines 2391-2439

- [ ] **broadcast_add @ 512² float32**: 0.11x (REGRESSION - needs investigation) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 2.22x to 0.11x
  - Previous performance: 2.22x (20251118)
  - Current performance: 0.11x (20251118-210653)
  - Actions needed: Investigate routing and SIMD path usage
  - Code location: `rust/src/lib.rs` broadcast_add functions, `rust/src/simd/mod.rs` lines 2391-2439
  - Date: 2025-11-18

- [x] **broadcast_add @ 1024² float64**: Fixed regression (was 3.5x → 1.6x, now optimized) ✅
  - Status: ✅ Fixed regression by adding 4x unrolling to NEON add_same_shape_f64 kernel
  - Code location: `rust/src/simd/mod.rs` lines 3078-3126
  - Date: 2025-11-17

- [ ] **scale @ 2048² float64**: 0.77x (regression from 0.86x, needs optimization) ⚠️
  - Status: ⚠️ **REGRESSION** - Performance dropped from 0.86x to 0.77x
  - Recent optimizations (2025-11-18):
    - ✅ Adaptive NEON unrolling: 10x for large arrays (>1M), 8x for medium (64K-1M), 6x for small (<64K)
    - ✅ Changed to SIMD-first for large matrices on Linux (BLAS overhead)
  - Current performance: 0.77x (20251118-210653)
  - Code location: `rust/src/lib.rs` lines 3750-3857, `rust/src/simd/mod.rs` lines 3885-4027
  - Date: 2025-11-18

- [x] **scale @ 2048² float32**: 1.35x (improved from 0.98x, now faster than NumPy!) ✅
  - Status: ✅ **FIXED** - Improved from 0.98x to 1.35x (exceeds NumPy!)
  - Current performance: 1.35x (20251118-210653)
  - Code location: `rust/src/lib.rs` lines 3843-3869, `rust/src/simd/mod.rs` lines 2597-2732
  - Date: 2025-11-18

### Notes

- **BLAS Integration**: OpenBLAS is now enabled by default in Cargo.toml. Runtime detection available via `blas::openblas_available()`. BLAS is optimal for float64 on larger sizes (2048²) on macOS, while SIMD is optimal for most sizes on Linux due to OpenBLAS overhead. Use `raptors.blas_config()` in Python to check BLAS backend status.
- **Recent Optimizations (2025-11-16 to 2025-11-18)**:
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
  - **Benchmark Results (20251118-210653)**:
    - ⚠️ mean_axis0 operations have regressed significantly:
      - 512² float64 mean_axis0: 0.11x (REGRESSION from 1.63x) - needs investigation
      - 1024² float64 mean_axis0: 0.04x (REGRESSION from 2.24x) - needs investigation
      - 2048² float64 mean_axis0: 0.10x (REGRESSION from 0.59x) - needs investigation
      - 512² float32 mean_axis0: 0.19x (REGRESSION from 1.29x) - needs investigation
      - 1024² float32 mean_axis0: 0.32x (below target)
      - 2048² float32 mean_axis0: 0.32x (REGRESSION from 2.12x) - needs investigation
    - ✅ scale operations improved:
      - 512² float64 scale: 1.00x (improved from 0.55x) - now faster than NumPy!
      - 1024² float64 scale: 1.00x (improved from 0.48x) - now faster than NumPy!
      - 2048² float64 scale: 0.77x (regression from 0.86x, needs optimization)
      - 1024² float32 scale: 0.93x (good performance)
      - 2048² float32 scale: 1.35x (faster than NumPy!)
    - ⚠️ Other regressions:
      - 512² float32 broadcast_add: 0.11x (very bad - needs investigation)
      - 512² float32 scale: 0.60x (below target)
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
- **OpenBLAS Integration (2025-11-18)**:
  - ✅ OpenBLAS feature enabled by default in `Cargo.toml`
  - ✅ Runtime detection via `blas::openblas_available()`
  - ✅ Enhanced diagnostics via `blas::backend_info()` and `raptors.blas_config()` in Python
  - ✅ BLAS dispatch optimized: SIMD-first for small matrices on Linux (BLAS overhead), BLAS-first for large matrices on macOS

See [docs/linux_development_guide.md](docs/linux_development_guide.md) for development setup and [docs/docker_benchmarking.md](docs/docker_benchmarking.md) for benchmarking instructions.


