# Plan: Surpass NumPy Performance (2025-11-10)

## Objective
Outperform NumPy across targeted 2-D workloads (denser reductions, broadcasts, and contiguous elementwise ops) on both AVX2/AVX-512 x86-64 and NEON/SVE ARM64 while preserving correctness, ergonomics, and portable fallbacks.

## Current Snapshot
- **Ahead:** row broadcasts and fused axis reductions when SIMD is enabled; SIMD-enabled column broadcasts approach parity.
- **Behind:** full-array reductions (`mean`, `sum`) lag (0.35–0.45 ms vs NumPy at 0.13–0.15 ms). Mean/row reduction still rely on SIMD but can spike due to cache behavior and Rayon overhead.
- **Dependencies:** baseline validation now tracks `1024x1024` float64 operations; we see variance up to ~1 ms in worst cases on ARM.

## Workstreams

### 1. Global Reduction Kernels
- Implement tiled, fused reduction loops for full-array `sum/mean` using chunked SIMD accumulation, avoiding repeated pass over memory.
- Add optional partial sum buffers to reduce cache thrashing.
- Integrate with Rayon: per-tile parallel map + reduce with deterministic summation order.
- Validation: add dedicated baselines for `mean`/`sum` and ensure <1.5× NumPy across repeated runs.

### 2. Cache-Aware Tiling & Wide Lanes
- Introduce tile descriptors (`TileSpec`) tuned per ISA (AVX2, AVX-512, NEON, SVE) with heuristics deriving row/column block sizes from shape.
- For x86-64 with AVX-512, reuse the new `_mm512` helpers to process 8 f64 elements per loop when available.
- On Apple Silicon, evaluate `arm64e` throughput; if SVE is unavailable, maintain NEON path but align tiling to 128-bit boundaries.

### 3. Broadcast & Column Improvements
- Implement column-broadcast kernels using shared loads plus register reuse, reducing repeated boundary checks.
- Batch multiple column stripes when launching Rayon tasks to amortize task creation cost.

### 4. Adaptive Threading Enhancements
- Extend `should_parallelize` to include measured throughput data (via running medians stored in `OnceLock`), allowing dynamic threshold tweaks.
- Expose a Python API (`raptors.threading_info()`) showing active thread pool size and heuristics for debugging CI runs.

### 5. Benchmark & CI Automation
- Expand baselines to include `512x512`, `2048x2048` shapes in float32/float64.
- Integrate benchmark validation into CI (GitHub Actions) with slack options per operation.
- Capture separate baseline JSONs for scalar fallback to track regressions when SIMD is disabled.

### 6. Cross-Platform Validation
- Secure access to x86-64 AVX2/AVX-512 hardware; run the updated suites and capture deltas.
- Document hardware-specific tuning knobs (e.g., env flags to force AVX-512 usage) and publish results alongside ARM metrics.

## Milestones
1. **Reduction Kernel Upgrade** (2-3 days): implement fused global reducers, update docs, baselines. Goal: `mean` <= 0.25 ms on M3.
2. **Tiling + AVX-512** (2 days): integrate tile specs and update x86 SIMD module; re-run benchmarks on x86 hardware.
3. **Threading Diagnostics** (1 day): expose introspection API and gather telemetry in CI logs.
4. **CI Baseline Expansion** (1 day): automate JSON validation across larger suite, support per-operation slack.

## Risks & Mitigations
- **Variance on small matrices:** gating heuristics must prevent spurious threading; fallback to scalar when in doubt.
- **Cross-platform differences:** maintain scalar fallback tests and allow per-platform baselines to avoid blocking PRs due to hardware variance.
- **Numerical drift:** ensure reductions use deterministic accumulation order or document accuracy trade-offs vs NumPy.

## Deliverables
- Updated SIMD/parallel kernels in `rust/src/lib.rs` and `rust/src/simd/mod.rs`.
- Revised benchmarking docs/README with new results.
- Baseline JSONs under `benchmarks/baselines/` for CI gating.
- CI workflow additions running validation mode per target.
- Diagnostics API for thread/pool status.
