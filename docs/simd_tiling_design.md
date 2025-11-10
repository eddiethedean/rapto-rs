# 2-D Tiled SIMD Kernel Design

## Profiling Summary

Recent benchmarks (Apple M3) show:

- Same-shape add (1024×1024) SIMD: ~0.60 ms vs scalar 0.61 ms (no real win).
- Row broadcast add: SIMD 0.60 ms vs scalar 0.65 ms (≈8% faster).
- Column broadcast add: ~48 ms — still scalar and the dominant pain point.
- NumPy executes the same workloads in ~0.7 ms (same-shape) and <1 ms (broadcast).

Perf captures highlight:

- High L1 miss rates during column broadcasts (due to strided loads).
- Under-utilized vector units (only 2-wide NEON lanes active).
- No multi-threading; single core capped at ~320 MB/s vs NumPy’s multi-threaded ~900 MB/s.

## Objectives

1. Achieve >1× NumPy throughput for 2-D elementwise add/scale and primary broadcast cases.
2. Maintain portability across x86-64 (AVX2/AVX-512) and ARM64 (NEON/SVE, when available).
3. Keep scalar fallback for exotic strides and older CPUs.

## Kernel Strategy

### 1. Data Layout Detection

- Ensure we know when arrays are row-major and contiguous (existing check).
- Extend shape metadata with stride classification: contiguous row, contiguous column, general.
- Derive tile-friendly metrics (`rows_per_tile`, `cols_per_tile`) based on strides.

### 2. Tiling Plan

| Target ISA | L1 Size | Proposed Tile | Rationale |
| --- | --- | --- | --- |
| AVX2 (x86-64) | 32 KB | 8×128 (double) | 8 rows × 128 cols × 8 bytes ≈ 8 KB per operand → fits L1 |
| AVX-512 | 32 KB | 16×128 | Wider lanes justify doubling rows |
| NEON (ARM64) | 64 KB | 8×64 | 8 rows × 64 cols × 8 bytes ≈ 4 KB per operand |

- Outer loop: block rows in tile-height steps.
- Inner loop: operate on contiguous column tiles with vector lanes.
- Broadcast handling: replicate RHS tile into vector registers once per tile.

### 3. SIMD Implementations

- Implement separate modules: `simd::avx2`, `simd::avx512`, `simd::neon`.
- Provide common trait `SimdKernel` with functions:
  - `add_tile(lhs, rhs, out)`
  - `add_row_broadcast(inout, row)`
  - `add_col_broadcast(inout, col)`
  - `scale_tile(matrix, factor)`
- Use vector load/store with tail handling (mask loads for AVX-512, scalar cleanup for AVX2/NEON).
- Fuse operations where possible (e.g., `out = a + b` vs `out = (a + b) * alpha`).

### 4. Column Broadcast

- Convert column vector to contiguous block per tile: load `cols` values into a temporary buffer, broadcast across rows using vector lane duplication.
- Use gather instructions only as fallback (AVX2 gather ≈ slow); prefer transposing tile or preloading repeated values.

### 5. Multi-threading

- Partition by row tiles across threads (Rayon on Rust side).
- Thread count derived from env var (`RAPTORS_THREADS`) or `num_cpus` default.
- Ensure tiles align to cache to avoid false sharing.

### 6. Scalar Fallback

- If strides are not tile-friendly (e.g., non-unit column stride), re-use existing scalar loops.
- Add diagnostic instrumentation to histogram fallback usage in benchmarks.

## Timeline / Deliverables

1. **Prototype AVX2 tile**: same-shape add/scale.
2. **Extend to NEON**: SVE not yet targeted; NEON fallback with 4-lane f64.
3. **Broadcast rows/columns**: share tiling infra; column broadcast uses preloaded tile buffer.
4. **Introduce threading**: once kernels stable.
5. **Benchmark harness**: automate comparisons vs NumPy (with CPU affinity, thread parity).

## Risks / Mitigations

- **Complexity:** Tiled kernels increase code size → hide in dedicated modules with tests.
- **Portability:** Feature detection must guard AVX-512; fallback to AVX2/NEON automatically.
- **Thread contention:** On small matrices, threading can hurt → add threshold for enabling parallel path.

## Next Steps

- Implement the `simd::tiles` scaffolding in Rust (`TileDim`, `TileIter`).
- Start with AVX2 8×128 tile and measure.
- Integrate column broadcast path with per-tile temp buffer.
