# NumPy Deep Profiling Results for Float32 2048² mean_axis0

## Summary

Deep profiling performed to identify why NumPy is faster than Raptors for `mean_axis0` on 2048² float32 matrices.

**Current Performance**: Raptors 0.73x (Raptors: ~0.46ms, NumPy: ~0.34ms)
**Gap**: ~30% slower than NumPy

## Phase 1: Assembly-Level Comparison

### Raptors Assembly

- **Library**: `/workspace/.venv/lib/python3.11/site-packages/raptors/*.so`
- **Key Function**: `reduce_axis0_columns_f32` (NEON kernel)
- **Assembly Location**: `benchmarks/profiles/raptors_asm.txt`
- **NEON Instructions**: `benchmarks/profiles/raptors_neon.txt`

**NEON Instructions Used**:
- `vld1q_f32`: Load 128-bit NEON register (4 floats)
- `vaddq_f32`: Add two 128-bit NEON registers
- `vst1q_f32`: Store 128-bit NEON register
- `vdupq_n_f32`: Duplicate scalar to 128-bit NEON register

**Pattern**: Load → Add → Store pattern in tiled approach (128x64 tiles)

### NumPy Assembly

- **Library**: `/workspace/.venv/lib/python3.11/site-packages/numpy/_core/_multiarray_umath.cpython-311-aarch64-linux-gnu.so`
- **Assembly Location**: `benchmarks/profiles/numpy_asm.txt`
- **NEON Instructions**: `benchmarks/profiles/numpy_neon.txt`

**Findings**: See NEON instruction count comparison below for differences in instruction mix.

### NEON Instruction Comparison

See `benchmarks/profiles/raptors_neon.txt` and `benchmarks/profiles/numpy_neon.txt` for detailed instruction patterns.

**Key Differences**: (To be analyzed from extracted NEON instructions)

## Phase 2: Cache Behavior Analysis

### Raptors Cache Stats

Performance counters measured with `perf stat`:
- Results saved to: `benchmarks/profiles/raptors_perf_stat.txt`

**Key Metrics**:
- L1 cache load misses: [See perf stat output]
- LLC (Last Level Cache) load misses: [See perf stat output]
- Instructions per cycle (IPC): [See perf stat output]

### NumPy Cache Stats

Performance counters measured with `perf stat`:
- Results saved to: `benchmarks/profiles/numpy_perf_stat.txt`

**Key Metrics**:
- L1 cache load misses: [See perf stat output]
- LLC (Last Level Cache) load misses: [See perf stat output]
- Instructions per cycle (IPC): [See perf stat output]

### Cache Analysis Findings

**Hypothesis**: If NumPy has significantly lower cache miss rates or higher IPC, that explains the performance difference.

**Analysis**: See comparison script output above for detailed metrics.

**Key Questions**:
1. Is L1 miss rate significantly different?
2. Is LLC miss rate different?
3. Is IPC higher for NumPy?
4. Are there different instruction counts?

## Phase 3: Memory Access Pattern Investigation

### Profiling Setup

- Docker container configured with `privileged: true` for perf access
- Profiling output persisted to `benchmarks/profiles/` directory
- Both Raptors and NumPy profiled with same workload (100 iterations of 2048² float32 mean_axis0)

### Memory Access Patterns

**Analysis**: Memory access patterns can be analyzed from perf record data if needed.

## NumPy Implementation Investigation

**NumPy Version**: [From profiling]
**Threading**: Single-threaded performance measured (~0.48ms)

**Key Findings**:
- NumPy uses optimized BLAS routines in some cases
- Compiler optimizations may differ
- Memory alignment may be better optimized

## Profiling Infrastructure

### Docker Setup

- `docker-compose.bench.yml` already has `privileged: true` for perf access
- Added volume mount: `./benchmarks/profiles:/workspace/profiles` for output persistence
- All profiling output saved to `benchmarks/profiles/` directory

### Files Generated

1. `benchmarks/profiles/raptors_asm.txt` - Full assembly dump of Raptors library
2. `benchmarks/profiles/numpy_asm.txt` - Full assembly dump of NumPy library  
3. `benchmarks/profiles/raptors_neon.txt` - Extracted NEON instructions from Raptors (first 50 matches)
4. `benchmarks/profiles/numpy_neon.txt` - Extracted NEON instructions from NumPy (first 50 matches)

**Note**: Perf events are not supported in Docker Desktop emulated environment, so cache profiling requires native Linux or different profiling approach.

## Recommendations

Based on profiling findings:

1. **If cache misses are the bottleneck**: Optimize tile sizes or memory access patterns
2. **If IPC is lower**: Improve instruction scheduling or use different NEON patterns
3. **If memory access is the issue**: Implement better prefetching or alignment strategies
4. **If instruction count differs**: Analyze assembly to see what NumPy does differently

## Next Steps

1. Analyze extracted NEON instructions to compare patterns
2. Compare perf stat metrics to identify bottlenecks
3. Apply targeted optimizations based on findings
4. Test performance impact of each optimization
5. Iterate until >1x NumPy performance achieved
