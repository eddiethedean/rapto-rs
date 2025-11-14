# Docker Linux Benchmarking Status

## Infrastructure Setup

✅ **Completed:**
- Docker image with all dependencies (Ubuntu 22.04, Python 3.11, Rust, profiling tools)
- Docker Compose configuration with volume mounts
- Helper scripts for benchmarking and profiling
- Profiling infrastructure (perf, py-spy, flamegraph)

## Baseline Results (Linux ARM64)

Benchmarks run on Linux (Ubuntu 22.04, ARM64) show the following laggards:

### Critical Laggards (< 1.0x speedup):

1. **mean_axis0 @ 2048x2048 float64**: 0.02x (SEVERE - needs investigation)
2. **mean_axis0 @ 2048x2048 float32**: 0.04x (SEVERE - needs investigation)
3. **broadcast_add @ 512x512 float32**: 0.81x (close to parity)
4. **broadcast_add @ 1024x1024 float32**: 0.80x
5. **scale @ 2048x2048 float64**: 0.86x
6. **scale @ 2048x2048 float32**: 0.92x (close to parity)

### Analysis

**Broadcast Row @ 512² float32 (0.81x):**
- Path: Sequential SIMD loop with `simd::add_same_shape_f32` per row
- Issue: NEON SIMD implementation may not be as optimized as NumPy's OpenBLAS
- Status: Analyzed, requires NEON kernel optimization or BLAS path
- Attempted fix: Tried parallelization but overhead was too high (0.26x)

**Scale operations:**
- Need profiling to identify bottlenecks
- May benefit from BLAS dispatch on Linux

**mean_axis0 operations:**
- Severe performance issues on Linux
- Likely different code path than macOS
- Needs investigation

## Next Steps

1. Optimize NEON SIMD kernel for broadcast_add_row
2. Profile scale operations to identify bottlenecks
3. Investigate mean_axis0 performance issues on Linux
4. Consider BLAS paths for Linux where appropriate

## Files Created

- `Dockerfile.bench` - Docker image definition
- `docker-compose.bench.yml` - Docker Compose configuration
- `scripts/docker_bench.sh` - Helper script for Docker operations
- `scripts/docker_run_benchmarks.sh` - Run benchmarks in Docker
- `scripts/profile_operation.sh` - Profile specific operations
- `scripts/profile_single.py` - Single operation profiling script
- `docs/docker_benchmarking.md` - Docker setup guide
- `docs/profiling_methodology.md` - Profiling methodology

