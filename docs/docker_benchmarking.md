# Docker-Based Linux Benchmarking Guide

This guide explains how to use Docker to run consistent Linux benchmarks and profiles for Raptors vs NumPy.

## Overview

The Docker setup provides:
- Consistent Linux environment (Ubuntu 22.04)
- All profiling tools pre-installed (perf, py-spy, flamegraph)
- Isolated environment for reproducible benchmarks
- Easy profiling and performance analysis

## Quick Start

### 1. Build the Docker Image

```bash
./scripts/docker_bench.sh build
```

This builds the `raptors-bench` image with all dependencies.

### 2. Run Benchmarks

Run a full benchmark suite:

```bash
./scripts/docker_bench.sh bench
```

Or run specific benchmarks:

```bash
./scripts/docker_bench.sh run "python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations broadcast_add"
```

### 3. Open Interactive Shell

```bash
./scripts/docker_bench.sh shell
```

## Profiling Operations

### Profile a Specific Operation

Profile both NumPy and Raptors:

```bash
./scripts/docker_bench.sh profile broadcast_add 512x512 float32
```

Profile only one backend:

```bash
./scripts/docker_bench.sh profile scale 1024x1024 float64 raptors
```

This generates:
- `perf` data and reports
- Flamegraphs (SVG)
- Py-spy profiles

Results are saved to `docs/profiles/`.

### Compare Profiles

After profiling both backends:

```bash
./scripts/compare_profiles.sh broadcast_add 512x512 float32
```

This shows side-by-side comparison of NumPy vs Raptors performance.

## Directory Structure

```
/workspace/
├── benchmarks/
│   ├── docker_results/    # Benchmark results from Docker runs
│   └── ...
├── docs/
│   ├── profiles/          # Profiling results (perf data, flamegraphs)
│   └── ...
└── scripts/
    ├── profile_operation.sh
    ├── profile_single.py
    └── ...
```

## Environment Variables

Set in Docker container:

- `PYTHONPATH=/workspace/python`
- `RAPTORS_THREADS=10`
- `RAPTORS_SIMD=auto`

## Validation

After making performance fixes:

```bash
./scripts/docker_validate_fix.sh broadcast_add 512x512 float32
```

Run full suite:

```bash
./scripts/docker_full_suite.sh
```

## Troubleshooting

### perf requires privileged mode

The Docker Compose configuration sets `privileged: true` to enable perf profiling.

### Flamegraph generation fails

Ensure `flamegraph` is installed in the venv. It should be installed automatically during Docker build.

### Permission errors

If you see permission errors, ensure Docker has proper permissions. On Linux, you may need to add your user to the docker group.

## Next Steps

1. Run baseline benchmarks to identify laggards
2. Profile each laggard operation
3. Analyze profiles to identify bottlenecks
4. Implement fixes
5. Validate fixes with benchmarks

