# Linux Development Guide

This guide covers how to develop, test, benchmark, and profile Raptors on Linux using Docker.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Environment Setup](#docker-environment-setup)
- [Development Workflow](#development-workflow)
- [Building and Installing](#building-and-installing)
- [Running Tests](#running-tests)
- [Benchmarking](#benchmarking)
- [Performance Profiling](#performance-profiling)
- [Debugging](#debugging)
- [Troubleshooting](#troubleshooting)

## Overview

The Docker-based Linux development environment provides:

- **Consistent Linux environment**: Ubuntu 22.04 with all dependencies pre-installed
- **Development tools**: Rust, Python 3.11, build tools, and profiling utilities
- **Performance tools**: `perf`, `py-spy`, and `flamegraph` for detailed profiling
- **Isolated environment**: Reproducible builds and benchmarks independent of your host system
- **Easy cleanup**: Remove container and volumes when done

## Prerequisites

- **Docker** and **Docker Compose** installed on your system
  - Docker Desktop for macOS/Windows: https://www.docker.com/products/docker-desktop
  - Docker Engine for Linux: https://docs.docker.com/engine/install/
- **Git** to clone the repository

## Quick Start

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/eddiethedean/rapto-rs.git
   cd rapto-rs
   ```

2. **Build the Docker image**:
   ```bash
   ./scripts/docker_bench.sh build
   ```
   This builds the `raptors-bench` image with all dependencies. It may take several minutes the first time.

3. **Open an interactive shell**:
   ```bash
   ./scripts/docker_bench.sh shell
   ```

4. **Build and install Raptors** (inside the container):
   ```bash
   cd /workspace/src
   /workspace/.venv/bin/maturin develop --release
   ```

5. **Verify installation**:
   ```bash
   /workspace/.venv/bin/python -c "import raptors; print(raptors.__version__)"
   ```

## Docker Environment Setup

### Image Structure

The Docker image includes:

- **Base OS**: Ubuntu 22.04
- **Python**: 3.11 with virtual environment at `/workspace/.venv`
- **Rust**: Latest stable via rustup
- **System libraries**: OpenBLAS, LAPACK, BLAS for optimized linear algebra
- **Profiling tools**: `perf`, `py-spy`, `flamegraph`
- **Build tools**: `build-essential`, `pkg-config`, `curl`, `git`

### Volume Mounts

The Docker Compose setup mounts:

- **Source code**: `.:/workspace/src` (your local repository)
- **Virtual environment**: `raptors-venv:/workspace/.venv` (persisted between runs)
- **Rust build cache**: `raptors-target:/workspace/src/rust/target` (faster rebuilds)
- **Benchmark results**: `./benchmarks/docker_results:/workspace/src/benchmarks/docker_results`
- **Profiling data**: `./docs/profiles:/workspace/src/docs/profiles`

### Environment Variables

Set in the container:

- `PYTHONPATH=/workspace/.venv/lib/python3.11/site-packages:/workspace/src/python`
- `RAPTORS_THREADS=10` (controls Rayon thread pool)
- `RAPTORS_SIMD=auto` (SIMD enablement)

You can override these when running commands:

```bash
docker-compose -f docker-compose.bench.yml run --rm bench \
  bash -c "export RAPTORS_THREADS=8 && /workspace/.venv/bin/python script.py"
```

## Development Workflow

### Interactive Development

1. **Start a container shell**:
   ```bash
   ./scripts/docker_bench.sh shell
   ```

2. **Make code changes** in your local editor (files are mounted from host)

3. **Rebuild Raptors** (inside container):
   ```bash
   cd /workspace/src
   /workspace/.venv/bin/maturin develop --release
   ```

4. **Test your changes**:
   ```bash
   /workspace/.venv/bin/python -m pytest tests/
   ```

5. **Exit the container**:
   ```bash
   exit
   ```

### Running Commands from Host

You can also run commands directly without entering the container:

```bash
# Run a Python script
./scripts/docker_bench.sh run "/workspace/.venv/bin/python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32"

# Run tests
./scripts/docker_bench.sh run "/workspace/.venv/bin/python -m pytest tests/ -v"

# Run any shell command
./scripts/docker_bench.sh run "cd /workspace/src && /workspace/.venv/bin/maturin develop --release"
```

## Building and Installing

### Development Build

For development with debug symbols and faster compilation:

```bash
# Inside container
cd /workspace/src
/workspace/.venv/bin/maturin develop
```

### Release Build

For performance testing and benchmarking:

```bash
# Inside container
cd /workspace/src
/workspace/.venv/bin/maturin develop --release
```

### Build with OpenBLAS

To enable OpenBLAS support (Linux-specific):

```bash
# Inside container
cd /workspace/src
/workspace/.venv/bin/maturin develop --release --features openblas
```

**Note**: The Docker image includes `libopenblas-dev`, but you need to enable the `openblas` feature during build.

### Rebuilding After Changes

After modifying Rust code, rebuild:

```bash
/workspace/.venv/bin/maturin develop --release
```

The Rust build cache is persisted in a volume, so incremental builds are fast.

## Running Tests

### Python Tests

Run the full test suite:

```bash
# Inside container
/workspace/.venv/bin/python -m pytest tests/ -v
```

Run specific tests:

```bash
/workspace/.venv/bin/python -m pytest tests/test_array.py::test_scale -v
```

### Rust Tests

Run Rust unit tests:

```bash
# Inside container
cd /workspace/src/rust
cargo test
```

Run Rust integration tests:

```bash
cd /workspace/src/rust
cargo test --features test-suite
```

### Quick Test Script

From the host:

```bash
./scripts/docker_bench.sh run "/workspace/.venv/bin/python -m pytest tests/ -v"
```

## Benchmarking

### Quick Benchmark

Compare a specific operation:

```bash
# Inside container
/workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
  --shape 2048x2048 \
  --dtype float64 \
  --operations mean_axis0 \
  --warmup 3 \
  --repeats 30
```

### Full Benchmark Suite

Run the complete 2D benchmark suite:

```bash
# Using helper script (from host)
./scripts/docker_run_benchmarks.sh

# Or manually (inside container)
/workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
  --suite 2d \
  --warmup 3 \
  --repeats 30 \
  --output-json benchmarks/docker_results/results.json
```

Results are saved to `benchmarks/docker_results/` on your host.

### Comparing Benchmarks

To validate performance fixes:

```bash
# Run baseline
./scripts/docker_run_benchmarks.sh benchmarks/docker_results/baseline

# Make changes and rebuild

# Run validation
./scripts/docker_run_benchmarks.sh benchmarks/docker_results/validation

# Compare results (on host)
python scripts/compare_numpy_raptors.py \
  --validate-json benchmarks/docker_results/baseline/results.json \
  --validate-slack 0.05
```

## Performance Profiling

### Using `perf`

Profile with Linux `perf`:

```bash
# Profile a specific operation
./scripts/profile_operation.sh mean_axis0 2048x2048 float64

# This generates:
# - docs/profiles/mean_axis0_2048_2048_float64_raptors.perf.data
# - docs/profiles/mean_axis0_2048_2048_float64_raptors.flamegraph.svg
# - docs/profiles/mean_axis0_2048_2048_float64_raptors.perf.txt
```

View the report:

```bash
# Inside container
perf report -i docs/profiles/mean_axis0_2048_2048_float64_raptors.perf.data
```

### Using `py-spy`

Profile with Python-specific profiler:

```bash
# Profile from host
./scripts/docker_bench.sh run "/workspace/.venv/bin/py-spy record \
  --rate 100 \
  --output /workspace/src/docs/profiles/profile.svg \
  --format flamegraph \
  -- /workspace/.venv/bin/python scripts/profile_single.py scale 2048x2048 float32 raptors 200"
```

### Manual Profiling

Create a profiling script:

```python
# profile_test.py
import numpy as np
import raptors

arr = np.random.randn(2048, 2048).astype(np.float64)
r_arr = raptors.from_numpy(arr)

# Warmup
for _ in range(10):
    _ = r_arr.mean_axis(0)

# Profile loop
for _ in range(200):
    result = r_arr.mean_axis(0)
```

Then profile it:

```bash
# Inside container
perf record -F 99 -g -- /workspace/.venv/bin/python profile_test.py
perf script | /workspace/.venv/bin/python -m flamegraph > profile.svg
```

### Comparing NumPy vs Raptors

Profile both backends:

```bash
# Profile NumPy
./scripts/profile_operation.sh mean_axis0 2048x2048 float64 numpy

# Profile Raptors
./scripts/profile_operation.sh mean_axis0 2048x2048 float64 raptors

# Compare (on host, open both SVG files)
open docs/profiles/mean_axis0_2048_2048_float64_numpy.flamegraph.svg
open docs/profiles/mean_axis0_2048_2048_float64_raptors.flamegraph.svg
```

## Debugging

### Enable Debug Logging

For `mean_axis0` operations, enable debug output:

```bash
# Inside container
export RAPTORS_DEBUG_AXIS0=1
/workspace/.venv/bin/python -c "
import numpy as np
import raptors
arr = np.random.randn(2048, 2048).astype(np.float64)
r_arr = raptors.from_numpy(arr)
result = r_arr.mean_axis(0)
"
```

This prints debug information about which code paths are executed.

### Check Backend Usage

Inspect which backends are being used:

```python
import raptors
from raptors.threading import threading_info

# Run some operations
arr = raptors.array2d([[1.0, 2.0], [3.0, 4.0]])
_ = arr.mean_axis(0)

# Check backend usage
info = threading_info()
print(f"Backend usage: {info.backend_usage}")
```

### Verify SIMD

Check if SIMD is enabled:

```python
import raptors
print(f"SIMD enabled: {raptors.simd_enabled()}")
```

### Check BLAS

Test BLAS availability (requires openblas feature):

```bash
# Inside container
/workspace/.venv/bin/python scripts/check_blas_simd.py
```

### Rust Debug Build

Build with debug assertions for more detailed error messages:

```bash
# Inside container
cd /workspace/src
export RUSTFLAGS='-C debug-assertions'
/workspace/.venv/bin/maturin develop --release
```

## Troubleshooting

### Container Won't Start

**Error**: `docker-compose: command not found`

**Solution**: Install Docker Compose:
- Docker Desktop includes it automatically
- On Linux: `sudo apt-get install docker-compose` or use `docker compose` (v2)

### Permission Denied

**Error**: `permission denied` when accessing Docker

**Solution**: Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in, or:
newgrp docker
```

### perf Not Working

**Error**: `perf` events not available

**Solution**: The Docker Compose config sets `privileged: true` which is required for perf. Ensure your Docker daemon supports privileged mode.

### Build Failures

**Error**: `maturin failed` or compilation errors

**Solution**:
1. Ensure Rust is installed: `rustc --version`
2. Check Rust toolchain: `rustup show`
3. Clean build cache: `rm -rf rust/target/` (on host)
4. Rebuild from scratch: `maturin develop --release`

### Slow Builds

**Issue**: Rebuilds are slow

**Solution**:
- The Rust build cache is persisted in a Docker volume
- Only the first build is slow
- Incremental builds should be fast
- If still slow, check disk space and Docker volume usage

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'raptors'`

**Solution**:
1. Rebuild: `/workspace/.venv/bin/maturin develop --release`
2. Check PYTHONPATH: `echo $PYTHONPATH`
3. Verify installation: `/workspace/.venv/bin/python -c "import raptors"`

### Volume Mount Issues

**Issue**: Changes not reflected in container

**Solution**:
1. Ensure files are saved on host
2. Check volume mounts: `docker-compose -f docker-compose.bench.yml config`
3. Restart container: `docker-compose -f docker-compose.bench.yml down && ./scripts/docker_bench.sh shell`

### Out of Disk Space

**Issue**: Docker using too much disk space

**Solution**:
```bash
# Clean up unused Docker resources
docker system prune -a

# Remove old images
docker rmi raptors-bench

# Clean up build volumes
docker-compose -f docker-compose.bench.yml down -v
```

### Port Conflicts

**Issue**: Port already in use

**Solution**: The benchmarking setup doesn't use ports, but if you add services that do, update `docker-compose.bench.yml` with port mappings:
```yaml
ports:
  - "8080:8080"
```

## Advanced Usage

### Custom Docker Image

To customize the Docker image:

1. Edit `Dockerfile.bench`
2. Rebuild: `./scripts/docker_bench.sh build`

### Multiple Containers

Run multiple containers for parallel testing:

```bash
docker-compose -f docker-compose.bench.yml up -d --scale bench=3
```

### Persistent Data

Data is automatically persisted in volumes:
- `raptors-venv`: Virtual environment
- `raptors-target`: Rust build cache

To reset everything:

```bash
docker-compose -f docker-compose.bench.yml down -v
./scripts/docker_bench.sh build
```

### CI/CD Integration

The Docker setup can be used in CI:

```yaml
# .github/workflows/linux-test.yml
- name: Run tests in Docker
  run: |
    docker-compose -f docker-compose.bench.yml run --rm bench \
      /workspace/.venv/bin/python -m pytest tests/
```

## Additional Resources

- [Docker Benchmarking Guide](docker_benchmarking.md) - Detailed benchmarking instructions
- [Profiling Methodology](profiling_methodology.md) - Profiling best practices
- [Performance Fixes](performance_fixes.md) - Performance optimization notes

## Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review Docker logs: `docker-compose -f docker-compose.bench.yml logs`
3. Check GitHub issues: https://github.com/eddiethedean/rapto-rs/issues
4. Create a new issue with:
   - Docker version: `docker --version`
   - Docker Compose version: `docker-compose --version`
   - Error messages and logs
   - Steps to reproduce

