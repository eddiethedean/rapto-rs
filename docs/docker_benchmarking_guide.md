# Docker Benchmarking Guide

This guide explains how to run Linux benchmarks using Docker for the raptors project.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2 (included with Docker Desktop)

Verify Docker is ready:
```bash
docker ps
docker compose version
```

## Quick Start

### 1. Build the Docker Image (First Time Only)

```bash
./scripts/docker_bench.sh build
```

This builds an Ubuntu 22.04 image with:
- Python 3.11
- Rust toolchain
- BLAS/LAPACK libraries (OpenBLAS)
- Benchmarking tools (perf, py-spy, flamegraph)
- All Python dependencies (maturin, numpy, asv, etc.)

**Note:** This may take 5-10 minutes the first time as it downloads and installs all dependencies.

### 2. Build and Install Raptors in the Container

You have two options:

#### Option A: Interactive Shell (Recommended for first-time setup)

```bash
./scripts/docker_bench.sh shell
```

Inside the container:
```bash
cd /workspace/src
/workspace/.venv/bin/maturin develop --release --features openblas
```

#### Option B: One-liner Build

```bash
./scripts/docker_bench.sh run bash -c "cd /workspace/src && /workspace/.venv/bin/maturin develop --release --features openblas"
```

### 3. Run Benchmarks

#### Full Benchmark Suite (2D operations)

```bash
./scripts/docker_bench.sh bench
```

This runs the full 2D benchmark suite with:
- Shapes: 512×512, 1024×1024, 2048×2048
- DTypes: float32, float64
- Operations: mean_axis0, scale, broadcast_add, etc.
- Results saved to: `benchmarks/docker_results/YYYYMMDD-HHMMSS/`

#### Specific Benchmark Configuration

Use `docker_run_benchmarks.sh` for more control:

```bash
./scripts/docker_run_benchmarks.sh [OUTPUT_DIR]
```

This script:
- Builds raptors automatically
- Runs the 2D suite with 3 warmup iterations and 30 repeats
- Saves results as JSON to the specified output directory

#### Custom Benchmark Command

Run any benchmark command directly:

```bash
./scripts/docker_bench.sh run "/workspace/.venv/bin/python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations mean_axis0"
```

#### Axis-0 Focused Suite

Run the specialized axis-0 benchmark suite:

```bash
./scripts/docker_bench.sh run "/workspace/.venv/bin/python benchmarks/run_axis0_suite.py --shapes 512 1024 2048 --dtypes float32 float64"
```

## Available Scripts

### `docker_bench.sh` - Main Helper Script

```bash
# Build the Docker image
./scripts/docker_bench.sh build

# Open an interactive shell
./scripts/docker_bench.sh shell

# Run full benchmark suite
./scripts/docker_bench.sh bench

# Run a custom command
./scripts/docker_bench.sh run "your command here"

# Profile a specific operation
./scripts/docker_bench.sh profile mean_axis0 2048x2048 float32

# Clean up Docker resources
./scripts/docker_bench.sh clean
```

### `docker_run_benchmarks.sh` - Automated Benchmark Runner

```bash
# Run with default output directory (timestamped)
./scripts/docker_run_benchmarks.sh

# Run with custom output directory
./scripts/docker_run_benchmarks.sh benchmarks/docker_results/my_test
```

### `docker_full_suite.sh` - Full Suite Runner

```bash
# Run full suite with default settings
./scripts/docker_full_suite.sh

# Run with custom output directory
./scripts/docker_full_suite.sh benchmarks/docker_results/my_test
```

## Benchmark Script Options

The main benchmark script `compare_numpy_raptors.py` supports many options:

```bash
# Run specific shape and dtype
--shape 2048x2048 --dtype float32

# Run specific operations
--operations mean_axis0 scale broadcast_add

# Control iterations
--warmup 3 --repeats 30

# Output options
--output-json path/to/results.json
--output-dir path/to/directory

# SIMD mode
--simd-mode auto|force|disable

# Run preset suites
--suite 2d|mixed
```

## Container Environment

The Docker container provides:

- **Working Directory:** `/workspace/src` (mounted from project root)
- **Python Environment:** `/workspace/.venv` (persistent volume)
- **Rust Target:** `/workspace/src/rust/target` (persistent volume for faster rebuilds)
- **Results Directory:** `benchmarks/docker_results/` (mounted from host)
- **Profiles Directory:** `docs/profiles/` (mounted from host)

### Environment Variables

- `PYTHONPATH`: Set to include project Python code
- `RAPTORS_THREADS`: Default 10 (configurable)
- `RAPTORS_SIMD`: Default "auto" (configurable)
- `CARGO_TARGET_DIR`: Set to use volume-mounted target directory

## Example Workflow

### Complete Benchmark Run

```bash
# 1. Build image (first time only)
./scripts/docker_bench.sh build

# 2. Build raptors
./scripts/docker_bench.sh run bash -c "cd /workspace/src && /workspace/.venv/bin/maturin develop --release --features openblas"

# 3. Run benchmarks
./scripts/docker_run_benchmarks.sh

# 4. Check results
ls -la benchmarks/docker_results/
cat benchmarks/docker_results/YYYYMMDD-HHMMSS/results.json
```

### Quick Test

```bash
# Build and test a single operation
./scripts/docker_bench.sh run bash -c "
  cd /workspace/src && \
  /workspace/.venv/bin/maturin develop --release --features openblas && \
  /workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
    --shape 1024x1024 \
    --dtype float32 \
    --operations mean_axis0 \
    --warmup 3 \
    --repeats 10
"
```

## Troubleshooting

### Docker Not Running
```bash
# Check Docker status
docker ps

# If error, start Docker Desktop and wait 30-60 seconds
```

### Permission Errors
- On macOS, Docker Desktop should work without additional permissions
- Ensure Docker Desktop has necessary permissions in System Preferences

### Build Fails
- Check internet connection (needs to download packages)
- Ensure Docker Desktop has enough resources (Memory: 4GB+, CPU: 2+ cores)
- Try cleaning and rebuilding:
  ```bash
  ./scripts/docker_bench.sh clean
  ./scripts/docker_bench.sh build
  ```

### Container Already Running
If you see "container already running" errors:
```bash
# List running containers
docker ps

# Stop specific container
docker stop <container_id>

# Or stop all raptors containers
docker ps -q --filter "ancestor=rapto-rs-bench" | xargs docker stop
```

### Rebuild Raptors After Code Changes
```bash
# Rebuild inside container
./scripts/docker_bench.sh run bash -c "cd /workspace/src && /workspace/.venv/bin/maturin develop --release --features openblas"
```

## Results Format

Benchmark results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "timestamp": "...",
    "host": "...",
    "platform": "...",
    "numpy_config": {...}
  },
  "cases": [
    {
      "shape": [2048, 2048],
      "dtype": "float32",
      "operations": [
        {
          "name": "mean_axis0",
          "raptors_mean_s": 0.00123,
          "numpy_mean_s": 0.00145,
          "speedup": 1.18
        }
      ]
    }
  ]
}
```

## Advanced Usage

### Profiling with perf

The container includes `perf` for performance profiling:

```bash
./scripts/docker_bench.sh shell
# Inside container:
perf record -g python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations mean_axis0
perf report
```

### Using py-spy

```bash
./scripts/docker_bench.sh run "py-spy record -o profile.svg -- python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32"
```

### Custom Environment Variables

```bash
./scripts/docker_bench.sh run bash -c "
  export RAPTORS_THREADS=16 && \
  export RAPTORS_SIMD=force && \
  python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32
"
```

## Next Steps

After running benchmarks:
1. Review JSON results in `benchmarks/docker_results/`
2. Compare against baselines in `benchmarks/baselines/`
3. Use profiling tools to identify bottlenecks
4. Iterate on optimizations and re-run benchmarks

