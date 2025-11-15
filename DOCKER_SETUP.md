# Docker Environment Setup

## Status

Docker Desktop has been started. Please wait for it to fully initialize (check the menu bar icon - it should show "Docker Desktop is running").

## Quick Start

Once Docker Desktop is fully running, execute:

```bash
# 1. Build the Docker image (first time setup - may take a few minutes)
./scripts/docker_bench.sh build

# 2. Open an interactive shell in the container
./scripts/docker_bench.sh shell

# 3. Inside the container, build and install raptors
cd /workspace/src
maturin develop --release

# 4. Run benchmarks
./scripts/docker_bench.sh bench

# Or run specific benchmarks
./scripts/docker_bench.sh run "python scripts/compare_numpy_raptors.py --shape 2048x2048 --dtype float32 --operations mean_axis0"
```

## Verify Docker is Ready

Check if Docker is available:

```bash
docker ps
```

If this command works, Docker Desktop is ready.

## Test the Setup

Once Docker is ready, test with:

```bash
./scripts/docker_bench.sh shell
```

Then inside the container:

```bash
# Verify Python environment
/workspace/.venv/bin/python --version

# Verify Rust
rustc --version

# Build raptors
cd /workspace/src
maturin develop --release

# Test mean_axis0 performance improvements
python -c "
import numpy as np
import raptors
import time

# Test mean_axis0 @ 2048x2048 float32
arr = np.random.randn(2048, 2048).astype(np.float32)
r_arr = raptors.from_numpy(arr)

# Warmup
for _ in range(5):
    _ = r_arr.mean_axis(0)

# Benchmark
times = []
for _ in range(20):
    start = time.perf_counter()
    result = r_arr.mean_axis(0)
    times.append(time.perf_counter() - start)

print(f'Raptors mean_axis0: {np.mean(times)*1000:.2f}ms')

# Compare with NumPy
numpy_times = []
for _ in range(20):
    start = time.perf_counter()
    _ = np.mean(arr, axis=0)
    numpy_times.append(time.perf_counter() - start)

print(f'NumPy mean_axis0: {np.mean(numpy_times)*1000:.2f}ms')
print(f'Speedup: {np.mean(numpy_times)/np.mean(times):.2f}x')
"
```

## Common Commands

```bash
# Build image
./scripts/docker_bench.sh build

# Open shell
./scripts/docker_bench.sh shell

# Run full benchmark suite
./scripts/docker_bench.sh bench

# Profile a specific operation
./scripts/docker_bench.sh profile mean_axis0 2048x2048 float32

# Clean up
./scripts/docker_bench.sh clean
```

## Troubleshooting

### Docker not found
- Ensure Docker Desktop is running (check menu bar icon)
- Wait 30-60 seconds after starting Docker Desktop for it to fully initialize
- Restart your terminal after Docker Desktop starts

### Permission errors
- On macOS, Docker Desktop should work without additional permissions
- Ensure Docker Desktop has necessary permissions in System Preferences

### Build fails
- Check internet connection (needs to download Ubuntu packages and Rust)
- Ensure Docker Desktop has enough resources (Memory: 4GB+, CPU: 2+ cores)

## Next Steps

After setup is complete:
1. Run baseline benchmarks to measure current performance
2. Validate the optimized mean_axis0 implementations
3. Compare results against NumPy

