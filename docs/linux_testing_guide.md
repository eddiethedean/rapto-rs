# Linux Testing Guide

This guide explains how to run all tests (Rust and Python) on Linux using Docker.

## Prerequisites

1. **Docker Desktop** must be installed and running
   - Check with: `docker ps`
   - If not running, start Docker Desktop and wait 30-60 seconds for it to fully initialize

2. **Docker Compose** (usually included with Docker Desktop)
   - Check with: `docker compose version` or `docker-compose version`

## Quick Start

### 1. Build the Docker Image (First Time Only)

```bash
./scripts/docker_bench.sh build
```

This builds the Ubuntu 22.04 image with:
- Python 3.11
- Rust toolchain
- OpenBLAS
- Required build tools

**Note**: This may take several minutes the first time.

### 2. Run All Tests

```bash
./scripts/docker_bench.sh run bash -c "cd /workspace/src/rust && cargo test --no-default-features --features test-suite && cd /workspace/src && /workspace/.venv/bin/python -m maturin develop --release --manifest-path rust/Cargo.toml && cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so && export PYTHONPATH=/workspace/src/python:\$PYTHONPATH && /workspace/.venv/bin/python -m pytest tests/"
```

## Step-by-Step Guide

### Step 1: Run Rust Tests

Rust tests can be run directly without building the Python extension:

```bash
./scripts/docker_bench.sh run bash -c "cd /workspace/src/rust && cargo test --no-default-features --features test-suite"
```

**Expected Output:**
```
running 19 tests
test reduce::tiled::tests::... ok
...
test result: ok. 19 passed; 0 failed

running 5 tests
test broadcast_add_handles_row_vector_inputs ... ok
...
test result: ok. 5 passed; 0 failed
```

### Step 2: Build Python Extension

Build the Python extension module using maturin:

```bash
./scripts/docker_bench.sh run bash -c "cd /workspace/src/rust && VIRTUAL_ENV=/workspace/.venv /workspace/.venv/bin/python -m maturin develop --release --manifest-path /workspace/src/rust/Cargo.toml"
```

**Note**: You may see a warning about `PyInit_raptors` - this is expected and can be ignored.

### Step 3: Copy Extension Module

Maturin installs the extension with a different naming convention. Copy it to the expected location:

```bash
./scripts/docker_bench.sh run bash -c "cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so /workspace/src/python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so"
```

### Step 4: Run Python Tests

Run the pytest suite:

```bash
./scripts/docker_bench.sh run bash -c "cd /workspace/src && export PYTHONPATH=/workspace/src/python:\$PYTHONPATH && /workspace/.venv/bin/python -m pytest tests/"
```

**Expected Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.11.0rc1, pytest-9.0.1, pluggy-1.6.0
collected 77 items

tests/test_placeholder.py ...........................                    [ 35%]
tests/test_reductions.py ..............................................  [ 94%]
tests/test_simd_env.py ....                                              [100%]

============================== 77 passed in 0.47s ==============================
```

## Complete Test Script

For convenience, here's a complete script that runs all tests:

```bash
#!/bin/bash
set -e

echo "==> Running Rust tests..."
./scripts/docker_bench.sh run bash -c "cd /workspace/src/rust && cargo test --no-default-features --features test-suite"

echo "==> Building Python extension..."
./scripts/docker_bench.sh run bash -c "cd /workspace/src/rust && VIRTUAL_ENV=/workspace/.venv /workspace/.venv/bin/python -m maturin develop --release --manifest-path /workspace/src/rust/Cargo.toml"

echo "==> Copying extension module..."
./scripts/docker_bench.sh run bash -c "cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so /workspace/src/python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so"

echo "==> Running Python tests..."
./scripts/docker_bench.sh run bash -c "cd /workspace/src && export PYTHONPATH=/workspace/src/python:\$PYTHONPATH && /workspace/.venv/bin/python -m pytest tests/"

echo "==> All tests passed!"
```

Save this as `scripts/run_linux_tests.sh` and make it executable:

```bash
chmod +x scripts/run_linux_tests.sh
./scripts/run_linux_tests.sh
```

## Interactive Testing

To open an interactive shell in the Docker container for manual testing:

```bash
./scripts/docker_bench.sh shell
```

Inside the container:

```bash
# Navigate to source
cd /workspace/src

# Run Rust tests
cd rust
cargo test --no-default-features --features test-suite

# Build Python extension
VIRTUAL_ENV=/workspace/.venv /workspace/.venv/bin/python -m maturin develop --release

# Copy extension module
cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so

# Run Python tests
export PYTHONPATH=/workspace/src/python:$PYTHONPATH
/workspace/.venv/bin/python -m pytest tests/
```

## Troubleshooting

### Issue: Docker not found

**Error**: `Error: Neither 'docker compose' nor 'docker-compose' found`

**Solution**:
1. Ensure Docker Desktop is installed and running
2. Wait 30-60 seconds after starting Docker Desktop
3. Restart your terminal
4. Verify with: `docker ps`

### Issue: Linker errors during build

**Error**: `cannot find dynamic_lookup: No such file or directory`

**Solution**: This happens when macOS-specific linker flags (`-undefined dynamic_lookup`) are used on Linux. The test script should not set `RUSTFLAGS` with these flags. If you see this error, ensure you're not exporting macOS-specific `RUSTFLAGS` in the Docker container.

### Issue: Python extension not found

**Error**: `ImportError: cannot import name '_raptors'`

**Solution**: 
1. Verify the extension was built: `ls -la /workspace/.venv/lib/python3.11/site-packages/raptors/`
2. Copy it to the correct location:
   ```bash
   cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so \
      /workspace/src/python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so
   ```
3. Verify `PYTHONPATH` includes `/workspace/src/python`

### Issue: Virtual environment not found

**Error**: `Your virtualenv at /workspace/src/.venv is broken`

**Solution**: The script may be detecting a `.venv` directory in the source. Use the container's venv at `/workspace/.venv`:

```bash
VIRTUAL_ENV=/workspace/.venv /workspace/.venv/bin/python -m maturin develop --release
```

### Issue: Tests fail with "partially initialized module"

**Error**: `ImportError: cannot import name '_raptors' from partially initialized module 'raptors'`

**Solution**: This usually means:
1. The extension module isn't in the right location (see above)
2. There's a circular import issue - ensure `PYTHONPATH` is set correctly
3. The extension wasn't built for Linux (check file name contains `linux`)

## Test Coverage

### Rust Tests

- **Unit tests**: 19 tests covering:
  - Tiled reductions
  - SIMD codegen
  - Tile specifications
  - Basic operations (add, scale, reduce)

- **Integration tests**: 5 tests covering:
  - Python API functionality
  - Broadcasting
  - In-place operations
  - Threading info

### Python Tests

- **77 pytest tests** covering:
  - Placeholder tests (35 tests)
  - Reduction operations (38 tests)
  - SIMD environment detection (4 tests)

## CI/CD Integration

For CI/CD pipelines, you can use:

```yaml
# Example GitHub Actions workflow
- name: Run Linux tests
  run: |
    ./scripts/docker_bench.sh build
    ./scripts/docker_bench.sh run bash -c "
      cd /workspace/src/rust && \
      cargo test --no-default-features --features test-suite && \
      cd /workspace/src && \
      /workspace/.venv/bin/python -m maturin develop --release --manifest-path rust/Cargo.toml && \
      cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so \
         python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so && \
      export PYTHONPATH=/workspace/src/python:\$PYTHONPATH && \
      /workspace/.venv/bin/python -m pytest tests/
    "
```

## Performance Testing

To run benchmarks (not tests) on Linux:

```bash
./scripts/docker_bench.sh bench
```

This runs the full benchmark suite comparing Raptors against NumPy.

## Additional Resources

- [Docker Setup Guide](DOCKER_SETUP.md) - Initial Docker environment setup
- [Pure Columnar Optimization Guide](pure_columnar_optimization.md) - Performance optimization techniques
- [TODO.md](../TODO.md) - Current development status

## Summary

**Quick Command** (all tests):
```bash
./scripts/docker_bench.sh run bash -c "cd /workspace/src/rust && cargo test --no-default-features --features test-suite && cd /workspace/src && /workspace/.venv/bin/python -m maturin develop --release --manifest-path rust/Cargo.toml && cp /workspace/.venv/lib/python3.11/site-packages/raptors/raptors.cpython-311-aarch64-linux-gnu.so python/raptors/_raptors.cpython-311-aarch64-linux-gnu.so && export PYTHONPATH=/workspace/src/python:\$PYTHONPATH && /workspace/.venv/bin/python -m pytest tests/"
```

**Expected Results**:
- ✅ 19 Rust unit tests passed
- ✅ 5 Rust integration tests passed  
- ✅ 77 Python tests passed
- **Total: 101 tests, 0 failures**

