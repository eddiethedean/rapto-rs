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

