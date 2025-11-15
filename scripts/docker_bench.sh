#!/usr/bin/env bash
set -euo pipefail

# Helper script for Docker benchmarking

usage() {
  cat <<'EOF'
Usage: docker_bench.sh [COMMAND]

Commands:
  build       Build the Docker image
  run         Run a command in the Docker container
  shell       Open a shell in the Docker container
  bench       Run benchmarks
  profile     Profile a specific operation
  clean       Clean up Docker resources

Examples:
  ./scripts/docker_bench.sh build
  ./scripts/docker_bench.sh shell
  ./scripts/docker_bench.sh bench
  ./scripts/docker_bench.sh run "python scripts/compare_numpy_raptors.py --shape 512x512 --dtype float32 --operations broadcast_add"
EOF
}

COMMAND="${1:-help}"

case "$COMMAND" in
  build)
    echo "Building Docker image..."
    # Try docker compose (v2) first, fall back to docker-compose (v1)
    if command -v docker &> /dev/null && docker compose version &> /dev/null 2>/dev/null; then
      docker compose -f docker-compose.bench.yml build
    elif command -v docker-compose &> /dev/null; then
      docker-compose -f docker-compose.bench.yml build
    else
      echo "Error: Neither 'docker compose' nor 'docker-compose' found. Please ensure Docker Desktop is running."
      exit 1
    fi
    ;;
  run)
    shift || true
    if [ $# -eq 0 ]; then
      echo "Error: No command provided"
      usage
      exit 1
    fi
    if command -v docker &> /dev/null && docker compose version &> /dev/null 2>/dev/null; then
      docker compose -f docker-compose.bench.yml run --rm bench "$@"
    elif command -v docker-compose &> /dev/null; then
      docker-compose -f docker-compose.bench.yml run --rm bench "$@"
    else
      echo "Error: Neither 'docker compose' nor 'docker-compose' found. Please ensure Docker Desktop is running."
      exit 1
    fi
    ;;
  shell)
    if command -v docker &> /dev/null && docker compose version &> /dev/null 2>/dev/null; then
      docker compose -f docker-compose.bench.yml run --rm bench /bin/bash
    elif command -v docker-compose &> /dev/null; then
      docker-compose -f docker-compose.bench.yml run --rm bench /bin/bash
    else
      echo "Error: Neither 'docker compose' nor 'docker-compose' found. Please ensure Docker Desktop is running."
      exit 1
    fi
    ;;
  bench)
    echo "Running benchmarks in Docker..."
    if command -v docker &> /dev/null && docker compose version &> /dev/null 2>/dev/null; then
      docker compose -f docker-compose.bench.yml run --rm bench \
        /workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
        --suite 2d \
        --output-dir benchmarks/docker_results/$(date +%Y%m%d-%H%M%S)
    elif command -v docker-compose &> /dev/null; then
      docker-compose -f docker-compose.bench.yml run --rm bench \
        /workspace/.venv/bin/python scripts/compare_numpy_raptors.py \
        --suite 2d \
        --output-dir benchmarks/docker_results/$(date +%Y%m%d-%H%M%S)
    else
      echo "Error: Neither 'docker compose' nor 'docker-compose' found. Please ensure Docker Desktop is running."
      exit 1
    fi
    ;;
  profile)
    shift || true
    if [ $# -eq 0 ]; then
      echo "Error: No operation specified"
      echo "Usage: docker_bench.sh profile <operation>"
      exit 1
    fi
    if command -v docker &> /dev/null && docker compose version &> /dev/null 2>/dev/null; then
      docker compose -f docker-compose.bench.yml run --rm bench \
        /workspace/scripts/profile_operation.sh "$@"
    elif command -v docker-compose &> /dev/null; then
      docker-compose -f docker-compose.bench.yml run --rm bench \
        /workspace/scripts/profile_operation.sh "$@"
    else
      echo "Error: Neither 'docker compose' nor 'docker-compose' found. Please ensure Docker Desktop is running."
      exit 1
    fi
    ;;
  clean)
    echo "Cleaning up Docker resources..."
    if command -v docker &> /dev/null && docker compose version &> /dev/null 2>/dev/null; then
      docker compose -f docker-compose.bench.yml down
      docker rmi raptors-bench 2>/dev/null || true
    elif command -v docker-compose &> /dev/null; then
      docker-compose -f docker-compose.bench.yml down
      docker rmi raptors-bench 2>/dev/null || true
    else
      echo "Error: Neither 'docker compose' nor 'docker-compose' found. Please ensure Docker Desktop is running."
      exit 1
    fi
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Error: Unknown command: $COMMAND"
    usage
    exit 1
    ;;
esac

