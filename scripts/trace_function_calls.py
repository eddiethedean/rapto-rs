#!/usr/bin/env python3
"""Trace function calls for NumPy and Raptors to understand dispatch paths.

This script calls NumPy and Raptors operations with minimal overhead
to allow tracing with dtruss/dtrace.
"""

import sys
import time
from pathlib import Path

# Add python directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import raptors


def trace_numpy_multiply():
    """Perform NumPy multiply operation for tracing."""
    arr = np.random.randn(2048, 2048).astype('float32')
    factor = 2.5
    
    # Warmup
    for _ in range(5):
        _ = arr * factor
    
    # Actual operation to trace
    result = arr * factor
    
    return result


def trace_raptors_scale():
    """Perform Raptors scale operation for tracing."""
    arr_np = np.random.randn(2048, 2048).astype('float32')
    arr_rp = raptors.from_numpy(arr_np)
    factor = 2.5
    
    # Warmup
    for _ in range(5):
        _ = arr_rp.scale(factor)
    
    # Actual operation to trace
    result = arr_rp.scale(factor)
    
    return result


def main():
    """Main tracing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trace function calls for profiling")
    parser.add_argument(
        "--operation",
        type=str,
        default="numpy",
        choices=["numpy", "raptors", "both"],
        help="Which operation to trace",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations",
    )
    
    args = parser.parse_args()
    
    print(f"Tracing {args.operation} operations ({args.iterations} iterations)")
    
    if args.operation in ("numpy", "both"):
        print("Tracing NumPy...")
        for _ in range(args.iterations):
            _ = trace_numpy_multiply()
        print("NumPy tracing complete")
    
    if args.operation in ("raptors", "both"):
        print("Tracing Raptors...")
        for _ in range(args.iterations):
            _ = trace_raptors_scale()
        print("Raptors tracing complete")
    
    # Add delay to make it easier to trace
    time.sleep(1)
    print("Tracing complete")


if __name__ == "__main__":
    main()

