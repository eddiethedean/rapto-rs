#!/usr/bin/env python3
"""Profile script for use with Xcode Instruments or py-spy.

This script performs the same operations as compare_numpy_raptors.py but is
designed to be profiled with Instruments or other profiling tools. It includes
proper warmup, measurement, and can profile both NumPy and Raptors operations.
"""

import sys
import time
from pathlib import Path

# Add python directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import raptors


def profile_numpy_scale(arr: np.ndarray, factor: float, iterations: int):
    """Profile NumPy scale operation."""
    # Warmup
    for _ in range(10):
        _ = arr * factor
    
    # Measurement
    start = time.perf_counter()
    for _ in range(iterations):
        result = arr * factor
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations, result


def profile_raptors_scale(arr, factor: float, iterations: int):
    """Profile Raptors scale operation."""
    arr_rp = raptors.from_numpy(arr)
    
    # Warmup
    for _ in range(10):
        _ = arr_rp.scale(factor)
    
    # Measurement
    start = time.perf_counter()
    for _ in range(iterations):
        result = arr_rp.scale(factor)
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations, result


def main():
    """Main profiling function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile scale operations for Instruments")
    parser.add_argument(
        "--shape",
        type=str,
        default="2048x2048",
        help="Array shape (e.g., 2048x2048)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations to profile",
    )
    parser.add_argument(
        "--operation",
        type=str,
        default="both",
        choices=["numpy", "raptors", "both"],
        help="Which operation to profile",
    )
    
    args = parser.parse_args()
    
    # Parse shape
    shape_parts = args.shape.lower().replace("*", "x").split("x")
    shape = tuple(int(p.strip()) for p in shape_parts if p.strip())
    
    # Create array
    dtype_map = {"float32": np.float32, "float64": np.float64}
    arr = np.random.randn(*shape).astype(dtype_map[args.dtype])
    factor = 2.5
    
    print(f"Profiling {args.shape} {args.dtype} scale operation")
    print(f"Array size: {arr.size:,} elements ({arr.nbytes / 1024 / 1024:.2f} MB)")
    print(f"Iterations: {args.iterations}")
    print()
    
    results = {}
    
    if args.operation in ("numpy", "both"):
        print("Profiling NumPy...")
        numpy_time, numpy_result = profile_numpy_scale(arr, factor, args.iterations)
        results["numpy"] = numpy_time * 1000  # Convert to ms
        print(f"NumPy: {results['numpy']:.3f}ms per iteration")
    
    if args.operation in ("raptors", "both"):
        print("Profiling Raptors...")
        raptors_time, raptors_result = profile_raptors_scale(arr, factor, args.iterations)
        results["raptors"] = raptors_time * 1000  # Convert to ms
        print(f"Raptors: {results['raptors']:.3f}ms per iteration")
    
    if len(results) == 2:
        speedup = results["numpy"] / results["raptors"]
        print()
        print(f"Speedup: {speedup:.3f}x (NumPy time / Raptors time)")
    
    return results


if __name__ == "__main__":
    main()

