#!/usr/bin/env python3
"""
Compare overhead between NumPy and Raptors scale operations.

This script breaks down the time spent in different parts of the operation
to identify bottlenecks.
"""

import sys
import time
import cProfile
import pstats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import raptors


def measure_allocation_time(size):
    """Measure time to allocate arrays."""
    times = []
    for _ in range(100):
        start = time.perf_counter()
        arr = np.zeros(size, dtype=np.float32)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times), min(times), max(times)


def measure_conversion_time(np_arr):
    """Measure time to convert NumPy array to Raptors."""
    times = []
    for _ in range(100):
        start = time.perf_counter()
        rp_arr = raptors.from_numpy(np_arr.copy())
        end = time.perf_counter()
        times.append((end - start) * 1000)
    del rp_arr
    return sum(times) / len(times), min(times), max(times)


def measure_scale_time(np_arr, factor):
    """Measure NumPy scale time."""
    times = []
    for _ in range(100):
        arr_cp = np_arr.copy()
        start = time.perf_counter()
        result = arr_cp * factor
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times), min(times), max(times)


def measure_raptors_scale_time(rp_arr, factor):
    """Measure Raptors scale time."""
    times = []
    for _ in range(100):
        start = time.perf_counter()
        result = rp_arr.scale(factor)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return sum(times) / len(times), min(times), max(times)


def main():
    """Main profiling function."""
    sizes = [
        (512, 512, "512²"),
        (1024, 1024, "1024²"),
        (2048, 2048, "2048²"),
    ]
    
    factor = 2.5
    
    print("=" * 80)
    print("Scale Operation Overhead Analysis")
    print("=" * 80)
    print()
    
    for rows, cols, label in sizes:
        size = (rows, cols)
        total_elements = rows * cols
        
        print(f"{label} ({rows}×{cols}):")
        print("-" * 80)
        
        # Warmup
        np_arr = np.random.randn(rows, cols).astype(np.float32)
        rp_arr = raptors.from_numpy(np_arr.copy())
        for _ in range(10):
            _ = np_arr * factor
            _ = rp_arr.scale(factor)
        
        # Measure allocation overhead
        alloc_mean, alloc_min, alloc_max = measure_allocation_time(total_elements)
        print(f"  Allocation time:        {alloc_mean:.4f} ms (min: {alloc_min:.4f}, max: {alloc_max:.4f})")
        
        # Measure conversion overhead
        conv_mean, conv_min, conv_max = measure_conversion_time(np_arr)
        print(f"  Conversion time:        {conv_mean:.4f} ms (min: {conv_min:.4f}, max: {conv_max:.4f})")
        
        # Measure NumPy scale
        np_mean, np_min, np_max = measure_scale_time(np_arr, factor)
        print(f"  NumPy scale time:       {np_mean:.4f} ms (min: {np_min:.4f}, max: {np_max:.4f})")
        
        # Measure Raptors scale (excluding conversion)
        rp_arr = raptors.from_numpy(np_arr.copy())
        rp_mean, rp_min, rp_max = measure_raptors_scale_time(rp_arr, factor)
        print(f"  Raptors scale time:     {rp_mean:.4f} ms (min: {rp_min:.4f}, max: {rp_max:.4f})")
        
        # Total overhead for Raptors (conversion + scale)
        total_raptors = conv_mean + rp_mean
        total_numpy = np_mean
        overhead = total_raptors - total_numpy
        overhead_pct = (overhead / total_numpy) * 100
        
        print()
        print(f"  NumPy total:            {total_numpy:.4f} ms")
        print(f"  Raptors total:          {total_raptors:.4f} ms")
        print(f"  Overhead:               {overhead:.4f} ms ({overhead_pct:+.1f}%)")
        print(f"  Speedup:                {total_numpy / total_raptors:.2f}x")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()

