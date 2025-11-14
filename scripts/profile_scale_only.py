#!/usr/bin/env python3
"""
Profile just the scale operation (excluding conversion overhead).
This helps identify if the bottleneck is in the operation itself.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import raptors


def profile_scale_only():
    """Profile just the scale operation, excluding conversion."""
    sizes = [
        (512, 512, "512²"),
        (1024, 1024, "1024²"),
        (2048, 2048, "2048²"),
    ]
    
    factor = 2.5
    iterations = 100
    
    print("=" * 80)
    print("Scale Operation Performance (excluding conversion)")
    print("=" * 80)
    print()
    
    for rows, cols, label in sizes:
        # Create arrays once
        arr_np = np.random.randn(rows, cols).astype(np.float32)
        arr_rp = raptors.from_numpy(arr_np.copy())
        
        # Warmup
        for _ in range(20):
            _ = arr_np * factor
            _ = arr_rp.scale(factor)
        
        # Measure NumPy
        np_times = []
        for _ in range(iterations):
            arr_cp = arr_np.copy()
            start = time.perf_counter()
            _ = arr_cp * factor
            end = time.perf_counter()
            np_times.append((end - start) * 1000)
        
        # Measure Raptors (reuse same array, scale creates new)
        rp_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = arr_rp.scale(factor)
            end = time.perf_counter()
            rp_times.append((end - start) * 1000)
        
        np_mean = sum(np_times) / len(np_times)
        rp_mean = sum(rp_times) / len(rp_times)
        np_std = (sum((t - np_mean)**2 for t in np_times) / len(np_times))**0.5
        rp_std = (sum((t - rp_mean)**2 for t in rp_times) / len(rp_times))**0.5
        
        print(f"{label} ({rows}×{cols}):")
        print(f"  NumPy:    {np_mean:.4f} ± {np_std:.4f} ms")
        print(f"  Raptors:  {rp_mean:.4f} ± {rp_std:.4f} ms")
        print(f"  Speedup:  {np_mean / rp_mean:.2f}x")
        print()
    
    print("=" * 80)
    print()
    print("Note: This measures the operation itself, excluding:")
    print("  - Array allocation")
    print("  - NumPy to Raptors conversion")
    print()
    print("If Raptors is slower here, the bottleneck is in:")
    print("  1. The scale operation implementation")
    print("  2. Python binding overhead")
    print("  3. Memory allocation for result array")


if __name__ == "__main__":
    profile_scale_only()

