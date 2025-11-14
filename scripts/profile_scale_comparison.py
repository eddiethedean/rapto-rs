#!/usr/bin/env python3
"""
Profile scale operations for NumPy vs Raptors comparison.

This script runs scale operations that can be profiled with Instruments
to identify performance differences.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import raptors


def profile_scale_operations():
    """Run scale operations for profiling."""
    print("Running scale operations for profiling...")
    print("=" * 70)
    
    # Create test arrays
    sizes = [
        (512, 512, "512²"),
        (1024, 1024, "1024²"),
        (2048, 2048, "2048²"),
    ]
    
    factor = 2.5
    
    for rows, cols, label in sizes:
        print(f"\n{label} ({rows}×{cols}):")
        print("-" * 70)
        
        # Create arrays
        arr_np = np.random.randn(rows, cols).astype(np.float32)
        arr_rp = raptors.from_numpy(arr_np.copy())
        
        # Warmup
        print("  Warming up...")
        for _ in range(10):
            _ = arr_np * factor
            _ = arr_rp.scale(factor)
        
        # NumPy operations
        print("  Running NumPy operations...")
        for _ in range(100):
            _ = arr_np * factor
        
        # Raptors operations
        print("  Running Raptors operations...")
        for _ in range(100):
            _ = arr_rp.scale(factor)
        
        print(f"  Done with {label}")
    
    print("\n" + "=" * 70)
    print("Profile run complete!")
    print("\nTo profile with Instruments:")
    print("  1. Open Instruments (Xcode)")
    print("  2. Choose 'Time Profiler' or 'System Trace'")
    print("  3. Run: python3 scripts/profile_scale_comparison.py")
    print("  4. Compare NumPy vs Raptors call stacks")


if __name__ == "__main__":
    profile_scale_operations()

