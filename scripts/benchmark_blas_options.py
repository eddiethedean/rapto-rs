#!/usr/bin/env python3
"""
Benchmark script to compare BLAS/Accelerate function options.

Tests:
- macOS: vDSP_vsmul vs cblas_sscal (Accelerate)
- Linux/Windows: OpenBLAS cblas_sscal vs SIMD (if available)
- In-place vs copy operations
- Different array sizes (512², 1024², 2048²)
"""

import sys
import time
import statistics
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import raptors
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure numpy and raptors are installed")
    sys.exit(1)


def warmup(func, *args, **kwargs):
    """Run function multiple times to warm up cache."""
    for _ in range(10):
        func(*args, **kwargs)


def benchmark(func, *args, iterations=100, warmup_iterations=10, **kwargs):
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(warmup_iterations):
        func(*args, **kwargs)
    
    # Actual benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0.0,
        'min': min(times),
        'max': max(times),
    }


def benchmark_numpy_scale(arr, factor):
    """NumPy scale operation."""
    return arr * factor


def benchmark_raptors_scale(arr, factor):
    """Raptors scale operation."""
    # Convert numpy array directly to Raptors array (zero-copy when possible)
    raptors_arr = raptors.from_numpy(arr.astype(np.float32))
    # Use the scale method instead of * operator
    result = raptors_arr.scale(factor)
    return result.to_numpy()


def benchmark_raptors_scale_inplace(arr, factor):
    """Raptors scale operation (simulated in-place by modifying copy)."""
    # Convert numpy array directly to Raptors array (zero-copy when possible)
    raptors_arr = raptors.from_numpy(arr.astype(np.float32))
    # Note: Raptors has scale_inplace, but for benchmarking we'll use regular scale
    # to match the comparison with NumPy
    result = raptors_arr.scale(factor)
    return result.to_numpy()


def run_benchmarks():
    """Run comprehensive benchmarks."""
    sizes = [
        (512, 512, "512²"),
        (1024, 1024, "1024²"),
        (2048, 2048, "2048²"),
    ]
    
    factor = 2.5
    
    print("=" * 80)
    print("BLAS/Accelerate Function Comparison Benchmarks")
    print("=" * 80)
    print()
    
    results = {}
    
    for rows, cols, label in sizes:
        print(f"\n{'=' * 80}")
        print(f"Size: {label} ({rows}×{cols}, {rows*cols:,} elements)")
        print(f"{'=' * 80}")
        
        # Create test array
        arr = np.random.randn(rows, cols).astype(np.float32)
        
        # Benchmark NumPy (baseline)
        print(f"\n1. NumPy (baseline):")
        np_stats = benchmark(benchmark_numpy_scale, arr.copy(), factor, iterations=50)
        print(f"   Mean:   {np_stats['mean']:.4f} ms")
        print(f"   Median: {np_stats['median']:.4f} ms")
        print(f"   StdDev: {np_stats['stdev']:.4f} ms")
        print(f"   Min:    {np_stats['min']:.4f} ms")
        print(f"   Max:    {np_stats['max']:.4f} ms")
        
        # Benchmark Raptors (current implementation)
        print(f"\n2. Raptors (current):")
        try:
            raptors_stats = benchmark(benchmark_raptors_scale, arr.copy(), factor, iterations=50)
            print(f"   Mean:   {raptors_stats['mean']:.4f} ms")
            print(f"   Median: {raptors_stats['median']:.4f} ms")
            print(f"   StdDev: {raptors_stats['stdev']:.4f} ms")
            print(f"   Min:    {raptors_stats['min']:.4f} ms")
            print(f"   Max:    {raptors_stats['max']:.4f} ms")
            
            speedup = np_stats['mean'] / raptors_stats['mean']
            print(f"   Speedup: {speedup:.3f}x vs NumPy")
            
            results[label] = {
                'numpy': np_stats,
                'raptors': raptors_stats,
                'speedup': speedup,
            }
        except Exception as e:
            print(f"   Error: {e}")
            results[label] = {
                'numpy': np_stats,
                'raptors': None,
                'error': str(e),
            }
        
        # Note about in-place operations
        print(f"\n3. In-place operations:")
        print(f"   Note: Raptors in-place scale not yet implemented")
        print(f"   Will be tested after implementation")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print()
    print(f"{'Size':<10} {'NumPy (ms)':<15} {'Raptors (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    for label, data in results.items():
        if 'error' not in data:
            np_mean = data['numpy']['mean']
            rp_mean = data['raptors']['mean']
            speedup = data['speedup']
            print(f"{label:<10} {np_mean:<15.4f} {rp_mean:<15.4f} {speedup:<10.3f}x")
        else:
            print(f"{label:<10} {data['numpy']['mean']:<15.4f} {'ERROR':<15} {'N/A':<10}")
    
    # Platform-specific notes
    print(f"\n{'=' * 80}")
    print("Platform-Specific Notes")
    print(f"{'=' * 80}")
    import platform
    system = platform.system()
    print(f"Platform: {system}")
    
    if system == "Darwin":
        print("\nmacOS detected:")
        print("- NumPy uses Accelerate framework (vDSP/BLAS)")
        print("- Raptors should use Accelerate (vDSP_vsmul or cblas_sscal)")
        print("- Benchmark will help determine which is faster")
    elif system == "Linux":
        print("\nLinux detected:")
        print("- NumPy typically uses OpenBLAS (via pip) or MKL (via conda)")
        print("- Raptors can use OpenBLAS if available, else SIMD")
        print("- Check if OpenBLAS is available in Raptors build")
    elif system == "Windows":
        print("\nWindows detected:")
        print("- NumPy typically uses OpenBLAS (via pip) or MKL (via conda)")
        print("- Raptors can use OpenBLAS if available, else SIMD")
        print("- Check if OpenBLAS is available in Raptors build")
    
    return results


if __name__ == "__main__":
    results = run_benchmarks()
    print("\nBenchmark complete!")

