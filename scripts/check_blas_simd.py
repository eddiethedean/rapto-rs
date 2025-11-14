#!/usr/bin/env python3
"""Diagnostic script to check BLAS and SIMD availability on Linux."""
import sys
import numpy as np

try:
    import raptors
except ImportError:
    print("ERROR: raptors module not found")
    sys.exit(1)

print("=" * 80)
print("BLAS and SIMD Diagnostic Report")
print("=" * 80)

# Check SIMD status
print("\n1. SIMD Status:")
print(f"   simd_enabled: {raptors.simd_enabled()}")

# Check threading info for backend usage
print("\n2. Threading Info:")
try:
    info = raptors.threading_info()
    print(f"   Thread pool: {info.thread_pool}")
    print(f"   Backend usage:")
    for usage in info.backend_usage:
        if "axis0" in usage.operation.lower() or "scale" in usage.operation.lower():
            print(f"     - {usage.operation} ({usage.dtype}): {usage.backend} (count: {usage.count})")
except Exception as e:
    print(f"   Error getting threading info: {e}")

# Test SIMD kernels by running operations and checking which backend is used
print("\n3. Testing Operations to Identify Backend:")

# Test reduce_axis0 operations
print("\n   Testing mean_axis0 @ 2048x2048 float64:")
try:
    arr = np.random.randn(2048, 2048).astype(np.float64)
    r_arr = raptors.from_numpy(arr)
    
    # Run operation
    result = r_arr.mean_axis(0)
    
    # Check backend usage after operation
    info = raptors.threading_info()
    axis0_backends = [u for u in info.backend_usage if "axis0" in u.operation.lower() and u.dtype == "float64"]
    if axis0_backends:
        print(f"     Backend used: {axis0_backends[-1].backend} (count: {axis0_backends[-1].count})")
    else:
        print("     No backend usage recorded for axis0 float64")
    
    print(f"     Result shape: {result.shape}")
    result_np = raptors.to_numpy(result)
    print(f"     First few values: {result_np[:5]}")
except Exception as e:
    print(f"     ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n   Testing mean_axis0 @ 2048x2048 float32:")
try:
    arr = np.random.randn(2048, 2048).astype(np.float32)
    r_arr = raptors.from_numpy(arr)
    
    # Run operation
    result = r_arr.mean_axis(0)
    
    # Check backend usage after operation
    info = raptors.threading_info()
    axis0_backends = [u for u in info.backend_usage if "axis0" in u.operation.lower() and u.dtype == "float32"]
    if axis0_backends:
        print(f"     Backend used: {axis0_backends[-1].backend} (count: {axis0_backends[-1].count})")
    else:
        print("     No backend usage recorded for axis0 float32")
    
    print(f"     Result shape: {result.shape}")
    result_np = raptors.to_numpy(result)
    print(f"     First few values: {result_np[:5]}")
except Exception as e:
    print(f"     ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test SIMD kernels directly by checking if they work
print("\n4. Testing SIMD Kernels:")
print("\n   Testing reduce_axis0_columns on small test data:")

try:
    # Test float64
    arr64 = np.random.randn(512, 512).astype(np.float64)
    r_arr64 = raptors.from_numpy(arr64)
    result64 = r_arr64.mean_axis(0)
    print(f"     float64 @ 512x512: SUCCESS (result shape: {result64.shape})")
except Exception as e:
    print(f"     float64 @ 512x512: FAILED - {e}")

try:
    # Test float32
    arr32 = np.random.randn(512, 512).astype(np.float32)
    r_arr32 = raptors.from_numpy(arr32)
    result32 = r_arr32.mean_axis(0)
    print(f"     float32 @ 512x512: SUCCESS (result shape: {result32.shape})")
except Exception as e:
    print(f"     float32 @ 512x512: FAILED - {e}")

# Check system info
print("\n5. System Information:")
import platform
print(f"   Platform: {platform.platform()}")
print(f"   Architecture: {platform.machine()}")
print(f"   Python: {sys.version}")

# Check if OpenBLAS is available (if we can detect it)
print("\n6. BLAS Detection:")
try:
    # Try to import scipy.linalg which should use BLAS
    try:
        import scipy.linalg
        print("   scipy.linalg available (indicates BLAS may be available)")
        # Try a simple BLAS operation
        a = np.random.randn(100, 100)
        b = np.random.randn(100, 100)
        c = scipy.linalg.blas.dgemm(1.0, a, b)
        print("   scipy.linalg.blas.dgemm works (BLAS is available)")
    except ImportError:
        print("   scipy not available, cannot verify BLAS via scipy")
    except Exception as e:
        print(f"   scipy.linalg available but dgemm failed: {e}")
except Exception:
    print("   Could not check BLAS via scipy")

print("\n" + "=" * 80)
print("Diagnostic complete")
print("=" * 80)

