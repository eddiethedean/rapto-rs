#!/usr/bin/env python3
"""Simple script to run a single operation for profiling."""
import sys
import time
import numpy as np

# Simple loop to run operation many times for profiling
operation = sys.argv[1] if len(sys.argv) > 1 else "scale"
shape_str = sys.argv[2] if len(sys.argv) > 2 else "512x512"
dtype = sys.argv[3] if len(sys.argv) > 3 else "float32"
backend = sys.argv[4] if len(sys.argv) > 4 else "both"
iterations = int(sys.argv[5]) if len(sys.argv) > 5 else 100

shape = tuple(int(x) for x in shape_str.split("x"))
np_dtype = getattr(np, dtype)

# Create arrays
np_arr = np.random.randn(*shape).astype(np_dtype)
if operation == "scale":
    factor = 1.0009765625
elif operation == "broadcast_add":
    # For 2D, use row vector (first row) as RHS
    if len(shape) == 2:
        rhs_np = np_arr[0]
    else:
        rhs_np = np_arr

if backend == "raptors" or backend == "both":
    import raptors as raptors_mod
    raptors_arr = raptors_mod.from_numpy(np_arr.copy())
    if operation == "broadcast_add":
        rhs_r = raptors_mod.from_numpy(rhs_np.copy())

# Warmup
for _ in range(10):
    if backend == "numpy" or backend == "both":
        if operation == "scale":
            _ = np_arr * factor
        elif operation == "broadcast_add":
            _ = np_arr + rhs_np
        elif operation == "mean_axis0":
            _ = np.mean(np_arr, axis=0)
    
    if backend == "raptors" or backend == "both":
        if operation == "scale":
            _ = raptors_arr.scale(factor)
        elif operation == "broadcast_add":
            _ = raptors_mod.broadcast_add(raptors_arr, rhs_r)
        elif operation == "mean_axis0":
            _ = raptors_arr.mean_axis(0)

# Profile loop
print(f"Running {operation} @ {shape} {dtype} ({backend}) for {iterations} iterations...")
start = time.time()

for i in range(iterations):
    if backend == "numpy" or backend == "both":
        if operation == "scale":
            result = np_arr * factor
        elif operation == "broadcast_add":
            result = np_arr + rhs_np
        elif operation == "mean_axis0":
            result = np.mean(np_arr, axis=0)
        _ = result  # Consume result
    
    if backend == "raptors" or backend == "both":
        if operation == "scale":
            result = raptors_arr.scale(factor)
        elif operation == "broadcast_add":
            result = raptors_mod.broadcast_add(raptors_arr, rhs_r)
        elif operation == "mean_axis0":
            result = raptors_arr.mean_axis(0)
        _ = result  # Consume result

elapsed = time.time() - start
print(f"Completed in {elapsed:.3f}s ({elapsed/iterations*1000:.3f}ms per iteration)")

