# NumPy Implementation Investigation

## Goal

Understand how NumPy implements `mean_axis0` to identify why it's faster than Raptors.

## Approach

1. Extract NumPy's mean function from assembly
2. Identify hot functions using perf
3. Compare with Raptors implementation
4. Analyze differences in approach

## Findings

### NumPy Library Structure

- **Library**: `numpy/_core/_multiarray_umath.cpython-311-aarch64-linux-gnu.so`
- **Size**: 971k lines of assembly
- **Location**: `/workspace/.venv/lib/python3.11/site-packages/numpy/_core/`

### Function Extraction

**Attempted**:
- Extract `PyArray_Mean` function
- Find symbols with `nm -D`
- Search for mean/reduce related functions

**Challenges**:
- NumPy uses complex Python-C binding layer
- Functions may be inlined or optimized away
- Large library makes searching difficult

### Performance Analysis

**NumPy Performance**:
- ~0.34ms for 2048×2048 float32 mean_axis0
- Single-threaded (BLAS threads set to 1)

**Raptors Performance**:
- ~0.46ms for same operation (0.73x NumPy)

**Gap**: ~30% slower

## Next Steps

1. **Use perf record** to identify actual hot functions
2. **Extract specific functions** called during mean_axis0
3. **Compare assembly** of hot loops
4. **Analyze algorithm differences**

## Potential Causes

1. **Different algorithm**: NumPy may use fundamentally different approach
2. **BLAS backend**: NumPy may use optimized BLAS routines (though BLAS tested slower)
3. **Compiler flags**: NumPy may use different optimization flags
4. **Memory access**: NumPy may have better memory access patterns
5. **Inlining**: NumPy may inline more aggressively

## Status

- ✅ Identified NumPy library location
- ⏳ Extracting hot functions with perf
- ⏳ Comparing assembly patterns
- ⏳ Analyzing algorithm differences

