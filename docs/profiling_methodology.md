# Profiling Methodology

This document describes the profiling approach used to identify and fix performance bottlenecks in Raptors.

## Tools

### 1. perf (Linux Performance Counter)

**Purpose**: CPU profiling, cache analysis, instruction-level profiling

**Usage**:
```bash
perf record -F 99 -g -o output.perf.data -- python script.py
perf report -i output.perf.data
```

**What it shows**:
- Function call stacks
- CPU cycle distribution
- Cache miss rates
- Instruction counts
- Branch prediction accuracy

### 2. py-spy

**Purpose**: Python-level profiling with flamegraph support

**Usage**:
```bash
py-spy record --rate 100 --output output.svg --format flamegraph -- python script.py
```

**What it shows**:
- Python call stack
- Time spent in Python functions
- Native code calls (Rust functions)

### 3. Flamegraphs

**Purpose**: Visual representation of call stacks

**Generation**:
```bash
perf script -i perf.data | flamegraph.pl > output.svg
```

**What it shows**:
- Hot paths (wider boxes = more time)
- Call stack depth
- Function relationships

## Profiling Workflow

### Step 1: Baseline Benchmark

Run benchmark to establish timing:

```bash
python scripts/compare_numpy_raptors.py \
    --shape 512x512 \
    --dtype float32 \
    --operations broadcast_add \
    --warmup 3 \
    --repeats 30
```

### Step 2: Profile NumPy

```bash
./scripts/profile_operation.sh broadcast_add 512x512 float32 numpy
```

Examine the flamegraph to understand:
- What functions NumPy calls
- Where time is spent
- Any optimizations NumPy uses

### Step 3: Profile Raptors

```bash
./scripts/profile_operation.sh broadcast_add 512x512 float32 raptors
```

Compare with NumPy to identify:
- Different code paths
- Missing optimizations
- Inefficient algorithms

### Step 4: Analyze Differences

Use `compare_profiles.sh` or manually compare flamegraphs:

- **Instruction count**: Are we doing more work?
- **Cache misses**: Are we accessing memory inefficiently?
- **Function calls**: Are there unnecessary function call overhead?
- **SIMD usage**: Are SIMD instructions being used?

### Step 5: Implement Fix

Based on analysis:
- Optimize hot paths
- Improve memory access patterns
- Reduce function call overhead
- Enable SIMD where appropriate
- Parallelize if beneficial

### Step 6: Re-profile and Validate

After implementing fix:
1. Re-profile to confirm optimization
2. Run benchmarks to measure improvement
3. Ensure no regressions in other operations

## Profile Interpretation

### Flamegraph Basics

- **Width**: Time spent (wider = more time)
- **Height**: Call stack depth
- **Color**: Usually random for differentiation
- **Top of stack**: Functions currently executing
- **Bottom of stack**: Entry points (main, Python calls)

### What to Look For

1. **Wide functions**: These are hot paths - optimize these first
2. **Deep stacks**: May indicate function call overhead
3. **Missing SIMD**: Look for scalar loops that could use SIMD
4. **Cache misses**: High L1/L2 miss rates indicate memory issues
5. **Branch mispredictions**: Conditional logic that hurts performance

### Comparison Strategy

When comparing NumPy vs Raptors:

1. **Same operation, different paths**: Are we taking a less efficient path?
2. **Instruction mix**: Are we using the right CPU instructions?
3. **Memory access**: Are we accessing memory more efficiently?
4. **Parallelism**: Is NumPy using parallelism we're missing?
5. **BLAS usage**: Is NumPy using optimized BLAS routines?

## Common Bottlenecks

### 1. Scalar Loops Instead of SIMD

**Symptom**: Wide functions with simple loops in assembly

**Fix**: Enable SIMD dispatch, check SIMD detection

### 2. Memory Bandwidth Limits

**Symptom**: High cache miss rates, low CPU utilization

**Fix**: Improve cache locality, use non-temporal stores for large arrays

### 3. Function Call Overhead

**Symptom**: Many small function calls in flamegraph

**Fix**: Inline functions, reduce abstraction layers

### 4. Unnecessary Copies

**Symptom**: Copy operations showing up in profile

**Fix**: Avoid copies, use in-place operations where possible

### 5. Suboptimal Dispatch

**Symptom**: Slower path chosen when faster path available

**Fix**: Review dispatch logic, adjust thresholds

## Example Analysis

### Broadcast Row @ 512² float32 (0.63x laggard)

**NumPy profile**:
- Uses optimized BLAS routine (cblas_saxpy or similar)
- Single-threaded but highly optimized
- Minimal function call overhead

**Raptors profile**:
- Uses SIMD kernel but may have overhead
- Check if dispatch chooses correct path
- May need BLAS fallback for this case

**Fix strategy**:
1. Profile to confirm current path
2. Compare with NumPy's approach
3. Optimize SIMD kernel or use BLAS
4. Validate improvement

## Best Practices

1. **Always profile both**: NumPy and Raptors for comparison
2. **Multiple runs**: Profile data can vary, run multiple times
3. **Sufficient iterations**: Use enough iterations for statistical validity
4. **Check assembly**: For critical paths, examine generated assembly
5. **Validate assumptions**: Don't assume - profile to confirm
6. **Measure, don't guess**: Always benchmark before and after

