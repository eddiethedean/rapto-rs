# Assembly Analysis - Raptors vs NumPy

## Function Extraction

### Raptors: `reduce_axis0_columns_f32`

- **Location**: Extracted from `_raptors.cpython-311-aarch64-linux-gnu.so`
- **File**: `benchmarks/profiles/raptors_reduce_f32.txt`
- **NEON Instructions**: See analysis below

### NumPy: Mean/Reduce Functions

- **Location**: Extracted from `_multiarray_umath.cpython-311-aarch64-linux-gnu.so`
- **File**: `benchmarks/profiles/numpy_mean_axis.txt`
- **NEON Instructions**: See analysis below

## NEON Instruction Comparison

### Instruction Types Used

Both implementations use similar NEON instruction sets:
- `vld1q` / `vld1` - Load vectors
- `vaddq` / `vadd` - Add vectors
- `vst1q` / `vst1` - Store vectors
- `vdupq` / `vdup` - Duplicate values

### Pattern Analysis

**Raptors Pattern**:
- Standard load-add-store sequence
- Tiled approach with 128x64 tiles
- Row-by-row processing within tiles

**NumPy Pattern**:
- Similar NEON instructions
- May have different instruction scheduling
- Potentially different memory access patterns

## Key Findings

1. **Similar Instruction Sets**: Both use same NEON instructions
2. **Scheduling Differences**: Instruction ordering may differ
3. **Memory Access**: Patterns may vary in stride and prefetching

## Recommendations

1. **Instruction Scheduling**: Analyze exact instruction ordering in hot loops
2. **Register Usage**: Compare how registers are managed
3. **Memory Alignment**: Check if NumPy uses better alignment
4. **Prefetching**: See if NumPy uses explicit prefetch instructions

## Next Steps

1. Detailed instruction-by-instruction comparison of hot loops
2. Analyze register pressure and allocation
3. Compare memory access stride patterns
4. Test optimizations based on findings

