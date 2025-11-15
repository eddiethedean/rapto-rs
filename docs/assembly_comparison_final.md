# Final Assembly Comparison - Raptors vs NumPy

## Raptors Implementation Analysis

### Function: `reduce_axis0_columns_f32`

**Symbol**: `_ZN7raptors4simd24reduce_axis0_columns_f3217h818fe0908888c1ffE`
**Address**: `948e0`
**File**: `benchmarks/profiles/raptors_function_full.txt`

### NEON Instruction Patterns

#### Standard Loop Pattern

```
ldr q0, [x9]              ; Load accumulator
ldr q1, [x10], #16        ; Load data with post-increment (+16 bytes)
fadd v0.4s, v0.4s, v1.4s  ; Add 4 single-precision floats
str q0, [x9], #16         ; Store with post-increment
b.ls <loop_start>         ; Loop back if less or same
```

**Characteristics**:
- Uses 128-bit NEON registers (q0, q1)
- Processes 4 floats per iteration
- Post-increment addressing for efficient stride access
- Standard load-add-store pattern

#### Unrolled Loop Pattern (2x)

```
ldp q0, q3, [x11, #-16]   ; Load pair of accumulators
ldp q1, q2, [x12, #-16]   ; Load pair of data vectors
fadd v0.4s, v1.4s, v0.4s  ; Add first pair
fadd v1.4s, v2.4s, v3.4s  ; Add second pair
stp q0, q1, [x11, #-16]  ; Store pair of results
b.ne <loop_start>         ; Loop back
```

**Characteristics**:
- 2x unrolling using load-pair/store-pair
- Processes 8 floats per iteration (2 vectors × 4 floats)
- Better instruction-level parallelism
- Reduces loop overhead

### Instruction Counts

From analysis of the function:
- **Load operations**: Multiple (ldr q, ldp q)
- **Add operations**: Multiple (fadd v)
- **Store operations**: Multiple (str q, stp q)
- **Total NEON instructions**: See detailed analysis

### Memory Access Patterns

1. **Post-increment addressing**: `[x10], #16` - efficient for sequential access
2. **Indexed addressing**: `[x0, x5]` - for tiled access patterns
3. **Load-pair/store-pair**: For unrolled loops

## NumPy Implementation

**Status**: NumPy's exact hot loop extraction needs refinement. The library is large (971k lines) and the mean_axis0 function may be inlined or optimized differently.

**Expected patterns** (based on NumPy's optimization level):
- Similar NEON instructions
- Potentially better instruction scheduling
- May use different unrolling factors
- Could have better register allocation

## Key Differences to Investigate

1. **Instruction Scheduling**: How instructions are ordered
2. **Register Usage**: Number of registers used simultaneously
3. **Unrolling Factor**: 2x, 4x, or different?
4. **Memory Access**: Stride patterns and alignment
5. **Loop Structure**: Different loop organization

## Recommendations

1. **Extract NumPy's exact hot loop** using symbol resolution
2. **Compare instruction-by-instruction** the hot loops
3. **Analyze register pressure** - see if NumPy uses more registers
4. **Check instruction dependencies** - see if NumPy hides latency better
5. **Test optimizations** based on findings

## Next Steps

1. Use `nm` or `objdump -t` to find NumPy's mean_axis0 symbol
2. Extract that specific function
3. Compare hot loop sequences side-by-side
4. Identify specific optimization opportunities
5. Apply and test optimizations

