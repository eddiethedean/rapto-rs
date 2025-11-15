# Hot Loop Analysis - Raptors vs NumPy

## Raptors Hot Function

**Function**: `_ZN7raptors4simd24reduce_axis0_columns_f3217h818fe0908888c1ffE`
**Address**: `948e0`
**File**: `benchmarks/profiles/raptors_hot_function.txt`

### NEON Instructions Found

From the assembly analysis, Raptors uses:
- `ldr q0, [x9]` - Load 128-bit register q0
- `ldr q1, [x10], #16` - Load q1 with post-increment
- `fadd v0.4s, v0.4s, v1.4s` - Add 4 single-precision floats
- `str q0, [x9], #16` - Store q0 with post-increment
- `ldp q0, q3, [x11, #-16]` - Load pair of 128-bit registers
- `stp q0, q1, [x11, #-16]` - Store pair of 128-bit registers

### Hot Loop Pattern

```
66d24:  ldr q0, [x9]           ; Load accumulator
66d28:  ldr q1, [x10], #16    ; Load data with post-increment
66d34:  fadd v0.4s, v0.4s, v1.4s  ; Add
66d40:  str q0, [x9], #16     ; Store with post-increment
66d44:  b.ls 66d24            ; Loop back
```

This is a standard load-add-store loop with post-increment addressing.

### Optimized Loop (Unrolled)

```
66d78:  ldp q0, q3, [x11, #-16]  ; Load pair
66d80:  ldp q1, q2, [x12, #-16]  ; Load pair
66d88:  fadd v0.4s, v1.4s, v0.4s ; Add
66d8c:  fadd v1.4s, v2.4s, v3.4s ; Add
66d90:  stp q0, q1, [x11, #-16]  ; Store pair
66d98:  b.ne 66d78               ; Loop back
```

This shows 2x unrolling with load-pair/store-pair instructions.

## NumPy Hot Loop

**File**: `benchmarks/profiles/numpy_hot_loop.txt`

### Analysis

NumPy's implementation may use:
- Similar NEON instructions
- Potentially different instruction scheduling
- Different memory access patterns

## Key Findings

1. **Raptors uses standard NEON patterns**: Load-add-store with post-increment
2. **Unrolling present**: 2x unrolling with load-pair/store-pair
3. **Instruction scheduling**: Standard pattern, may benefit from optimization

## Recommendations

1. **Compare with NumPy**: See if NumPy uses different instruction ordering
2. **Register pressure**: Check if we can use more registers
3. **Instruction interleaving**: See if we can hide latency better
4. **Memory access**: Compare stride patterns

## Next Steps

1. Extract NumPy's exact hot loop sequence
2. Compare instruction-by-instruction
3. Identify scheduling differences
4. Apply optimizations based on findings

