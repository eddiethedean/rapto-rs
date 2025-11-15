# NEON Pattern Analysis - Raptors Implementation

## Key Discovery

The compiler has optimized the Raptors `reduce_axis0_columns_f32` function to process **16 columns simultaneously** using parallel accumulator registers.

## Multi-Column Parallel Pattern

### Pattern Found in Assembly

```
94c28:  ldr q27, [x14]           ; Load from row, column 0
94c30:  fadd v23.4s, v23.4s, v27.4s  ; Accumulate into v23 (column 0)

94c38:  ldr q27, [x14, #16]      ; Load from row, column 1 (+16 bytes)
94c40:  fadd v22.4s, v22.4s, v27.4s  ; Accumulate into v22 (column 1)

94c48:  ldr q27, [x14, #32]      ; Load from row, column 2 (+32 bytes)
94c50:  fadd v21.4s, v21.4s, v27.4s  ; Accumulate into v21 (column 2)

... (continues for 16 columns, using v23 down to v0)
```

### Characteristics

1. **16 parallel accumulators**: Uses registers v23, v22, v21, ..., v0
2. **Same row, different columns**: All loads from same base address (x14) with offsets
3. **Column offsets**: +0, +16, +32, +48, ..., +240 (16 columns × 16 bytes = 256 bytes)
4. **Parallel processing**: All 16 columns processed for the same row simultaneously

### Output Accumulation Pattern

After processing a row, accumulates with previous results:

```
94d74:  ldr q24, [x9]            ; Load previous accumulator for column
94d80:  fadd v23.4s, v23.4s, v24.4s  ; Add to current accumulator
94d84:  str q23, [x9]            ; Store result
```

## Comparison with Source Code

**Source code** (rust/src/simd/mod.rs):
- Simple row-by-row processing
- One column at a time within a tile
- Vector loop processes 4 floats at a time

**Compiled assembly**:
- Compiler optimized to process 16 columns in parallel
- Uses 16 accumulator registers simultaneously
- Better instruction-level parallelism

## Implications

1. **Compiler optimization**: Rust compiler (LLVM) has already optimized the code significantly
2. **Register pressure**: Using 16 accumulator registers (v0-v23) - near maximum
3. **Memory access**: Sequential access within row (good for cache)
4. **ILP**: High instruction-level parallelism

## Why NumPy Might Be Faster

Potential reasons:
1. **Different unrolling factor**: NumPy might use different column parallelism
2. **Better instruction scheduling**: NumPy might interleave instructions better
3. **Memory alignment**: NumPy might have better alignment
4. **Cache prefetching**: NumPy might use explicit prefetch instructions
5. **Different tile strategy**: NumPy might use different tiling approach

## Next Steps

1. **Extract NumPy's exact pattern** to compare
2. **Test different column parallelism** (8, 32 columns instead of 16)
3. **Analyze instruction dependencies** to see if we can improve scheduling
4. **Check memory alignment** - ensure 16-byte alignment
5. **Test explicit prefetching** for next row

## Optimization Opportunities

1. **Manual register allocation**: Try using more/fewer accumulator registers
2. **Instruction reordering**: Manually schedule to hide latency
3. **Prefetching**: Add explicit prefetch for next row
4. **Alignment**: Ensure input data is 16-byte aligned
5. **Different unrolling**: Test 8-column or 32-column parallelism

