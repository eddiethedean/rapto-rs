# NEON Implementation Analysis - Raptors vs NumPy

## Function Extraction

### Raptors: `reduce_axis0_columns_f32`

- **Symbol**: `_ZN7raptors4simd24reduce_axis0_columns_f3217h818fe0908888c1ffE`
- **File**: `benchmarks/profiles/raptors_neon_impl.txt`
- **Lines**: ~250 lines of assembly

### NumPy: Mean/Reduce Functions

- **File**: `benchmarks/profiles/numpy_neon_impl.txt`
- **Extracted from**: Mean/reduce related functions

## NEON Instruction Analysis

### Raptors NEON Instructions

From the extracted implementation:
- Uses ARM64 NEON instructions: `ldr q`, `fadd v`, `str q` (128-bit registers)
- Pattern: Load → Add → Store in tiled approach

### NumPy NEON Instructions

- Similar NEON instruction set
- May have different instruction scheduling

## Key Findings

1. **Both use NEON**: Both implementations use ARM64 NEON SIMD instructions
2. **Similar instructions**: Both use load-add-store patterns
3. **Scheduling differences**: Instruction ordering may differ
4. **Register usage**: May differ in how registers are managed

## Hot Loop Analysis

The hot loop in Raptors implementation:
- Processes data in tiles (128x64)
- Uses NEON 128-bit registers (q0, q1, etc.)
- Load-add-store pattern for accumulation

## Recommendations

1. **Compare exact instruction sequences** in hot loops
2. **Analyze register allocation** - see if NumPy uses more registers
3. **Check instruction scheduling** - see if NumPy interleaves better
4. **Memory access patterns** - compare stride and alignment

## Next Steps

1. Extract and compare exact hot loop sequences
2. Count register usage in both implementations
3. Analyze instruction dependencies and scheduling
4. Apply optimizations based on findings

