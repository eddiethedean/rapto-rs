# NumPy Profiling Findings - Assembly Analysis

## Summary

Assembly-level comparison between Raptors and NumPy for `mean_axis0` on 2048² float32 matrices.

## NEON Instruction Analysis

### Raptors NEON Instructions

From `benchmarks/profiles/raptors_neon.txt` (first 100 lines):

**Instruction Count**:
- See analysis output for detailed counts

**Pattern**: Standard load-add-store pattern with tiled approach

### NumPy NEON Instructions

From `benchmarks/profiles/numpy_neon.txt` (first 100 lines):

**Instruction Count**:
- See analysis output for detailed counts

**Pattern**: Analyze for differences in instruction scheduling

## Key Differences

1. **Instruction Mix**: Compare which NEON instructions are used
2. **Instruction Scheduling**: How instructions are interleaved
3. **Register Usage**: How registers are managed
4. **Memory Access Patterns**: Load/store ordering

## Findings

- Both use similar NEON instruction sets (vld1q, vaddq, vst1q)
- Differences likely in instruction scheduling and register management
- NumPy may use fused operations or better instruction interleaving

## Next Steps

1. Analyze complete assembly sequences (not just first 100 lines)
2. Compare hot loop patterns in detail
3. Identify specific instruction scheduling differences
4. Apply learned patterns to Raptors implementation

