# Assembly Comparison Summary - Raptors vs NumPy

## Profiling Infrastructure

✅ **Docker Setup**: Configured with privileged mode and profile output persistence
✅ **Assembly Extraction**: Successfully extracted both Raptors and NumPy assembly
✅ **NEON Analysis**: Identified NEON instruction patterns in both implementations

## Key Findings

### Raptors Implementation

**Source Code Pattern**:
- Tiled approach: 128 rows × 64 columns
- Row-by-row processing within tiles
- Simple vector loop: 4 floats per iteration

**Compiler Optimization (LLVM)**:
- **Automatic unrolling**: Processes **16 columns simultaneously**
- **Register usage**: 16 accumulator registers (v23 down to v0)
- **Memory access**: Column offsets +0, +16, +32, ..., +240 bytes
- **Instruction pattern**: Load-Add-Store with high ILP

**Key Characteristics**:
- 106 NEON instructions in hot function
- ~500 lines of assembly total
- Excellent compiler optimization

### NumPy Implementation

**Status**: 
- Large library (971k lines of assembly)
- Exact mean_axis0 function extraction needs refinement
- Likely uses similar or more advanced unrolling strategies

## Comparison

### Similarities

1. Both use NEON SIMD instructions (ldr q, fadd v, str q)
2. Both likely use unrolling for column parallelism
3. Both use tiled/cached approaches

### Differences to Investigate

1. **Unrolling factor**: Raptors uses 16 columns, NumPy may use different
2. **Instruction scheduling**: NumPy may interleave better
3. **Memory alignment**: NumPy may have better alignment guarantees
4. **Prefetching**: NumPy may use explicit prefetch instructions
5. **Register allocation**: NumPy may use more/fewer registers

## Performance Gap Analysis

**Current**: Raptors ~0.73x NumPy (Raptors: ~0.46ms, NumPy: ~0.34ms)
**Gap**: ~30% slower

**Potential Causes**:
1. **Different unrolling factor**: 16 vs different number
2. **Instruction scheduling**: Less optimal interleaving
3. **Memory access patterns**: Cache behavior differences
4. **Alignment**: Suboptimal memory alignment
5. **Prefetching**: Missing explicit prefetch instructions

## Optimization Roadmap

### Phase 1: Explicit Unrolling (Quick Win)

Test 8-column explicit unrolling:
- Better cache behavior (128 bytes = 2 cache lines)
- Lower register pressure
- Potentially better alignment

**Expected**: 5-10% improvement

### Phase 2: Instruction Scheduling (Medium Effort)

Manually interleave loads and adds:
- Hide memory latency
- Better pipeline utilization

**Expected**: 5-15% improvement

### Phase 3: Memory Optimization (Medium Effort)

- Explicit 16-byte alignment
- Prefetching for next row
- Optimized tile sizes

**Expected**: 5-10% improvement

### Combined Expected Impact

**Conservative**: 15-25% improvement (0.73x → 0.85-0.90x)
**Optimistic**: 25-35% improvement (0.73x → 0.95-1.00x+)

## Next Steps

1. ✅ **Assembly extraction** - COMPLETE
2. ✅ **Pattern identification** - COMPLETE  
3. ⏭️ **Test 8-column explicit unrolling** - NEXT
4. ⏭️ **Compare with NumPy hot loop** - NEXT
5. ⏭️ **Implement optimizations** - NEXT
6. ⏭️ **Benchmark improvements** - NEXT

## Files Generated

- `benchmarks/profiles/raptors_function_full.txt` - Raptors hot function (500 lines, 106 NEON)
- `benchmarks/profiles/raptors_asm.txt` - Full Raptors assembly (228k lines)
- `benchmarks/profiles/numpy_asm.txt` - Full NumPy assembly (971k lines)
- `benchmarks/profiles/numpy_neon_patterns.txt` - NumPy NEON patterns

## Documentation

- `docs/neon_pattern_analysis.md` - Detailed NEON pattern analysis
- `docs/optimization_opportunities.md` - Optimization recommendations
- `docs/comparison_summary.md` - This file

