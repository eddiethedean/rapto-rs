# Assembly Comparison Analysis

## Overview

This document compares the assembly output from Raptors and NumPy for the `float32 @ 2048²` scale operation to identify optimization opportunities.

## Raptors Assembly

### Extraction Method

- Used `otool -tvV` on compiled `.so` file
- Searched for scale-related symbols
- Extracted NEON instruction patterns

### Key Findings

#### NEON Instruction Usage

The Raptors implementation uses:
- `ld1` (load vectors) - loading 4 float32s per instruction
- `fmul` (multiply) - vector multiply
- `st1` (store vectors) - storing results
- `dup` (duplicate) - broadcasting the factor
- `prfm` (prefetch) - memory prefetching

#### Instruction Pattern (16× unrolling for 2048²)

For the 16× unrolled loop (64 elements per iteration):
- Loads: 16 × `ld1` instructions
- Multiplies: 16 × `fmul` instructions  
- Stores: 16 × `st1` instructions
- Prefetch: 2 × `prfm` instructions (read + write)
- Loop overhead: branch, add, compare

**Total per iteration**: ~50-60 instructions for 64 elements
**Instructions per element**: ~0.8-0.9

### Register Usage

- Uses NEON registers v0-v15 (and potentially more)
- Factor is broadcast to a vector register and reused
- Input and output pointers kept in general-purpose registers

### Memory Access Pattern

- Sequential memory access (contiguous arrays)
- Prefetch distance: 28 vectors ≈ 3.5KB ahead
- L1 cache prefetch (`pldl1keep`)

## NumPy Assembly (Accelerate Framework)

### Limitations

- NumPy uses Apple's Accelerate framework
- Accelerate is a closed-source library with hand-tuned assembly
- Direct disassembly may not be available or meaningful
- Accelerate likely uses proprietary optimizations

### Inferred Characteristics

Based on performance analysis:

1. **Instruction Efficiency**
   - NumPy achieves ~116 GB/s memory bandwidth
   - Raptors achieves ~79 GB/s
   - Suggests better instruction-level parallelism

2. **Best-Case Performance**
   - NumPy: 0.288ms (very fast)
   - Raptors: 0.422ms
   - Gap: 46% in best-case
   - Suggests specialized fast-path code

3. **Performance Variance**
   - NumPy: High variance (CV ~27-39%)
   - Raptors: Lower variance (CV ~12-14%)
   - Suggests NumPy has conditional optimizations

## Comparison

### Instruction Count Analysis

**Raptors (theoretical minimum for 64 elements)**:
- Loads: 16 instructions
- Multiplies: 16 instructions
- Stores: 16 instructions
- Loop overhead: ~5-10 instructions
- **Total**: ~53-58 instructions

**Optimization opportunities**:
- Reduce loop overhead through better scheduling
- Overlap memory operations with computation
- Use more registers to minimize spills

### Memory Access Patterns

**Similarities**:
- Both use sequential access
- Both likely use prefetching
- Both work with contiguous memory

**Differences (inferred)**:
- Accelerate may use non-temporal stores
- Accelerate may have better cache management
- Accelerate may use different prefetch strategies

### Register Usage

**Raptors**:
- Uses 16+ NEON registers effectively
- Good register pressure management

**Accelerate (inferred)**:
- May use all 32 NEON registers more aggressively
- Possibly better register allocation
- May use specialized register pairing

## Optimization Opportunities

### 1. Instruction Scheduling

**Current**: Load → Multiply → Store pattern
**Opportunity**: Deeper pipelining with more overlap

### 2. Register Usage

**Current**: 16 registers used in unrolled loop
**Opportunity**: Use all 32 registers, minimize dependencies

### 3. Non-Temporal Stores

**Current**: Regular stores (pollute cache)
**Opportunity**: Non-temporal stores for large arrays (2048² = 16MB)

### 4. Prefetch Strategy

**Current**: L1 prefetch, 28 vectors ahead
**Opportunity**: Test L2 prefetch, adjust distance

### 5. Instruction Count Reduction

**Current**: ~0.8-0.9 instructions per element
**Opportunity**: Reduce loop overhead, eliminate redundant operations

## Next Steps

1. Implement non-temporal stores (see Phase 3)
2. Create hand-tuned assembly with optimal scheduling (see Phase 5)
3. Experiment with register allocation strategies
4. Test different prefetch strategies

## References

- ARM NEON Instruction Set Reference
- Apple Accelerate Framework Documentation
- ARM Architecture Reference Manual

