# Optimization Opportunities Based on Assembly Analysis

## Current State

### Raptors Implementation

**Source Code**:
- Tile size: 128 rows × 64 columns
- Row-by-row processing within tiles
- Vector loop processes 4 floats at a time

**Compiler Optimization** (LLVM):
- Unrolls to process **16 columns simultaneously**
- Uses 16 accumulator registers (v23 down to v0)
- Column offsets: +0, +16, +32, ..., +240 bytes (256 bytes total per iteration)
- Excellent instruction-level parallelism

### NumPy Implementation

**Status**: NumPy's exact pattern needs further extraction, but likely uses similar or different unrolling strategies.

## Optimization Opportunities

### 1. Explicit Column Unrolling

**Current**: Compiler automatically unrolls to 16 columns
**Opportunity**: Explicitly unroll in source code to control the unrolling factor

**Benefits**:
- Control over register usage
- Can test different unrolling factors (8, 16, 32)
- Better cache alignment

**Implementation**:
```rust
// Explicitly unroll 16 columns
for col_block in (0..cols).step_by(16) {
    let col_end = (col_block + 16).min(cols);
    // Process 16 columns in parallel using explicit registers
    let vec0 = vld1q_f32(ptr.add(col_block + 0));
    let vec1 = vld1q_f32(ptr.add(col_block + 4));
    // ... up to 16
    vec_acc[0] = vaddq_f32(vec_acc[0], vec0);
    vec_acc[1] = vaddq_f32(vec_acc[1], vec1);
    // ...
}
```

### 2. Test Different Unrolling Factors

**8 columns**:
- Memory per iteration: 128 bytes (2 cache lines)
- Registers needed: 8
- Better cache locality
- Lower register pressure

**32 columns**:
- Memory per iteration: 512 bytes (8 cache lines)
- Registers needed: 32 (may exceed available registers)
- More parallelism
- Potential register spilling

### 3. Instruction Scheduling Optimization

**Current**: Load-Add-Store pattern
**Opportunity**: Interleave loads and adds to hide latency

**Example**:
```
Load row 0, cols 0-15  →  Add row 0, cols 0-15
Load row 1, cols 0-15  →  Add row 1, cols 0-15 (while row 0 adds execute)
Store row 0, cols 0-15 →  Load row 2, cols 0-15 (while row 1 adds execute)
```

### 4. Memory Alignment Optimization

**Current**: May not be explicitly aligned
**Opportunity**: Ensure 16-byte alignment for NEON loads

**Implementation**:
- Use `#[repr(align(16))]` for data structures
- Use aligned allocators
- Explicit alignment checks in hot loops

### 5. Prefetching Strategy

**Current**: May rely on hardware prefetcher
**Opportunity**: Explicit prefetch for next row/column

**Implementation**:
```rust
// Prefetch next row while processing current row
prefetch::<Prefetch::Read>(ptr.add(stride)); // Next row
prefetch::<Prefetch::Read>(out_ptr.add(64)); // Next output block
```

### 6. Tile Size Optimization

**Current**: 128×64 tiles
**Opportunity**: Test different tile sizes optimized for L1/L2 cache

**Options**:
- 64×128: Smaller row tile, larger column tile
- 96×96: Square tiles
- 256×32: Larger row tile, smaller column tile

## Recommended Testing Order

1. **Test 8-column explicit unrolling** - Better cache behavior
2. **Test explicit alignment** - Ensure optimal memory access
3. **Test instruction interleaving** - Hide latency
4. **Test prefetching** - Improve memory access patterns
5. **Test different tile sizes** - Optimize cache usage

## Expected Impact

- **8-column unrolling**: 5-10% improvement (better cache usage)
- **Explicit alignment**: 2-5% improvement (faster loads)
- **Instruction interleaving**: 5-15% improvement (better ILP)
- **Prefetching**: 3-8% improvement (reduced memory stalls)
- **Tile size optimization**: 2-10% improvement (better cache locality)

**Combined**: Could achieve 15-30% improvement, potentially exceeding NumPy performance

