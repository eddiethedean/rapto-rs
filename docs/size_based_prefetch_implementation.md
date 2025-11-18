# Size-Based Prefetch Control Implementation

**Date**: 2025-11-18  
**Status**: ✅ Implemented

## Overview

Based on comprehensive prefetch testing, we discovered that optimal prefetch behavior varies significantly by matrix size in Docker. This document describes the size-based prefetch control implementation.

## Test Results Summary

| Size | Best Config | Performance Gain |
|------|------------|------------------|
| 512x512 | L1 prefetch | +10.6% vs no-prefetch |
| 1024x1024 | No prefetch | +8.2% vs L1 prefetch |
| 2048x2048 | L3 prefetch | +2.6% vs L1 prefetch |

## Implementation

### New Functions

1. **`prefetch_level_for_size(rows: usize, cols: usize) -> Option<u8>`**
   - Returns optimal prefetch level based on size
   - Returns `None` to disable prefetch (for 1024x1024)
   - Returns `Some(1)`, `Some(2)`, or `Some(3)` for L1/L2/L3
   - Respects `RAPTORS_PREFETCH_LEVEL` environment variable override

2. **`emit_prefetch_load_sized(addr: *const f64, rows: usize, cols: usize)`**
   - Size-aware version of `emit_prefetch_load`
   - Uses `prefetch_level_for_size()` to determine optimal level
   - Emits no prefetch instruction if `None` is returned

### Code Changes

**File**: `rust/src/simd/mod.rs`

- Added `prefetch_level_for_size()` function (lines ~2010-2031)
- Added `emit_prefetch_load_sized()` function (lines ~2075-2104)
- Updated 512x512 specialized path to use `emit_prefetch_load_sized()`
- Updated 1024x1024 specialized path to use `emit_prefetch_load_sized()` (disables prefetch)
- Updated 2048x2048 specialized path to use `emit_prefetch_load_sized()` (uses L3)

### Size-Based Strategy

```rust
if rows == 1024 && cols == 1024 {
    // Disable prefetch for 1024x1024 (8.2% faster)
    return None;
} else if rows == 2048 && cols == 2048 {
    // Use L3 prefetch for 2048x2048 (2.6% faster than L1)
    return Some(3);
}
// For other sizes (including 512x512), use default L1
Some(1)
```

## Benefits

1. **Automatic Optimization**: No manual configuration needed
2. **Environment Override**: `RAPTORS_PREFETCH_LEVEL` still works for testing
3. **Size-Specific**: Each size gets optimal prefetch strategy
4. **Performance Gains**: Expected 8.2% improvement for 1024x1024

## Testing

To verify the implementation:

1. Run benchmarks in Docker
2. Compare 1024x1024 performance (should be ~8.2% faster)
3. Compare 2048x2048 performance (should be ~2.6% faster)
4. Verify 512x512 maintains performance (L1 prefetch)

## Future Enhancements

- Extend to float32 variants
- Add support for other sizes (256², 4096², etc.)
- Consider platform-specific optimizations (macOS vs Linux)

---

**Last Updated**: 2025-11-18 15:25

