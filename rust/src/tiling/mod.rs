use std::cmp;

/// Describes cache- and SIMD-aware tiling parameters for 2-D kernels.
///
/// The fields intentionally avoid exposing ISA-specific details so that callers
/// can reason about high-level loop structures.  The `lane_width` field
/// represents the number of scalar elements processed per SIMD lane, while
/// `accumulators` controls how many independent vector accumulators should be
/// maintained in the hot loop to hide latencies on wide machines.
#[derive(Debug, Clone, Copy)]
pub struct TileSpec {
    pub row_block: usize,
    pub col_block: usize,
    pub lane_width: usize,
    pub accumulators: usize,
}

impl TileSpec {
    /// Heuristic tiling tuned per architecture with cache-aware sizing.
    /// Row and column blocks are chosen to keep working sets inside L1/L2
    /// while amortising SIMD setup. Cache sizes:
    /// - L1: ~32KB (typical) → ~4096 f64 or ~8192 f32 elements
    /// - L2: ~256KB-1MB → larger tiles for multi-threaded work
    /// - L3: ~8MB+ → maximize tile reuse
    /// 
    /// This is parameterized by SIMD width (not hard-coded) to match NumPy's approach.
    pub fn for_shape(rows: usize, cols: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                // AVX-512: 8 f64 or 16 f32 per lane
                // L1 cache-aware: ~512 f64 or ~1024 f32 per tile
                let row_hint = if rows * cols >= 1 << 22 {
                    128 // L2 cache-aware for large arrays
                } else {
                    96 // L1 cache-aware
                };
                let col_hint = if rows * cols >= 1 << 22 {
                    512 // L2 cache-aware
                } else {
                    256 // L1 cache-aware (~32KB for f64)
                };
                return Self::new(rows, cols, 8, 8, row_hint, col_hint);
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                // AVX2: 4 f64 or 8 f32 per lane
                let row_hint = if rows * cols >= 1 << 22 { 96 } else { 64 };
                let col_hint = if rows * cols >= 1 << 22 {
                    384 // L2 cache-aware
                } else {
                    192 // L1 cache-aware
                };
                return Self::new(rows, cols, 4, 6, row_hint, col_hint);
            }
            return Self::new(rows, cols, 2, 4, 64, 160);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON: 128-bit vectors = 2 f64 or 4 f32 per lane
            // Parameterize tile sizes based on SIMD width and cache hierarchy
            // L1 cache: ~64KB on ARM64 → ~8192 f64 or ~16384 f32 elements
            // L2 cache: ~256KB-1MB → larger tiles for bigger arrays
            
            let elements = rows.saturating_mul(cols);
            
            // Calculate optimal tile sizes based on cache and SIMD width
            // For f64: 2 elements per vector, aim for ~32KB working set per tile
            // For f32: 4 elements per vector, aim for ~32KB working set per tile
            let (row_hint, col_hint) = if elements >= 1 << 22 {
                // Large arrays: use L2 cache-aware tiles
                // Row tile: ~128 rows, Column tile: ~256-384 cols (depending on dtype)
                (128, 384)
            } else if elements >= 1 << 18 {
                // Medium arrays: balance L1/L2
                (96, 256)
            } else {
                // Small arrays: L1 cache-aware
                (64, 128)
            };
            
            // Use 5 accumulators for NEON (good balance of ILP and register pressure)
            return Self::new(rows, cols, 2, 5, row_hint, col_hint);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Conservative fallback for scalar-only builds.
            return Self::new(rows, cols, 1, 2, 64, 128);
        }
    }

    /// Helper for contiguous buffers where only the element count is known.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn for_len(len: usize) -> Self {
        let cols = (len as f64).sqrt().round() as usize;
        Self::for_shape(cmp::max(1, len / cmp::max(1, cols)), cmp::max(1, cols))
    }

    fn new(
        rows: usize,
        cols: usize,
        lane_width: usize,
        accumulators: usize,
        row_hint: usize,
        col_hint: usize,
    ) -> Self {
        let row_block = cmp::max(1, cmp::min(rows, row_hint));
        let col_block = cmp::max(1, cmp::min(cols, col_hint));
        Self {
            row_block,
            col_block,
            lane_width,
            accumulators,
        }
    }

    /// Returns the number of scalar elements covered by a single tile.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn tile_elems(&self) -> usize {
        self.row_block * self.col_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_spec_tile_elems_matches_product() {
        let spec = TileSpec::for_shape(128, 256);
        assert_eq!(spec.tile_elems(), spec.row_block * spec.col_block);
    }

    #[test]
    fn tile_spec_for_len_produces_non_zero_blocks() {
        let spec = TileSpec::for_len(10_000);
        assert!(spec.row_block >= 1);
        assert!(spec.col_block >= 1);
        assert!(spec.lane_width >= 1);
        assert!(spec.accumulators >= 1);
    }

    #[test]
    fn tile_spec_for_shape_never_exceeds_dimensions() {
        let spec = TileSpec::for_shape(32, 48);
        assert!(spec.row_block <= 32);
        assert!(spec.col_block <= 48);
    }
}
