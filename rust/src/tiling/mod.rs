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
    /// Heuristic tiling tuned per architecture.  Row and column blocks are
    /// chosen to keep working sets inside L1/L2 while amortising SIMD setup.
    pub fn for_shape(rows: usize, cols: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Self::new(rows, cols, 8, 8, 96, 256);
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::new(rows, cols, 4, 6, 64, 192);
            }
            return Self::new(rows, cols, 2, 4, 64, 160);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Apple M-series NEON prefers 128-bit lanes, so 2-wide f64 and 4-wide f32.
            return Self::new(rows, cols, 2, 5, 96, 192);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Conservative fallback for scalar-only builds.
            return Self::new(rows, cols, 1, 2, 64, 128);
        }
    }

    /// Helper for contiguous buffers where only the element count is known.
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
