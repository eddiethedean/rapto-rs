//! Code generation for SIMD reduction kernels
//!
//! This module provides functions to generate optimized reduction kernels
//! parameterized by SIMD width, tile dimensions, and accumulator count,
//! similar to NumPy's code-generated kernels from `loops.c.src`.

#![allow(dead_code)]

use crate::tiling::TileSpec;

/// Parameters for kernel generation
#[derive(Debug, Clone, Copy)]
pub struct KernelParams {
    pub dtype: Dtype,
    pub simd_width: usize,  // Number of elements per SIMD vector
    pub row_tile: usize,
    pub col_tile: usize,
    pub accumulators: usize, // Number of accumulator registers
    pub use_higher_precision: bool, // Use f64 for f32 operations
    pub use_nontemporal_stores: bool, // Use non-temporal stores
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F64,
}

impl KernelParams {
    /// Create parameters from TileSpec and dtype
    pub fn from_tilespec(spec: TileSpec, dtype: Dtype) -> Self {
        let simd_width = match dtype {
            Dtype::F32 => 4, // NEON: 4 f32 per 128-bit vector
            Dtype::F64 => 2, // NEON: 2 f64 per 128-bit vector
        };
        Self {
            dtype,
            simd_width,
            row_tile: spec.row_block,
            col_tile: spec.col_block,
            accumulators: spec.accumulators,
            use_higher_precision: dtype == Dtype::F32, // Use f64 for f32
            use_nontemporal_stores: true, // Use non-temporal stores for final writeback
        }
    }
    
    /// Calculate optimal tile size for given shape
    pub fn optimal_tile_size(rows: usize, cols: usize, _dtype: Dtype) -> (usize, usize) {
        let spec = TileSpec::for_shape(rows, cols);
        (spec.row_block, spec.col_block)
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;
    use core::arch::aarch64::*;
    
    /// Fallback for very large tiles (uses heap allocation)
    #[target_feature(enable = "neon")]
    unsafe fn reduce_axis0_f64_generated_fallback(
        data: &[f64],
        rows: usize,
        cols: usize,
        _params: KernelParams,
    ) -> Vec<f64> {
        // For very large tiles, use simpler approach
        let mut out = vec![0.0f64; cols];
        for col in 0..cols {
            let mut sum = 0.0f64;
            for row in 0..rows {
                sum += data[row * cols + col];
            }
            out[col] = sum;
        }
        out
    }
    
    /// Generate a register-resident reduction kernel for f64
    /// 
    /// This keeps accumulators in registers across entire tiles,
    /// only writing at tile boundaries to avoid load-modify-store cycles.
    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_axis0_f64_generated(
        data: &[f64],
        rows: usize,
        cols: usize,
        params: KernelParams,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; cols];
        if rows == 0 || cols == 0 {
            return out;
        }
        
        let stride = cols;
        let base_ptr = data.as_ptr();
        let out_ptr = out.as_mut_ptr();
        const LANES: usize = 2; // NEON f64: 2 elements per vector
        
        // Process in tiles: keep accumulators in registers across entire row tile
        let mut row_start = 0usize;
        while row_start < rows {
            let block_rows = (rows - row_start).min(params.row_tile);
            let mut col = 0usize;
            
            while col < cols {
                let width = (cols - col).min(params.col_tile);
                let vec_count = width / LANES;
                let tail_start = vec_count * LANES;
                let tail = width - tail_start;
                
                // Register-resident accumulators: keep in SIMD registers for entire tile
                // One accumulator per column vector - keeps sums in registers until tile complete
                // Use stack allocation for reasonable tile sizes
                // Note: Using Vec for now to avoid stack overflow - can optimize later
                // The key optimization is register-resident accumulation, not stack vs heap
                let mut vec_acc = vec![vdupq_n_f64(0.0); vec_count];
                let mut tail_acc = [0.0f64; LANES];
                
                // Process all rows in this tile, accumulating in registers
                // This is the key optimization: accumulate across entire tile before writing
                for r in 0..block_rows {
                    let ptr = base_ptr.add((row_start + r) * stride + col);
                    
                    // Process vectorized columns - accumulate in registers
                    for v in 0..vec_count {
                        let offset = v * LANES;
                        let vec = vld1q_f64(ptr.add(offset));
                        vec_acc[v] = vaddq_f64(vec_acc[v], vec);
                    }
                    
                    // Handle tail
                    if tail > 0 {
                        for t in 0..tail {
                            tail_acc[t] += *ptr.add(tail_start + t);
                        }
                    }
                }
                
                // Write results only at tile boundary (register-resident accumulation)
                // This eliminates load-modify-store cycles - we only write once per tile
                
                // Write vectorized results - only at tile boundary
                for v in 0..vec_count {
                    let dst = out_ptr.add(col + v * LANES);
                    
                    if row_start == 0 {
                        // First tile: write directly (no previous value to load)
                        if params.use_nontemporal_stores {
                            // Use prefetch hint for store (non-temporal hint)
                            // Note: True non-temporal stores (stnp) are for scalar pairs
                            // For vectors, we use regular store with prefetch hint
                            core::arch::asm!(
                                "prfm pstl1keep, [{addr}]",
                                addr = in(reg) dst,
                                options(nostack)
                            );
                            vst1q_f64(dst, vec_acc[v]);
                        } else {
                            vst1q_f64(dst, vec_acc[v]);
                        }
                    } else {
                        // Subsequent tiles: accumulate with previous
                        // For non-temporal stores, we still need to load previous value
                        let prev = vld1q_f64(dst);
                        let total = vaddq_f64(prev, vec_acc[v]);
                        
                        // Check if this is the final row tile
                        let is_final_tile = row_start + block_rows >= rows;
                        if params.use_nontemporal_stores && is_final_tile {
                            // Final write: use prefetch hint (non-temporal hint)
                            core::arch::asm!(
                                "prfm pstl1keep, [{addr}]",
                                addr = in(reg) dst,
                                options(nostack)
                            );
                        }
                        vst1q_f64(dst, total);
                    }
                }
                
                // Write tail
                if tail > 0 {
                    for t in 0..tail {
                        let idx = col + tail_start + t;
                        if row_start == 0 {
                            *out_ptr.add(idx) = tail_acc[t];
                        } else {
                            *out_ptr.add(idx) += tail_acc[t];
                        }
                    }
                }
                
                col += width;
            }
            row_start += block_rows;
        }
        
        out
    }
    
    /// Generate a register-resident reduction kernel for f32
    /// 
    /// Optionally uses f64 accumulators for higher precision (like NumPy's long double).
    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_axis0_f32_generated(
        data: &[f32],
        rows: usize,
        cols: usize,
        params: KernelParams,
    ) -> Vec<f32> {
        if params.use_higher_precision {
            // Use f64 accumulators for f32 operations
            reduce_axis0_f32_f64_accum(data, rows, cols, params)
        } else {
            // Use f32 accumulators
            reduce_axis0_f32_f32_accum(data, rows, cols, params)
        }
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn reduce_axis0_f32_f64_accum(
        data: &[f32],
        rows: usize,
        cols: usize,
        _params: KernelParams,
    ) -> Vec<f32> {
        // For now, use scalar f64 accumulation for higher precision
        // TODO: Implement proper NEON f32→f64 conversion with vectorized accumulation
        let mut out = vec![0.0f32; cols];
        if rows == 0 || cols == 0 {
            return out;
        }
        
        let stride = cols;
        let base_ptr = data.as_ptr();
        let out_ptr = out.as_mut_ptr();
        
        // Process each column with f64 accumulator
        for col in 0..cols {
            let mut sum = 0.0f64; // Use f64 for accumulation (higher precision)
            
            // Process all rows for this column
            for row in 0..rows {
                sum += *base_ptr.add(row * stride + col) as f64;
            }
            
            // Convert back to f32 and store
            *out_ptr.add(col) = sum as f32;
        }
        
        out
    }
    
    /// Fallback for very large tiles
    #[target_feature(enable = "neon")]
    unsafe fn reduce_axis0_f32_f32_accum_fallback(
        data: &[f32],
        rows: usize,
        cols: usize,
        _params: KernelParams,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for col in 0..cols {
            let mut sum = 0.0f32;
            for row in 0..rows {
                sum += data[row * cols + col];
            }
            out[col] = sum;
        }
        out
    }
    
    #[target_feature(enable = "neon")]
    unsafe fn reduce_axis0_f32_f32_accum(
        data: &[f32],
        rows: usize,
        cols: usize,
        params: KernelParams,
    ) -> Vec<f32> {
        // Similar to f64 version but with f32 accumulators
        let mut out = vec![0.0f32; cols];
        if rows == 0 || cols == 0 {
            return out;
        }
        
        let stride = cols;
        let base_ptr = data.as_ptr();
        let out_ptr = out.as_mut_ptr();
        const LANES: usize = 4; // NEON f32: 4 elements per vector
        
        let mut row_start = 0usize;
        while row_start < rows {
            let block_rows = (rows - row_start).min(params.row_tile);
            let mut col = 0usize;
            
            while col < cols {
                let width = (cols - col).min(params.col_tile);
                let vec_count = width / LANES;
                let tail_start = vec_count * LANES;
                let tail = width - tail_start;
                
                // Register-resident accumulators: keep in SIMD registers for entire tile
                // One accumulator per column vector - keeps sums in registers until tile complete
                // Use Vec for accumulator storage
                // Note: The key optimization is register-resident accumulation across tiles
                // Stack vs heap allocation is secondary
                let mut vec_acc = vec![vdupq_n_f32(0.0); vec_count];
                let mut tail_acc = [0.0f32; LANES];
                
                // Process all rows in this tile, accumulating in registers
                // This is the key optimization: accumulate across entire tile before writing
                for r in 0..block_rows {
                    let ptr = base_ptr.add((row_start + r) * stride + col);
                    
                    // Process vectorized columns - accumulate in registers
                    for v in 0..vec_count {
                        let offset = v * LANES;
                        let vec = vld1q_f32(ptr.add(offset));
                        vec_acc[v] = vaddq_f32(vec_acc[v], vec);
                    }
                    
                    // Handle tail
                    if tail > 0 {
                        for t in 0..tail {
                            tail_acc[t] += *ptr.add(tail_start + t);
                        }
                    }
                }
                
                // Write results only at tile boundary (register-resident accumulation)
                // This eliminates load-modify-store cycles - we only write once per tile
                
                // Write vectorized results - only at tile boundary
                for v in 0..vec_count {
                    let dst = out_ptr.add(col + v * LANES);
                    
                    if row_start == 0 {
                        // First tile: write directly (no previous value to load)
                        if params.use_nontemporal_stores {
                            // Use prefetch hint for store (non-temporal hint)
                            core::arch::asm!(
                                "prfm pstl1keep, [{addr}]",
                                addr = in(reg) dst,
                                options(nostack)
                            );
                            vst1q_f32(dst, vec_acc[v]);
                        } else {
                            vst1q_f32(dst, vec_acc[v]);
                        }
                    } else {
                        // Subsequent tiles: accumulate with previous
                        let prev = vld1q_f32(dst);
                        let total = vaddq_f32(prev, vec_acc[v]);
                        
                        // Check if this is the final row tile
                        let is_final_tile = row_start + block_rows >= rows;
                        if params.use_nontemporal_stores && is_final_tile {
                            // Final write: use prefetch hint (non-temporal hint)
                            core::arch::asm!(
                                "prfm pstl1keep, [{addr}]",
                                addr = in(reg) dst,
                                options(nostack)
                            );
                        }
                        vst1q_f32(dst, total);
                    }
                }
                
                if tail > 0 {
                    for t in 0..tail {
                        let idx = col + tail_start + t;
                        if row_start == 0 {
                            *out_ptr.add(idx) = tail_acc[t];
                        } else {
                            *out_ptr.add(idx) += tail_acc[t];
                        }
                    }
                }
                
                col += width;
            }
            row_start += block_rows;
        }
        
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_params_from_tilespec() {
        let spec = TileSpec::for_shape(1024, 1024);
        let params = KernelParams::from_tilespec(spec, Dtype::F64);
        assert_eq!(params.simd_width, 2); // NEON f64
        assert_eq!(params.dtype, Dtype::F64);
    }
    
    #[test]
    fn kernel_params_f32_uses_higher_precision() {
        let spec = TileSpec::for_shape(1024, 1024);
        let params = KernelParams::from_tilespec(spec, Dtype::F32);
        assert!(params.use_higher_precision); // Should use f64 for f32
    }
}
