#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
pub fn reduce_sum_f64(input: &[f64], accumulators: usize) -> Option<f64> {
    if std::arch::is_x86_feature_detected!("avx512f") {
        return Some(unsafe { x86::avx512::reduce_sum_f64(input, accumulators) });
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::reduce_sum_f64(input, accumulators) });
    }
    None
}

#[cfg(target_arch = "aarch64")]
pub fn reduce_sum_f64(input: &[f64], accumulators: usize) -> Option<f64> {
    Some(unsafe { neon::reduce_sum_f64(input, accumulators) })
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn reduce_sum_f64(input: &[f64], accumulators: usize) -> Option<f64> {
    let _ = (input, accumulators);
    None
}

#[cfg(target_arch = "x86_64")]
pub fn reduce_sum_f32(input: &[f32], accumulators: usize) -> Option<f64> {
    if std::arch::is_x86_feature_detected!("avx2") {
        return Some(unsafe { x86::reduce_sum_f32(input, accumulators) });
    }
    None
}

#[cfg(target_arch = "aarch64")]
pub fn reduce_sum_f32(input: &[f32], accumulators: usize) -> Option<f64> {
    Some(unsafe { neon::reduce_sum_f32(input, accumulators) })
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn reduce_sum_f32(input: &[f32], accumulators: usize) -> Option<f64> {
    let _ = (input, accumulators);
    None
}

#[cfg(target_arch = "x86_64")]
pub fn add_column_broadcast_f64(
    input: &[f64],
    col_values: &[f64],
    rows: usize,
    cols: usize,
    out: &mut [f64],
) -> bool {
    if input.len() != out.len()
        || input.len() != rows.saturating_mul(cols)
        || col_values.len() != rows
    {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_columnar_f64(input, col_values, rows, cols, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_column_broadcast_f64(
    input: &[f64],
    col_values: &[f64],
    rows: usize,
    cols: usize,
    out: &mut [f64],
) -> bool {
    if input.len() != out.len()
        || input.len() != rows.saturating_mul(cols)
        || col_values.len() != rows
    {
        return false;
    }
    unsafe {
        neon::add_columnar_f64(input, col_values, rows, cols, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_column_broadcast_f64(
    input: &[f64],
    col_values: &[f64],
    rows: usize,
    cols: usize,
    out: &mut [f64],
) -> bool {
    let _ = (input, col_values, rows, cols, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn add_column_broadcast_f32(
    input: &[f32],
    col_values: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if input.len() != out.len()
        || input.len() != rows.saturating_mul(cols)
        || col_values.len() != rows
    {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_columnar_f32(input, col_values, rows, cols, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_column_broadcast_f32(
    input: &[f32],
    col_values: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    if input.len() != out.len()
        || input.len() != rows.saturating_mul(cols)
        || col_values.len() != rows
    {
        return false;
    }
    unsafe {
        neon::add_columnar_f32(input, col_values, rows, cols, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_column_broadcast_f32(
    input: &[f32],
    col_values: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let _ = (input, col_values, rows, cols, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn add_assign_inplace_f64(acc: &mut [f64], row: &[f64]) -> bool {
    if acc.len() != row.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            x86::avx512::add_assign_inplace_f64(acc, row);
        }
        return true;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_assign_inplace_f64(acc, row);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_assign_inplace_f64(acc: &mut [f64], row: &[f64]) -> bool {
    if acc.len() != row.len() {
        return false;
    }
    unsafe {
        neon::add_assign_inplace_f64(acc, row);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_assign_inplace_f64(acc: &mut [f64], row: &[f64]) -> bool {
    let _ = (acc, row);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn add_assign_inplace_f32(acc: &mut [f32], row: &[f32]) -> bool {
    if acc.len() != row.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_assign_inplace_f32(acc, row);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_assign_inplace_f32(acc: &mut [f32], row: &[f32]) -> bool {
    if acc.len() != row.len() {
        return false;
    }
    unsafe {
        neon::add_assign_inplace_f32(acc, row);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_assign_inplace_f32(acc: &mut [f32], row: &[f32]) -> bool {
    let _ = (acc, row);
    false
}

pub fn reduce_axis0_columns_f32(data: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    if cols == 0 {
        return Some(Vec::new());
    }
    let elements = rows.checked_mul(cols)?;
    if elements != data.len() {
        return None;
    }
    if rows == 0 {
        return Some(vec![0.0; cols]);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Some(unsafe { x86::reduce_axis0_columns_f32(data, rows, cols) });
        }
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Some(unsafe { neon::reduce_axis0_columns_f32(data, rows, cols) });
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (data, rows, cols);
        None
    }
}

pub fn reduce_axis0_tiled_f32(data: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    if cols == 0 {
        return Some(Vec::new());
    }
    if rows == 0 {
        return Some(vec![0.0; cols]);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if rows >= x86::TILED_MIN_ROWS_F32
            && cols >= x86::TILED_MIN_COLS_F32
            && std::arch::is_x86_feature_detected!("avx2")
        {
            return Some(unsafe { x86::reduce_axis0_tiled_f32(data, rows, cols) });
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if rows >= neon::TILED_MIN_ROWS_F32 && cols >= neon::TILED_MIN_COLS_F32 {
            return Some(unsafe { neon::reduce_axis0_tiled_f32(data, rows, cols) });
        }
    }
    let _ = (data, rows, cols);
    None
}

#[cfg(target_arch = "x86_64")]
pub fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> bool {
    if lhs.len() != rhs.len() || lhs.len() != out.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            x86::avx512::add_same_shape_f64(lhs, rhs, out);
        }
        return true;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_same_shape_f64(lhs, rhs, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> bool {
    if lhs.len() != rhs.len() || lhs.len() != out.len() {
        return false;
    }
    unsafe {
        neon::add_same_shape_f64(lhs, rhs, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> bool {
    let _ = (lhs, rhs, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn add_same_shape_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> bool {
    if lhs.len() != rhs.len() || lhs.len() != out.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_same_shape_f32(lhs, rhs, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_same_shape_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> bool {
    if lhs.len() != rhs.len() || lhs.len() != out.len() {
        return false;
    }
    unsafe {
        neon::add_same_shape_f32(lhs, rhs, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_same_shape_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> bool {
    let _ = (lhs, rhs, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn add_row_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            x86::avx512::add_row_scalar_f64(input, scalar, out);
        }
        return true;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_row_scalar_f64(input, scalar, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_row_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    unsafe {
        neon::add_row_scalar_f64(input, scalar, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_row_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) -> bool {
    let _ = (input, scalar, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn add_row_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::add_row_scalar_f32(input, scalar, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn add_row_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    unsafe {
        neon::add_row_scalar_f32(input, scalar, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn add_row_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) -> bool {
    let _ = (input, scalar, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn scale_same_shape_f64(input: &[f64], factor: f64, out: &mut [f64]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe {
            x86::avx512::scale_same_shape_f64(input, factor, out);
        }
        return true;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::scale_same_shape_f64(input, factor, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn scale_same_shape_f64(input: &[f64], factor: f64, out: &mut [f64]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    unsafe {
        neon::scale_same_shape_f64(input, factor, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn scale_same_shape_f64(input: &[f64], factor: f64, out: &mut [f64]) -> bool {
    let _ = (input, factor, out);
    false
}

#[cfg(target_arch = "x86_64")]
pub fn scale_same_shape_f32(input: &[f32], factor: f32, out: &mut [f32]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            x86::scale_same_shape_f32(input, factor, out);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "aarch64")]
pub fn scale_same_shape_f32(input: &[f32], factor: f32, out: &mut [f32]) -> bool {
    if input.len() != out.len() {
        return false;
    }
    unsafe {
        neon::scale_same_shape_f32(input, factor, out);
    }
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn scale_same_shape_f32(input: &[f32], factor: f32, out: &mut [f32]) -> bool {
    let _ = (input, factor, out);
    false
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    const LANES_F64: usize = 4;
    const LANES_F32: usize = 8;
    const PREFETCH_DISTANCE_F32: usize = LANES_F32 * 8;
    const PREFETCH_DISTANCE_F64: usize = LANES_F64 * 8;
    const MAX_ACCUMULATORS_F64: usize = 6;
    const MAX_ACCUMULATORS_F32: usize = 8;
    const COLUMN_BLOCK: usize = LANES_F32 * 4;
    const PREFETCH_ROWS: usize = 4;
    const ROW_TILE_F32: usize = 128;
    const COL_TILE_F32: usize = COLUMN_BLOCK;
    const MAX_TILE_VECTORS_F32: usize = COL_TILE_F32 / LANES_F32;
    pub(super) const TILED_MIN_ROWS_F32: usize = 64;
    pub(super) const TILED_MIN_COLS_F32: usize = COL_TILE_F32;

    #[target_feature(enable = "avx2")]
    pub unsafe fn reduce_sum_f64(input: &[f64], accumulators: usize) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let acc_count = accumulators.clamp(1, MAX_ACCUMULATORS_F64);
        let mut regs = [_mm256_setzero_pd(); MAX_ACCUMULATORS_F64];
        let mut index = 0usize;
        let step = acc_count * LANES_F64;

        while index + step <= len {
            let mut offset = index;
            for slot in 0..acc_count {
                let vec = _mm256_loadu_pd(input.as_ptr().add(offset));
                regs[slot] = _mm256_add_pd(regs[slot], vec);
                offset += LANES_F64;
            }
            index += step;
        }

        let mut carry = _mm256_setzero_pd();
        while index + LANES_F64 <= len {
            let vec = _mm256_loadu_pd(input.as_ptr().add(index));
            carry = _mm256_add_pd(carry, vec);
            index += LANES_F64;
        }
        regs[0] = _mm256_add_pd(regs[0], carry);

        let mut total = 0.0;
        for slot in 0..acc_count {
            let mut buf = [0.0f64; LANES_F64];
            _mm256_storeu_pd(buf.as_mut_ptr(), regs[slot]);
            total += buf.iter().sum::<f64>();
        }
        while index < len {
            total += *input.get_unchecked(index);
            index += 1;
        }
        total
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn reduce_axis0_columns_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        debug_assert_eq!(rows.saturating_mul(cols), data.len());
        let mut out = vec![0.0f32; cols];
        let mut col = 0usize;
        let stride = cols;
        let base_ptr = data.as_ptr();

        while col + COLUMN_BLOCK <= cols {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut row_ptr = base_ptr.add(col);
            for row_idx in 0..rows {
                if row_idx + PREFETCH_ROWS < rows {
                    _mm_prefetch(
                        row_ptr.add(stride * PREFETCH_ROWS) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(row_ptr));
                acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(row_ptr.add(LANES_F32)));
                acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(row_ptr.add(LANES_F32 * 2)));
                acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(row_ptr.add(LANES_F32 * 3)));
                row_ptr = row_ptr.add(stride);
            }

            let mut buf0 = [0.0f32; LANES_F32];
            let mut buf1 = [0.0f32; LANES_F32];
            let mut buf2 = [0.0f32; LANES_F32];
            let mut buf3 = [0.0f32; LANES_F32];
            _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
            _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
            _mm256_storeu_ps(buf2.as_mut_ptr(), acc2);
            _mm256_storeu_ps(buf3.as_mut_ptr(), acc3);

            for lane in 0..LANES_F32 {
                out[col + lane] = buf0[lane];
                out[col + LANES_F32 + lane] = buf1[lane];
                out[col + LANES_F32 * 2 + lane] = buf2[lane];
                out[col + LANES_F32 * 3 + lane] = buf3[lane];
            }

            col += COLUMN_BLOCK;
        }

        while col + (LANES_F32 * 2) <= cols {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut row_ptr = base_ptr.add(col);
            for row_idx in 0..rows {
                if row_idx + PREFETCH_ROWS < rows {
                    _mm_prefetch(
                        row_ptr.add(stride * PREFETCH_ROWS) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(row_ptr));
                acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(row_ptr.add(LANES_F32)));
                row_ptr = row_ptr.add(stride);
            }
            let mut buf0 = [0.0f32; LANES_F32];
            let mut buf1 = [0.0f32; LANES_F32];
            _mm256_storeu_ps(buf0.as_mut_ptr(), acc0);
            _mm256_storeu_ps(buf1.as_mut_ptr(), acc1);
            for lane in 0..LANES_F32 {
                out[col + lane] = buf0[lane];
                out[col + LANES_F32 + lane] = buf1[lane];
            }
            col += LANES_F32 * 2;
        }

        while col + LANES_F32 <= cols {
            let mut acc = _mm256_setzero_ps();
            let mut row_ptr = base_ptr.add(col);
            for row_idx in 0..rows {
                if row_idx + PREFETCH_ROWS < rows {
                    _mm_prefetch(
                        row_ptr.add(stride * PREFETCH_ROWS) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                acc = _mm256_add_ps(acc, _mm256_loadu_ps(row_ptr));
                row_ptr = row_ptr.add(stride);
            }
            let mut buf = [0.0f32; LANES_F32];
            _mm256_storeu_ps(buf.as_mut_ptr(), acc);
            for lane in 0..LANES_F32 {
                out[col + lane] = buf[lane];
            }
            col += LANES_F32;
        }

        if col < cols {
            let remaining = cols - col;
            let mut row_ptr = base_ptr.add(col);
            for _ in 0..rows {
                for offset in 0..remaining {
                    *out.get_unchecked_mut(col + offset) += *row_ptr.add(offset);
                }
                row_ptr = row_ptr.add(stride);
            }
        }

        out
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn reduce_axis0_tiled_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        debug_assert_eq!(rows.saturating_mul(cols), data.len());
        let mut out = vec![0.0f32; cols];
        if rows == 0 || cols == 0 {
            return out;
        }
        let stride = cols;
        let base_ptr = data.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let mut row_start = 0usize;
        while row_start < rows {
            let block_rows = (rows - row_start).min(ROW_TILE_F32);
            let mut col = 0usize;
            while col < cols {
                let width = (cols - col).min(COL_TILE_F32);
                let vec_count = width / LANES_F32;
                let tail_start = vec_count * LANES_F32;
                let tail = width - tail_start;
                let mut vec_acc = [_mm256_setzero_ps(); MAX_TILE_VECTORS_F32];
                let mut tail_acc = [0.0f32; LANES_F32];

                let mut r = 0usize;
                while r < block_rows {
                    let ptr = base_ptr.add((row_start + r) * stride + col);
                    for v in 0..vec_count {
                        let offset = v * LANES_F32;
                        let vec = _mm256_loadu_ps(ptr.add(offset));
                        vec_acc[v] = _mm256_add_ps(vec_acc[v], vec);
                    }
                    if tail > 0 {
                        for t in 0..tail {
                            tail_acc[t] += *ptr.add(tail_start + t);
                        }
                    }
                    r += 1;
                }

                for v in 0..vec_count {
                    let dst = out_ptr.add(col + v * LANES_F32);
                    let prev = _mm256_loadu_ps(dst);
                    let sum = _mm256_add_ps(prev, vec_acc[v]);
                    _mm256_storeu_ps(dst, sum);
                }
                if tail > 0 {
                    for t in 0..tail {
                        let idx = col + tail_start + t;
                        *out_ptr.add(idx) += tail_acc[t];
                    }
                }

                col += width;
            }
            row_start += block_rows;
        }

        out
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn reduce_sum_f32(input: &[f32], accumulators: usize) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let acc_count = accumulators.clamp(1, MAX_ACCUMULATORS_F32);
        let mut regs = [_mm256_setzero_ps(); MAX_ACCUMULATORS_F32];
        let mut index = 0usize;
        let step = acc_count * LANES_F32;

        while index + step <= len {
            let mut offset = index;
            for slot in 0..acc_count {
                let vec = _mm256_loadu_ps(input.as_ptr().add(offset));
                regs[slot] = _mm256_add_ps(regs[slot], vec);
                offset += LANES_F32;
            }
            index += step;
        }

        let mut carry = _mm256_setzero_ps();
        while index + LANES_F32 <= len {
            let vec = _mm256_loadu_ps(input.as_ptr().add(index));
            carry = _mm256_add_ps(carry, vec);
            index += LANES_F32;
        }
        regs[0] = _mm256_add_ps(regs[0], carry);

        let mut total = 0.0f64;
        for slot in 0..acc_count {
            let mut buf = [0.0f32; LANES_F32];
            _mm256_storeu_ps(buf.as_mut_ptr(), regs[slot]);
            total += buf.iter().map(|&v| v as f64).sum::<f64>();
        }
        while index < len {
            total += *input.get_unchecked(index) as f64;
            index += 1;
        }
        total
    }

    pub(crate) mod avx512 {
        use super::*;

        const LANES: usize = 8;
        const MAX_ACCUMULATORS: usize = 8;

        #[target_feature(enable = "avx512f")]
        pub unsafe fn reduce_sum_f64(input: &[f64], accumulators: usize) -> f64 {
            let len = input.len();
            if len == 0 {
                return 0.0;
            }

            let acc_count = accumulators.clamp(1, MAX_ACCUMULATORS);
            let mut regs = [_mm512_setzero_pd(); MAX_ACCUMULATORS];
            let mut index = 0usize;
            let step = acc_count * LANES;

            while index + step <= len {
                let mut offset = index;
                for slot in 0..acc_count {
                    let vec = _mm512_loadu_pd(input.as_ptr().add(offset));
                    regs[slot] = _mm512_add_pd(regs[slot], vec);
                    offset += LANES;
                }
                index += step;
            }

            let mut carry = _mm512_setzero_pd();
            while index + LANES <= len {
                let vec = _mm512_loadu_pd(input.as_ptr().add(index));
                carry = _mm512_add_pd(carry, vec);
                index += LANES;
            }
            regs[0] = _mm512_add_pd(regs[0], carry);

            let mut total = 0.0;
            for slot in 0..acc_count {
                let mut buf = [0.0f64; LANES];
                _mm512_storeu_pd(buf.as_mut_ptr(), regs[slot]);
                total += buf.iter().sum::<f64>();
            }
            while index < len {
                total += *input.get_unchecked(index);
                index += 1;
            }
            total
        }

        #[target_feature(enable = "avx512f")]
        pub unsafe fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
            let len = lhs.len();
            let mut i = 0usize;
            while i + LANES <= len {
                let a = _mm512_loadu_pd(lhs.as_ptr().add(i));
                let b = _mm512_loadu_pd(rhs.as_ptr().add(i));
                let c = _mm512_add_pd(a, b);
                _mm512_storeu_pd(out.as_mut_ptr().add(i), c);
                i += LANES;
            }
            while i < len {
                *out.get_unchecked_mut(i) = lhs.get_unchecked(i) + rhs.get_unchecked(i);
                i += 1;
            }
        }

        #[target_feature(enable = "avx512f")]
        pub unsafe fn add_row_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) {
            let len = input.len();
            let scalar_v = _mm512_set1_pd(scalar);
            let mut i = 0usize;
            while i + LANES <= len {
                let a = _mm512_loadu_pd(input.as_ptr().add(i));
                let c = _mm512_add_pd(a, scalar_v);
                _mm512_storeu_pd(out.as_mut_ptr().add(i), c);
                i += LANES;
            }
            while i < len {
                *out.get_unchecked_mut(i) = input.get_unchecked(i) + scalar;
                i += 1;
            }
        }

        #[target_feature(enable = "avx512f")]
        pub unsafe fn scale_same_shape_f64(input: &[f64], factor: f64, out: &mut [f64]) {
            let len = input.len();
            let factor_v = _mm512_set1_pd(factor);
            let mut i = 0usize;
            while i + LANES <= len {
                let a = _mm512_loadu_pd(input.as_ptr().add(i));
                let c = _mm512_mul_pd(a, factor_v);
                _mm512_storeu_pd(out.as_mut_ptr().add(i), c);
                i += LANES;
            }
            while i < len {
                *out.get_unchecked_mut(i) = input.get_unchecked(i) * factor;
                i += 1;
            }
        }

        #[target_feature(enable = "avx512f")]
        pub unsafe fn add_columnar_f64(
            input: &[f64],
            col_values: &[f64],
            rows: usize,
            cols: usize,
            out: &mut [f64],
        ) {
            for row in 0..rows {
                let scalar = *col_values.get_unchecked(row);
                let scalar_v = _mm512_set1_pd(scalar);
                let base = row * cols;
                let mut col = 0usize;
                while col + LANES <= cols {
                    let offset = base + col;
                    let a = _mm512_loadu_pd(input.as_ptr().add(offset));
                    let c = _mm512_add_pd(a, scalar_v);
                    _mm512_storeu_pd(out.as_mut_ptr().add(offset), c);
                    col += LANES;
                }
                while col < cols {
                    let offset = base + col;
                    *out.get_unchecked_mut(offset) = *input.get_unchecked(offset) + scalar;
                    col += 1;
                }
            }
        }

        #[target_feature(enable = "avx512f")]
        pub unsafe fn add_assign_inplace_f64(acc: &mut [f64], row: &[f64]) {
            let len = acc.len();
            let mut i = 0usize;
            while i + LANES <= len {
                let a = _mm512_loadu_pd(acc.as_ptr().add(i));
                let b = _mm512_loadu_pd(row.as_ptr().add(i));
                let c = _mm512_add_pd(a, b);
                _mm512_storeu_pd(acc.as_mut_ptr().add(i), c);
                i += LANES;
            }
            while i < len {
                *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
        let len = lhs.len();
        let ptr_l = lhs.as_ptr();
        let ptr_r = rhs.as_ptr();
        let ptr_o = out.as_mut_ptr();

        let mut i = 0usize;
        while i + LANES_F64 <= len {
            let a = _mm256_loadu_pd(ptr_l.add(i));
            let b = _mm256_loadu_pd(ptr_r.add(i));
            let c = _mm256_add_pd(a, b);
            _mm256_storeu_pd(ptr_o.add(i), c);
            i += LANES_F64;
        }

        while i < len {
            *ptr_o.add(i) = *ptr_l.add(i) + *ptr_r.add(i);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_same_shape_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        let len = lhs.len();
        let ptr_l = lhs.as_ptr();
        let ptr_r = rhs.as_ptr();
        let ptr_o = out.as_mut_ptr();

        let mut i = 0usize;
        while i + LANES_F32 <= len {
            let a = _mm256_loadu_ps(ptr_l.add(i));
            let b = _mm256_loadu_ps(ptr_r.add(i));
            let c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(ptr_o.add(i), c);
            i += LANES_F32;
        }

        while i < len {
            *ptr_o.add(i) = *ptr_l.add(i) + *ptr_r.add(i);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_row_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let scalar_v = _mm256_set1_pd(scalar);

        let mut i = 0usize;
        while i + LANES_F64 <= len {
            let a = _mm256_loadu_pd(ptr_in.add(i));
            let c = _mm256_add_pd(a, scalar_v);
            _mm256_storeu_pd(ptr_out.add(i), c);
            i += LANES_F64;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) + scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_row_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let scalar_v = _mm256_set1_ps(scalar);

        let mut i = 0usize;
        let unroll = LANES_F32 * 4;
        while i + unroll <= len {
            let base = ptr_in.add(i);
            if i + PREFETCH_DISTANCE_F32 < len {
                _mm_prefetch(
                    ptr_in.add(i + PREFETCH_DISTANCE_F32) as *const i8,
                    _MM_HINT_T0,
                );
            }
            let a0 = _mm256_loadu_ps(base);
            let a1 = _mm256_loadu_ps(base.add(LANES_F32));
            let a2 = _mm256_loadu_ps(base.add(LANES_F32 * 2));
            let a3 = _mm256_loadu_ps(base.add(LANES_F32 * 3));
            let c0 = _mm256_add_ps(a0, scalar_v);
            let c1 = _mm256_add_ps(a1, scalar_v);
            let c2 = _mm256_add_ps(a2, scalar_v);
            let c3 = _mm256_add_ps(a3, scalar_v);
            let out_base = ptr_out.add(i);
            _mm256_storeu_ps(out_base, c0);
            _mm256_storeu_ps(out_base.add(LANES_F32), c1);
            _mm256_storeu_ps(out_base.add(LANES_F32 * 2), c2);
            _mm256_storeu_ps(out_base.add(LANES_F32 * 3), c3);
            i += unroll;
        }
        while i + LANES_F32 <= len {
            let a = _mm256_loadu_ps(ptr_in.add(i));
            let c = _mm256_add_ps(a, scalar_v);
            _mm256_storeu_ps(ptr_out.add(i), c);
            i += LANES_F32;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) + scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scale_same_shape_f64(input: &[f64], factor: f64, out: &mut [f64]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let factor_v = _mm256_set1_pd(factor);

        let mut i = 0usize;
        let unroll = LANES_F64 * 4;
        while i + unroll <= len {
            let base = ptr_in.add(i);
            if i + PREFETCH_DISTANCE_F64 < len {
                _mm_prefetch(
                    ptr_in.add(i + PREFETCH_DISTANCE_F64) as *const i8,
                    _MM_HINT_T0,
                );
            }
            let a0 = _mm256_loadu_pd(base);
            let a1 = _mm256_loadu_pd(base.add(LANES_F64));
            let a2 = _mm256_loadu_pd(base.add(LANES_F64 * 2));
            let a3 = _mm256_loadu_pd(base.add(LANES_F64 * 3));
            let c0 = _mm256_mul_pd(a0, factor_v);
            let c1 = _mm256_mul_pd(a1, factor_v);
            let c2 = _mm256_mul_pd(a2, factor_v);
            let c3 = _mm256_mul_pd(a3, factor_v);
            let out_base = ptr_out.add(i);
            _mm256_storeu_pd(out_base, c0);
            _mm256_storeu_pd(out_base.add(LANES_F64), c1);
            _mm256_storeu_pd(out_base.add(LANES_F64 * 2), c2);
            _mm256_storeu_pd(out_base.add(LANES_F64 * 3), c3);
            i += unroll;
        }
        while i + LANES_F64 <= len {
            let a = _mm256_loadu_pd(ptr_in.add(i));
            let c = _mm256_mul_pd(a, factor_v);
            _mm256_storeu_pd(ptr_out.add(i), c);
            i += LANES_F64;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) * factor;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn scale_same_shape_f32(input: &[f32], factor: f32, out: &mut [f32]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let factor_v = _mm256_set1_ps(factor);

        let mut i = 0usize;
        while i + LANES_F32 * 4 <= len {
            let base = ptr_in.add(i);
            if i + PREFETCH_DISTANCE_F32 < len {
                _mm_prefetch(
                    ptr_in.add(i + PREFETCH_DISTANCE_F32) as *const i8,
                    _MM_HINT_T0,
                );
            }
            let a0 = _mm256_loadu_ps(base);
            let a1 = _mm256_loadu_ps(base.add(LANES_F32));
            let a2 = _mm256_loadu_ps(base.add(LANES_F32 * 2));
            let a3 = _mm256_loadu_ps(base.add(LANES_F32 * 3));
            let c0 = _mm256_mul_ps(a0, factor_v);
            let c1 = _mm256_mul_ps(a1, factor_v);
            let c2 = _mm256_mul_ps(a2, factor_v);
            let c3 = _mm256_mul_ps(a3, factor_v);
            let out_base = ptr_out.add(i);
            _mm256_storeu_ps(out_base, c0);
            _mm256_storeu_ps(out_base.add(LANES_F32), c1);
            _mm256_storeu_ps(out_base.add(LANES_F32 * 2), c2);
            _mm256_storeu_ps(out_base.add(LANES_F32 * 3), c3);
            i += LANES_F32 * 4;
        }

        while i + LANES_F32 <= len {
            let a = _mm256_loadu_ps(ptr_in.add(i));
            let c = _mm256_mul_ps(a, factor_v);
            _mm256_storeu_ps(ptr_out.add(i), c);
            i += LANES_F32;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) * factor;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_columnar_f64(
        input: &[f64],
        col_values: &[f64],
        rows: usize,
        cols: usize,
        out: &mut [f64],
    ) {
        for row in 0..rows {
            let scalar = *col_values.get_unchecked(row);
            let scalar_v = _mm256_set1_pd(scalar);
            let base = row * cols;
            let mut col = 0usize;
            while col + LANES_F64 <= cols {
                let offset = base + col;
                let a = _mm256_loadu_pd(input.as_ptr().add(offset));
                let c = _mm256_add_pd(a, scalar_v);
                _mm256_storeu_pd(out.as_mut_ptr().add(offset), c);
                col += LANES_F64;
            }
            while col < cols {
                let offset = base + col;
                *out.get_unchecked_mut(offset) = *input.get_unchecked(offset) + scalar;
                col += 1;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_columnar_f32(
        input: &[f32],
        col_values: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) {
        let enable_prefetch = rows >= 1024 || cols >= 1024;
        for row in 0..rows {
            let scalar = *col_values.get_unchecked(row);
            let scalar_v = _mm256_set1_ps(scalar);
            let base = row * cols;
            let mut col = 0usize;
            if enable_prefetch && row + PREFETCH_ROWS < rows {
                _mm_prefetch(
                    input.as_ptr().add((row + PREFETCH_ROWS) * cols) as *const i8,
                    _MM_HINT_T0,
                );
            }
            while col + LANES_F32 * 4 <= cols {
                let offset = base + col;
                let ptr_in = input.as_ptr().add(offset);
                if enable_prefetch && col + PREFETCH_DISTANCE_F32 < cols {
                    _mm_prefetch(ptr_in.add(PREFETCH_DISTANCE_F32) as *const i8, _MM_HINT_T0);
                }
                let a0 = _mm256_loadu_ps(ptr_in);
                let a1 = _mm256_loadu_ps(ptr_in.add(LANES_F32));
                let a2 = _mm256_loadu_ps(ptr_in.add(LANES_F32 * 2));
                let a3 = _mm256_loadu_ps(ptr_in.add(LANES_F32 * 3));
                let c0 = _mm256_add_ps(a0, scalar_v);
                let c1 = _mm256_add_ps(a1, scalar_v);
                let c2 = _mm256_add_ps(a2, scalar_v);
                let c3 = _mm256_add_ps(a3, scalar_v);
                let ptr_out = out.as_mut_ptr().add(offset);
                _mm256_storeu_ps(ptr_out, c0);
                _mm256_storeu_ps(ptr_out.add(LANES_F32), c1);
                _mm256_storeu_ps(ptr_out.add(LANES_F32 * 2), c2);
                _mm256_storeu_ps(ptr_out.add(LANES_F32 * 3), c3);
                col += LANES_F32 * 4;
            }
            while col + LANES_F32 <= cols {
                let offset = base + col;
                let ptr_in = input.as_ptr().add(offset);
                let a = _mm256_loadu_ps(ptr_in);
                let c = _mm256_add_ps(a, scalar_v);
                _mm256_storeu_ps(out.as_mut_ptr().add(offset), c);
                col += LANES_F32;
            }
            while col < cols {
                let offset = base + col;
                *out.get_unchecked_mut(offset) = *input.get_unchecked(offset) + scalar;
                col += 1;
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_assign_inplace_f64(acc: &mut [f64], row: &[f64]) {
        let len = acc.len();
        let mut i = 0usize;
        while i + LANES_F64 <= len {
            let a = _mm256_loadu_pd(acc.as_ptr().add(i));
            let b = _mm256_loadu_pd(row.as_ptr().add(i));
            let c = _mm256_add_pd(a, b);
            _mm256_storeu_pd(acc.as_mut_ptr().add(i), c);
            i += LANES_F64;
        }
        while i < len {
            *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn add_assign_inplace_f32(acc: &mut [f32], row: &[f32]) {
        let len = acc.len();
        let mut i = 0usize;
        while i + LANES_F32 <= len {
            let a = _mm256_loadu_ps(acc.as_ptr().add(i));
            let b = _mm256_loadu_ps(row.as_ptr().add(i));
            let c = _mm256_add_ps(a, b);
            _mm256_storeu_ps(acc.as_mut_ptr().add(i), c);
            i += LANES_F32;
        }
        while i < len {
            *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    const LANES_F64: usize = 2;
    const LANES_F32: usize = 4;
    const MAX_ACCUMULATORS_F64: usize = 4;
    const MAX_ACCUMULATORS_F32: usize = 8;
    const PREFETCH_DISTANCE_F32: usize = LANES_F32 * 16;
    const PREFETCH_DISTANCE_F64: usize = LANES_F64 * 16;
    const COLUMN_BLOCK: usize = LANES_F32 * 4;
    const PREFETCH_ROWS: usize = 4;
    const ROW_TILE_F32: usize = 128;
    const COL_TILE_F32: usize = COLUMN_BLOCK;
    const MAX_TILE_VECTORS_F32: usize = COL_TILE_F32 / LANES_F32;
    pub(super) const TILED_MIN_ROWS_F32: usize = 64;
    pub(super) const TILED_MIN_COLS_F32: usize = COL_TILE_F32;

    #[inline(always)]
    unsafe fn reduce_sum_f64_fixed<const ACC: usize>(input: &[f64]) -> f64 {
        debug_assert!(ACC >= 1 && ACC <= MAX_ACCUMULATORS_F64);

        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let mut regs: [float64x2_t; ACC] = [vdupq_n_f64(0.0); ACC];
        let ptr = input.as_ptr();
        let mut index = 0usize;
        let step = ACC * LANES_F64;

        while index + step <= len {
            let mut offset = 0usize;
            while offset < ACC {
                let vec = vld1q_f64(ptr.add(index + offset * LANES_F64));
                regs[offset] = vaddq_f64(regs[offset], vec);
                offset += 1;
            }
            index += step;
        }

        let mut tail = vdupq_n_f64(0.0);
        while index + LANES_F64 <= len {
            let vec = vld1q_f64(ptr.add(index));
            tail = vaddq_f64(tail, vec);
            index += LANES_F64;
        }
        regs[0] = vaddq_f64(regs[0], tail);

        let mut total = 0.0;
        let mut slot = 0usize;
        while slot < ACC {
            total += vaddvq_f64(regs[slot]);
            slot += 1;
        }
        while index < len {
            total += *ptr.add(index);
            index += 1;
        }
        total
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_sum_f64(input: &[f64], accumulators: usize) -> f64 {
        match accumulators.clamp(1, MAX_ACCUMULATORS_F64) {
            4 => reduce_sum_f64_fixed::<4>(input),
            3 => reduce_sum_f64_fixed::<3>(input),
            2 => reduce_sum_f64_fixed::<2>(input),
            _ => reduce_sum_f64_fixed::<1>(input),
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_axis0_columns_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        debug_assert_eq!(rows.saturating_mul(cols), data.len());
        let mut out = vec![0.0f32; cols];
        let stride = cols;
        let base_ptr = data.as_ptr();
        let mut col = 0usize;

        while col + COLUMN_BLOCK <= cols {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut row_ptr = base_ptr.add(col);
            for row_idx in 0..rows {
                if row_idx + PREFETCH_ROWS < rows {
                    #[cfg(target_arch = "aarch64")]
                    {
                        core::arch::asm!(
                            "prfm pldl1keep, [{addr}]",
                            addr = in(reg) row_ptr.add(stride * PREFETCH_ROWS),
                            options(readonly, nostack)
                        );
                    }
                }
                acc0 = vaddq_f32(acc0, vld1q_f32(row_ptr));
                acc1 = vaddq_f32(acc1, vld1q_f32(row_ptr.add(LANES_F32)));
                acc2 = vaddq_f32(acc2, vld1q_f32(row_ptr.add(LANES_F32 * 2)));
                acc3 = vaddq_f32(acc3, vld1q_f32(row_ptr.add(LANES_F32 * 3)));
                row_ptr = row_ptr.add(stride);
            }
            let mut buf0 = [0.0f32; LANES_F32];
            let mut buf1 = [0.0f32; LANES_F32];
            let mut buf2 = [0.0f32; LANES_F32];
            let mut buf3 = [0.0f32; LANES_F32];
            vst1q_f32(buf0.as_mut_ptr(), acc0);
            vst1q_f32(buf1.as_mut_ptr(), acc1);
            vst1q_f32(buf2.as_mut_ptr(), acc2);
            vst1q_f32(buf3.as_mut_ptr(), acc3);
            for lane in 0..LANES_F32 {
                out[col + lane] = buf0[lane];
                out[col + LANES_F32 + lane] = buf1[lane];
                out[col + LANES_F32 * 2 + lane] = buf2[lane];
                out[col + LANES_F32 * 3 + lane] = buf3[lane];
            }
            col += COLUMN_BLOCK;
        }

        while col + (LANES_F32 * 2) <= cols {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut row_ptr = base_ptr.add(col);
            for row_idx in 0..rows {
                if row_idx + PREFETCH_ROWS < rows {
                    #[cfg(target_arch = "aarch64")]
                    {
                        core::arch::asm!(
                            "prfm pldl1keep, [{addr}]",
                            addr = in(reg) row_ptr.add(stride * PREFETCH_ROWS),
                            options(readonly, nostack)
                        );
                    }
                }
                acc0 = vaddq_f32(acc0, vld1q_f32(row_ptr));
                acc1 = vaddq_f32(acc1, vld1q_f32(row_ptr.add(LANES_F32)));
                row_ptr = row_ptr.add(stride);
            }
            let mut buf0 = [0.0f32; LANES_F32];
            let mut buf1 = [0.0f32; LANES_F32];
            vst1q_f32(buf0.as_mut_ptr(), acc0);
            vst1q_f32(buf1.as_mut_ptr(), acc1);
            for lane in 0..LANES_F32 {
                out[col + lane] = buf0[lane];
                out[col + LANES_F32 + lane] = buf1[lane];
            }
            col += LANES_F32 * 2;
        }

        while col + LANES_F32 <= cols {
            let mut acc = vdupq_n_f32(0.0);
            let mut row_ptr = base_ptr.add(col);
            for row_idx in 0..rows {
                if row_idx + PREFETCH_ROWS < rows {
                    #[cfg(target_arch = "aarch64")]
                    {
                        core::arch::asm!(
                            "prfm pldl1keep, [{addr}]",
                            addr = in(reg) row_ptr.add(stride * PREFETCH_ROWS),
                            options(readonly, nostack)
                        );
                    }
                }
                acc = vaddq_f32(acc, vld1q_f32(row_ptr));
                row_ptr = row_ptr.add(stride);
            }
            let mut buf = [0.0f32; LANES_F32];
            vst1q_f32(buf.as_mut_ptr(), acc);
            for lane in 0..LANES_F32 {
                out[col + lane] = buf[lane];
            }
            col += LANES_F32;
        }

        if col < cols {
            let remaining = cols - col;
            let mut row_ptr = base_ptr.add(col);
            for _ in 0..rows {
                for offset in 0..remaining {
                    *out.get_unchecked_mut(col + offset) += *row_ptr.add(offset);
                }
                row_ptr = row_ptr.add(stride);
            }
        }

        out
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_axis0_tiled_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        debug_assert_eq!(rows.saturating_mul(cols), data.len());
        let mut out = vec![0.0f32; cols];
        if rows == 0 || cols == 0 {
            return out;
        }
        let stride = cols;
        let base_ptr = data.as_ptr();
        let out_ptr = out.as_mut_ptr();

        let mut row_start = 0usize;
        while row_start < rows {
            let block_rows = (rows - row_start).min(ROW_TILE_F32);
            let mut col = 0usize;
            while col < cols {
                let width = (cols - col).min(COL_TILE_F32);
                let vec_count = width / LANES_F32;
                let tail_start = vec_count * LANES_F32;
                let tail = width - tail_start;
                let mut vec_acc = [vdupq_n_f32(0.0); MAX_TILE_VECTORS_F32];
                let mut tail_acc = [0.0f32; LANES_F32];

                let mut r = 0usize;
                while r < block_rows {
                    let ptr = base_ptr.add((row_start + r) * stride + col);
                    for v in 0..vec_count {
                        let offset = v * LANES_F32;
                        let vec = vld1q_f32(ptr.add(offset));
                        vec_acc[v] = vaddq_f32(vec_acc[v], vec);
                    }
                    if tail > 0 {
                        for t in 0..tail {
                            tail_acc[t] += *ptr.add(tail_start + t);
                        }
                    }
                    r += 1;
                }

                for v in 0..vec_count {
                    let dst = out_ptr.add(col + v * LANES_F32);
                    let prev = vld1q_f32(dst);
                    let sum = vaddq_f32(prev, vec_acc[v]);
                    vst1q_f32(dst, sum);
                }
                if tail > 0 {
                    for t in 0..tail {
                        let idx = col + tail_start + t;
                        *out_ptr.add(idx) += tail_acc[t];
                    }
                }

                col += width;
            }
            row_start += block_rows;
        }

        out
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_sum_f32(input: &[f32], accumulators: usize) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let acc_count = accumulators.clamp(1, MAX_ACCUMULATORS_F32);
        let mut regs = [vdupq_n_f32(0.0); MAX_ACCUMULATORS_F32];
        let mut index = 0usize;
        let ptr = input.as_ptr();
        let step = acc_count * LANES_F32;

        while index + step <= len {
            let mut offset = index;
            for slot in 0..acc_count {
                let vec = vld1q_f32(ptr.add(offset));
                regs[slot] = vaddq_f32(regs[slot], vec);
                offset += LANES_F32;
            }
            index += step;
        }

        let mut tail = vdupq_n_f32(0.0);
        while index + LANES_F32 <= len {
            let vec = vld1q_f32(ptr.add(index));
            tail = vaddq_f32(tail, vec);
            index += LANES_F32;
        }
        regs[0] = vaddq_f32(regs[0], tail);

        let mut total = 0.0f64;
        for slot in 0..acc_count {
            total += vaddvq_f32(regs[slot]) as f64;
        }
        while index < len {
            total += *ptr.add(index) as f64;
            index += 1;
        }
        total
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) {
        let len = lhs.len();
        let ptr_l = lhs.as_ptr();
        let ptr_r = rhs.as_ptr();
        let ptr_o = out.as_mut_ptr();

        let mut i = 0usize;
        while i + LANES_F64 <= len {
            let a = vld1q_f64(ptr_l.add(i));
            let b = vld1q_f64(ptr_r.add(i));
            let c = vaddq_f64(a, b);
            vst1q_f64(ptr_o.add(i), c);
            i += LANES_F64;
        }

        while i < len {
            *ptr_o.add(i) = *ptr_l.add(i) + *ptr_r.add(i);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_same_shape_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        let len = lhs.len();
        let ptr_l = lhs.as_ptr();
        let ptr_r = rhs.as_ptr();
        let ptr_o = out.as_mut_ptr();

        let mut i = 0usize;
        while i + LANES_F32 <= len {
            let a = vld1q_f32(ptr_l.add(i));
            let b = vld1q_f32(ptr_r.add(i));
            let c = vaddq_f32(a, b);
            vst1q_f32(ptr_o.add(i), c);
            i += LANES_F32;
        }

        while i < len {
            *ptr_o.add(i) = *ptr_l.add(i) + *ptr_r.add(i);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_row_scalar_f64(input: &[f64], scalar: f64, out: &mut [f64]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let scalar_v = vdupq_n_f64(scalar);

        let mut i = 0usize;
        while i + LANES_F64 <= len {
            let a = vld1q_f64(ptr_in.add(i));
            let c = vaddq_f64(a, scalar_v);
            vst1q_f64(ptr_out.add(i), c);
            i += LANES_F64;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) + scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_row_scalar_f32(input: &[f32], scalar: f32, out: &mut [f32]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let scalar_v = vdupq_n_f32(scalar);

        let mut i = 0usize;
        let unroll = LANES_F32 * 4;
        while i + unroll <= len {
            let base = ptr_in.add(i);
            #[cfg(target_arch = "aarch64")]
            {
                if i + PREFETCH_DISTANCE_F64 < len {
                    core::arch::asm!(
                        "prfm pldl1keep, [{addr}]",
                        addr = in(reg) base.add(PREFETCH_DISTANCE_F64),
                        options(readonly, nostack)
                    );
                }
            }
            let a0 = vld1q_f32(base);
            let a1 = vld1q_f32(base.add(LANES_F32));
            let a2 = vld1q_f32(base.add(LANES_F32 * 2));
            let a3 = vld1q_f32(base.add(LANES_F32 * 3));
            let c0 = vaddq_f32(a0, scalar_v);
            let c1 = vaddq_f32(a1, scalar_v);
            let c2 = vaddq_f32(a2, scalar_v);
            let c3 = vaddq_f32(a3, scalar_v);
            let out_base = ptr_out.add(i);
            vst1q_f32(out_base, c0);
            vst1q_f32(out_base.add(LANES_F32), c1);
            vst1q_f32(out_base.add(LANES_F32 * 2), c2);
            vst1q_f32(out_base.add(LANES_F32 * 3), c3);
            i += unroll;
        }
        while i + LANES_F32 <= len {
            let a = vld1q_f32(ptr_in.add(i));
            let c = vaddq_f32(a, scalar_v);
            vst1q_f32(ptr_out.add(i), c);
            i += LANES_F32;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) + scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn scale_same_shape_f64(input: &[f64], factor: f64, out: &mut [f64]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let factor_v = vdupq_n_f64(factor);

        let mut i = 0usize;
        let unroll = LANES_F64 * 4;
        while i + unroll <= len {
            let base = ptr_in.add(i);
            #[cfg(target_arch = "aarch64")]
            {
                if i + PREFETCH_DISTANCE_F32 < len {
                    core::arch::asm!(
                        "prfm pldl1keep, [{addr}]",
                        addr = in(reg) base.add(PREFETCH_DISTANCE_F32),
                        options(readonly, nostack)
                    );
                }
            }
            let a0 = vld1q_f64(base);
            let a1 = vld1q_f64(base.add(LANES_F64));
            let a2 = vld1q_f64(base.add(LANES_F64 * 2));
            let a3 = vld1q_f64(base.add(LANES_F64 * 3));
            let c0 = vmulq_f64(a0, factor_v);
            let c1 = vmulq_f64(a1, factor_v);
            let c2 = vmulq_f64(a2, factor_v);
            let c3 = vmulq_f64(a3, factor_v);
            let out_base = ptr_out.add(i);
            vst1q_f64(out_base, c0);
            vst1q_f64(out_base.add(LANES_F64), c1);
            vst1q_f64(out_base.add(LANES_F64 * 2), c2);
            vst1q_f64(out_base.add(LANES_F64 * 3), c3);
            i += unroll;
        }
        while i + LANES_F64 <= len {
            let a = vld1q_f64(ptr_in.add(i));
            let c = vmulq_f64(a, factor_v);
            vst1q_f64(ptr_out.add(i), c);
            i += LANES_F64;
        }
        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) * factor;
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn scale_same_shape_f32(input: &[f32], factor: f32, out: &mut [f32]) {
        let len = input.len();
        let ptr_in = input.as_ptr();
        let ptr_out = out.as_mut_ptr();
        let factor_v = vdupq_n_f32(factor);

        let mut i = 0usize;
        let unroll = LANES_F32 * 4;
        while i + unroll <= len {
            let base = ptr_in.add(i);
            #[cfg(target_arch = "aarch64")]
            {
                if i + PREFETCH_DISTANCE_F32 < len {
                    core::arch::asm!(
                        "prfm pldl1keep, [{addr}]",
                        addr = in(reg) base.add(PREFETCH_DISTANCE_F32),
                        options(readonly, nostack)
                    );
                }
            }
            let a0 = vld1q_f32(base);
            let a1 = vld1q_f32(base.add(LANES_F32));
            let a2 = vld1q_f32(base.add(LANES_F32 * 2));
            let a3 = vld1q_f32(base.add(LANES_F32 * 3));
            let c0 = vmulq_f32(a0, factor_v);
            let c1 = vmulq_f32(a1, factor_v);
            let c2 = vmulq_f32(a2, factor_v);
            let c3 = vmulq_f32(a3, factor_v);
            let out_base = ptr_out.add(i);
            vst1q_f32(out_base, c0);
            vst1q_f32(out_base.add(LANES_F32), c1);
            vst1q_f32(out_base.add(LANES_F32 * 2), c2);
            vst1q_f32(out_base.add(LANES_F32 * 3), c3);
            i += unroll;
        }
        while i + LANES_F32 <= len {
            let a = vld1q_f32(ptr_in.add(i));
            let c = vmulq_f32(a, factor_v);
            vst1q_f32(ptr_out.add(i), c);
            i += LANES_F32;
        }

        while i < len {
            *ptr_out.add(i) = *ptr_in.add(i) * factor;
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_columnar_f64(
        input: &[f64],
        col_values: &[f64],
        rows: usize,
        cols: usize,
        out: &mut [f64],
    ) {
        for row in 0..rows {
            let scalar = *col_values.get_unchecked(row);
            let scalar_v = vdupq_n_f64(scalar);
            let base = row * cols;
            let mut col = 0usize;
            while col + LANES_F64 <= cols {
                let offset = base + col;
                let a = vld1q_f64(input.as_ptr().add(offset));
                let c = vaddq_f64(a, scalar_v);
                vst1q_f64(out.as_mut_ptr().add(offset), c);
                col += LANES_F64;
            }
            while col < cols {
                let offset = base + col;
                *out.get_unchecked_mut(offset) = *input.get_unchecked(offset) + scalar;
                col += 1;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_columnar_f32(
        input: &[f32],
        col_values: &[f32],
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) {
        let enable_prefetch = rows >= 1024 || cols >= 1024;
        for row in 0..rows {
            let scalar = *col_values.get_unchecked(row);
            let scalar_v = vdupq_n_f32(scalar);
            let base = row * cols;
            let mut col = 0usize;
            if enable_prefetch && row + PREFETCH_ROWS < rows {
                #[cfg(target_arch = "aarch64")]
                {
                    core::arch::asm!(
                        "prfm pldl1keep, [{addr}]",
                        addr = in(reg) input.as_ptr().add((row + PREFETCH_ROWS) * cols),
                        options(readonly, nostack)
                    );
                }
            }
            let unroll = LANES_F32 * 4;
            while col + unroll <= cols {
                let offset = base + col;
                let ptr_in = input.as_ptr().add(offset);
                #[cfg(target_arch = "aarch64")]
                {
                    if enable_prefetch && col + PREFETCH_DISTANCE_F32 < cols {
                        core::arch::asm!(
                            "prfm pldl1keep, [{addr}]",
                            addr = in(reg) ptr_in.add(PREFETCH_DISTANCE_F32),
                            options(readonly, nostack)
                        );
                    }
                }
                let a0 = vld1q_f32(ptr_in);
                let a1 = vld1q_f32(ptr_in.add(LANES_F32));
                let a2 = vld1q_f32(ptr_in.add(LANES_F32 * 2));
                let a3 = vld1q_f32(ptr_in.add(LANES_F32 * 3));
                let c0 = vaddq_f32(a0, scalar_v);
                let c1 = vaddq_f32(a1, scalar_v);
                let c2 = vaddq_f32(a2, scalar_v);
                let c3 = vaddq_f32(a3, scalar_v);
                let ptr_out = out.as_mut_ptr().add(offset);
                vst1q_f32(ptr_out, c0);
                vst1q_f32(ptr_out.add(LANES_F32), c1);
                vst1q_f32(ptr_out.add(LANES_F32 * 2), c2);
                vst1q_f32(ptr_out.add(LANES_F32 * 3), c3);
                col += unroll;
            }
            while col + LANES_F32 <= cols {
                let offset = base + col;
                let ptr_in = input.as_ptr().add(offset);
                let a = vld1q_f32(ptr_in);
                let c = vaddq_f32(a, scalar_v);
                vst1q_f32(out.as_mut_ptr().add(offset), c);
                col += LANES_F32;
            }
            while col < cols {
                let offset = base + col;
                *out.get_unchecked_mut(offset) = *input.get_unchecked(offset) + scalar;
                col += 1;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_assign_inplace_f64(acc: &mut [f64], row: &[f64]) {
        let len = acc.len();
        let mut i = 0usize;
        while i + LANES_F64 <= len {
            let a = vld1q_f64(acc.as_ptr().add(i));
            let b = vld1q_f64(row.as_ptr().add(i));
            let c = vaddq_f64(a, b);
            vst1q_f64(acc.as_mut_ptr().add(i), c);
            i += LANES_F64;
        }
        while i < len {
            *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn add_assign_inplace_f32(acc: &mut [f32], row: &[f32]) {
        let len = acc.len();
        let mut i = 0usize;
        while i + LANES_F32 <= len {
            let a = vld1q_f32(acc.as_ptr().add(i));
            let b = vld1q_f32(row.as_ptr().add(i));
            let c = vaddq_f32(a, b);
            vst1q_f32(acc.as_mut_ptr().add(i), c);
            i += LANES_F32;
        }
        while i < len {
            *acc.get_unchecked_mut(i) += *row.get_unchecked(i);
            i += 1;
        }
    }
}
