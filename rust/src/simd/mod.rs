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
    const MAX_ACCUMULATORS_F64: usize = 6;
    const MAX_ACCUMULATORS_F32: usize = 8;

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
        for row in 0..rows {
            let scalar = *col_values.get_unchecked(row);
            let scalar_v = _mm256_set1_ps(scalar);
            let base = row * cols;
            let mut col = 0usize;
            while col + LANES_F32 <= cols {
                let offset = base + col;
                let a = _mm256_loadu_ps(input.as_ptr().add(offset));
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

    #[target_feature(enable = "neon")]
    pub unsafe fn reduce_sum_f64(input: &[f64], accumulators: usize) -> f64 {
        let len = input.len();
        if len == 0 {
            return 0.0;
        }

        let acc_count = accumulators.clamp(1, MAX_ACCUMULATORS_F64);
        let mut regs = [vdupq_n_f64(0.0); MAX_ACCUMULATORS_F64];
        let mut index = 0usize;
        let ptr = input.as_ptr();
        let step = acc_count * LANES_F64;

        while index + step <= len {
            let mut offset = index;
            for slot in 0..acc_count {
                let vec = vld1q_f64(ptr.add(offset));
                regs[slot] = vaddq_f64(regs[slot], vec);
                offset += LANES_F64;
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
        for slot in 0..acc_count {
            total += vaddvq_f64(regs[slot]);
        }
        while index < len {
            total += *ptr.add(index);
            index += 1;
        }
        total
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
        for row in 0..rows {
            let scalar = *col_values.get_unchecked(row);
            let scalar_v = vdupq_n_f32(scalar);
            let base = row * cols;
            let mut col = 0usize;
            while col + LANES_F32 <= cols {
                let offset = base + col;
                let a = vld1q_f32(input.as_ptr().add(offset));
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
