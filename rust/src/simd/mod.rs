#![allow(dead_code)]

#[cfg(target_arch = "x86_64")]
pub fn add_same_shape_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> bool {
    if lhs.len() != rhs.len() || lhs.len() != out.len() {
        return false;
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
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    const LANES_F64: usize = 2;
    const LANES_F32: usize = 4;

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
}
