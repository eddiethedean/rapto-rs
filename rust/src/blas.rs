use std::env;
use std::sync::OnceLock;

pub trait BlasProvider: Send + Sync {
    fn name(&self) -> &'static str;

    fn sgemv_axis0_sum(&self, rows: usize, cols: usize, data: &[f32], out: &mut [f32]) -> bool {
        let _ = (rows, cols, data, out);
        false
    }

    fn dgemv_axis0_sum(&self, rows: usize, cols: usize, data: &[f64], out: &mut [f64]) -> bool {
        let _ = (rows, cols, data, out);
        false
    }

    fn sscal_f32(&self, len: usize, alpha: f32, data: &mut [f32]) -> bool {
        let _ = (len, alpha, data);
        false
    }

    fn dscal_f64(&self, len: usize, alpha: f64, data: &mut [f64]) -> bool {
        let _ = (len, alpha, data);
        false
    }

}

struct NoBlas;

impl BlasProvider for NoBlas {
    fn name(&self) -> &'static str {
        "none"
    }
}

#[cfg(target_os = "macos")]
struct AccelerateBlas;

#[cfg(target_os = "macos")]
impl BlasProvider for AccelerateBlas {
    fn name(&self) -> &'static str {
        "accelerate"
    }

    fn sgemv_axis0_sum(&self, rows: usize, cols: usize, data: &[f32], out: &mut [f32]) -> bool {
        if rows == 0 || cols == 0 || out.len() < cols {
            return false;
        }
        if rows > i32::MAX as usize || cols > i32::MAX as usize {
            return false;
        }
        if data.len() != rows.saturating_mul(cols) {
            return false;
        }
        let ones = vec![1.0f32; rows];
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANSPOSE,
                rows as i32,
                cols as i32,
                1.0,
                data.as_ptr(),
                rows as i32,
                ones.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr(),
                1,
            );
        }
        true
    }

    fn sscal_f32(&self, len: usize, alpha: f32, data: &mut [f32]) -> bool {
        if len == 0 || data.len() < len || len > i32::MAX as usize {
            return false;
        }
        unsafe {
            cblas_sscal(len as i32, alpha, data.as_mut_ptr(), 1);
        }
        true
    }

    fn dgemv_axis0_sum(&self, rows: usize, cols: usize, data: &[f64], out: &mut [f64]) -> bool {
        if rows == 0 || cols == 0 || out.len() < cols {
            return false;
        }
        if rows > i32::MAX as usize || cols > i32::MAX as usize {
            return false;
        }
        if data.len() != rows.saturating_mul(cols) {
            return false;
        }
        let ones = vec![1.0f64; rows];
        unsafe {
            cblas_dgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANSPOSE,
                rows as i32,
                cols as i32,
                1.0,
                data.as_ptr(),
                rows as i32,
                ones.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr(),
                1,
            );
        }
        true
    }

    fn dscal_f64(&self, len: usize, alpha: f64, data: &mut [f64]) -> bool {
        if len == 0 || data.len() < len || len > i32::MAX as usize {
            return false;
        }
        unsafe {
            cblas_dscal(len as i32, alpha, data.as_mut_ptr(), 1);
        }
        true
    }

}

static BACKEND: OnceLock<Box<dyn BlasProvider>> = OnceLock::new();
static SCALE_OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
static AXIS0_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn current_backend() -> &'static dyn BlasProvider {
    BACKEND.get_or_init(select_backend).as_ref()
}

pub fn backend_name() -> &'static str {
    current_backend().name()
}

pub fn scale_override() -> Option<bool> {
    *SCALE_OVERRIDE.get_or_init(|| {
        env_flag("RAPTORS_BLAS_SCALE").or_else(|| env_list_flag("RAPTORS_BLAS_OPS", "scale"))
    })
}

pub fn scale_enabled() -> bool {
    scale_override().unwrap_or(false)
}

pub fn axis0_enabled() -> bool {
    *AXIS0_ENABLED.get_or_init(|| {
        env_flag("RAPTORS_BLAS_AXIS0")
            .or_else(|| env_list_flag("RAPTORS_BLAS_OPS", "axis0"))
            .unwrap_or_else(|| backend_name() != "none")
    })
}

fn env_flag(name: &str) -> Option<bool> {
    let value = env::var(name).ok()?;
    parse_bool(&value)
}

fn env_list_flag(name: &str, token: &str) -> Option<bool> {
    let value = env::var(name).ok()?;
    let trimmed = value.trim().to_lowercase();
    if let Some(result) = parse_bool(&trimmed) {
        return Some(result);
    }
    if trimmed == "all" {
        return Some(true);
    }
    let enabled = trimmed
        .split(|c| matches!(c, ',' | ';' | ' '))
        .filter(|part| !part.is_empty())
        .any(|part| part == token);
    if enabled {
        Some(true)
    } else {
        None
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn select_backend() -> Box<dyn BlasProvider> {
    match env::var("RAPTORS_BLAS")
        .ok()
        .unwrap_or_else(|| "auto".to_string())
        .to_lowercase()
        .as_str()
    {
        "none" => Box::new(NoBlas),
        "accelerate" => build_accelerate(),
        "openblas" => build_openblas(),
        _ => {
            if let Some(accel) = build_accelerate_option() {
                accel
            } else if let Some(open) = build_openblas_option() {
                open
            } else {
                Box::new(NoBlas)
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn build_accelerate() -> Box<dyn BlasProvider> {
    Box::new(AccelerateBlas)
}

#[cfg(not(target_os = "macos"))]
fn build_accelerate() -> Box<dyn BlasProvider> {
    Box::new(NoBlas)
}

#[cfg(target_os = "macos")]
fn build_accelerate_option() -> Option<Box<dyn BlasProvider>> {
    Some(Box::new(AccelerateBlas))
}

#[cfg(not(target_os = "macos"))]
fn build_accelerate_option() -> Option<Box<dyn BlasProvider>> {
    None
}

#[cfg(all(feature = "openblas", not(target_os = "macos")))]
struct OpenBlasBackend;

#[cfg(all(feature = "openblas", not(target_os = "macos")))]
impl BlasProvider for OpenBlasBackend {
    fn name(&self) -> &'static str {
        "openblas"
    }

    fn sgemv_axis0_sum(&self, rows: usize, cols: usize, data: &[f32], out: &mut [f32]) -> bool {
        if rows == 0 || cols == 0 || out.len() < cols {
            return false;
        }
        if rows > i32::MAX as usize || cols > i32::MAX as usize {
            return false;
        }
        if data.len() != rows.saturating_mul(cols) {
            return false;
        }
        let ones = vec![1.0f32; rows];
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANSPOSE,
                rows as i32,
                cols as i32,
                1.0,
                data.as_ptr(),
                rows as i32,
                ones.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr(),
                1,
            );
        }
        true
    }

    fn dgemv_axis0_sum(&self, rows: usize, cols: usize, data: &[f64], out: &mut [f64]) -> bool {
        if rows == 0
            || cols == 0
            || out.len() < cols
            || rows > i32::MAX as usize
            || cols > i32::MAX as usize
            || data.len() != rows.saturating_mul(cols)
        {
            return false;
        }
        let ones = vec![1.0f64; rows];
        unsafe {
            cblas_dgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_TRANSPOSE,
                rows as i32,
                cols as i32,
                1.0,
                data.as_ptr(),
                rows as i32,
                ones.as_ptr(),
                1,
                0.0,
                out.as_mut_ptr(),
                1,
            );
        }
        true
    }

    fn sscal_f32(&self, len: usize, alpha: f32, data: &mut [f32]) -> bool {
        if len == 0 || data.len() < len || len > i32::MAX as usize {
            return false;
        }
        unsafe {
            cblas_sscal(len as i32, alpha, data.as_mut_ptr(), 1);
        }
        true
    }

    fn dscal_f64(&self, len: usize, alpha: f64, data: &mut [f64]) -> bool {
        if len == 0 || data.len() < len || len > i32::MAX as usize {
            return false;
        }
        unsafe {
            cblas_dscal(len as i32, alpha, data.as_mut_ptr(), 1);
        }
        true
    }

}

#[cfg(not(all(feature = "openblas", not(target_os = "macos"))))]
fn build_openblas() -> Box<dyn BlasProvider> {
    Box::new(NoBlas)
}

#[cfg(all(feature = "openblas", not(target_os = "macos")))]
fn build_openblas() -> Box<dyn BlasProvider> {
    Box::new(OpenBlasBackend)
}

#[cfg(all(feature = "openblas", not(target_os = "macos")))]
fn build_openblas_option() -> Option<Box<dyn BlasProvider>> {
    Some(Box::new(OpenBlasBackend))
}

#[cfg(not(all(feature = "openblas", not(target_os = "macos"))))]
fn build_openblas_option() -> Option<Box<dyn BlasProvider>> {
    None
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );

    fn cblas_dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );

    fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32);
    fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
}

#[cfg(all(feature = "openblas", not(target_os = "macos")))]
#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );

    fn cblas_dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );

    fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32);
    fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_TRANSPOSE: i32 = 112;
