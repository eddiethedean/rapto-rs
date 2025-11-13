use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use crate::{SCALE_BLAS_MIN_COLS, SCALE_BLAS_MIN_LEN, SCALE_BLAS_MIN_ROWS};

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
static THRESHOLDS: OnceLock<BlasThresholds> = OnceLock::new();

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
    scale_override().unwrap_or_else(|| backend_name() != "none")
}

pub fn axis0_enabled() -> bool {
    *AXIS0_ENABLED.get_or_init(|| {
        env_flag("RAPTORS_BLAS_AXIS0")
            .or_else(|| env_list_flag("RAPTORS_BLAS_OPS", "axis0"))
            .unwrap_or_else(|| backend_name() != "none")
    })
}

#[derive(Debug, Clone, Copy)]
pub enum BlasOp {
    Scale,
}

#[derive(Debug, Clone, Deserialize)]
struct ScaleThreshold {
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    min_len: Option<usize>,
    #[serde(default)]
    min_rows: Option<usize>,
    #[serde(default)]
    min_cols: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct BlasThresholds {
    #[serde(default)]
    scale: HashMap<String, ScaleThreshold>,
}

impl Default for BlasThresholds {
    fn default() -> Self {
        let mut scale = HashMap::new();
        scale.insert(
            "float32".to_string(),
            ScaleThreshold::default_for_dtype("float32"),
        );
        scale.insert(
            "float64".to_string(),
            ScaleThreshold::default_for_dtype("float64"),
        );
        Self { scale }
    }
}

impl BlasThresholds {
    fn with_defaults(mut self) -> Self {
        for dtype in ["float32", "float64"] {
            self.scale
                .entry(dtype.to_string())
                .or_insert_with(|| ScaleThreshold::default_for_dtype(dtype));
        }
        self
    }

    fn scale_for(&self, dtype: &str) -> ScaleThreshold {
        self.scale
            .get(dtype)
            .cloned()
            .unwrap_or_else(|| ScaleThreshold::default_for_dtype(dtype))
    }
}

impl ScaleThreshold {
    fn default_for_dtype(_dtype: &str) -> Self {
        Self {
            enabled: true,
            min_len: Some(SCALE_BLAS_MIN_LEN),
            min_rows: Some(SCALE_BLAS_MIN_ROWS),
            min_cols: Some(SCALE_BLAS_MIN_COLS),
        }
    }

    fn should_use(&self, len: usize, rows: usize, cols: usize) -> bool {
        if !self.enabled {
            return false;
        }
        if let Some(min) = self.min_len {
            if len < min {
                return false;
            }
        }
        if let Some(min) = self.min_rows {
            if rows < min {
                return false;
            }
        }
        if let Some(min) = self.min_cols {
            if cols < min {
                return false;
            }
        }
        true
    }
}

fn thresholds() -> &'static BlasThresholds {
    THRESHOLDS.get_or_init(load_thresholds)
}

fn load_thresholds() -> BlasThresholds {
    let path = env::var("RAPTORS_BLAS_THRESHOLDS").ok();
    if let Some(path) = path {
        if let Ok(text) = fs::read_to_string(Path::new(&path)) {
            if let Ok(parsed) = serde_json::from_str::<BlasThresholds>(&text) {
                return parsed.with_defaults();
            }
        }
    }
    BlasThresholds::default()
}

const fn default_true() -> bool {
    true
}

pub fn should_use(
    op: BlasOp,
    dtype: &'static str,
    len: usize,
    rows: usize,
    cols: usize,
    force: bool,
) -> bool {
    if force {
        return true;
    }
    match op {
        BlasOp::Scale => thresholds().scale_for(dtype).should_use(len, rows, cols),
    }
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
