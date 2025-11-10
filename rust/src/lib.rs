use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    convert::TryInto,
    env,
    ops::Add,
    slice,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

mod reduce;
mod simd;
mod tiling;

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use numpy::{Element, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyList, PyModule, PySequence, PyTuple};
use pyo3::FromPyObject;
use pyo3::IntoPy;
use pyo3::PyObject;

type PyResultF64 = PyResult<f64>;

trait NumericElement:
    Copy + Clone + Send + Sync + 'static + Add<Output = Self> + Zero + One + PartialEq
{
    const DTYPE_NAME: &'static str;
    const SUPPORTS_FRACTIONS: bool;
    const NUMPY_TYPESTR: &'static str;

    fn try_to_f64(self) -> Option<f64>;
    fn try_from_f64(value: f64) -> Option<Self>;

    fn try_scale(self, factor: f64) -> Option<Self> {
        let base = self.try_to_f64()?;
        Self::try_from_f64(base * factor)
    }

    fn scalar_sum(slice: &[Self]) -> Option<f64> {
        slice
            .iter()
            .try_fold(0.0, |acc, &value| value.try_to_f64().map(|v| acc + v))
    }

    fn simd_sum(slice: &[Self]) -> Option<f64> {
        Self::scalar_sum(slice)
    }

    fn scalar_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        for ((dest, &l), &r) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
            *dest = l + r;
        }
    }

    fn simd_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        Self::scalar_add(lhs, rhs, out);
    }

    fn scalar_scale(slice: &[Self], factor: f64, out: &mut [Self]) -> Result<(), ()> {
        for (dest, &value) in out.iter_mut().zip(slice.iter()) {
            *dest = value.try_scale(factor).ok_or(())?;
        }
        Ok(())
    }

    fn simd_scale(slice: &[Self], factor: f64, out: &mut [Self]) -> Result<(), ()> {
        Self::scalar_scale(slice, factor, out)
    }
}

fn simd_is_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if let Ok(flag) = env::var("RAPTORS_SIMD") {
            match flag.trim().to_ascii_lowercase().as_str() {
                "0" | "false" | "off" => return false,
                "1" | "true" | "on" => return true,
                _ => {}
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            return std::arch::is_x86_feature_detected!("sse2");
        }
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    })
}

const PARALLEL_MIN_ELEMENTS: usize = 1 << 15;
const ADAPTIVE_SAMPLE_WINDOW: usize = 9;
const TRACKED_DTYPES: &[&str] = &["float64", "float32", "int32"];

#[derive(Default)]
struct AdaptiveThreadingState {
    throughput: HashMap<&'static str, VecDeque<f64>>,
    last_event: Option<ThreadingEvent>,
}

#[derive(Clone, Default)]
struct ThreadingSnapshot {
    thresholds: Vec<ThresholdEntry>,
    last_event: Option<ThreadingEvent>,
}

#[derive(Clone, Default)]
struct ThresholdEntry {
    dtype: &'static str,
    median_elements_per_ms: f64,
    sample_count: usize,
    recommended_cutover: Option<usize>,
    samples: Vec<f64>,
}

#[derive(Clone, Default)]
struct ThreadingEvent {
    dtype: &'static str,
    elements: usize,
    duration_ms: f64,
    tiles: usize,
    partial_buffer: usize,
    parallel: bool,
    operation: &'static str,
}

static ADAPTIVE_STATE: OnceLock<Mutex<AdaptiveThreadingState>> = OnceLock::new();

fn adaptive_state() -> &'static Mutex<AdaptiveThreadingState> {
    ADAPTIVE_STATE.get_or_init(|| Mutex::new(AdaptiveThreadingState::default()))
}

impl AdaptiveThreadingState {
    fn record(&mut self, event: ThreadingEvent) {
        if event.parallel && event.duration_ms > 0.0 && event.elements > 0 {
            let throughput = event.elements as f64 / event.duration_ms;
            if throughput.is_finite() && throughput > 0.0 {
                let deque = self.throughput.entry(event.dtype).or_default();
                if deque.len() == ADAPTIVE_SAMPLE_WINDOW {
                    deque.pop_front();
                }
                deque.push_back(throughput);
            }
        }
        self.last_event = Some(event);
    }

    fn median(&self, dtype: &'static str) -> Option<f64> {
        let values = self.throughput.get(dtype)?;
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 && mid > 0 {
            Some((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Some(sorted[mid])
        }
    }

    fn recommend_cutover(&self, dtype: &'static str) -> Option<usize> {
        let median = self.median(dtype)?;
        if median <= 0.0 {
            return None;
        }
        let target = target_latency_ms(dtype);
        if target <= 0.0 {
            return None;
        }
        let cutover = (median * target).ceil() as usize;
        Some(cutover.max(1))
    }

    fn snapshot(&self) -> ThreadingSnapshot {
        let mut thresholds = Vec::with_capacity(TRACKED_DTYPES.len());
        for &dtype in TRACKED_DTYPES {
            let samples_list = self
                .throughput
                .get(dtype)
                .map(|deque| deque.iter().copied().collect::<Vec<f64>>())
                .unwrap_or_default();
            let median = self.median(dtype).unwrap_or(0.0);
            let recommended = self.recommend_cutover(dtype);
            thresholds.push(ThresholdEntry {
                dtype,
                median_elements_per_ms: median,
                sample_count: samples_list.len(),
                recommended_cutover: recommended,
                samples: samples_list,
            });
        }
        ThreadingSnapshot {
            thresholds,
            last_event: self.last_event.clone(),
        }
    }
}

fn target_latency_ms(dtype: &'static str) -> f64 {
    match dtype {
        "float64" => 0.28,
        "float32" => 0.32,
        _ => 0.40,
    }
}

fn baseline_cutover(dtype: &'static str) -> usize {
    match dtype {
        "float64" => PARALLEL_MIN_ELEMENTS,
        "float32" => PARALLEL_MIN_ELEMENTS * 3 / 2,
        _ => PARALLEL_MIN_ELEMENTS * 2,
    }
    .max(PARALLEL_MIN_ELEMENTS)
}

fn dtype_dim_threshold(dtype: &'static str) -> (usize, usize) {
    match dtype {
        "float64" => (64, 64),
        "float32" => (128, 64),
        _ => (128, 128),
    }
}

fn dimension_gate(rows: usize, cols: usize, dtype: &'static str) -> bool {
    if rows == 0 || cols == 0 {
        return false;
    }
    let (min_rows, min_cols) = dtype_dim_threshold(dtype);
    rows >= min_rows && cols >= min_cols
}

fn record_threading_event(
    dtype: &'static str,
    elements: usize,
    elapsed: Duration,
    outcome: &reduce::tiled::ReduceOutcome,
    operation: &'static str,
) {
    if elements == 0 {
        return;
    }
    let duration_ms = elapsed.as_secs_f64() * 1_000.0;
    let event = ThreadingEvent {
        dtype,
        elements,
        duration_ms,
        tiles: outcome.tiles_processed,
        partial_buffer: outcome.partial_buffer,
        parallel: outcome.parallel,
        operation,
    };
    if let Ok(mut guard) = adaptive_state().lock() {
        guard.record(event);
    }
}

fn record_axis_event(
    dtype: &'static str,
    rows: usize,
    cols: usize,
    elapsed: Duration,
    parallel: bool,
    operation: &'static str,
) {
    let outcome = reduce::tiled::ReduceOutcome {
        value: 0.0,
        tiles_processed: rows,
        parallel,
        partial_buffer: cols,
    };
    record_threading_event(
        dtype,
        rows.saturating_mul(cols),
        elapsed,
        &outcome,
        operation,
    );
}
fn threading_snapshot() -> ThreadingSnapshot {
    adaptive_state()
        .lock()
        .map(|guard| guard.snapshot())
        .unwrap_or_default()
}

pub(crate) fn thread_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let explicit = env::var("RAPTORS_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&threads| threads > 1);
        let builder = if let Some(threads) = explicit {
            rayon::ThreadPoolBuilder::new().num_threads(threads)
        } else {
            rayon::ThreadPoolBuilder::new()
        };
        match builder.build() {
            Ok(pool) if pool.current_num_threads() > 1 => Some(pool),
            _ => None,
        }
    })
    .as_ref()
}

fn try_parallel<F>(work_len: usize, f: F) -> bool
where
    F: FnOnce() + Send,
{
    if work_len < PARALLEL_MIN_ELEMENTS {
        return false;
    }
    if let Some(pool) = thread_pool() {
        pool.install(f);
        true
    } else {
        false
    }
}

fn should_parallelize(rows: usize, cols: usize, dtype: &'static str) -> bool {
    if !dimension_gate(rows, cols, dtype) {
        return false;
    }
    let elements = rows.saturating_mul(cols);
    let baseline = baseline_cutover(dtype);
    let threshold = adaptive_state()
        .lock()
        .ok()
        .and_then(|guard| guard.recommend_cutover(dtype))
        .map(|value| value.max(baseline))
        .unwrap_or(baseline);
    elements >= threshold
}

#[derive(Clone, Copy)]
enum AxisKind {
    Axis0,
    Axis1,
}

fn axis_parallel_cutover(axis: AxisKind, dtype: &'static str) -> usize {
    match (axis, dtype) {
        (AxisKind::Axis0, "float64") => 2048,
        (AxisKind::Axis0, "float32") => 4096,
        (AxisKind::Axis1, "float64") => 1536,
        (AxisKind::Axis1, "float32") => 2048,
        _ => 2048,
    }
}

fn should_parallelize_axis(axis: AxisKind, rows: usize, cols: usize, dtype: &'static str) -> bool {
    let primary = match axis {
        AxisKind::Axis0 => rows,
        AxisKind::Axis1 => cols,
    };
    if primary < axis_parallel_cutover(axis, dtype) {
        return false;
    }
    should_parallelize(rows, cols, dtype)
}

struct AxisOutcome {
    values: Vec<f64>,
    parallel: bool,
}

fn parallel_scale_f64(
    input: &[f64],
    factor: f64,
    out: &mut [f64],
    rows: usize,
    cols: usize,
) -> bool {
    if rows <= 1 || cols == 0 || input.len() != out.len() || input.len() != rows * cols {
        return false;
    }
    if let Some(pool) = thread_pool() {
        pool.install(|| {
            use rayon::prelude::*;
            input
                .par_chunks(cols)
                .zip(out.par_chunks_mut(cols))
                .for_each(|(src_row, dst_row)| {
                    if !simd::scale_same_shape_f64(src_row, factor, dst_row) {
                        for (dst, &value) in dst_row.iter_mut().zip(src_row.iter()) {
                            *dst = value * factor;
                        }
                    }
                });
        });
        true
    } else {
        false
    }
}

fn parallel_scale_f32(
    input: &[f32],
    factor: f64,
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> bool {
    if rows <= 1 || cols == 0 || input.len() != out.len() || input.len() != rows * cols {
        return false;
    }
    let factor_f32 = factor as f32;
    if let Some(pool) = thread_pool() {
        pool.install(|| {
            use rayon::prelude::*;
            input
                .par_chunks(cols)
                .zip(out.par_chunks_mut(cols))
                .for_each(|(src_row, dst_row)| {
                    if !simd::scale_same_shape_f32(src_row, factor_f32, dst_row) {
                        for (dst, &value) in dst_row.iter_mut().zip(src_row.iter()) {
                            *dst = value * factor_f32;
                        }
                    }
                });
        });
        true
    } else {
        false
    }
}

#[inline]
fn add_assign_f64(acc: &mut [f64], row: &[f64]) {
    if simd::add_assign_inplace_f64(acc, row) {
        return;
    }
    for (dst, &value) in acc.iter_mut().zip(row.iter()) {
        *dst += value;
    }
}

#[inline]
fn accumulate_row_f32(acc: &mut [f32], row: &[f32]) {
    if simd::add_assign_inplace_f32(acc, row) {
        return;
    }
    for (dst, &value) in acc.iter_mut().zip(row.iter()) {
        *dst += value;
    }
}

impl NumericElement for f64 {
    const DTYPE_NAME: &'static str = "float64";
    const SUPPORTS_FRACTIONS: bool = true;
    const NUMPY_TYPESTR: &'static str = "<f8";

    fn try_to_f64(self) -> Option<f64> {
        Some(self)
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        Some(value)
    }

    fn simd_sum(slice: &[Self]) -> Option<f64> {
        if !simd_is_enabled() {
            return Self::scalar_sum(slice);
        }
        use wide::f64x2;

        let lanes = 2;
        let mut acc = f64x2::splat(0.0);

        let mut chunks = slice.chunks_exact(lanes);
        for chunk in chunks.by_ref() {
            let arr: [f64; 2] = chunk.try_into().unwrap();
            let v = f64x2::from(arr);
            acc = acc + v;
        }

        let acc_array: [f64; 2] = acc.into();
        let mut total = acc_array.iter().sum();

        for &value in chunks.remainder() {
            total += value;
        }
        Some(total)
    }

    fn simd_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        if !simd_is_enabled() {
            return Self::scalar_add(lhs, rhs, out);
        }
        use wide::f64x2;

        let lanes = 2;
        let mut lhs_chunks = lhs.chunks_exact(lanes);
        let mut rhs_chunks = rhs.chunks_exact(lanes);
        let mut out_chunks = out.chunks_exact_mut(lanes);

        for ((l_chunk, r_chunk), out_chunk) in lhs_chunks
            .by_ref()
            .zip(rhs_chunks.by_ref())
            .zip(out_chunks.by_ref())
        {
            let l_arr: [f64; 2] = l_chunk.try_into().unwrap();
            let r_arr: [f64; 2] = r_chunk.try_into().unwrap();
            let a = f64x2::from(l_arr);
            let b = f64x2::from(r_arr);
            let result: [f64; 2] = (a + b).into();
            out_chunk.copy_from_slice(&result);
        }

        let lhs_rem = lhs_chunks.remainder();
        let rhs_rem = rhs_chunks.remainder();
        let out_rem = out_chunks.into_remainder();
        for i in 0..lhs_rem.len() {
            out_rem[i] = lhs_rem[i] + rhs_rem[i];
        }
    }

    fn simd_scale(slice: &[Self], factor: f64, out: &mut [Self]) -> Result<(), ()> {
        if !simd_is_enabled() {
            return Self::scalar_scale(slice, factor, out);
        }
        use wide::f64x2;

        let lanes = 2;
        let factor_vec = f64x2::splat(factor);

        let mut in_chunks = slice.chunks_exact(lanes);
        let mut out_chunks = out.chunks_exact_mut(lanes);

        for (in_chunk, out_chunk) in in_chunks.by_ref().zip(out_chunks.by_ref()) {
            let arr: [f64; 2] = in_chunk.try_into().unwrap();
            let v = f64x2::from(arr);
            let result: [f64; 2] = (v * factor_vec).into();
            out_chunk.copy_from_slice(&result);
        }

        let in_rem = in_chunks.remainder();
        let out_rem = out_chunks.into_remainder();
        for i in 0..in_rem.len() {
            out_rem[i] = in_rem[i] * factor;
        }
        Ok(())
    }
}

impl NumericElement for f32 {
    const DTYPE_NAME: &'static str = "float32";
    const SUPPORTS_FRACTIONS: bool = true;
    const NUMPY_TYPESTR: &'static str = "<f4";

    fn try_to_f64(self) -> Option<f64> {
        Some(self as f64)
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        Some(value as f32)
    }

    fn simd_sum(slice: &[Self]) -> Option<f64> {
        if !simd_is_enabled() {
            return Self::scalar_sum(slice);
        }
        use wide::f32x4;

        let lanes = 4;
        let mut acc = f32x4::splat(0.0);

        let mut chunks = slice.chunks_exact(lanes);
        for chunk in chunks.by_ref() {
            let arr: [f32; 4] = chunk.try_into().unwrap();
            let v = f32x4::from(arr);
            acc = acc + v;
        }

        let acc_array: [f32; 4] = acc.into();
        let mut total = acc_array.iter().fold(0.0_f64, |sum, &v| sum + v as f64);

        for &value in chunks.remainder() {
            total += value as f64;
        }
        Some(total)
    }

    fn simd_add(lhs: &[Self], rhs: &[Self], out: &mut [Self]) {
        if !simd_is_enabled() {
            return Self::scalar_add(lhs, rhs, out);
        }
        use wide::f32x4;

        let lanes = 4;
        let mut lhs_chunks = lhs.chunks_exact(lanes);
        let mut rhs_chunks = rhs.chunks_exact(lanes);
        let mut out_chunks = out.chunks_exact_mut(lanes);

        for ((l_chunk, r_chunk), out_chunk) in lhs_chunks
            .by_ref()
            .zip(rhs_chunks.by_ref())
            .zip(out_chunks.by_ref())
        {
            let l_arr: [f32; 4] = l_chunk.try_into().unwrap();
            let r_arr: [f32; 4] = r_chunk.try_into().unwrap();
            let a = f32x4::from(l_arr);
            let b = f32x4::from(r_arr);
            let result: [f32; 4] = (a + b).into();
            out_chunk.copy_from_slice(&result);
        }

        let lhs_rem = lhs_chunks.remainder();
        let rhs_rem = rhs_chunks.remainder();
        let out_rem = out_chunks.into_remainder();
        for i in 0..lhs_rem.len() {
            out_rem[i] = lhs_rem[i] + rhs_rem[i];
        }
    }

    fn simd_scale(slice: &[Self], factor: f64, out: &mut [Self]) -> Result<(), ()> {
        if !simd_is_enabled() {
            return Self::scalar_scale(slice, factor, out);
        }
        use wide::f32x4;

        let lanes = 4;
        let factor_vec = f32x4::splat(factor as f32);

        let mut in_chunks = slice.chunks_exact(lanes);
        let mut out_chunks = out.chunks_exact_mut(lanes);

        for (in_chunk, out_chunk) in in_chunks.by_ref().zip(out_chunks.by_ref()) {
            let arr: [f32; 4] = in_chunk.try_into().unwrap();
            let v = f32x4::from(arr);
            let result: [f32; 4] = (v * factor_vec).into();
            out_chunk.copy_from_slice(&result);
        }

        let in_rem = in_chunks.remainder();
        let out_rem = out_chunks.into_remainder();
        for i in 0..in_rem.len() {
            out_rem[i] = in_rem[i] * factor as f32;
        }
        Ok(())
    }
}

impl NumericElement for i32 {
    const DTYPE_NAME: &'static str = "int32";
    const SUPPORTS_FRACTIONS: bool = false;
    const NUMPY_TYPESTR: &'static str = "<i4";

    fn try_to_f64(self) -> Option<f64> {
        ToPrimitive::to_f64(&self)
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        if value.is_finite() {
            FromPrimitive::from_f64(value)
        } else {
            None
        }
    }

    fn try_scale(self, factor: f64) -> Option<Self> {
        if factor.fract() != 0.0 {
            return None;
        }
        let base = self.try_to_f64()?;
        Self::try_from_f64(base * factor)
    }
}

#[derive(Clone)]
enum NumericStorage<T> {
    Owned(Vec<T>),
    Borrowed {
        ptr: *const T,
        len: usize,
        _owner: Py<PyAny>,
    },
}

#[derive(Clone)]
struct NumericArray<T: NumericElement> {
    storage: NumericStorage<T>,
    shape: Vec<usize>,
}

impl<T> NumericArray<T>
where
    T: NumericElement + Element + for<'py> FromPyObject<'py>,
{
    fn is_contiguous(&self) -> bool {
        match &self.storage {
            NumericStorage::Owned(_) | NumericStorage::Borrowed { .. } => true,
        }
    }

    fn matrix_dims(&self) -> (usize, usize) {
        match self.shape.as_slice() {
            [] => (1, self.data_len()),
            [cols] => (1, *cols),
            [rows, cols] => (*rows, *cols),
            shape if !shape.is_empty() => {
                let rows = shape[0];
                let cols = self.data_len() / rows.max(1);
                (rows, cols.max(1))
            }
            _ => (1, self.data_len()),
        }
    }

    fn global_reduce(&self, op: reduce::tiled::GlobalOp) -> Option<f64> {
        if !self.is_contiguous() {
            return None;
        }
        let len = self.data_len();
        if len == 0 {
            return Some(0.0);
        }
        let (rows, cols) = self.matrix_dims();
        match T::DTYPE_NAME {
            "float64" => {
                let data = unsafe {
                    std::slice::from_raw_parts(self.data_slice().as_ptr() as *const f64, len)
                };
                let allow_parallel = should_parallelize(rows, cols, T::DTYPE_NAME);
                let pool = if allow_parallel { thread_pool() } else { None };
                let start = Instant::now();
                let outcome =
                    reduce::tiled::reduce_full_f64(data, rows, cols, op, pool, allow_parallel);
                let elapsed = start.elapsed();
                record_threading_event(
                    T::DTYPE_NAME,
                    len,
                    elapsed,
                    &outcome,
                    match op {
                        reduce::tiled::GlobalOp::Sum => "sum",
                        reduce::tiled::GlobalOp::Mean => "mean",
                    },
                );
                Some(outcome.value)
            }
            "float32" => {
                let data = unsafe {
                    std::slice::from_raw_parts(self.data_slice().as_ptr() as *const f32, len)
                };
                let allow_parallel = should_parallelize(rows, cols, T::DTYPE_NAME);
                let pool = if allow_parallel { thread_pool() } else { None };
                let start = Instant::now();
                let outcome =
                    reduce::tiled::reduce_full_f32(data, rows, cols, op, pool, allow_parallel);
                let elapsed = start.elapsed();
                record_threading_event(
                    T::DTYPE_NAME,
                    len,
                    elapsed,
                    &outcome,
                    match op {
                        reduce::tiled::GlobalOp::Sum => "sum",
                        reduce::tiled::GlobalOp::Mean => "mean",
                    },
                );
                Some(outcome.value)
            }
            _ => None,
        }
    }

    fn try_simd_add_same_shape(&self, other: &NumericArray<T>) -> Option<NumericArray<T>> {
        if self.shape != other.shape {
            return None;
        }
        if !self.is_contiguous() || !other.is_contiguous() {
            return None;
        }
        let len = self.data_len();
        if len != other.data_len() {
            return None;
        }
        let lhs_slice = self.data_slice();
        let rhs_slice = other.data_slice();
        let mut data = vec![T::zero(); len];
        if T::DTYPE_NAME == "float64" {
            let lhs = unsafe { std::slice::from_raw_parts(lhs_slice.as_ptr() as *const f64, len) };
            let rhs = unsafe { std::slice::from_raw_parts(rhs_slice.as_ptr() as *const f64, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, len) };
            if simd::add_same_shape_f64(lhs, rhs, out) {
                return Some(NumericArray::new_owned(data, self.shape.clone()));
            }
        } else if T::DTYPE_NAME == "float32" {
            let lhs = unsafe { std::slice::from_raw_parts(lhs_slice.as_ptr() as *const f32, len) };
            let rhs = unsafe { std::slice::from_raw_parts(rhs_slice.as_ptr() as *const f32, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, len) };
            if simd::add_same_shape_f32(lhs, rhs, out) {
                return Some(NumericArray::new_owned(data, self.shape.clone()));
            }
        }
        if try_parallel(len, || {
            use rayon::prelude::*;
            data.par_iter_mut()
                .zip(lhs_slice.par_iter())
                .zip(rhs_slice.par_iter())
                .for_each(|((out, lhs), rhs)| {
                    *out = *lhs + *rhs;
                });
        }) {
            return Some(NumericArray::new_owned(data, self.shape.clone()));
        }
        T::simd_add(lhs_slice, rhs_slice, &mut data);
        Some(NumericArray::new_owned(data, self.shape.clone()))
    }

    fn try_simd_add_row(&self, other: &NumericArray<T>) -> Option<NumericArray<T>> {
        if self.shape.len() != 2 || !self.is_contiguous() || !other.is_contiguous() {
            return None;
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        match other.shape.as_slice() {
            [len] if *len == cols => {
                let lhs_slice = self.data_slice();
                let rhs_slice = other.data_slice();
                let mut data = vec![T::zero(); lhs_slice.len()];
                if T::DTYPE_NAME == "float64" {
                    let lhs = unsafe {
                        std::slice::from_raw_parts(
                            lhs_slice.as_ptr() as *const f64,
                            lhs_slice.len(),
                        )
                    };
                    let rhs = unsafe {
                        std::slice::from_raw_parts(
                            rhs_slice.as_ptr() as *const f64,
                            rhs_slice.len(),
                        )
                    };
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, data.len())
                    };
                    let mut used = true;
                    for row in 0..rows {
                        let start = row * cols;
                        let end = start + cols;
                        if !simd::add_same_shape_f64(&lhs[start..end], rhs, &mut out[start..end]) {
                            used = false;
                            break;
                        }
                    }
                    if used {
                        return Some(NumericArray::new_owned(data, self.shape.clone()));
                    }
                } else if T::DTYPE_NAME == "float32" {
                    let lhs = unsafe {
                        std::slice::from_raw_parts(
                            lhs_slice.as_ptr() as *const f32,
                            lhs_slice.len(),
                        )
                    };
                    let rhs = unsafe {
                        std::slice::from_raw_parts(
                            rhs_slice.as_ptr() as *const f32,
                            rhs_slice.len(),
                        )
                    };
                    let out = unsafe {
                        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len())
                    };
                    let mut used = true;
                    for row in 0..rows {
                        let start = row * cols;
                        let end = start + cols;
                        if !simd::add_same_shape_f32(&lhs[start..end], rhs, &mut out[start..end]) {
                            used = false;
                            break;
                        }
                    }
                    if used {
                        return Some(NumericArray::new_owned(data, self.shape.clone()));
                    }
                }
                if try_parallel(lhs_slice.len(), || {
                    use rayon::prelude::*;
                    data.par_chunks_mut(cols)
                        .zip(lhs_slice.par_chunks(cols))
                        .for_each(|(out_row, lhs_row)| {
                            for (idx, dst) in out_row.iter_mut().enumerate() {
                                *dst = lhs_row[idx] + rhs_slice[idx];
                            }
                        });
                }) {
                    return Some(NumericArray::new_owned(data, self.shape.clone()));
                }
                for row in 0..rows {
                    let start = row * cols;
                    let end = start + cols;
                    T::simd_add(&lhs_slice[start..end], rhs_slice, &mut data[start..end]);
                }
                Some(NumericArray::new_owned(data, self.shape.clone()))
            }
            _ => None,
        }
    }

    fn try_simd_add_column(&self, other: &NumericArray<T>) -> Option<NumericArray<T>> {
        if self.shape.len() != 2 || !self.is_contiguous() || !other.is_contiguous() {
            return None;
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let valid = matches!(other.shape.as_slice(), [len] if *len == rows)
            || matches!(other.shape.as_slice(), [len, 1] if *len == rows);
        if !valid {
            return None;
        }
        let lhs_slice = self.data_slice();
        let rhs_slice = other.data_slice();
        if rhs_slice.len() != rows {
            return None;
        }
        let mut data = vec![T::zero(); lhs_slice.len()];
        if cols == 0 {
            return Some(NumericArray::new_owned(data, self.shape.clone()));
        }
        let spec = tiling::TileSpec::for_shape(rows, cols);
        if T::DTYPE_NAME == "float64" {
            let lhs = unsafe {
                std::slice::from_raw_parts(lhs_slice.as_ptr() as *const f64, lhs_slice.len())
            };
            let rhs = unsafe {
                std::slice::from_raw_parts(rhs_slice.as_ptr() as *const f64, rhs_slice.len())
            };
            let out = unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, data.len())
            };
            if simd::add_column_broadcast_f64(lhs, rhs, rows, cols, out) {
                return Some(NumericArray::new_owned(data, self.shape.clone()));
            }
        } else if T::DTYPE_NAME == "float32" {
            let lhs = unsafe {
                std::slice::from_raw_parts(lhs_slice.as_ptr() as *const f32, lhs_slice.len())
            };
            let rhs = unsafe {
                std::slice::from_raw_parts(rhs_slice.as_ptr() as *const f32, rhs_slice.len())
            };
            let out = unsafe {
                std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len())
            };
            if simd::add_column_broadcast_f32(lhs, rhs, rows, cols, out) {
                return Some(NumericArray::new_owned(data, self.shape.clone()));
            }
        }
        if try_parallel(lhs_slice.len(), || {
            use rayon::prelude::*;
            match T::DTYPE_NAME {
                "float64" => {
                    let lhs_rows = unsafe {
                        std::slice::from_raw_parts(
                            lhs_slice.as_ptr() as *const f64,
                            lhs_slice.len(),
                        )
                    };
                    let rhs_vals = unsafe {
                        std::slice::from_raw_parts(
                            rhs_slice.as_ptr() as *const f64,
                            rhs_slice.len(),
                        )
                    };
                    data.par_chunks_mut(cols)
                        .zip(lhs_rows.par_chunks(cols))
                        .zip(rhs_vals.par_iter().copied())
                        .for_each(|((out_row, lhs_row), scalar)| {
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(
                                    out_row.as_mut_ptr() as *mut f64,
                                    out_row.len(),
                                )
                            };
                            if !simd::add_row_scalar_f64(lhs_row, scalar, out) {
                                for (dst, &src) in out.iter_mut().zip(lhs_row.iter()) {
                                    *dst = src + scalar;
                                }
                            }
                        });
                }
                "float32" => {
                    let lhs_rows = unsafe {
                        std::slice::from_raw_parts(
                            lhs_slice.as_ptr() as *const f32,
                            lhs_slice.len(),
                        )
                    };
                    let rhs_vals = unsafe {
                        std::slice::from_raw_parts(
                            rhs_slice.as_ptr() as *const f32,
                            rhs_slice.len(),
                        )
                    };
                    data.par_chunks_mut(cols)
                        .zip(lhs_rows.par_chunks(cols))
                        .zip(rhs_vals.par_iter().copied())
                        .for_each(|((out_row, lhs_row), scalar)| {
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(
                                    out_row.as_mut_ptr() as *mut f32,
                                    out_row.len(),
                                )
                            };
                            if !simd::add_row_scalar_f32(lhs_row, scalar, out) {
                                for (dst, &src) in out.iter_mut().zip(lhs_row.iter()) {
                                    *dst = src + scalar;
                                }
                            }
                        });
                }
                _ => {
                    let batch_rows = spec.row_block.max(1);
                    let chunk_len = cols * batch_rows;
                    data.par_chunks_mut(chunk_len.max(cols))
                        .zip(lhs_slice.par_chunks(chunk_len.max(cols)))
                        .enumerate()
                        .for_each(|(chunk_index, (out_chunk, lhs_chunk))| {
                            let row_start = chunk_index * batch_rows;
                            let rows_in_chunk = rows.saturating_sub(row_start).min(batch_rows);
                            for row_offset in 0..rows_in_chunk {
                                let rhs_value = rhs_slice[row_start + row_offset];
                                let start = row_offset * cols;
                                let end = start + cols;
                                let src_row = &lhs_chunk[start..end];
                                let dst_row = &mut out_chunk[start..end];
                                for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()) {
                                    *dst = src + rhs_value;
                                }
                            }
                        });
                }
            }
        }) {
            return Some(NumericArray::new_owned(data, self.shape.clone()));
        }
        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            if T::DTYPE_NAME == "float64" {
                let lhs = unsafe {
                    std::slice::from_raw_parts(lhs_slice[start..end].as_ptr() as *const f64, cols)
                };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(data[start..end].as_mut_ptr() as *mut f64, cols)
                };
                if simd::add_row_scalar_f64(lhs, rhs_slice[row].try_to_f64().unwrap(), out) {
                    continue;
                }
            } else if T::DTYPE_NAME == "float32" {
                let lhs = unsafe {
                    std::slice::from_raw_parts(lhs_slice[start..end].as_ptr() as *const f32, cols)
                };
                let out = unsafe {
                    std::slice::from_raw_parts_mut(data[start..end].as_mut_ptr() as *mut f32, cols)
                };
                if simd::add_row_scalar_f32(lhs, rhs_slice[row].try_to_f64().unwrap() as f32, out) {
                    continue;
                }
            }
            for col in 0..cols {
                data[start + col] = lhs_slice[start + col] + rhs_slice[row];
            }
        }
        Some(NumericArray::new_owned(data, self.shape.clone()))
    }

    fn try_simd_add_scalar(&self, other: &NumericArray<T>) -> Option<NumericArray<T>> {
        if !self.is_contiguous() || !other.is_contiguous() {
            return None;
        }
        if other.shape == [1] {
            let scalar = other.data_slice().get(0)?;
            let mut out = vec![T::zero(); self.data_len()];
            for (dest, &value) in out.iter_mut().zip(self.data_slice().iter()) {
                *dest = value + *scalar;
            }
            return Some(NumericArray::new_owned(out, self.shape.clone()));
        }
        None
    }

    fn new_owned(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            storage: NumericStorage::Owned(data),
            shape,
        }
    }

    fn new_borrowed(
        ptr: *const T,
        len: usize,
        owner: Py<PyAny>,
        shape: Vec<usize>,
    ) -> PyResult<Self> {
        if product(&shape) != len {
            return Err(PyValueError::new_err(
                "borrowed buffer length does not match provided shape",
            ));
        }
        Ok(Self {
            storage: NumericStorage::Borrowed {
                ptr,
                len,
                _owner: owner,
            },
            shape,
        })
    }

    fn data_len(&self) -> usize {
        match &self.storage {
            NumericStorage::Owned(data) => data.len(),
            NumericStorage::Borrowed { len, .. } => *len,
        }
    }

    fn data_slice(&self) -> &[T] {
        match &self.storage {
            NumericStorage::Owned(data) => data.as_slice(),
            NumericStorage::Borrowed { ptr, len, .. } => {
                if *len == 0 {
                    &[]
                } else {
                    unsafe { slice::from_raw_parts(*ptr, *len) }
                }
            }
        }
    }

    fn data_ptr(&self) -> *const T {
        match &self.storage {
            NumericStorage::Owned(data) => data.as_ptr(),
            NumericStorage::Borrowed { ptr, .. } => *ptr,
        }
    }

    fn to_vec(&self) -> Vec<T> {
        self.data_slice().to_vec()
    }

    fn get(&self, index: usize) -> T {
        self.data_slice()[index]
    }

    fn array_interface_dict<'py>(
        &self,
        py: Python<'py>,
        typestr: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("version", 3)?;

        let shape_objs: Vec<PyObject> = self
            .shape
            .iter()
            .map(|&dim| (dim as isize).into_py(py))
            .collect();
        let shape_tuple = PyTuple::new_bound(py, shape_objs);
        dict.set_item("shape", shape_tuple)?;

        let ptr_value = if self.data_len() == 0 {
            0usize
        } else {
            self.data_ptr() as usize
        };
        let data_tuple = PyTuple::new_bound(py, [ptr_value.into_py(py), false.into_py(py)]);
        dict.set_item("data", data_tuple)?;
        dict.set_item("typestr", typestr)?;
        dict.set_item("strides", py.None())?;

        Ok(dict)
    }

    fn from_python(iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (data, shape) = parse_python_iterable::<T>(iterable)?;
        Ok(Self::new_owned(data, shape))
    }

    fn zeros(dims: &[usize]) -> PyResult<Self> {
        let total = dims.iter().product();
        Ok(Self::new_owned(vec![T::zero(); total], dims.to_vec()))
    }

    fn ones(dims: &[usize]) -> PyResult<Self> {
        let total = dims.iter().product();
        Ok(Self::new_owned(vec![T::one(); total], dims.to_vec()))
    }

    fn from_numpy(array: PyReadonlyArrayDyn<'_, T>, owner: Py<PyAny>) -> PyResult<Self> {
        let shape = array.shape().to_vec();
        if array.is_c_contiguous() {
            let len = array.len();
            let view = array.as_array();
            let ptr = view.as_ptr() as *const T;
            NumericArray::new_borrowed(ptr, len, owner, shape)
        } else {
            let data = array.as_array().to_owned().into_raw_vec();
            Ok(Self::new_owned(data, shape))
        }
    }

    fn total_len(&self) -> usize {
        self.data_len()
    }

    fn first_axis_len(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn sum_f64(&self) -> PyResultF64 {
        if let Some(value) = self.global_reduce(reduce::tiled::GlobalOp::Sum) {
            return Ok(value);
        }
        T::simd_sum(self.data_slice())
            .ok_or_else(|| PyValueError::new_err("value cannot be represented as float"))
    }

    fn mean_f64(&self) -> PyResultF64 {
        if self.data_len() == 0 {
            Err(PyValueError::new_err("cannot compute mean of empty array"))
        } else {
            if let Some(value) = self.global_reduce(reduce::tiled::GlobalOp::Mean) {
                Ok(value)
            } else {
                Ok(self.sum_f64()? / self.data_len() as f64)
            }
        }
    }

    fn add(&self, other: &NumericArray<T>) -> PyResult<NumericArray<T>> {
        if let Some(result) = self.try_simd_add_same_shape(other) {
            return Ok(result);
        }
        if let Some(result) = self.try_simd_add_row(other) {
            return Ok(result);
        }
        if let Some(result) = other.try_simd_add_row(self) {
            return Ok(result);
        }
        if let Some(result) = self.try_simd_add_column(other) {
            return Ok(result);
        }
        if let Some(result) = other.try_simd_add_column(self) {
            return Ok(result);
        }
        if let Some(result) = self.try_simd_add_scalar(other) {
            return Ok(result);
        }
        if let Some(result) = other.try_simd_add_scalar(self) {
            return Ok(result);
        }

        let broadcast = BroadcastPair::new(self, other)?;
        let data = broadcast
            .rows()
            .map(|result| result.map(|(a, b)| a + b))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NumericArray::new_owned(
            data,
            broadcast.output_shape.clone(),
        ))
    }

    fn scale(&self, factor: f64) -> PyResult<NumericArray<T>> {
        if !T::SUPPORTS_FRACTIONS && factor.fract() != 0.0 {
            return Err(PyValueError::new_err(format!(
                "dtype {} only supports integer scale factors",
                T::DTYPE_NAME
            )));
        }

        let len = self.data_len();
        let mut data = vec![T::zero(); len];
        let (rows, cols) = self.matrix_dims();
        if T::DTYPE_NAME == "float64" {
            let input = self.data_slice();
            let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, len) };
            if should_parallelize(rows, cols, T::DTYPE_NAME)
                && parallel_scale_f64(input, factor, out, rows, cols)
            {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if simd::scale_same_shape_f64(input, factor, out) {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
        } else if T::DTYPE_NAME == "float32" {
            let input = self.data_slice();
            let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, len) };
            if should_parallelize(rows, cols, T::DTYPE_NAME)
                && parallel_scale_f32(input, factor, out, rows, cols)
            {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if simd::scale_same_shape_f32(input, factor as f32, out) {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
        }
        T::simd_scale(self.data_slice(), factor, &mut data).map_err(|_| {
            PyValueError::new_err(format!(
                "scaling produced values outside {} representable range",
                T::DTYPE_NAME
            ))
        })?;
        Ok(NumericArray::new_owned(data, self.shape.clone()))
    }

    fn reduce_axis(&self, axis: usize, op: Reduction) -> PyResult<NumericArray<T>> {
        match self.shape.len() {
            0 => Err(PyValueError::new_err(
                "cannot reduce an array with no shape information",
            )),
            1 => self.reduce_axis_1d(axis, op),
            2 => self.reduce_axis_2d(axis, op),
            _ => Err(PyValueError::new_err(
                "axis reductions are currently supported for up to 2-D arrays",
            )),
        }
    }

    fn convert_from_f64(value: f64, op: Reduction) -> PyResult<T> {
        if !T::SUPPORTS_FRACTIONS && (value.fract().abs() > f64::EPSILON) {
            return Err(PyValueError::new_err(format!(
                "{} result cannot be represented exactly in {} arrays",
                match op {
                    Reduction::Sum => "sum",
                    Reduction::Mean => "mean",
                },
                T::DTYPE_NAME
            )));
        }
        T::try_from_f64(value).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{} result cannot be represented exactly in {} arrays",
                match op {
                    Reduction::Sum => "sum",
                    Reduction::Mean => "mean",
                },
                T::DTYPE_NAME
            ))
        })
    }

    fn reduce_axis_1d(&self, axis: usize, op: Reduction) -> PyResult<NumericArray<T>> {
        if axis != 0 {
            return Err(PyValueError::new_err("axis out of bounds for 1-D array"));
        }

        let value = match op {
            Reduction::Sum => self.sum_f64()?,
            Reduction::Mean => self.mean_f64()?,
        };

        let converted = Self::convert_from_f64(value, op)?;

        Ok(NumericArray::new_owned(vec![converted], vec![1]))
    }

    fn reduce_axis_2d(&self, axis: usize, op: Reduction) -> PyResult<NumericArray<T>> {
        if axis > 1 {
            return Err(PyValueError::new_err("axis out of bounds for 2-D array"));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let strides = row_major_strides(&self.shape);
        let row_stride = strides[0];
        let col_stride = strides[1];

        match axis {
            0 => {
                if self.is_contiguous()
                    && (T::DTYPE_NAME == "float64" || T::DTYPE_NAME == "float32")
                {
                    let allow_parallel =
                        should_parallelize_axis(AxisKind::Axis0, rows, cols, T::DTYPE_NAME);
                    let start = Instant::now();
                    let outcome = if T::DTYPE_NAME == "float64" {
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                self.data_slice().as_ptr() as *const f64,
                                rows * cols,
                            )
                        };
                        reduce_axis0_f64(data, rows, cols, op, allow_parallel)
                    } else {
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                self.data_slice().as_ptr() as *const f32,
                                rows * cols,
                            )
                        };
                        reduce_axis0_f32(data, rows, cols, op, allow_parallel)
                    };
                    let elapsed = start.elapsed();
                    record_axis_event(
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        elapsed,
                        outcome.parallel,
                        "axis0",
                    );
                    return self.convert_reduction_from_f64(outcome.values, vec![cols], op);
                }
                let mut out = Vec::with_capacity(cols);
                for c in 0..cols {
                    let mut acc = 0.0;
                    for r in 0..rows {
                        let base = r * row_stride;
                        let idx = base + c * col_stride;
                        acc += self.get(idx).try_to_f64().ok_or_else(|| {
                            PyValueError::new_err("value cannot be represented as float")
                        })?;
                    }
                    if matches!(op, Reduction::Mean) && rows > 0 {
                        acc /= rows as f64;
                    }
                    let converted = Self::convert_from_f64(acc, op)?;
                    out.push(converted);
                }
                Ok(NumericArray::new_owned(out, vec![cols]))
            }
            1 => {
                if self.is_contiguous()
                    && (T::DTYPE_NAME == "float64" || T::DTYPE_NAME == "float32")
                {
                    let allow_parallel =
                        should_parallelize_axis(AxisKind::Axis1, rows, cols, T::DTYPE_NAME);
                    let start = Instant::now();
                    let outcome = if T::DTYPE_NAME == "float64" {
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                self.data_slice().as_ptr() as *const f64,
                                rows * cols,
                            )
                        };
                        reduce_axis1_f64(data, rows, cols, op, allow_parallel)
                    } else {
                        let data = unsafe {
                            std::slice::from_raw_parts(
                                self.data_slice().as_ptr() as *const f32,
                                rows * cols,
                            )
                        };
                        reduce_axis1_f32(data, rows, cols, op, allow_parallel)
                    };
                    let elapsed = start.elapsed();
                    record_axis_event(
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        elapsed,
                        outcome.parallel,
                        "axis1",
                    );
                    return self.convert_reduction_from_f64(outcome.values, vec![rows], op);
                }
                let mut out = Vec::with_capacity(rows);
                for r in 0..rows {
                    let base = r * row_stride;
                    let mut acc = 0.0;
                    for c in 0..cols {
                        let idx = base + c * col_stride;
                        acc += self.get(idx).try_to_f64().ok_or_else(|| {
                            PyValueError::new_err("value cannot be represented as float")
                        })?;
                    }
                    if matches!(op, Reduction::Mean) && cols > 0 {
                        acc /= cols as f64;
                    }
                    let converted = Self::convert_from_f64(acc, op)?;
                    out.push(converted);
                }
                Ok(NumericArray::new_owned(out, vec![rows]))
            }
            _ => unreachable!(),
        }
    }

    fn convert_reduction_from_f64(
        &self,
        values: Vec<f64>,
        shape: Vec<usize>,
        op: Reduction,
    ) -> PyResult<NumericArray<T>> {
        let mut out = Vec::with_capacity(values.len());
        for value in values {
            out.push(Self::convert_from_f64(value, op)?);
        }
        Ok(NumericArray::new_owned(out, shape))
    }
}

fn reduce_axis0_f64(
    data: &[f64],
    rows: usize,
    cols: usize,
    op: Reduction,
    allow_parallel: bool,
) -> AxisOutcome {
    if cols == 0 || rows == 0 {
        return AxisOutcome {
            values: vec![0.0; cols],
            parallel: false,
        };
    }
    let total = rows * cols;
    let mut parallel_used = false;
    let mut sums = if allow_parallel && total >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols)
                    .fold(
                        || vec![0.0; cols],
                        |mut acc, row| {
                            add_assign_f64(&mut acc, row);
                            acc
                        },
                    )
                    .reduce(
                        || vec![0.0; cols],
                        |mut acc, partial| {
                            add_assign_f64(&mut acc, &partial);
                            acc
                        },
                    )
            })
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    if sums.is_empty() {
        parallel_used = false;
        sums = vec![0.0; cols];
        for row in data.chunks(cols) {
            add_assign_f64(&mut sums, row);
        }
    }
    if matches!(op, Reduction::Mean) {
        let inv = 1.0 / rows as f64;
        for value in &mut sums {
            *value *= inv;
        }
    }
    AxisOutcome {
        values: sums,
        parallel: parallel_used,
    }
}

fn reduce_axis0_f32(
    data: &[f32],
    rows: usize,
    cols: usize,
    op: Reduction,
    allow_parallel: bool,
) -> AxisOutcome {
    if cols == 0 || rows == 0 {
        return AxisOutcome {
            values: vec![0.0; cols],
            parallel: false,
        };
    }
    let total = rows * cols;
    let mut parallel_used = false;
    let mut sums_f32 = if allow_parallel && total >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols.saturating_mul(64).max(cols))
                    .map(|chunk| {
                        let mut partial = vec![0.0f32; cols];
                        for row in chunk.chunks(cols) {
                            accumulate_row_f32(&mut partial, row);
                        }
                        partial
                    })
                    .reduce(
                        || vec![0.0f32; cols],
                        |mut acc, partial| {
                            accumulate_row_f32(&mut acc, &partial);
                            acc
                        },
                    )
            })
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    if sums_f32.is_empty() {
        parallel_used = false;
        sums_f32 = vec![0.0f32; cols];
        for row in data.chunks(cols) {
            accumulate_row_f32(&mut sums_f32, row);
        }
    }
    let mut sums: Vec<f64> = sums_f32.iter().map(|&value| value as f64).collect();
    if matches!(op, Reduction::Mean) {
        let inv = 1.0 / rows as f64;
        for value in &mut sums {
            *value *= inv;
        }
    }
    AxisOutcome {
        values: sums,
        parallel: parallel_used,
    }
}

fn reduce_axis1_f64(
    data: &[f64],
    rows: usize,
    cols: usize,
    op: Reduction,
    allow_parallel: bool,
) -> AxisOutcome {
    if rows == 0 {
        return AxisOutcome {
            values: Vec::new(),
            parallel: false,
        };
    }
    let total = rows * cols;
    let mut parallel_used = false;
    let mut sums = if allow_parallel && total >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols)
                    .map(|row| row.iter().sum::<f64>())
                    .collect::<Vec<f64>>()
            })
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    if sums.is_empty() {
        parallel_used = false;
        let mut out = Vec::with_capacity(rows);
        for row in data.chunks(cols) {
            let sum = simd::reduce_sum_f64(row, 1).unwrap_or_else(|| row.iter().sum::<f64>());
            out.push(sum);
        }
        sums = out;
    }
    if matches!(op, Reduction::Mean) && cols > 0 {
        let inv = 1.0 / cols as f64;
        for value in &mut sums {
            *value *= inv;
        }
    }
    AxisOutcome {
        values: sums,
        parallel: parallel_used,
    }
}

fn reduce_axis1_f32(
    data: &[f32],
    rows: usize,
    cols: usize,
    op: Reduction,
    allow_parallel: bool,
) -> AxisOutcome {
    if rows == 0 {
        return AxisOutcome {
            values: Vec::new(),
            parallel: false,
        };
    }
    let total = rows * cols;
    let mut parallel_used = false;
    let mut sums = if allow_parallel && total >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols)
                    .map(|row| {
                        simd::reduce_sum_f32(row, 1)
                            .unwrap_or_else(|| row.iter().map(|&value| value as f64).sum::<f64>())
                    })
                    .collect::<Vec<f64>>()
            })
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    if sums.is_empty() {
        parallel_used = false;
        let mut out = Vec::with_capacity(rows);
        for row in data.chunks(cols) {
            let sum = simd::reduce_sum_f32(row, 1)
                .unwrap_or_else(|| row.iter().map(|&value| value as f64).sum::<f64>());
            out.push(sum);
        }
        sums = out;
    }
    if matches!(op, Reduction::Mean) && cols > 0 {
        let inv = 1.0 / cols as f64;
        for value in &mut sums {
            *value *= inv;
        }
    }
    AxisOutcome {
        values: sums,
        parallel: parallel_used,
    }
}
#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    Mean,
}

struct BroadcastPair<'a, T>
where
    T: NumericElement + Element + for<'py> FromPyObject<'py>,
{
    left: &'a NumericArray<T>,
    right: &'a NumericArray<T>,
    output_shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
}

impl<'a, T> BroadcastPair<'a, T>
where
    T: NumericElement + Element + for<'py> FromPyObject<'py>,
{
    fn new(left: &'a NumericArray<T>, right: &'a NumericArray<T>) -> PyResult<Self> {
        let (shape, left_strides, right_strides) = broadcast_shapes(&left.shape, &right.shape)?;
        Ok(Self {
            left,
            right,
            output_shape: shape,
            left_strides,
            right_strides,
        })
    }

    fn rows(&self) -> impl Iterator<Item = Result<(T, T), PyErr>> + '_ {
        let total = self.output_shape.iter().product::<usize>();
        (0..total).map(move |idx| {
            let left_idx = map_index(&self.output_shape, &self.left_strides, idx)?;
            let right_idx = map_index(&self.output_shape, &self.right_strides, idx)?;
            Ok((self.left.get(left_idx), self.right.get(right_idx)))
        })
    }
}

fn broadcast_shapes(
    left: &[usize],
    right: &[usize],
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let ndim = left.len().max(right.len());
    let mut shape = vec![1; ndim];
    let mut left_strides = vec![0; ndim];
    let mut right_strides = vec![0; ndim];

    let left_strides_raw = row_major_strides(left);
    let right_strides_raw = row_major_strides(right);
    let left_offset = ndim.saturating_sub(left.len());
    let right_offset = ndim.saturating_sub(right.len());

    for i in 0..ndim {
        let l_dim = if i >= left_offset {
            left[i - left_offset]
        } else {
            1
        };
        let r_dim = if i >= right_offset {
            right[i - right_offset]
        } else {
            1
        };

        match (l_dim, r_dim) {
            (a, b) if a == b => shape[i] = a,
            (1, b) => shape[i] = b,
            (a, 1) => shape[i] = a,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "operands could not be broadcast together: left={:?}, right={:?}",
                    left, right
                )))
            }
        }

        left_strides[i] = if i < left_offset || l_dim == 1 {
            0
        } else {
            left_strides_raw[i - left_offset]
        };
        right_strides[i] = if i < right_offset || r_dim == 1 {
            0
        } else {
            right_strides_raw[i - right_offset]
        };
    }

    Ok((shape, left_strides, right_strides))
}

fn map_index(shape: &[usize], strides: &[usize], flat_index: usize) -> PyResult<usize> {
    if shape.is_empty() {
        return Ok(0);
    }
    let mut coords = vec![0usize; shape.len()];
    let mut remainder = flat_index;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim == 0 {
            coords[i] = 0;
            continue;
        }
        coords[i] = remainder % dim;
        remainder /= dim;
    }

    coords
        .into_iter()
        .zip(strides.iter())
        .try_fold(0usize, |acc, (coord, stride)| {
            acc.checked_add(coord * stride)
                .ok_or_else(|| PyValueError::new_err("index overflow during broadcasting"))
        })
}

fn product(values: &[usize]) -> usize {
    values.iter().product()
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![0; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i]);
    }
    strides
}

fn parse_shape_argument(arg: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(value) = arg.extract::<usize>() {
        return Ok(vec![value]);
    }

    let arg_clone = arg.clone();
    let seq = arg_clone
        .downcast::<PySequence>()
        .map_err(|_| PyTypeError::new_err("shape must be an int or a sequence of ints"))?;
    let dims: Vec<usize> = seq
        .iter()?
        .map(|item| {
            item.and_then(|value| {
                value
                    .extract::<usize>()
                    .map_err(|_| PyTypeError::new_err("shape sequence must contain integers"))
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    if dims.is_empty() {
        return Err(PyValueError::new_err(
            "shape sequences must contain at least one dimension",
        ));
    }

    Ok(dims)
}

fn parse_python_iterable<T>(iterable: &Bound<'_, PyAny>) -> PyResult<(Vec<T>, Vec<usize>)>
where
    T: NumericElement + for<'py> FromPyObject<'py>,
{
    let items: Vec<Bound<'_, PyAny>> = iterable.iter()?.collect::<PyResult<Vec<_>>>()?;

    if items.is_empty() {
        return Ok((Vec::new(), vec![0]));
    }

    if items[0].extract::<T>().is_ok() {
        let mut data = Vec::with_capacity(items.len());
        for item in items {
            data.push(item.extract::<T>()?);
        }
        let len = data.len();
        return Ok((data, vec![len]));
    }

    let mut data = Vec::new();
    let outer_len = items.len();
    let mut inner_len: Option<usize> = None;

    for row in items {
        let seq = row.downcast::<PySequence>().map_err(|_| {
            PyTypeError::new_err("expected a sequence of sequences for multi-dimensional input")
        })?;
        let row_items: Vec<Bound<'_, PyAny>> = seq.iter()?.collect::<PyResult<Vec<_>>>()?;

        let current_len = row_items.len();
        if let Some(expected) = inner_len {
            if current_len != expected {
                return Err(PyValueError::new_err(
                    "all inner sequences must have the same length",
                ));
            }
        } else {
            inner_len = Some(current_len);
        }

        for item in row_items {
            data.push(item.extract::<T>()?);
        }
    }

    let cols = inner_len.unwrap_or(0);
    if cols == 0 && !data.is_empty() {
        return Err(PyValueError::new_err(
            "inner sequences cannot be empty for multi-dimensional arrays",
        ));
    }

    Ok((data, vec![outer_len, cols]))
}

macro_rules! impl_pyarray {
    ($name:ident, $pyname:literal, $t:ty) => {
        #[pyclass(unsendable, name = $pyname, module = "raptors")]
        pub struct $name {
            inner: NumericArray<$t>,
        }

        impl $name {
            fn from_inner(inner: NumericArray<$t>) -> Self {
                Self { inner }
            }
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(Self::from_inner(NumericArray::from_python(iterable)?))
            }

            #[getter]
            fn len(&self) -> usize {
                self.inner.total_len()
            }

            fn __len__(&self) -> usize {
                self.inner.first_axis_len()
            }

            #[getter]
            fn shape(&self) -> Vec<usize> {
                self.inner.shape.clone()
            }

            #[getter]
            fn ndim(&self) -> usize {
                self.inner.ndim()
            }

            fn to_list(&self) -> Vec<$t> {
                self.inner.to_vec()
            }

            #[getter]
            #[pyo3(name = "__array_interface__")]
            fn get_array_interface<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<PyObject> {
                let dict = slf
                    .inner
                    .array_interface_dict(py, <$t as NumericElement>::NUMPY_TYPESTR)?;
                let obj: PyObject = slf.into_py(py);
                let bound = obj.bind(py);
                dict.set_item("obj", bound)?;
                Ok(dict.into())
            }

            fn sum(&self) -> PyResult<f64> {
                self.inner.sum_f64()
            }

            fn mean(&self) -> PyResult<f64> {
                self.inner.mean_f64()
            }

            #[pyo3(signature = (axis))]
            fn sum_axis(&self, axis: usize) -> PyResult<Self> {
                Ok(Self::from_inner(
                    self.inner.reduce_axis(axis, Reduction::Sum)?,
                ))
            }

            #[pyo3(signature = (axis))]
            fn mean_axis(&self, axis: usize) -> PyResult<Self> {
                Ok(Self::from_inner(
                    self.inner.reduce_axis(axis, Reduction::Mean)?,
                ))
            }

            fn add(&self, other: &Self) -> PyResult<Self> {
                Ok(Self::from_inner(self.inner.add(&other.inner)?))
            }

            fn scale(&self, factor: f64) -> PyResult<Self> {
                Ok(Self::from_inner(self.inner.scale(factor)?))
            }

            fn to_numpy<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Py<PyArrayDyn<$t>>> {
                let numpy = py.import_bound("numpy")?;
                let asarray = numpy.getattr("asarray")?;
                let obj: PyObject = slf.into_py(py);
                let bound = obj.bind(py);
                let result = asarray.call1((bound,))?;
                let array = result.downcast_into::<PyArrayDyn<$t>>()?;
                Ok(array.unbind())
            }
        }
    };
}

impl_pyarray!(RustArray, "RustArray", f64);
impl_pyarray!(RustArrayF32, "RustArrayF32", f32);
impl_pyarray!(RustArrayI32, "RustArrayI32", i32);

#[pyfunction]
fn array(iterable: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    RustArray::new(iterable)
}

#[pyfunction]
fn array_f32(iterable: &Bound<'_, PyAny>) -> PyResult<RustArrayF32> {
    RustArrayF32::new(iterable)
}

#[pyfunction]
fn array_i32(iterable: &Bound<'_, PyAny>) -> PyResult<RustArrayI32> {
    RustArrayI32::new(iterable)
}

#[pyfunction]
fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let dims = parse_shape_argument(shape)?;
    Ok(RustArray::from_inner(NumericArray::<f64>::zeros(&dims)?))
}

#[pyfunction]
fn zeros_f32(shape: &Bound<'_, PyAny>) -> PyResult<RustArrayF32> {
    let dims = parse_shape_argument(shape)?;
    Ok(RustArrayF32::from_inner(NumericArray::<f32>::zeros(&dims)?))
}

#[pyfunction]
fn zeros_i32(shape: &Bound<'_, PyAny>) -> PyResult<RustArrayI32> {
    let dims = parse_shape_argument(shape)?;
    Ok(RustArrayI32::from_inner(NumericArray::<i32>::zeros(&dims)?))
}

#[pyfunction]
fn ones(shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let dims = parse_shape_argument(shape)?;
    Ok(RustArray::from_inner(NumericArray::<f64>::ones(&dims)?))
}

#[pyfunction]
fn ones_f32(shape: &Bound<'_, PyAny>) -> PyResult<RustArrayF32> {
    let dims = parse_shape_argument(shape)?;
    Ok(RustArrayF32::from_inner(NumericArray::<f32>::ones(&dims)?))
}

#[pyfunction]
fn ones_i32(shape: &Bound<'_, PyAny>) -> PyResult<RustArrayI32> {
    let dims = parse_shape_argument(shape)?;
    Ok(RustArrayI32::from_inner(NumericArray::<i32>::ones(&dims)?))
}

#[pyfunction]
fn from_numpy(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let numpy_array: PyReadonlyArrayDyn<'_, f64> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype float64"))?;
    let owner = array.to_object(py);
    Ok(RustArray::from_inner(NumericArray::from_numpy(
        numpy_array,
        owner,
    )?))
}

#[pyfunction]
fn from_numpy_f32(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArrayF32> {
    let numpy_array: PyReadonlyArrayDyn<'_, f32> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype float32"))?;
    let owner = array.to_object(py);
    Ok(RustArrayF32::from_inner(NumericArray::from_numpy(
        numpy_array,
        owner,
    )?))
}

#[pyfunction]
fn from_numpy_i32(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArrayI32> {
    let numpy_array: PyReadonlyArrayDyn<'_, i32> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype int32"))?;
    let owner = array.to_object(py);
    Ok(RustArrayI32::from_inner(NumericArray::from_numpy(
        numpy_array,
        owner,
    )?))
}

#[pyfunction]
fn broadcast_add(lhs: &RustArray, rhs: &RustArray) -> PyResult<RustArray> {
    Ok(RustArray::from_inner(lhs.inner.add(&rhs.inner)?))
}

#[pyfunction]
fn broadcast_add_f32(lhs: &RustArrayF32, rhs: &RustArrayF32) -> PyResult<RustArrayF32> {
    Ok(RustArrayF32::from_inner(lhs.inner.add(&rhs.inner)?))
}

#[pyfunction]
fn broadcast_add_i32(lhs: &RustArrayI32, rhs: &RustArrayI32) -> PyResult<RustArrayI32> {
    Ok(RustArrayI32::from_inner(lhs.inner.add(&rhs.inner)?))
}

#[pyfunction(name = "simd_enabled")]
fn simd_enabled_py() -> bool {
    simd_is_enabled()
}

#[pyfunction(name = "threading_info")]
fn threading_info_py(py: Python<'_>) -> PyResult<PyObject> {
    let snapshot = threading_snapshot();
    let info = PyDict::new_bound(py);
    info.set_item("parallel_min_elements", PARALLEL_MIN_ELEMENTS)?;

    let baseline_dict = PyDict::new_bound(py);
    for &dtype in TRACKED_DTYPES {
        baseline_dict.set_item(dtype, baseline_cutover(dtype))?;
    }
    info.set_item("baseline_cutovers", baseline_dict)?;

    let dims_dict = PyDict::new_bound(py);
    for &dtype in TRACKED_DTYPES {
        let (rows, cols) = dtype_dim_threshold(dtype);
        dims_dict.set_item(dtype, PyList::new_bound(py, [rows, cols]))?;
    }
    info.set_item("dimension_thresholds", dims_dict)?;

    if let Some(pool) = thread_pool() {
        let pool_dict = PyDict::new_bound(py);
        pool_dict.set_item("active_threads", pool.current_num_threads())?;
        info.set_item("thread_pool", pool_dict)?;
    } else {
        info.set_item("thread_pool", py.None())?;
    }

    let thresholds_dict = PyDict::new_bound(py);
    for entry in snapshot.thresholds {
        let entry_dict = PyDict::new_bound(py);
        entry_dict.set_item("median_elements_per_ms", entry.median_elements_per_ms)?;
        entry_dict.set_item("sample_count", entry.sample_count)?;
        entry_dict.set_item("samples", PyList::new_bound(py, entry.samples))?;
        match entry.recommended_cutover {
            Some(value) => entry_dict.set_item("recommended_cutover", value)?,
            None => entry_dict.set_item("recommended_cutover", py.None())?,
        }
        entry_dict.set_item("baseline_cutover", baseline_cutover(entry.dtype))?;
        entry_dict.set_item("target_latency_ms", target_latency_ms(entry.dtype))?;
        thresholds_dict.set_item(entry.dtype, entry_dict)?;
    }
    info.set_item("adaptive_thresholds", thresholds_dict)?;

    if let Some(event) = snapshot.last_event {
        let event_dict = PyDict::new_bound(py);
        event_dict.set_item("dtype", event.dtype)?;
        event_dict.set_item("elements", event.elements)?;
        event_dict.set_item("duration_ms", event.duration_ms)?;
        event_dict.set_item("tiles_processed", event.tiles)?;
        event_dict.set_item("partial_buffer", event.partial_buffer)?;
        event_dict.set_item("parallel", event.parallel)?;
        event_dict.set_item("operation", event.operation)?;
        info.set_item("last_event", event_dict)?;
    } else {
        info.set_item("last_event", py.None())?;
    }

    Ok(info.into())
}

/// Python module initialization for `_raptors`.
#[pymodule]
fn _raptors(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustArray>()?;
    m.add_class::<RustArrayF32>()?;
    m.add_class::<RustArrayI32>()?;

    m.add_wrapped(pyo3::wrap_pyfunction!(array))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(array_f32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(array_i32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(zeros))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(zeros_f32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(zeros_i32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(ones))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(ones_f32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(ones_i32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(from_numpy))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(from_numpy_f32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(from_numpy_i32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(broadcast_add))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(broadcast_add_f32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(broadcast_add_i32))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(simd_enabled_py))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(threading_info_py))?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Odos Matthews <odosmatthews@gmail.com>")?;
    m.add("__github__", "https://github.com/eddiethedean")?;
    m.add("__doc__", "Rust-backed array core for the Raptors project.")?;

    Ok(())
}
