use std::{
    cell::RefCell,
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    convert::TryInto,
    env,
    hash::Hash,
    ops::Add,
    slice,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

mod blas;
mod reduce;
mod simd;
mod tiling;

#[cfg(test)]
mod test_support;

use crate::reduce::tiled::SMALL_DIRECT_THRESHOLD;

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

#[derive(Clone, Copy)]
struct RowParallelSetting {
    enabled: bool,
    force: bool,
}

fn axis0_row_parallel_setting() -> RowParallelSetting {
    static SETTING: OnceLock<RowParallelSetting> = OnceLock::new();
    *SETTING.get_or_init(|| {
        if let Ok(flag) = env::var("RAPTORS_AXIS0_ROW_CHUNK") {
            match flag.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" => {
                    return RowParallelSetting {
                        enabled: true,
                        force: false,
                    }
                }
                "force" => {
                    return RowParallelSetting {
                        enabled: true,
                        force: true,
                    }
                }
                "0" | "false" | "off" => {
                    return RowParallelSetting {
                        enabled: false,
                        force: false,
                    }
                }
                _ => {}
            }
        }
        RowParallelSetting {
            enabled: true,
            force: false,
        }
    })
}

fn axis0_row_parallel_enabled() -> bool {
    axis0_row_parallel_setting().enabled
}

fn axis0_row_parallel_force() -> bool {
    axis0_row_parallel_setting().force
}

const PARALLEL_MIN_ELEMENTS: usize = 1 << 15;
const ADAPTIVE_SAMPLE_WINDOW: usize = 9;
const TRACKED_DTYPES: &[&str] = &["float64", "float32", "int32"];

const SMALL_AXIS_PARALLEL_ROWS: usize = 512;
const AXIS0_SIMD_COL_LIMIT: usize = 2048;
const AXIS0_PAR_MIN_ROWS: usize = 64;
const AXIS1_PAR_MIN_ROWS: usize = 32;
const AXIS0_PAR_MIN_COL_CHUNK: usize = 64;
const AXIS0_PAR_MAX_COL_CHUNK: usize = 512;
#[allow(dead_code)]
const AXIS0_LARGE_TILED_ROWS: usize = 1024;
#[allow(dead_code)]
const AXIS0_LARGE_TILED_COLS: usize = 1024;
const MATRIX_AXIS0_ROW_THRESHOLD: usize = 1024;
const MATRIX_AXIS0_COL_THRESHOLD: usize = 1024;
const MATRIX_AXIS0_MATRIX_MAX_COLS: usize = 1536;
const MATRIX_AXIS0_ROW_THRESHOLD_F64: usize = 1536;
const AXIS0_PAR_MIN_ROWS_F64: usize = 1536;
const SCALE_PAR_MIN_ROWS: usize = 32;
const SCALE_PAR_MIN_CHUNK_ELEMS: usize = 1 << 14;
const SCALE_PAR_MAX_CHUNK_ELEMS: usize = 1 << 18;
const SCALE_BLAS_MIN_LEN: usize = 1 << 12;
const SCALE_BLAS_MIN_ROWS: usize = 1024;
const SCALE_BLAS_MIN_COLS: usize = 1024;
const SCALE_FORCE_PARALLEL_ROWS: usize = 1024;
const SCALE_FORCE_PARALLEL_COLS: usize = 1024;
const SCALE_FORCE_PARALLEL_ELEMS: usize = 1 << 20;
const AXIS0_BLAS_MAX_ROWS_F64: usize = 1536;
const AXIS0_BLAS_MIN_ROWS: usize = 512;
const AXIS0_BLAS_MIN_ROWS_F64: usize = 768;
const AXIS0_BLAS_MIN_COLS: usize = 512;
const BROADCAST_PAR_MIN_ROWS: usize = 48;
const BROADCAST_PAR_MIN_COLS: usize = 64;
const AXIS0_COLUMN_SIMD_MIN_ROWS: usize = 64;
const AXIS0_COLUMN_SIMD_MIN_COLS: usize = 32;
const COLUMN_BROADCAST_DIRECT_MIN_ELEMS: usize = 1 << 20;
const SMALL_MATRIX_FAST_DIM: usize = 256;
const SCALE_TINY_DIM: usize = 512;
const SCALE_PAR_MIN_ELEMS: usize = 1 << 19;
const SCALE_PAR_MIN_ELEMS_F64: usize = 1 << 19;
const OPERATION_SCALE: &str = "scale";
const OPERATION_BROADCAST_ROW: &str = "broadcast_row";
const OPERATION_BROADCAST_COL: &str = "broadcast_col";
const SMALL_MICRO_DIM: usize = 512;
const AXIS0_ROW_CHUNK_MIN_ROWS: usize = 1 << 20;
const OPERATION_AXIS0_ROW: &str = "axis0_row";
const OPERATION_AXIS0_COL: &str = "axis0_col";

thread_local! {
    static SMALL_F32_SCRATCH: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

#[cfg(feature = "matrixmultiply-backend")]
thread_local! {
    static F32_ONES: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

#[cfg(feature = "matrixmultiply-backend")]
thread_local! {
    static F64_ONES: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

#[inline]
fn with_f32_scratch<R, F>(len: usize, f: F) -> R
where
    F: FnOnce(&mut [f32]) -> R,
{
    SMALL_F32_SCRATCH.with(|cell| {
        let mut buffer = cell.borrow_mut();
        if buffer.len() < len {
            buffer.resize(len, 0.0);
        }
        f(&mut buffer[..len])
    })
}

#[cfg(feature = "matrixmultiply-backend")]
#[inline]
fn with_f32_ones<R, F>(len: usize, f: F) -> R
where
    F: FnOnce(&[f32]) -> R,
{
    F32_ONES.with(|cell| {
        let mut buffer = cell.borrow_mut();
        if buffer.len() < len {
            buffer.resize(len, 1.0);
        }
        f(&buffer[..len])
    })
}

#[cfg(feature = "matrixmultiply-backend")]
#[inline]
fn with_f64_ones<R, F>(len: usize, f: F) -> R
where
    F: FnOnce(&[f64]) -> R,
{
    F64_ONES.with(|cell| {
        let mut buffer = cell.borrow_mut();
        if buffer.len() < len {
            buffer.resize(len, 1.0);
        }
        f(&buffer[..len])
    })
}

#[cfg(feature = "matrixmultiply-backend")]
#[inline]
fn matrix_backend_enabled() -> bool {
    true
}

#[cfg(not(feature = "matrixmultiply-backend"))]
#[inline]
fn matrix_backend_enabled() -> bool {
    false
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
enum AxisKind {
    Axis0,
    Axis1,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
enum BroadcastKind {
    Row,
    Column,
}

#[derive(Default)]
struct AdaptiveThreadingState {
    throughput: HashMap<&'static str, VecDeque<f64>>,
    seq_throughput: HashMap<&'static str, VecDeque<f64>>,
    axis_parallel: HashMap<(AxisKind, &'static str), VecDeque<f64>>,
    axis_sequential: HashMap<(AxisKind, &'static str), VecDeque<f64>>,
    op_parallel: HashMap<(&'static str, &'static str), VecDeque<f64>>,
    op_sequential: HashMap<(&'static str, &'static str), VecDeque<f64>>,
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
    seq_median_elements_per_ms: f64,
    p95_elements_per_ms: Option<f64>,
    seq_p95_elements_per_ms: Option<f64>,
    variance_ratio: Option<f64>,
    seq_variance_ratio: Option<f64>,
    sample_count: usize,
    seq_sample_count: usize,
    recommended_cutover: Option<usize>,
    samples: Vec<f64>,
    seq_samples: Vec<f64>,
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

#[derive(Clone, Default)]
struct StrideCounter {
    contiguous: u64,
    strided: u64,
}

static STRIDE_STATE: OnceLock<Mutex<HashMap<&'static str, StrideCounter>>> = OnceLock::new();

type TileHistogram = HashMap<usize, u64>;
static AXIS_TILE_STATE: OnceLock<Mutex<HashMap<AxisKind, TileHistogram>>> = OnceLock::new();

fn stride_state() -> &'static Mutex<HashMap<&'static str, StrideCounter>> {
    STRIDE_STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_stride_event(operation: &'static str, contiguous: bool) {
    if let Ok(mut guard) = stride_state().lock() {
        let entry = guard.entry(operation).or_default();
        if contiguous {
            entry.contiguous = entry.contiguous.saturating_add(1);
        } else {
            entry.strided = entry.strided.saturating_add(1);
        }
    }
}

fn stride_snapshot() -> Vec<(String, StrideCounter)> {
    STRIDE_STATE
        .get()
        .and_then(|mutex| mutex.lock().ok().map(|map| map.clone()))
        .map(|map| {
            map.into_iter()
                .map(|(kind, counter)| (kind.to_string(), counter))
                .collect()
        })
        .unwrap_or_default()
}

fn axis_tile_state() -> &'static Mutex<HashMap<AxisKind, TileHistogram>> {
    AXIS_TILE_STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record_axis_tile(axis: AxisKind, width: usize) {
    if width == 0 {
        return;
    }
    if let Ok(mut guard) = axis_tile_state().lock() {
        let histogram = guard.entry(axis).or_insert_with(HashMap::new);
        *histogram.entry(width).or_insert(0) += 1;
    }
}

fn axis_tile_snapshot() -> HashMap<AxisKind, TileHistogram> {
    AXIS_TILE_STATE
        .get()
        .and_then(|mutex| mutex.lock().ok().map(|map| map.clone()))
        .unwrap_or_default()
}

fn adaptive_state() -> &'static Mutex<AdaptiveThreadingState> {
    ADAPTIVE_STATE.get_or_init(|| Mutex::new(AdaptiveThreadingState::default()))
}

impl AdaptiveThreadingState {
    fn record(&mut self, event: ThreadingEvent) {
        if event.duration_ms > 0.0 && event.elements > 0 {
            let throughput = event.elements as f64 / event.duration_ms;
            if throughput.is_finite() && throughput > 0.0 {
                if event.parallel {
                    Self::push_sample(&mut self.throughput, event.dtype, throughput);
                    if !event.operation.is_empty() {
                        Self::push_sample(
                            &mut self.op_parallel,
                            (event.operation, event.dtype),
                            throughput,
                        );
                    }
                } else {
                    Self::push_sample(&mut self.seq_throughput, event.dtype, throughput);
                    if !event.operation.is_empty() {
                        Self::push_sample(
                            &mut self.op_sequential,
                            (event.operation, event.dtype),
                            throughput,
                        );
                    }
                }
            }
        }
        self.last_event = Some(event);
    }

    fn record_axis(
        &mut self,
        axis: AxisKind,
        dtype: &'static str,
        elements: usize,
        duration_ms: f64,
        parallel: bool,
    ) {
        if duration_ms <= 0.0 || elements == 0 {
            return;
        }
        let throughput = elements as f64 / duration_ms;
        if !throughput.is_finite() || throughput <= 0.0 {
            return;
        }
        let key = (axis, dtype);
        if parallel {
            Self::push_sample(&mut self.axis_parallel, key, throughput);
        } else {
            Self::push_sample(&mut self.axis_sequential, key, throughput);
        }
    }

    fn push_sample<K>(map: &mut HashMap<K, VecDeque<f64>>, key: K, value: f64)
    where
        K: Eq + Hash,
    {
        let deque = map.entry(key).or_default();
        if deque.len() == ADAPTIVE_SAMPLE_WINDOW {
            deque.pop_front();
        }
        deque.push_back(value);
    }

    fn median_from_map<K>(&self, map: &HashMap<K, VecDeque<f64>>, key: K) -> Option<f64>
    where
        K: Eq + Hash + Copy,
    {
        let values = map.get(&key)?;
        Self::median_from_deque(values)
    }

    fn median_from_deque(values: &VecDeque<f64>) -> Option<f64> {
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

    fn percentile_from_deque(values: &VecDeque<f64>, percentile: f64) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.iter().copied().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let pct = percentile.clamp(0.0, 1.0);
        let max_index = sorted.len().saturating_sub(1);
        let position = pct * max_index as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        if lower == upper {
            Some(sorted[lower])
        } else {
            let weight = position - lower as f64;
            Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
        }
    }

    fn median(&self, dtype: &'static str) -> Option<f64> {
        self.median_from_map(&self.throughput, dtype)
    }

    fn median_seq(&self, dtype: &'static str) -> Option<f64> {
        self.median_from_map(&self.seq_throughput, dtype)
    }

    fn percentile_from_map<K>(
        &self,
        map: &HashMap<K, VecDeque<f64>>,
        key: K,
        percentile: f64,
    ) -> Option<f64>
    where
        K: Eq + Hash + Copy,
    {
        let values = map.get(&key)?;
        Self::percentile_from_deque(values, percentile)
    }

    fn percentile(&self, dtype: &'static str, percentile: f64, parallel: bool) -> Option<f64> {
        if parallel {
            self.percentile_from_map(&self.throughput, dtype, percentile)
        } else {
            self.percentile_from_map(&self.seq_throughput, dtype, percentile)
        }
    }

    fn axis_median(&self, axis: AxisKind, dtype: &'static str, parallel: bool) -> Option<f64> {
        let key = (axis, dtype);
        if parallel {
            self.median_from_map(&self.axis_parallel, key)
        } else {
            self.median_from_map(&self.axis_sequential, key)
        }
    }

    fn axis_sample_count(&self, axis: AxisKind, dtype: &'static str, parallel: bool) -> usize {
        let key = (axis, dtype);
        let map = if parallel {
            &self.axis_parallel
        } else {
            &self.axis_sequential
        };
        map.get(&key).map(VecDeque::len).unwrap_or(0)
    }

    fn operation_median(
        &self,
        operation: &'static str,
        dtype: &'static str,
        parallel: bool,
    ) -> Option<f64> {
        let key = (operation, dtype);
        if parallel {
            self.median_from_map(&self.op_parallel, key)
        } else {
            self.median_from_map(&self.op_sequential, key)
        }
    }

    fn operation_sample_count(
        &self,
        operation: &'static str,
        dtype: &'static str,
        parallel: bool,
    ) -> usize {
        let key = (operation, dtype);
        let map = if parallel {
            &self.op_parallel
        } else {
            &self.op_sequential
        };
        map.get(&key).map(VecDeque::len).unwrap_or(0)
    }

    fn operation_percentile(
        &self,
        operation: &'static str,
        dtype: &'static str,
        parallel: bool,
        percentile: f64,
    ) -> Option<f64> {
        let key = (operation, dtype);
        if parallel {
            self.percentile_from_map(&self.op_parallel, key, percentile)
        } else {
            self.percentile_from_map(&self.op_sequential, key, percentile)
        }
    }

    fn recommend_cutover(&self, dtype: &'static str) -> Option<usize> {
        let median = self.median(dtype)?;
        if median <= 0.0 {
            return None;
        }
        let seq_median = self.median_seq(dtype).unwrap_or(0.0);
        let effective_median = if seq_median > 0.0 && seq_median >= median {
            seq_median
        } else {
            median
        };
        let target = target_latency_ms(dtype);
        if target <= 0.0 {
            return None;
        }
        let cutover = (effective_median * target).ceil() as usize;
        Some(cutover.max(1))
    }

    fn recommend_cutover_op(&self, operation: &'static str, dtype: &'static str) -> Option<usize> {
        let median = self.operation_median(operation, dtype, true)?;
        if median <= 0.0 {
            return None;
        }
        let seq_median = self
            .operation_median(operation, dtype, false)
            .unwrap_or(0.0);
        let effective_median = if seq_median > 0.0 && seq_median >= median {
            seq_median
        } else {
            median
        };
        let target = target_latency_ms(dtype);
        if target <= 0.0 {
            return None;
        }
        let cutover = (effective_median * target).ceil() as usize;
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
            let p95 = self.percentile(dtype, 0.95, true);
            let seq_samples_list = self
                .seq_throughput
                .get(dtype)
                .map(|deque| deque.iter().copied().collect::<Vec<f64>>())
                .unwrap_or_default();
            let seq_median = self.median_seq(dtype).unwrap_or(0.0);
            let seq_p95 = self.percentile(dtype, 0.95, false);
            let recommended = self.recommend_cutover(dtype);
            thresholds.push(ThresholdEntry {
                dtype,
                median_elements_per_ms: median,
                seq_median_elements_per_ms: seq_median,
                p95_elements_per_ms: p95,
                seq_p95_elements_per_ms: seq_p95,
                variance_ratio: p95.and_then(|value| {
                    if median > 0.0 {
                        Some(value / median)
                    } else {
                        None
                    }
                }),
                seq_variance_ratio: seq_p95.and_then(|value| {
                    if seq_median > 0.0 {
                        Some(value / seq_median)
                    } else {
                        None
                    }
                }),
                sample_count: samples_list.len(),
                seq_sample_count: seq_samples_list.len(),
                recommended_cutover: recommended,
                samples: samples_list,
                seq_samples: seq_samples_list,
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

fn record_axis_strategy_event(
    strategy: &'static str,
    dtype: &'static str,
    rows: usize,
    cols: usize,
    elapsed: Duration,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let elements = rows.saturating_mul(cols);
    if elements == 0 {
        return;
    }
    let duration_ms = elapsed.as_secs_f64() * 1_000.0;
    if duration_ms <= 0.0 || !duration_ms.is_finite() {
        return;
    }
    let event = ThreadingEvent {
        dtype,
        elements,
        duration_ms,
        tiles: rows,
        partial_buffer: cols,
        parallel: true,
        operation: strategy,
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
    axis: AxisKind,
) {
    if rows == 0 || cols == 0 {
        return;
    }

    let elements = rows.saturating_mul(cols);
    let duration_ms = elapsed.as_secs_f64() * 1_000.0;
    if duration_ms <= 0.0 {
        return;
    }

    if let Ok(mut guard) = adaptive_state().lock() {
        guard.record_axis(axis, dtype, elements, duration_ms, parallel);
    }

    let outcome = reduce::tiled::ReduceOutcome {
        value: 0.0,
        tiles_processed: rows,
        parallel,
        partial_buffer: cols,
    };
    let operation = match axis {
        AxisKind::Axis0 => "axis0",
        AxisKind::Axis1 => "axis1",
    };
    record_threading_event(dtype, elements, elapsed, &outcome, operation);
}

fn record_scale_event(
    dtype: &'static str,
    rows: usize,
    cols: usize,
    elapsed: Duration,
    parallel: bool,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let elements = rows.saturating_mul(cols);
    let duration_ms = elapsed.as_secs_f64() * 1_000.0;
    if duration_ms <= 0.0 {
        return;
    }
    let event = ThreadingEvent {
        dtype,
        elements,
        duration_ms,
        tiles: rows,
        partial_buffer: cols,
        parallel,
        operation: "scale",
    };
    if let Ok(mut guard) = adaptive_state().lock() {
        guard.record(event);
    }
}

fn broadcast_operation_name(kind: BroadcastKind) -> &'static str {
    match kind {
        BroadcastKind::Row => OPERATION_BROADCAST_ROW,
        BroadcastKind::Column => OPERATION_BROADCAST_COL,
    }
}

fn record_broadcast_event(
    dtype: &'static str,
    rows: usize,
    cols: usize,
    elapsed: Duration,
    parallel: bool,
    kind: BroadcastKind,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let elements = rows.saturating_mul(cols);
    let duration_ms = elapsed.as_secs_f64() * 1_000.0;
    if duration_ms <= 0.0 {
        return;
    }
    let event = ThreadingEvent {
        dtype,
        elements,
        duration_ms,
        tiles: rows,
        partial_buffer: cols,
        parallel,
        operation: broadcast_operation_name(kind),
    };
    if let Ok(mut guard) = adaptive_state().lock() {
        guard.record(event);
    }
}

fn scale_parallel_policy(dtype: &'static str) -> (usize, bool) {
    let mut cutover = match dtype {
        "float64" => SCALE_PAR_MIN_ELEMS_F64,
        _ => SCALE_PAR_MIN_ELEMS,
    };
    let mut prefer_parallel = true;
    if let Ok(guard) = adaptive_state().lock() {
        if let Some(recommended) = guard.recommend_cutover_op(OPERATION_SCALE, dtype) {
            cutover = cutover.max(recommended);
        } else if let Some(recommended) = guard.recommend_cutover(dtype) {
            cutover = cutover.max(recommended);
        }
        let par_samples = guard.operation_sample_count(OPERATION_SCALE, dtype, true);
        let seq_samples = guard.operation_sample_count(OPERATION_SCALE, dtype, false);
        if seq_samples >= 6 && par_samples == 0 {
            prefer_parallel = false;
        }
        if par_samples >= 2 && seq_samples >= 1 {
            if let (Some(par_median), Some(seq_median)) = (
                guard.operation_median(OPERATION_SCALE, dtype, true),
                guard.operation_median(OPERATION_SCALE, dtype, false),
            ) {
                if par_median <= seq_median * 1.05 {
                    prefer_parallel = false;
                }
                if let Some(par_p95) =
                    guard.operation_percentile(OPERATION_SCALE, dtype, true, 0.95)
                {
                    if par_median > 0.0 && par_p95 >= par_median * 1.35 {
                        prefer_parallel = false;
                    }
                }
                if let Some(seq_p95) =
                    guard.operation_percentile(OPERATION_SCALE, dtype, false, 0.95)
                {
                    if seq_median > 0.0 && seq_p95 <= seq_median * 1.05 {
                        prefer_parallel = false;
                    }
                }
                let target = target_latency_ms(dtype);
                if target > 0.0 {
                    let effective = par_median.max(seq_median);
                    let op_cutover = (effective * target).ceil() as usize;
                    cutover = cutover.max(op_cutover.max(1));
                }
                if let (Some(par_p95), Some(seq_p95)) = (
                    guard.operation_percentile(OPERATION_SCALE, dtype, true, 0.95),
                    guard.operation_percentile(OPERATION_SCALE, dtype, false, 0.95),
                ) {
                    if par_p95 <= seq_p95 * 1.08 {
                        prefer_parallel = false;
                    }
                    let target = target_latency_ms(dtype);
                    if target > 0.0 {
                        let worst = par_p95.max(seq_p95);
                        let op_cutover = (worst * target).ceil() as usize;
                        cutover = cutover.max(op_cutover.max(1));
                    }
                } else if let Some(par_p95) =
                    guard.operation_percentile(OPERATION_SCALE, dtype, true, 0.95)
                {
                    let target = target_latency_ms(dtype);
                    if target > 0.0 {
                        let op_cutover = (par_p95 * target).ceil() as usize;
                        cutover = cutover.max(op_cutover.max(1));
                    }
                } else if let Some(seq_p95) =
                    guard.operation_percentile(OPERATION_SCALE, dtype, false, 0.95)
                {
                    let target = target_latency_ms(dtype);
                    if target > 0.0 {
                        let op_cutover = (seq_p95 * target).ceil() as usize;
                        cutover = cutover.max(op_cutover.max(1));
                    }
                    prefer_parallel = false;
                }
            }
        } else if let (Some(par_median), Some(seq_median)) =
            (guard.median(dtype), guard.median_seq(dtype))
        {
            if par_median <= seq_median * 1.05 {
                prefer_parallel = false;
            }
            if let Some(par_p95) = guard.percentile(dtype, 0.95, true) {
                if par_median > 0.0 && par_p95 >= par_median * 1.35 {
                    prefer_parallel = false;
                }
            }
            if let Some(seq_p95) = guard.percentile(dtype, 0.95, false) {
                if seq_median > 0.0 && seq_p95 <= seq_median * 1.05 {
                    prefer_parallel = false;
                }
            }
        }
    }
    (cutover, prefer_parallel)
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

fn axis_parallel_cutover(axis: AxisKind, dtype: &'static str) -> usize {
    match (axis, dtype) {
        (AxisKind::Axis0, "float64") => 2048,
        (AxisKind::Axis0, "float32") => 1536,
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

fn axis_parallel_policy(
    axis: AxisKind,
    dtype: &'static str,
    rows: usize,
    cols: usize,
    allow_parallel: bool,
) -> bool {
    if !allow_parallel || rows == 0 || cols == 0 {
        return false;
    }

    if let Ok(guard) = adaptive_state().lock() {
        let seq_samples = guard.axis_sample_count(axis, dtype, false);
        // Prefer sequential execution for smaller float32 axis-0 workloads when data suggests parity.
        if axis == AxisKind::Axis0 && dtype == "float32" && rows <= 1024 {
            if seq_samples >= 3 {
                if let (Some(par_median), Some(seq_median)) = (
                    guard.axis_median(axis, dtype, true),
                    guard.axis_median(axis, dtype, false),
                ) {
                    if par_median <= seq_median * 1.05 {
                        return false;
                    }
                }
            }
        }

        if let (Some(par), Some(seq)) = (
            guard.axis_median(axis, dtype, true),
            guard.axis_median(axis, dtype, false),
        ) {
            if seq >= par * 0.98 && rows.saturating_mul(cols) <= PARALLEL_MIN_ELEMENTS * 2 {
                return false;
            }
        }
    }

    true
}

fn broadcast_parallel_policy(
    kind: BroadcastKind,
    dtype: &'static str,
    rows: usize,
    cols: usize,
    allow_parallel: bool,
) -> bool {
    if !allow_parallel {
        return false;
    }

    let elements = rows.saturating_mul(cols);
    match kind {
        BroadcastKind::Row => {
            if cols < BROADCAST_PAR_MIN_COLS || elements < PARALLEL_MIN_ELEMENTS {
                return false;
            }
        }
        BroadcastKind::Column => {
            if rows < BROADCAST_PAR_MIN_ROWS && cols < BROADCAST_PAR_MIN_COLS {
                return false;
            }
        }
    }

    let operation = broadcast_operation_name(kind);
    if let Ok(guard) = adaptive_state().lock() {
        if kind == BroadcastKind::Row && dtype == "float32" && rows >= 1024 {
            return true;
        }

        let seq_samples = guard.operation_sample_count(operation, dtype, false);
        let par_samples = guard.operation_sample_count(operation, dtype, true);

        if par_samples == 0 && seq_samples >= 6 && rows <= 768 && cols <= 768 {
            return false;
        }

        if par_samples >= 2 && seq_samples >= 1 {
            if let (Some(par), Some(seq)) = (
                guard.operation_median(operation, dtype, true),
                guard.operation_median(operation, dtype, false),
            ) {
                if kind == BroadcastKind::Column && dtype == "float32" && rows <= 1024 {
                    if par <= seq * 1.08 {
                        return false;
                    }
                }
                if seq >= par * 0.99 {
                    return false;
                }
            }
        }
    }

    true
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
        let threads = pool.current_num_threads().max(1);
        let base_rows = ((rows + threads - 1) / threads).max(SCALE_PAR_MIN_ROWS);
        let max_rows = (SCALE_PAR_MAX_CHUNK_ELEMS / cols)
            .max(SCALE_PAR_MIN_ROWS)
            .min(rows.max(1));
        let min_rows = (SCALE_PAR_MIN_CHUNK_ELEMS / cols)
            .max(SCALE_PAR_MIN_ROWS)
            .min(rows.max(1));
        let chunk_rows = base_rows.clamp(min_rows, max_rows);
        let chunk_elems = cols.saturating_mul(chunk_rows.max(1)).min(input.len());
        let start = Instant::now();
        pool.install(|| {
            use rayon::prelude::*;
            out.par_chunks_mut(chunk_elems)
                .enumerate()
                .for_each(|(chunk_index, dst_block)| {
                    let start = chunk_index * chunk_elems;
                    let end = start + dst_block.len();
                    let src_block = &input[start..end];
                    if !simd::scale_same_shape_f64(src_block, factor, dst_block) {
                        for (dst, &value) in dst_block.iter_mut().zip(src_block.iter()) {
                            *dst = value * factor;
                        }
                    }
                });
        });
        record_scale_event("float64", rows, cols, start.elapsed(), true);
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
        let threads = pool.current_num_threads().max(1);
        let base_rows = ((rows + threads - 1) / threads).max(SCALE_PAR_MIN_ROWS);
        let max_rows = (SCALE_PAR_MAX_CHUNK_ELEMS / cols)
            .max(SCALE_PAR_MIN_ROWS)
            .min(rows.max(1));
        let min_rows = (SCALE_PAR_MIN_CHUNK_ELEMS / cols)
            .max(SCALE_PAR_MIN_ROWS)
            .min(rows.max(1));
        let chunk_rows = base_rows.clamp(min_rows, max_rows);
        let chunk_elems = cols.saturating_mul(chunk_rows.max(1)).min(input.len());
        let start = Instant::now();
        pool.install(|| {
            use rayon::prelude::*;
            out.par_chunks_mut(chunk_elems)
                .enumerate()
                .for_each(|(chunk_index, dst_block)| {
                    let start = chunk_index * chunk_elems;
                    let end = start + dst_block.len();
                    let src_block = &input[start..end];
                    if !simd::scale_same_shape_f32(src_block, factor_f32, dst_block) {
                        for (dst, &value) in dst_block.iter_mut().zip(src_block.iter()) {
                            *dst = value * factor_f32;
                        }
                    }
                });
        });
        record_scale_event("float32", rows, cols, start.elapsed(), true);
        true
    } else {
        false
    }
}

#[allow(dead_code)]
#[inline]
fn add_assign_f64(acc: &mut [f64], row: &[f64]) {
    if !simd::add_assign_inplace_f64(acc, row) {
        for (dst, &value) in acc.iter_mut().zip(row.iter()) {
            *dst += value;
        }
    }
}

#[allow(dead_code)]
#[inline]
fn accumulate_row_f32(acc: &mut [f64], row: &[f32]) {
    for (dst, &value) in acc.iter_mut().zip(row.iter()) {
        *dst += value as f64;
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
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
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
                let total_len = lhs_slice.len();
                let base_parallel =
                    should_parallelize(rows, cols, T::DTYPE_NAME) || cols >= BROADCAST_PAR_MIN_COLS;
                let allow_parallel = broadcast_parallel_policy(
                    BroadcastKind::Row,
                    T::DTYPE_NAME,
                    rows,
                    cols,
                    base_parallel,
                );
                if T::DTYPE_NAME == "float64" {
                    let mut data = Vec::with_capacity(total_len);
                    unsafe {
                        data.set_len(total_len);
                    }
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
                    let start = Instant::now();
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
                        record_broadcast_event(
                            T::DTYPE_NAME,
                            rows,
                            cols,
                            start.elapsed(),
                            false,
                            BroadcastKind::Row,
                        );
                        return Some(NumericArray::new_owned(data, self.shape.clone()));
                    }
                } else if T::DTYPE_NAME == "float32" {
                    if rows <= 1024 {
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
                        let mut data = Vec::<T>::with_capacity(total_len);
                        unsafe {
                            data.set_len(total_len);
                        }
                        let out = unsafe {
                            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, total_len)
                        };
                        let start = Instant::now();
                        for (out_row, lhs_row) in out.chunks_mut(cols).zip(lhs.chunks(cols)) {
                            if !simd::add_same_shape_f32(lhs_row, rhs, out_row) {
                                for (dst, (&l, &r)) in
                                    out_row.iter_mut().zip(lhs_row.iter().zip(rhs.iter()))
                                {
                                    *dst = l + r;
                                }
                            }
                        }
                        record_broadcast_event(
                            T::DTYPE_NAME,
                            rows,
                            cols,
                            start.elapsed(),
                            false,
                            BroadcastKind::Row,
                        );
                        return Some(NumericArray::new_owned(data, self.shape.clone()));
                    }
                    let mut data = Vec::with_capacity(total_len);
                    unsafe {
                        data.set_len(total_len);
                    }
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
                    if allow_parallel && rows >= 1536 {
                        if let Some(pool) = thread_pool() {
                            let start = Instant::now();
                            let row_block = 32usize;
                            pool.install(|| {
                                use rayon::prelude::*;
                                out.par_chunks_mut(cols * row_block)
                                    .zip(lhs.par_chunks(cols * row_block))
                                    .for_each(|(out_block, lhs_block)| {
                                        for (out_row, lhs_row) in
                                            out_block.chunks_mut(cols).zip(lhs_block.chunks(cols))
                                        {
                                            if !simd::add_same_shape_f32(lhs_row, rhs, out_row) {
                                                for ((dst, &l), &r) in out_row
                                                    .iter_mut()
                                                    .zip(lhs_row.iter())
                                                    .zip(rhs.iter())
                                                {
                                                    *dst = l + r;
                                                }
                                            }
                                        }
                                    });
                            });
                            record_broadcast_event(
                                T::DTYPE_NAME,
                                rows,
                                cols,
                                start.elapsed(),
                                true,
                                BroadcastKind::Row,
                            );
                            return Some(NumericArray::new_owned(data, self.shape.clone()));
                        }
                    }
                    let start = Instant::now();
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
                        record_broadcast_event(
                            T::DTYPE_NAME,
                            rows,
                            cols,
                            start.elapsed(),
                            false,
                            BroadcastKind::Row,
                        );
                        return Some(NumericArray::new_owned(data, self.shape.clone()));
                    }
                }

                let mut data = Vec::with_capacity(total_len);
                unsafe {
                    data.set_len(total_len);
                }
                if allow_parallel {
                    let start = Instant::now();
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
                        record_broadcast_event(
                            T::DTYPE_NAME,
                            rows,
                            cols,
                            start.elapsed(),
                            true,
                            BroadcastKind::Row,
                        );
                        return Some(NumericArray::new_owned(data, self.shape.clone()));
                    }
                }
                let start = Instant::now();
                for row in 0..rows {
                    let start = row * cols;
                    let end = start + cols;
                    T::simd_add(&lhs_slice[start..end], rhs_slice, &mut data[start..end]);
                }
                record_broadcast_event(
                    T::DTYPE_NAME,
                    rows,
                    cols,
                    start.elapsed(),
                    false,
                    BroadcastKind::Row,
                );
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
        let total_len = lhs_slice.len();
        if total_len == 0 {
            return Some(NumericArray::new_owned(Vec::new(), self.shape.clone()));
        }
        if T::DTYPE_NAME == "float64" {
            let lhs = unsafe { slice::from_raw_parts(lhs_slice.as_ptr() as *const f64, total_len) };
            let rhs = unsafe { slice::from_raw_parts(rhs_slice.as_ptr() as *const f64, rows) };
            let total_elems = rows.saturating_mul(cols);
            if total_elems < COLUMN_BROADCAST_DIRECT_MIN_ELEMS {
                let mut data = Vec::<f64>::with_capacity(total_len);
                unsafe {
                    data.set_len(total_len);
                    std::ptr::copy_nonoverlapping(lhs.as_ptr(), data.as_mut_ptr(), total_len);
                }
                for (row_idx, scalar) in rhs.iter().enumerate() {
                    let start = row_idx * cols;
                    let row_slice = &mut data[start..start + cols];
                    for value in row_slice.iter_mut() {
                        *value += *scalar;
                    }
                }
                let out_vec = unsafe {
                    let ptr = data.as_mut_ptr() as *mut T;
                    let len = data.len();
                    let cap = data.capacity();
                    std::mem::forget(data);
                    Vec::from_raw_parts(ptr, len, cap)
                };
                return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
            }
            let mut data = Vec::<f64>::with_capacity(total_len);
            unsafe {
                data.set_len(total_len);
            }
            let base_parallel =
                should_parallelize(rows, cols, T::DTYPE_NAME) || rows >= BROADCAST_PAR_MIN_ROWS;
            let allow_parallel = broadcast_parallel_policy(
                BroadcastKind::Column,
                T::DTYPE_NAME,
                rows,
                cols,
                base_parallel,
            );
            if allow_parallel {
                if let Some(pool) = thread_pool() {
                    let threads = pool.current_num_threads().max(1);
                    let min_rows = ((rows + threads - 1) / threads).max(BROADCAST_PAR_MIN_ROWS);
                    let start = Instant::now();
                    pool.install(|| {
                        use rayon::prelude::*;
                        data.par_chunks_mut(cols)
                            .with_min_len(min_rows)
                            .zip(lhs.par_chunks(cols).with_min_len(min_rows))
                            .zip(rhs.par_iter().with_min_len(min_rows).copied())
                            .for_each(|((out_row, lhs_row), scalar)| {
                                if !simd::add_row_scalar_f64(lhs_row, scalar, out_row) {
                                    for (dst, &value) in out_row.iter_mut().zip(lhs_row.iter()) {
                                        *dst = value + scalar;
                                    }
                                }
                            });
                    });
                    record_broadcast_event(
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        start.elapsed(),
                        true,
                        BroadcastKind::Column,
                    );
                    let out_vec = unsafe {
                        let ptr = data.as_mut_ptr() as *mut T;
                        let len = data.len();
                        let cap = data.capacity();
                        std::mem::forget(data);
                        Vec::from_raw_parts(ptr, len, cap)
                    };
                    return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
                }
            }
            if total_elems >= COLUMN_BROADCAST_DIRECT_MIN_ELEMS {
                let start = Instant::now();
                if simd::add_column_broadcast_f64(lhs, rhs, rows, cols, data.as_mut_slice()) {
                    record_broadcast_event(
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        start.elapsed(),
                        false,
                        BroadcastKind::Column,
                    );
                    let out_vec = unsafe {
                        let ptr = data.as_mut_ptr() as *mut T;
                        let len = data.len();
                        let cap = data.capacity();
                        std::mem::forget(data);
                        Vec::from_raw_parts(ptr, len, cap)
                    };
                    return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
                }
            }
            let start = Instant::now();
            for (row_idx, chunk) in lhs.chunks(cols).enumerate() {
                let scalar = rhs[row_idx];
                let out_row = &mut data[row_idx * cols..(row_idx + 1) * cols];
                if !simd::add_row_scalar_f64(chunk, scalar, out_row) {
                    out_row.copy_from_slice(chunk);
                    for value in out_row.iter_mut() {
                        *value += scalar;
                    }
                }
            }
            record_broadcast_event(
                T::DTYPE_NAME,
                rows,
                cols,
                start.elapsed(),
                false,
                BroadcastKind::Column,
            );
            let out_vec = unsafe {
                let ptr = data.as_mut_ptr() as *mut T;
                let len = data.len();
                let cap = data.capacity();
                std::mem::forget(data);
                Vec::from_raw_parts(ptr, len, cap)
            };
            return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
        }
        if T::DTYPE_NAME == "float32" {
            let lhs = unsafe { slice::from_raw_parts(lhs_slice.as_ptr() as *const f32, total_len) };
            let rhs = unsafe { slice::from_raw_parts(rhs_slice.as_ptr() as *const f32, rows) };
            let total_elems = rows.saturating_mul(cols);
            if total_elems < COLUMN_BROADCAST_DIRECT_MIN_ELEMS {
                let mut data = Vec::<f32>::with_capacity(total_len);
                unsafe {
                    data.set_len(total_len);
                    std::ptr::copy_nonoverlapping(lhs.as_ptr(), data.as_mut_ptr(), total_len);
                }
                for (row_idx, scalar) in rhs.iter().enumerate() {
                    let start = row_idx * cols;
                    let row_slice = &mut data[start..start + cols];
                    for value in row_slice.iter_mut() {
                        *value += *scalar;
                    }
                }
                let out_vec = unsafe {
                    let ptr = data.as_mut_ptr() as *mut T;
                    let len = data.len();
                    let cap = data.capacity();
                    std::mem::forget(data);
                    Vec::from_raw_parts(ptr, len, cap)
                };
                return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
            }
            let mut data = Vec::<f32>::with_capacity(total_len);
            unsafe {
                data.set_len(total_len);
            }
            let base_parallel =
                should_parallelize(rows, cols, T::DTYPE_NAME) || rows >= BROADCAST_PAR_MIN_ROWS;
            let allow_parallel = broadcast_parallel_policy(
                BroadcastKind::Column,
                T::DTYPE_NAME,
                rows,
                cols,
                base_parallel,
            );
            if allow_parallel {
                if let Some(pool) = thread_pool() {
                    let threads = pool.current_num_threads().max(1);
                    let min_rows = ((rows + threads - 1) / threads).max(BROADCAST_PAR_MIN_ROWS);
                    let start = Instant::now();
                    pool.install(|| {
                        use rayon::prelude::*;
                        data.par_chunks_mut(cols)
                            .with_min_len(min_rows)
                            .zip(lhs.par_chunks(cols).with_min_len(min_rows))
                            .zip(rhs.par_iter().with_min_len(min_rows).copied())
                            .for_each(|((out_row, lhs_row), scalar)| {
                                if !simd::add_row_scalar_f32(lhs_row, scalar, out_row) {
                                    for (dst, &value) in out_row.iter_mut().zip(lhs_row.iter()) {
                                        *dst = value + scalar;
                                    }
                                }
                            });
                    });
                    record_broadcast_event(
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        start.elapsed(),
                        true,
                        BroadcastKind::Column,
                    );
                    let out_vec = unsafe {
                        let ptr = data.as_mut_ptr() as *mut T;
                        let len = data.len();
                        let cap = data.capacity();
                        std::mem::forget(data);
                        Vec::from_raw_parts(ptr, len, cap)
                    };
                    return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
                }
            }
            if total_elems >= COLUMN_BROADCAST_DIRECT_MIN_ELEMS {
                let start = Instant::now();
                if simd::add_column_broadcast_f32(lhs, rhs, rows, cols, data.as_mut_slice()) {
                    record_broadcast_event(
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        start.elapsed(),
                        false,
                        BroadcastKind::Column,
                    );
                    let out_vec = unsafe {
                        let ptr = data.as_mut_ptr() as *mut T;
                        let len = data.len();
                        let cap = data.capacity();
                        std::mem::forget(data);
                        Vec::from_raw_parts(ptr, len, cap)
                    };
                    return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
                }
            }
            let start = Instant::now();
            for (row_idx, chunk) in lhs.chunks(cols).enumerate() {
                let scalar = rhs[row_idx];
                let out_row = &mut data[row_idx * cols..(row_idx + 1) * cols];
                if !simd::add_row_scalar_f32(chunk, scalar, out_row) {
                    out_row.copy_from_slice(chunk);
                    for value in out_row.iter_mut() {
                        *value += scalar;
                    }
                }
            }
            record_broadcast_event(
                T::DTYPE_NAME,
                rows,
                cols,
                start.elapsed(),
                false,
                BroadcastKind::Column,
            );
            let out_vec = unsafe {
                let ptr = data.as_mut_ptr() as *mut T;
                let len = data.len();
                let cap = data.capacity();
                std::mem::forget(data);
                Vec::from_raw_parts(ptr, len, cap)
            };
            return Some(NumericArray::new_owned(out_vec, self.shape.clone()));
        }
        let mut data = vec![T::zero(); total_len];
        for row in 0..rows {
            let start = row * cols;
            let scalar = rhs_slice[row];
            for col in 0..cols {
                let idx = start + col;
                data[idx] = lhs_slice[idx] + scalar;
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

    fn data_slice_mut(&mut self) -> Option<&mut [T]> {
        match &mut self.storage {
            NumericStorage::Owned(data) => Some(data.as_mut_slice()),
            NumericStorage::Borrowed { .. } => None,
        }
    }

    fn is_owned(&self) -> bool {
        matches!(self.storage, NumericStorage::Owned(_))
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

        record_stride_event("scale", self.is_contiguous());

        let len = self.data_len();
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
        let (rows, cols) = self.matrix_dims();
        let start = Instant::now();
        if T::DTYPE_NAME == "float64" {
            let input = self.data_slice();
            let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, len) };
            if rows >= SCALE_FORCE_PARALLEL_ROWS
                && cols >= SCALE_FORCE_PARALLEL_COLS
                && parallel_scale_f64(input, factor, out, rows, cols)
            {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if blas::scale_enabled()
                && len >= SCALE_BLAS_MIN_LEN
                && rows >= SCALE_BLAS_MIN_ROWS
                && cols >= SCALE_BLAS_MIN_COLS
            {
                out.copy_from_slice(input);
                if blas::current_backend().dscal_f64(len, factor, out) {
                    record_scale_event("float64", rows, cols, start.elapsed(), false);
                    return Ok(NumericArray::new_owned(data, self.shape.clone()));
                }
            }
            if len <= SMALL_DIRECT_THRESHOLD {
                if !simd::scale_same_shape_f64(input, factor, out) {
                    for (dst, &value) in out.iter_mut().zip(input.iter()) {
                        *dst = value * factor;
                    }
                }
                record_scale_event("float64", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if rows <= SMALL_MATRIX_FAST_DIM && cols <= SMALL_MATRIX_FAST_DIM {
                if !simd::scale_same_shape_f64(input, factor, out) {
                    for (dst, &value) in out.iter_mut().zip(input.iter()) {
                        *dst = value * factor;
                    }
                }
                record_scale_event("float64", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if rows <= SCALE_TINY_DIM && cols <= SCALE_TINY_DIM {
                if !simd::scale_same_shape_f64(input, factor, out) {
                    for (dst, &value) in out.iter_mut().zip(input.iter()) {
                        *dst = value * factor;
                    }
                }
                record_scale_event("float64", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            let (parallel_cutover, prefer_parallel) = scale_parallel_policy("float64");
            let base_parallel = should_parallelize(rows, cols, T::DTYPE_NAME);
            let elements = rows.saturating_mul(cols);
            let should_try_parallel =
                base_parallel && elements >= parallel_cutover && prefer_parallel;
            let force_parallel = base_parallel
                && elements >= SCALE_FORCE_PARALLEL_ELEMS
                && elements >= parallel_cutover;
            let allow_parallel = (should_try_parallel || force_parallel)
                && parallel_scale_f64(input, factor, out, rows, cols);
            if allow_parallel {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if simd::scale_same_shape_f64(input, factor, out) {
                record_scale_event("float64", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
        } else if T::DTYPE_NAME == "float32" {
            let input = self.data_slice();
            let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, len) };
            let factor_f32 = factor as f32;
            let use_blas = blas::scale_override().unwrap_or(false)
                && len >= SCALE_BLAS_MIN_LEN
                && rows >= SCALE_BLAS_MIN_ROWS
                && cols >= SCALE_BLAS_MIN_COLS;
            if use_blas {
                out.copy_from_slice(input);
                if blas::current_backend().sscal_f32(len, factor_f32, out) {
                    record_scale_event("float32", rows, cols, start.elapsed(), false);
                    return Ok(NumericArray::new_owned(data, self.shape.clone()));
                }
            }
            if len <= SMALL_DIRECT_THRESHOLD {
                if !simd::scale_same_shape_f32(input, factor_f32, out) {
                    for (dst, &value) in out.iter_mut().zip(input.iter()) {
                        *dst = value * factor_f32;
                    }
                }
                record_scale_event("float32", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if rows <= SMALL_MATRIX_FAST_DIM && cols <= SMALL_MATRIX_FAST_DIM {
                if !simd::scale_same_shape_f32(input, factor_f32, out) {
                    for (dst, &value) in out.iter_mut().zip(input.iter()) {
                        *dst = value * factor_f32;
                    }
                }
                record_scale_event("float32", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if rows <= SCALE_TINY_DIM && cols <= SCALE_TINY_DIM {
                if !simd::scale_same_shape_f32(input, factor_f32, out) {
                    for (dst, &value) in out.iter_mut().zip(input.iter()) {
                        *dst = value * factor_f32;
                    }
                }
                record_scale_event("float32", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            let (parallel_cutover, prefer_parallel) = scale_parallel_policy("float32");
            let elements = rows.saturating_mul(cols);
            let base_parallel = should_parallelize(rows, cols, T::DTYPE_NAME)
                || (rows >= SCALE_PAR_MIN_ROWS * 4 && cols >= BROADCAST_PAR_MIN_COLS)
                || (rows >= SCALE_PAR_MIN_ROWS * 2 && elements >= PARALLEL_MIN_ELEMENTS / 2);
            let should_try_parallel =
                base_parallel && elements >= parallel_cutover && prefer_parallel;
            let force_parallel = base_parallel
                && elements >= SCALE_FORCE_PARALLEL_ELEMS
                && elements >= parallel_cutover;
            let allow_parallel = (should_try_parallel || force_parallel)
                && parallel_scale_f32(input, factor, out, rows, cols);
            if allow_parallel {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
            if simd::scale_same_shape_f32(input, factor_f32, out) {
                record_scale_event("float32", rows, cols, start.elapsed(), false);
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
        }
        T::simd_scale(self.data_slice(), factor, &mut data).map_err(|_| {
            PyValueError::new_err(format!(
                "scaling produced values outside {} representable range",
                T::DTYPE_NAME
            ))
        })?;
        record_scale_event(T::DTYPE_NAME, rows, cols, start.elapsed(), false);
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
                    record_stride_event("axis0", true);
                    let base_allow =
                        should_parallelize_axis(AxisKind::Axis0, rows, cols, T::DTYPE_NAME);
                    let allow_parallel = axis_parallel_policy(
                        AxisKind::Axis0,
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        base_allow,
                    );
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
                        AxisKind::Axis0,
                    );
                    return self.convert_reduction_from_f64(outcome.values, vec![cols], op);
                }
                record_stride_event("axis0", false);
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
                    record_stride_event("axis1", true);
                    let base_allow =
                        should_parallelize_axis(AxisKind::Axis1, rows, cols, T::DTYPE_NAME);
                    let allow_parallel = axis_parallel_policy(
                        AxisKind::Axis1,
                        T::DTYPE_NAME,
                        rows,
                        cols,
                        base_allow,
                    );
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
                        AxisKind::Axis1,
                    );
                    return self.convert_reduction_from_f64(outcome.values, vec![rows], op);
                }
                record_stride_event("axis1", false);
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

fn scale_microkernel_inplace_f32(data: &mut [f32], factor: f32) {
    if factor == 1.0 {
        return;
    }
    let len = data.len();
    let input = unsafe { std::slice::from_raw_parts(data.as_ptr(), len) };
    if !simd::scale_same_shape_f32(input, factor, data) {
        for value in data.iter_mut() {
            *value *= factor;
        }
    }
}

fn add_row_scalar_f32(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

fn broadcast_row_microkernel_inplace_f32(data: &mut [f32], rhs: &[f32], rows: usize, cols: usize) {
    if cols == 0 || rows == 0 {
        return;
    }
    let mut iter = data.chunks_exact_mut(cols);
    for row in &mut iter {
        if !simd::add_assign_inplace_f32(row, rhs) {
            add_row_scalar_f32(row, rhs);
        }
    }
    let remainder = iter.into_remainder();
    if !remainder.is_empty() {
        let len = remainder.len().min(rhs.len());
        with_f32_scratch(len, |scratch| {
            scratch[..len].copy_from_slice(&rhs[..len]);
            for (dst, src) in remainder.iter_mut().zip(&scratch[..len]) {
                *dst += *src;
            }
        });
    }
}

impl NumericArray<f32> {
    fn small_scale_inplace(&mut self, factor: f64) -> PyResult<bool> {
        if !self.is_owned() || !self.is_contiguous() {
            return Ok(false);
        }
        record_stride_event("scale_inplace", true);
        let factor_f32 = factor as f32;
        if !factor_f32.is_finite() {
            return Ok(false);
        }
        let len = self.data_len();
        if len == 0 {
            return Ok(false);
        }
        let (rows, cols) = self.matrix_dims();
        if rows > SMALL_MICRO_DIM || cols > SMALL_MICRO_DIM {
            return Ok(false);
        }
        let data = self
            .data_slice_mut()
            .ok_or_else(|| PyValueError::new_err("scale_inplace requires an owned buffer"))?;
        let start = Instant::now();
        scale_microkernel_inplace_f32(data, factor_f32);
        record_scale_event("float32", rows, cols, start.elapsed(), false);
        Ok(true)
    }

    fn small_broadcast_row_inplace(&mut self, rhs: &NumericArray<f32>) -> PyResult<bool> {
        if !self.is_owned() || !self.is_contiguous() || !rhs.is_contiguous() {
            return Ok(false);
        }
        let (rows, cols) = self.matrix_dims();
        if rows == 0 || cols == 0 {
            return Ok(false);
        }
        if rows > SMALL_MICRO_DIM || cols > SMALL_MICRO_DIM {
            return Ok(false);
        }
        let rhs_slice = match rhs.shape.as_slice() {
            [len] if *len == cols => rhs.data_slice(),
            [1, len] if *len == cols => rhs.data_slice(),
            [len, 1] if *len == rows => return Ok(false),
            [len, col] if len * col == cols => {
                return Ok(false);
            }
            _ => return Ok(false),
        };
        let data = self.data_slice_mut().ok_or_else(|| {
            PyValueError::new_err("broadcast_add_inplace requires an owned buffer")
        })?;
        let start = Instant::now();
        broadcast_row_microkernel_inplace_f32(data, rhs_slice, rows, cols);
        record_broadcast_event(
            "float32",
            rows,
            cols,
            start.elapsed(),
            false,
            BroadcastKind::Row,
        );
        Ok(true)
    }
}

fn add_row_inplace_f64(dst: &mut [f64], src: &[f64]) {
    if !simd::add_assign_inplace_f64(dst, src) {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += s;
        }
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

    if rows <= SMALL_MATRIX_FAST_DIM && cols <= SMALL_MATRIX_FAST_DIM {
        let mut sums_stack = [0.0f64; SMALL_MATRIX_FAST_DIM];
        let mut comps_stack = [0.0f64; SMALL_MATRIX_FAST_DIM];
        {
            let sums_slice = &mut sums_stack[..cols];
            let comps_slice = &mut comps_stack[..cols];
            for row in data.chunks_exact(cols) {
                kahan_accumulate_row_f64(sums_slice, comps_slice, row);
            }
        }
        let mut sums = sums_stack[..cols].to_vec();
        for (idx, value) in comps_stack[..cols].iter().enumerate() {
            sums[idx] += *value;
        }
        if matches!(op, Reduction::Mean) {
            let inv = 1.0 / rows as f64;
            for value in &mut sums {
                *value *= inv;
            }
        }
        return AxisOutcome {
            values: sums,
            parallel: false,
        };
    }

    if blas::axis0_enabled()
        && rows >= AXIS0_BLAS_MIN_ROWS_F64
        && cols >= AXIS0_BLAS_MIN_COLS
        && rows <= AXIS0_BLAS_MAX_ROWS_F64
    {
        let mut sums = vec![0.0f64; cols];
        if blas::current_backend().dgemv_axis0_sum(rows, cols, data, &mut sums) {
            if matches!(op, Reduction::Mean) {
                let inv = 1.0 / rows as f64;
                for value in &mut sums {
                    *value *= inv;
                }
            }
            return AxisOutcome {
                values: sums,
                parallel: false,
            };
        }
    }

    if matrix_backend_enabled()
        && rows >= MATRIX_AXIS0_ROW_THRESHOLD_F64
        && cols >= MATRIX_AXIS0_COL_THRESHOLD
        && cols <= MATRIX_AXIS0_MATRIX_MAX_COLS
    {
        if allow_parallel {
            if let Some(pool) = thread_pool() {
                if let Some(mut sums) =
                    reduce_axis0_parallel_matrix_f64(&pool, data, rows, cols)
                {
                    if matches!(op, Reduction::Mean) {
                        let inv = 1.0 / rows as f64;
                        for value in &mut sums {
                            *value *= inv;
                        }
                    }
                    return AxisOutcome {
                        values: sums,
                        parallel: true,
                    };
                }
            }
        }
        let mut sums = reduce_axis0_columns_matrix_f64(data, rows, cols);
        if matches!(op, Reduction::Mean) {
            let inv = 1.0 / rows as f64;
            for value in &mut sums {
                *value *= inv;
            }
        }
        return AxisOutcome {
            values: sums,
            parallel: false,
        };
    }

    if cols <= AXIS0_SIMD_COL_LIMIT {
        let mut parallel_used = false;
        let mut sums = Vec::<f64>::new();
        if allow_parallel
            && rows >= AXIS0_PAR_MIN_ROWS_F64
            && rows >= AXIS0_PAR_MIN_ROWS
            && rows * cols >= PARALLEL_MIN_ELEMENTS
        {
            if let Some(pool) = thread_pool() {
                parallel_used = true;
                sums = pool.install(|| {
                    use rayon::prelude::*;
                    data.par_chunks(cols)
                        .with_min_len(AXIS0_PAR_MIN_ROWS)
                        .fold(
                            || vec![0.0f64; cols],
                            |mut acc, row| {
                                add_row_inplace_f64(&mut acc, row);
                                acc
                            },
                        )
                        .reduce(
                            || vec![0.0f64; cols],
                            |mut acc, partial| {
                                add_row_inplace_f64(&mut acc, partial.as_slice());
                                acc
                            },
                        )
                });
            }
        }
        if sums.is_empty() {
            let mut seq = vec![0.0f64; cols];
            for row in data.chunks_exact(cols) {
                add_row_inplace_f64(&mut seq, row);
            }
            sums = seq;
            parallel_used = false;
        }
        if matches!(op, Reduction::Mean) {
            let inv = 1.0 / rows as f64;
            for value in &mut sums {
                *value *= inv;
            }
        }
        return AxisOutcome {
            values: sums,
            parallel: parallel_used,
        };
    }

    if rows <= SMALL_AXIS_PARALLEL_ROWS {
        let mut sums = vec![0.0f64; cols];
        for row in data.chunks_exact(cols) {
            add_row_inplace_f64(&mut sums, row);
        }
        if matches!(op, Reduction::Mean) {
            let inv = 1.0 / rows as f64;
            for value in &mut sums {
                *value *= inv;
            }
        }
        return AxisOutcome {
            values: sums,
            parallel: false,
        };
    }

    let mut parallel_used = false;
    let (mut sums, mut comps) = if allow_parallel && rows * cols >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols)
                    .with_min_len(AXIS0_PAR_MIN_ROWS)
                    .fold(
                        || (vec![0.0f64; cols], vec![0.0f64; cols]),
                        |mut acc, row| {
                            kahan_accumulate_row_f64(&mut acc.0, &mut acc.1, row);
                            acc
                        },
                    )
                    .reduce(
                        || (vec![0.0f64; cols], vec![0.0f64; cols]),
                        |mut acc, partial| {
                            for idx in 0..cols {
                                kahan_step(&mut acc.0[idx], &mut acc.1[idx], partial.0[idx]);
                                if partial.1[idx] != 0.0 {
                                    kahan_step(&mut acc.0[idx], &mut acc.1[idx], partial.1[idx]);
                                }
                            }
                            acc
                        },
                    )
            })
        } else {
            (Vec::new(), Vec::new())
        }
    } else {
        (Vec::new(), Vec::new())
    };

    if sums.is_empty() {
        parallel_used = false;
        sums = vec![0.0f64; cols];
        comps = vec![0.0f64; cols];
        for row in data.chunks_exact(cols) {
            kahan_accumulate_row_f64(&mut sums, &mut comps, row);
        }
    }

    for idx in 0..cols {
        sums[idx] += comps[idx];
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
        return finalize_axis0_f32(vec![0.0; cols], None, rows, cols, op, false);
    }

    if rows <= SMALL_MATRIX_FAST_DIM && cols <= SMALL_MATRIX_FAST_DIM {
        let mut sums = vec![0.0f32; cols];
        let mut comps = vec![0.0f32; cols];
        for row in data.chunks_exact(cols) {
            kahan_accumulate_row_f32_native(&mut sums, &mut comps, row);
        }
        return finalize_axis0_f32(sums, Some(comps), rows, cols, op, false);
    }

    if blas::axis0_enabled() && rows >= AXIS0_BLAS_MIN_ROWS && cols >= AXIS0_BLAS_MIN_COLS {
        let mut sums = vec![0.0f32; cols];
        if blas::current_backend().sgemv_axis0_sum(rows, cols, data, &mut sums) {
            return finalize_axis0_f32(sums, None, rows, cols, op, false);
        }
    }

    if let Some(simd_totals) = simd::reduce_axis0_columns_f32(data, rows, cols) {
        return finalize_axis0_f32(simd_totals, None, rows, cols, op, false);
    }

    if matrix_backend_enabled()
        && rows >= MATRIX_AXIS0_ROW_THRESHOLD
        && cols >= MATRIX_AXIS0_COL_THRESHOLD
        && cols <= MATRIX_AXIS0_MATRIX_MAX_COLS
    {
        #[cfg(feature = "matrixmultiply-backend")]
        {
            if allow_parallel {
                if let Some(pool) = thread_pool() {
                    if let Some(sums) = reduce_axis0_parallel_matrix_f32(&pool, data, rows, cols) {
                        return finalize_axis0_f32(sums, None, rows, cols, op, true);
                    }
                }
            }
            let sums = reduce_axis0_columns_matrix(data, rows, cols);
            return finalize_axis0_f32(sums, None, rows, cols, op, false);
        }
    }

    if rows >= AXIS0_LARGE_TILED_ROWS && cols >= AXIS0_LARGE_TILED_COLS {
        if let Some(simd_totals) = simd::reduce_axis0_tiled_f32(data, rows, cols) {
            return finalize_axis0_f32(simd_totals, None, rows, cols, op, false);
        }
    }

    if cols <= AXIS0_SIMD_COL_LIMIT {
        if let Some(simd_totals) = simd::reduce_axis0_tiled_f32(data, rows, cols) {
            return finalize_axis0_f32(simd_totals, None, rows, cols, op, false);
        }
        if let Some(simd_totals) = accumulate_axis0_simd_f32(data, rows, cols) {
            return finalize_axis0_f32(simd_totals, None, rows, cols, op, false);
        }
    }

    if allow_parallel && cols >= AXIS0_PAR_MIN_COL_CHUNK && rows >= AXIS0_PAR_MIN_ROWS {
        if rows.saturating_mul(cols) >= PARALLEL_MIN_ELEMENTS {
            if let Some(pool) = thread_pool() {
                let row_force = axis0_row_parallel_force();
                let mut use_row = (row_force
                    || (axis0_row_parallel_enabled() && rows >= AXIS0_ROW_CHUNK_MIN_ROWS))
                    && rows >= AXIS0_PAR_MIN_ROWS
                    && cols >= AXIS0_PAR_MIN_COL_CHUNK;
                if use_row && !row_force {
                    if let Ok(guard) = adaptive_state().lock() {
                        let row_samples =
                            guard.operation_sample_count(OPERATION_AXIS0_ROW, "float32", true);
                        let col_samples =
                            guard.operation_sample_count(OPERATION_AXIS0_COL, "float32", true);
                        if row_samples >= 3 && col_samples >= 3 {
                            if let (Some(row_med), Some(col_med)) = (
                                guard.operation_median(OPERATION_AXIS0_ROW, "float32", true),
                                guard.operation_median(OPERATION_AXIS0_COL, "float32", true),
                            ) {
                                if row_med < col_med * 0.98 {
                                    use_row = false;
                                }
                            }
                        }
                    }
                }
                let strategy = if use_row {
                    OPERATION_AXIS0_ROW
                } else {
                    OPERATION_AXIS0_COL
                };
                let start_strategy = Instant::now();
                let sums = if use_row {
                    reduce_axis0_parallel_row_f32(pool, data, rows, cols)
                } else {
                    reduce_axis0_parallel_tiled_f32(pool, data, rows, cols)
                };
                record_axis_strategy_event(
                    strategy,
                    "float32",
                    rows,
                    cols,
                    start_strategy.elapsed(),
                );
                return finalize_axis0_f32(sums, None, rows, cols, op, true);
            }
        }
    }
    if cols <= AXIS0_SIMD_COL_LIMIT {
        let mut parallel_used = false;
        let mut sums: Vec<f32> = Vec::new();
        let mut comps: Option<Vec<f32>> = None;
        if allow_parallel && rows >= AXIS0_PAR_MIN_ROWS && rows * cols >= PARALLEL_MIN_ELEMENTS {
            if let Some(pool) = thread_pool() {
                parallel_used = true;
                let (sum_vec, comp_vec) = pool.install(|| {
                    use rayon::prelude::*;
                    data.par_chunks(cols)
                        .with_min_len(AXIS0_PAR_MIN_ROWS)
                        .fold(
                            || (vec![0.0f32; cols], vec![0.0f32; cols]),
                            |mut acc, row| {
                                kahan_accumulate_row_f32_native(&mut acc.0, &mut acc.1, row);
                                acc
                            },
                        )
                        .reduce(
                            || (vec![0.0f32; cols], vec![0.0f32; cols]),
                            |mut acc, partial| {
                                for idx in 0..cols {
                                    kahan_step_f32(
                                        &mut acc.0[idx],
                                        &mut acc.1[idx],
                                        partial.0[idx],
                                    );
                                    if partial.1[idx] != 0.0 {
                                        kahan_step_f32(
                                            &mut acc.0[idx],
                                            &mut acc.1[idx],
                                            partial.1[idx],
                                        );
                                    }
                                }
                                acc
                            },
                        )
                });
                sums = sum_vec;
                comps = Some(comp_vec);
            }
        }
        if sums.is_empty() {
            if let Some(simd_totals) = simd::reduce_axis0_columns_f32(data, rows, cols) {
                sums = simd_totals;
                comps = Some(vec![0.0f32; cols]);
            } else if let Some(simd_totals) = reduce_axis0_seq_simd_add_f32(data, rows, cols) {
                sums = simd_totals;
                comps = Some(vec![0.0f32; cols]);
            } else {
                sums = vec![0.0f32; cols];
                let mut corrections = vec![0.0f32; cols];
                for row in data.chunks_exact(cols) {
                    kahan_accumulate_row_f32_native(&mut sums, &mut corrections, row);
                }
                comps = Some(corrections);
            }
            parallel_used = false;
        }
        return finalize_axis0_f32(sums, comps, rows, cols, op, parallel_used);
    }

    if rows <= SMALL_AXIS_PARALLEL_ROWS {
        let sums = if let Some(simd_totals) = simd::reduce_axis0_tiled_f32(data, rows, cols) {
            simd_totals
        } else if let Some(simd_totals) = accumulate_axis0_simd_f32(data, rows, cols) {
            simd_totals
        } else {
            let mut seq = vec![0.0f32; cols];
            for row in data.chunks_exact(cols) {
                add_row_inplace_f32(&mut seq, row);
            }
            seq
        };
        return finalize_axis0_f32(sums, None, rows, cols, op, false);
    }

    let mut parallel_used = false;
    let (mut sums, mut comps) = if allow_parallel && rows * cols >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols)
                    .with_min_len(AXIS0_PAR_MIN_ROWS)
                    .fold(
                        || (vec![0.0f32; cols], vec![0.0f32; cols]),
                        |mut acc, row| {
                            kahan_accumulate_row_f32_native(&mut acc.0, &mut acc.1, row);
                            acc
                        },
                    )
                    .reduce(
                        || (vec![0.0f32; cols], vec![0.0f32; cols]),
                        |mut acc, partial| {
                            for idx in 0..cols {
                                kahan_step_f32(&mut acc.0[idx], &mut acc.1[idx], partial.0[idx]);
                                if partial.1[idx] != 0.0 {
                                    kahan_step_f32(
                                        &mut acc.0[idx],
                                        &mut acc.1[idx],
                                        partial.1[idx],
                                    );
                                }
                            }
                            acc
                        },
                    )
            })
        } else {
            (Vec::new(), Vec::new())
        }
    } else {
        (Vec::new(), Vec::new())
    };

    if sums.is_empty() {
        parallel_used = false;
        if let Some(simd_totals) = simd::reduce_axis0_tiled_f32(data, rows, cols) {
            sums = simd_totals;
            comps = vec![0.0f32; cols];
        } else if let Some(simd_totals) = accumulate_axis0_simd_f32(data, rows, cols) {
            sums = simd_totals;
            comps = vec![0.0f32; cols];
        } else {
            sums = vec![0.0f32; cols];
            comps = vec![0.0f32; cols];
            for row in data.chunks_exact(cols) {
                kahan_accumulate_row_f32_native(&mut sums, &mut comps, row);
            }
        }
    }

    finalize_axis0_f32(sums, Some(comps), rows, cols, op, parallel_used)
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

    let mut parallel_used = false;
    let mut sums = if allow_parallel && rows * cols >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                data.par_chunks(cols)
                    .with_min_len(AXIS1_PAR_MIN_ROWS)
                    .map(reduce_row_simd_f64)
                    .collect()
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
        for row in data.chunks_exact(cols) {
            out.push(reduce_row_simd_f64(row));
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

    let mut parallel_used = false;
    let mut sums = if allow_parallel && rows * cols >= PARALLEL_MIN_ELEMENTS {
        if let Some(pool) = thread_pool() {
            parallel_used = true;
            pool.install(|| {
                use rayon::prelude::*;
                (0..rows)
                    .into_par_iter()
                    .map(|row_idx| {
                        let offset = row_idx * cols;
                        reduce_row_simd_f32(&data[offset..offset + cols])
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
        for row in data.chunks_exact(cols) {
            out.push(reduce_row_simd_f32(row));
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
    ($name:ident, $pyname:literal, $t:ty; $($extra:item)*) => {
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

            $($extra)*
        }
    };
}

impl_pyarray!(RustArray, "RustArray", f64;);
impl_pyarray!(RustArrayF32, "RustArrayF32", f32;
    #[pyo3(name = "scale_inplace")]
    fn py_scale_inplace(&mut self, factor: f64) -> PyResult<()> {
        if !self.inner.small_scale_inplace(factor)? {
            self.inner = self.inner.scale(factor)?;
        }
        Ok(())
    }

    #[pyo3(name = "broadcast_add_inplace")]
    fn py_broadcast_add_inplace(&mut self, rhs: &Self) -> PyResult<()> {
        if !self.inner.small_broadcast_row_inplace(&rhs.inner)? {
            self.inner = self.inner.add(&rhs.inner)?;
        }
        Ok(())
    }
);
impl_pyarray!(RustArrayI32, "RustArrayI32", i32;);

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

    info.set_item("blas_backend", blas::backend_name())?;

    let thresholds_dict = PyDict::new_bound(py);
    for entry in snapshot.thresholds {
        let entry_dict = PyDict::new_bound(py);
        entry_dict.set_item("median_elements_per_ms", entry.median_elements_per_ms)?;
        entry_dict.set_item(
            "seq_median_elements_per_ms",
            entry.seq_median_elements_per_ms,
        )?;
        match entry.p95_elements_per_ms {
            Some(value) => entry_dict.set_item("p95_elements_per_ms", value)?,
            None => entry_dict.set_item("p95_elements_per_ms", py.None())?,
        }
        match entry.seq_p95_elements_per_ms {
            Some(value) => entry_dict.set_item("seq_p95_elements_per_ms", value)?,
            None => entry_dict.set_item("seq_p95_elements_per_ms", py.None())?,
        }
        match entry.variance_ratio {
            Some(value) => entry_dict.set_item("variance_ratio", value)?,
            None => entry_dict.set_item("variance_ratio", py.None())?,
        }
        match entry.seq_variance_ratio {
            Some(value) => entry_dict.set_item("seq_variance_ratio", value)?,
            None => entry_dict.set_item("seq_variance_ratio", py.None())?,
        }
        entry_dict.set_item("sample_count", entry.sample_count)?;
        entry_dict.set_item("seq_sample_count", entry.seq_sample_count)?;
        entry_dict.set_item("samples", PyList::new_bound(py, entry.samples))?;
        entry_dict.set_item("seq_samples", PyList::new_bound(py, entry.seq_samples))?;
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

    let simd_caps = PyDict::new_bound(py);
    let caps = simd::capabilities();
    simd_caps.set_item("arch", caps.arch)?;
    simd_caps.set_item("feature_level", caps.feature_level())?;
    simd_caps.set_item("lane_width_bits", caps.lane_width_bits)?;
    simd_caps.set_item("avx512", caps.avx512)?;
    simd_caps.set_item("avx2", caps.avx2)?;
    simd_caps.set_item("avx", caps.avx)?;
    simd_caps.set_item("fma", caps.fma)?;
    simd_caps.set_item("sse41", caps.sse41)?;
    simd_caps.set_item("neon", caps.neon)?;
    simd_caps.set_item("sve", caps.sve)?;
    info.set_item("simd_capabilities", simd_caps)?;

    let stride_dict = PyDict::new_bound(py);
    for (kind, counter) in stride_snapshot() {
        let entry = PyDict::new_bound(py);
        entry.set_item("contiguous_calls", counter.contiguous)?;
        entry.set_item("strided_calls", counter.strided)?;
        stride_dict.set_item(kind, entry)?;
    }
    info.set_item("stride_counters", stride_dict)?;

    let tile_dict = PyDict::new_bound(py);
    for (axis, histogram) in axis_tile_snapshot() {
        let axis_key = match axis {
            AxisKind::Axis0 => "axis0",
            AxisKind::Axis1 => "axis1",
        };
        let axis_dict = PyDict::new_bound(py);
        for (width, count) in histogram {
            axis_dict.set_item(width, count)?;
        }
        tile_dict.set_item(axis_key, axis_dict)?;
    }
    info.set_item("axis_tile_histogram", tile_dict)?;

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

#[inline]
fn kahan_step(sum: &mut f64, comp: &mut f64, value: f64) {
    let y = value - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

#[inline]
fn kahan_step_f32(sum: &mut f32, comp: &mut f32, value: f32) {
    let y = value - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

#[inline]
fn add_row_inplace_f32(dst: &mut [f32], src: &[f32]) {
    if !simd::add_assign_inplace_f32(dst, src) {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::ensure_python_initialized;
    use super::*;

    fn array_f64(data: Vec<f64>, shape: Vec<usize>) -> NumericArray<f64> {
        ensure_python_initialized();
        NumericArray::new_owned(data, shape)
    }

    fn array_f32(data: Vec<f32>, shape: Vec<usize>) -> NumericArray<f32> {
        ensure_python_initialized();
        NumericArray::new_owned(data, shape)
    }

    fn array_i32(data: Vec<i32>, shape: Vec<usize>) -> NumericArray<i32> {
        ensure_python_initialized();
        NumericArray::new_owned(data, shape)
    }

    #[test]
    fn sum_and_mean_handle_basic_matrix() {
        let array = array_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(array.sum_f64().unwrap(), 10.0);
        assert_eq!(array.mean_f64().unwrap(), 2.5);
    }

    #[test]
    fn add_same_shape_produces_expected_values() {
        let lhs = array_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = array_f64(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2]);
        let result = lhs.add(&rhs).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.to_vec(), vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn add_supports_row_broadcasts() {
        let lhs = array_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let rhs = array_f64(vec![10.0, 20.0, 30.0], vec![3]);
        let result = lhs.add(&rhs).unwrap();
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_supports_column_broadcasts() {
        let lhs = array_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let rhs = array_f64(vec![5.0, 6.0], vec![2, 1]);
        let result = lhs.add(&rhs).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.to_vec(), vec![6.0, 7.0, 9.0, 10.0]);
    }

    #[test]
    fn add_supports_scalar_broadcasts() {
        let lhs = array_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let rhs = array_f32(vec![2.5], vec![1]);
        let result = lhs.add(&rhs).unwrap();
        assert_eq!(result.shape, vec![4]);
        assert_eq!(result.to_vec(), vec![3.5_f32, 4.5_f32, 5.5_f32, 6.5_f32]);
    }

    #[test]
    fn scale_rejects_fractional_factors_for_i32() {
        let array = array_i32(vec![1, 2, 3, 4], vec![4]);
        assert!(array.scale(0.5).is_err());
        let doubled = array.scale(2.0).unwrap();
        assert_eq!(doubled.to_vec(), vec![2, 4, 6, 8]);
    }

    #[test]
    fn reduce_axis_0_and_1_match_expected_results() {
        let matrix = array_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let axis0 = matrix
            .reduce_axis(0, Reduction::Sum)
            .expect("axis-0 reduction succeeds");
        assert_eq!(axis0.shape, vec![3]);
        assert_eq!(axis0.to_vec(), vec![5.0, 7.0, 9.0]);

        let axis1 = matrix
            .reduce_axis(1, Reduction::Mean)
            .expect("axis-1 reduction succeeds");
        assert_eq!(axis1.shape, vec![2]);
        assert_eq!(axis1.to_vec(), vec![2.0, 5.0]);
    }

    #[test]
    fn broadcast_shapes_reports_incompatible_operands() {
        let err = broadcast_shapes(&[2, 2], &[3, 2]).unwrap_err();
        Python::with_gil(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }
}

pub fn init_test_module(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    _raptors(py, module)
}

fn kahan_accumulate_row_f64(sums: &mut [f64], comps: &mut [f64], row: &[f64]) {
    for (idx, &value) in row.iter().enumerate() {
        kahan_step(&mut sums[idx], &mut comps[idx], value);
    }
}

fn kahan_accumulate_row_f32_native(sums: &mut [f32], comps: &mut [f32], row: &[f32]) {
    for (idx, &value) in row.iter().enumerate() {
        kahan_step_f32(&mut sums[idx], &mut comps[idx], value);
    }
}

fn accumulate_axis0_simd_f32(data: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Some(vec![0.0; cols]);
    }
    let elements = rows.checked_mul(cols)?;
    if elements != data.len() {
        return None;
    }
    if let Some(values) = simd::reduce_axis0_tiled_f32(data, rows, cols) {
        return Some(values);
    }
    if rows >= AXIS0_COLUMN_SIMD_MIN_ROWS
        && cols >= AXIS0_COLUMN_SIMD_MIN_COLS
        && cols <= AXIS0_SIMD_COL_LIMIT
    {
        let large_enough_for_parallel = rows >= 768 || cols >= 1024;
        if large_enough_for_parallel
            && rows >= AXIS0_PAR_MIN_ROWS
            && rows * cols >= PARALLEL_MIN_ELEMENTS
        {
            if let Some(pool) = thread_pool() {
                use rayon::prelude::*;
                let threads = pool.current_num_threads().max(1);
                let chunk_rows = ((rows + threads - 1) / threads).max(AXIS0_PAR_MIN_ROWS);
                let totals = pool.install(|| {
                    data.par_chunks(chunk_rows * cols)
                        .map(|chunk| {
                            let chunk_row_count = chunk.len() / cols;
                            simd::reduce_axis0_columns_f32(chunk, chunk_row_count, cols)
                                .unwrap_or_else(|| {
                                    let mut tmp = vec![0.0f32; cols];
                                    for row in chunk.chunks_exact(cols) {
                                        add_row_inplace_f32(&mut tmp, row);
                                    }
                                    tmp
                                })
                        })
                        .reduce(
                            || vec![0.0f32; cols],
                            |mut acc, partial| {
                                if !simd::add_assign_inplace_f32(&mut acc, partial.as_slice()) {
                                    for (dst, &value) in acc.iter_mut().zip(partial.iter()) {
                                        *dst += value;
                                    }
                                }
                                acc
                            },
                        )
                });
                return Some(totals);
            }
        }
        if let Some(values) = simd::reduce_axis0_columns_f32(data, rows, cols) {
            return Some(values);
        }
    }
    let mut totals = vec![0.0f32; cols];
    let mut iter = data.chunks_exact(cols);
    if let Some(first_row) = iter.next() {
        if !simd::add_assign_inplace_f32(&mut totals, first_row) {
            return None;
        }
    }
    for row in iter {
        if !simd::add_assign_inplace_f32(&mut totals, row) {
            return None;
        }
    }
    Some(totals)
}

#[inline]
fn finalize_axis0_f32(
    sums: Vec<f32>,
    comps: Option<Vec<f32>>,
    rows: usize,
    cols: usize,
    op: Reduction,
    parallel_used: bool,
) -> AxisOutcome {
    finalize_axis0_f32_impl(sums, comps, rows, cols, op, parallel_used)
}

fn finalize_axis0_f32_impl(
    sums: Vec<f32>,
    comps: Option<Vec<f32>>,
    rows: usize,
    cols: usize,
    op: Reduction,
    parallel_used: bool,
) -> AxisOutcome {
    let mut values = Vec::with_capacity(cols);
    match comps {
        Some(mut corrections) => {
            if corrections.len() != cols {
                corrections.resize(cols, 0.0);
            }
            for idx in 0..cols {
                values.push((sums[idx] + corrections[idx]) as f64);
            }
        }
        None => {
            for value in sums {
                values.push(value as f64);
            }
        }
    }
    if matches!(op, Reduction::Mean) {
        let inv = 1.0 / rows as f64;
        for total in &mut values {
            *total *= inv;
        }
    }
    AxisOutcome {
        values,
        parallel: parallel_used,
    }
}

fn recommended_accumulators(len: usize) -> usize {
    if len >= 4096 {
        8
    } else if len >= 2048 {
        6
    } else if len >= 1024 {
        4
    } else if len >= 512 {
        2
    } else {
        1
    }
}

fn reduce_row_simd_f64(row: &[f64]) -> f64 {
    let accumulators = recommended_accumulators(row.len());
    if let Some(sum) = simd::reduce_sum_f64(row, accumulators) {
        sum
    } else {
        row.iter().sum()
    }
}

fn reduce_row_simd_f32(row: &[f32]) -> f64 {
    let accumulators = recommended_accumulators(row.len());
    if let Some(sum) = simd::reduce_sum_f32(row, accumulators) {
        sum
    } else {
        row.iter().map(|&value| value as f64).sum()
    }
}

struct AxisTilePlan {
    tile_cols: usize,
}

impl AxisTilePlan {
    fn new(cols: usize, threads: usize) -> Self {
        let caps = simd::capabilities();
        let lanes = (caps.lane_width_bits / 32).max(4);
        let mut tile = (cols + threads - 1) / threads;
        tile = tile
            .max(AXIS0_PAR_MIN_COL_CHUNK)
            .min(AXIS0_PAR_MAX_COL_CHUNK);
        tile = tile.max(lanes);
        let lane_multiple = lanes;
        if tile % lane_multiple != 0 {
            tile = ((tile + lane_multiple - 1) / lane_multiple) * lane_multiple;
        }
        tile = tile.min(cols.max(lanes));
        Self {
            tile_cols: tile.max(lanes),
        }
    }
}

fn reduce_axis0_parallel_tiled_f32(
    pool: &rayon::ThreadPool,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let threads = pool.current_num_threads().max(1);
    let plan = AxisTilePlan::new(cols, threads);
    let chunk_cols = plan.tile_cols.min(cols).max(1);
    record_axis_tile(AxisKind::Axis0, chunk_cols);

    let mut output = vec![0.0f32; cols];
    pool.install(|| {
        use rayon::prelude::*;
        output
            .par_chunks_mut(chunk_cols)
            .enumerate()
            .for_each(|(index, out_chunk)| {
                let start_col = index * chunk_cols;
                accumulate_axis0_chunk_f32(data, rows, cols, start_col, out_chunk);
            });
    });

    output
}

fn reduce_axis0_seq_simd_add_f32(data: &[f32], rows: usize, cols: usize) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Some(vec![0.0; cols]);
    }
    let mut totals = vec![0.0f32; cols];
    for row in data.chunks_exact(cols) {
        if !simd::add_assign_inplace_f32(&mut totals, row) {
            return None;
        }
    }
    Some(totals)
}

#[cfg(feature = "matrixmultiply-backend")]
#[allow(dead_code)]
fn reduce_axis0_parallel_matrix_f32(
    pool: &rayon::ThreadPool,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 || rows.saturating_mul(cols) != data.len() {
        return None;
    }
    if rows < AXIS0_PAR_MIN_ROWS {
        return None;
    }
    let threads = pool.current_num_threads().max(1);
    let chunk_rows = ((rows + threads - 1) / threads).max(AXIS0_PAR_MIN_ROWS);

    let totals = pool.install(|| {
        use rayon::prelude::*;
        data.par_chunks(chunk_rows * cols)
            .map(|chunk| {
                let chunk_row_count = chunk.len() / cols;
                if chunk_row_count == 0 {
                    return vec![0.0f32; cols];
                }
                with_f32_ones(chunk_row_count, |ones| {
                    let mut partial = vec![0.0f32; cols];
                    unsafe {
                        matrixmultiply::sgemm(
                            1,
                            chunk_row_count,
                            cols,
                            1.0,
                            ones.as_ptr(),
                            chunk_row_count as isize,
                            1,
                            chunk.as_ptr(),
                            cols as isize,
                            1,
                            0.0,
                            partial.as_mut_ptr(),
                            cols as isize,
                            1,
                        );
                    }
                    partial
                })
            })
            .reduce(
                || vec![0.0f32; cols],
                |mut acc, partial| {
                    if !simd::add_assign_inplace_f32(&mut acc, partial.as_slice()) {
                        for (dst, &value) in acc.iter_mut().zip(partial.iter()) {
                            *dst += value;
                        }
                    }
                    acc
                },
            )
    });
    Some(totals)
}

#[cfg(not(feature = "matrixmultiply-backend"))]
fn reduce_axis0_parallel_matrix_f32(
    _pool: &rayon::ThreadPool,
    _data: &[f32],
    _rows: usize,
    _cols: usize,
) -> Option<Vec<f32>> {
    None
}

#[cfg(feature = "matrixmultiply-backend")]
#[allow(dead_code)]
fn reduce_axis0_parallel_matrix_f64(
    pool: &rayon::ThreadPool,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> Option<Vec<f64>> {
    if rows == 0 || cols == 0 || rows.saturating_mul(cols) != data.len() {
        return None;
    }
    if rows < AXIS0_PAR_MIN_ROWS {
        return None;
    }
    let threads = pool.current_num_threads().max(1);
    let chunk_rows = ((rows + threads - 1) / threads).max(AXIS0_PAR_MIN_ROWS);

    let totals = pool.install(|| {
        use rayon::prelude::*;
        data.par_chunks(chunk_rows * cols)
            .map(|chunk| {
                let chunk_row_count = chunk.len() / cols;
                if chunk_row_count == 0 {
                    return vec![0.0f64; cols];
                }
                with_f64_ones(chunk_row_count, |ones| {
                    let mut partial = vec![0.0f64; cols];
                    unsafe {
                        matrixmultiply::dgemm(
                            1,
                            chunk_row_count,
                            cols,
                            1.0,
                            ones.as_ptr(),
                            chunk_row_count as isize,
                            1,
                            chunk.as_ptr(),
                            cols as isize,
                            1,
                            0.0,
                            partial.as_mut_ptr(),
                            cols as isize,
                            1,
                        );
                    }
                    partial
                })
            })
            .reduce(
                || vec![0.0f64; cols],
                |mut acc, partial| {
                    add_row_inplace_f64(&mut acc, partial.as_slice());
                    acc
                },
            )
    });
    Some(totals)
}

#[cfg(not(feature = "matrixmultiply-backend"))]
fn reduce_axis0_parallel_matrix_f64(
    _pool: &rayon::ThreadPool,
    _data: &[f64],
    _rows: usize,
    _cols: usize,
) -> Option<Vec<f64>> {
    None
}

fn reduce_axis0_parallel_row_f32(
    pool: &rayon::ThreadPool,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    let threads = pool.current_num_threads().max(1);
    let chunk_rows = ((rows + threads - 1) / threads).max(AXIS0_PAR_MIN_ROWS);
    pool.install(|| {
        use rayon::prelude::*;
        data.par_chunks(chunk_rows * cols)
            .map(|chunk| {
                let chunk_rows = chunk.len() / cols;
                if chunk_rows == 0 {
                    return vec![0.0f32; cols];
                }
                if let Some(values) = simd::reduce_axis0_tiled_f32(chunk, chunk_rows, cols) {
                    return values;
                }
                if let Some(values) = reduce_axis0_seq_simd_add_f32(chunk, chunk_rows, cols) {
                    return values;
                }
                let mut partial = vec![0.0f32; cols];
                for row in chunk.chunks_exact(cols) {
                    if !simd::add_assign_inplace_f32(&mut partial, row) {
                        for (dst, &value) in partial.iter_mut().zip(row.iter()) {
                            *dst += value;
                        }
                    }
                }
                partial
            })
            .reduce(
                || vec![0.0f32; cols],
                |mut acc, partial| {
                    if !simd::add_assign_inplace_f32(&mut acc, partial.as_slice()) {
                        for (dst, &value) in acc.iter_mut().zip(partial.iter()) {
                            *dst += value;
                        }
                    }
                    acc
                },
            )
    })
}

fn accumulate_axis0_chunk_f32(
    data: &[f32],
    rows: usize,
    cols: usize,
    start_col: usize,
    out_chunk: &mut [f32],
) {
    if rows == 0 || out_chunk.is_empty() {
        out_chunk.fill(0.0);
        return;
    }

    out_chunk.fill(0.0);
    for row in 0..rows {
        let offset = row * cols + start_col;
        let slice = &data[offset..offset + out_chunk.len()];
        if !simd::add_assign_inplace_f32(out_chunk, slice) {
            for (dst, &value) in out_chunk.iter_mut().zip(slice.iter()) {
                *dst += value;
            }
        }
    }
}

#[cfg(feature = "matrixmultiply-backend")]
#[allow(dead_code)]
fn reduce_axis0_columns_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    if cols == 0 {
        return Vec::new();
    }
    if rows == 0 {
        return vec![0.0f32; cols];
    }

    let mut output = vec![0.0f32; cols];
    if blas::axis0_enabled()
        && blas::current_backend().sgemv_axis0_sum(rows, cols, data, &mut output)
    {
        return output;
    }

    with_f32_ones(rows, |ones| unsafe {
        matrixmultiply::sgemm(
            1,
            rows,
            cols,
            1.0,
            ones.as_ptr(),
            rows as isize,
            1,
            data.as_ptr(),
            cols as isize,
            1,
            0.0,
            output.as_mut_ptr(),
            cols as isize,
            1,
        );
    });
    output
}

#[cfg(not(feature = "matrixmultiply-backend"))]
fn reduce_axis0_columns_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    if cols == 0 {
        return Vec::new();
    }
    if rows == 0 {
        return vec![0.0f32; cols];
    }
    let mut output = vec![0.0f32; cols];
    if blas::axis0_enabled()
        && blas::current_backend().sgemv_axis0_sum(rows, cols, data, &mut output)
    {
        return output;
    }
    if let Some(vec) = simd::reduce_axis0_tiled_f32(data, rows, cols) {
        return vec;
    }
    if let Some(vec) = accumulate_axis0_simd_f32(data, rows, cols) {
        return vec;
    }
    for row in data.chunks_exact(cols) {
        add_row_inplace_f32(&mut output, row);
    }
    output
}

#[cfg(feature = "matrixmultiply-backend")]
fn reduce_axis0_columns_matrix_f64(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    if cols == 0 {
        return Vec::new();
    }
    if rows == 0 {
        return vec![0.0f64; cols];
    }

    let mut output = vec![0.0f64; cols];
    with_f64_ones(rows, |ones| unsafe {
        matrixmultiply::dgemm(
            1,
            rows,
            cols,
            1.0,
            ones.as_ptr(),
            rows as isize,
            1,
            data.as_ptr(),
            cols as isize,
            1,
            0.0,
            output.as_mut_ptr(),
            cols as isize,
            1,
        );
    });
    output
}

#[cfg(not(feature = "matrixmultiply-backend"))]
fn reduce_axis0_columns_matrix_f64(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    if cols == 0 {
        return Vec::new();
    }
    if rows == 0 {
        return vec![0.0f64; cols];
    }
    let mut output = vec![0.0f64; cols];
    for row in data.chunks_exact(cols) {
        add_row_inplace_f64(&mut output, row);
    }
    output
}
