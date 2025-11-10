use std::{convert::TryInto, env, ops::Add, slice, sync::OnceLock};

mod simd;

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use numpy::{Element, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyModule, PySequence, PyTuple};
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

fn thread_pool() -> Option<&'static rayon::ThreadPool> {
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
            let mut used = true;
            for row in 0..rows {
                let start = row * cols;
                let row_slice = &lhs[start..start + cols];
                let out_slice = &mut out[start..start + cols];
                if !simd::add_row_scalar_f64(row_slice, rhs[row], out_slice) {
                    used = false;
                    break;
                }
            }
            if used {
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
            let mut used = true;
            for row in 0..rows {
                let start = row * cols;
                let row_slice = &lhs[start..start + cols];
                let out_slice = &mut out[start..start + cols];
                if !simd::add_row_scalar_f32(row_slice, rhs[row], out_slice) {
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
                .zip(rhs_slice.par_iter().copied())
                .for_each(|((out_row, lhs_row), col_val)| {
                    for (dst, &l) in out_row.iter_mut().zip(lhs_row.iter()) {
                        *dst = l + col_val;
                    }
                });
        }) {
            return Some(NumericArray::new_owned(data, self.shape.clone()));
        }
        for row in 0..rows {
            let start = row * cols;
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
        T::simd_sum(self.data_slice())
            .ok_or_else(|| PyValueError::new_err("value cannot be represented as float"))
    }

    fn mean_f64(&self) -> PyResultF64 {
        if self.data_len() == 0 {
            Err(PyValueError::new_err("cannot compute mean of empty array"))
        } else {
            Ok(self.sum_f64()? / self.data_len() as f64)
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
        if T::DTYPE_NAME == "float64" {
            let input = self.data_slice();
            let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f64, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f64, len) };
            if simd::scale_same_shape_f64(input, factor, out) {
                return Ok(NumericArray::new_owned(data, self.shape.clone()));
            }
        } else if T::DTYPE_NAME == "float32" {
            let input = self.data_slice();
            let input = unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, len) };
            let out = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, len) };
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

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Odos Matthews <odosmatthews@gmail.com>")?;
    m.add("__github__", "https://github.com/eddiethedean")?;
    m.add("__doc__", "Rust-backed array core for the Raptors project.")?;

    Ok(())
}
