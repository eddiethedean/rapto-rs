use std::ops::Add;

use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use numpy::{Element, IxDyn, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyModule, PySequence};
use pyo3::FromPyObject;

type PyResultF64 = PyResult<f64>;

trait NumericElement:
    Copy + Clone + Send + Sync + 'static + Add<Output = Self> + Zero + One + PartialEq
{
    const DTYPE_NAME: &'static str;
    const SUPPORTS_FRACTIONS: bool;

    fn try_to_f64(self) -> Option<f64>;
    fn try_from_f64(value: f64) -> Option<Self>;

    fn try_scale(self, factor: f64) -> Option<Self> {
        let base = self.try_to_f64()?;
        Self::try_from_f64(base * factor)
    }
}

impl NumericElement for f64 {
    const DTYPE_NAME: &'static str = "float64";
    const SUPPORTS_FRACTIONS: bool = true;

    fn try_to_f64(self) -> Option<f64> {
        Some(self)
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        Some(value)
    }
}

impl NumericElement for f32 {
    const DTYPE_NAME: &'static str = "float32";
    const SUPPORTS_FRACTIONS: bool = true;

    fn try_to_f64(self) -> Option<f64> {
        Some(self as f64)
    }

    fn try_from_f64(value: f64) -> Option<Self> {
        Some(value as f32)
    }
}

impl NumericElement for i32 {
    const DTYPE_NAME: &'static str = "int32";
    const SUPPORTS_FRACTIONS: bool = false;

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
struct NumericArray<T: NumericElement> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> NumericArray<T>
where
    T: NumericElement + Element + for<'py> FromPyObject<'py>,
{
    fn from_python(iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (data, shape) = parse_python_iterable::<T>(iterable)?;
        Ok(Self { data, shape })
    }

    fn zeros(dims: &[usize]) -> PyResult<Self> {
        let total = dims.iter().product();
        Ok(Self {
            data: vec![T::zero(); total],
            shape: dims.to_vec(),
        })
    }

    fn ones(dims: &[usize]) -> PyResult<Self> {
        let total = dims.iter().product();
        Ok(Self {
            data: vec![T::one(); total],
            shape: dims.to_vec(),
        })
    }

    fn from_numpy(array: PyReadonlyArrayDyn<'_, T>) -> PyResult<Self> {
        let shape = array.shape().to_vec();
        let data = array.as_array().to_owned().into_raw_vec();
        Ok(Self { data, shape })
    }

    fn total_len(&self) -> usize {
        self.data.len()
    }

    fn first_axis_len(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn sum_f64(&self) -> PyResultF64 {
        self.data.iter().try_fold(0.0, |acc, &value| {
            value
                .try_to_f64()
                .map(|v| acc + v)
                .ok_or_else(|| PyValueError::new_err("value cannot be represented as float"))
        })
    }

    fn mean_f64(&self) -> PyResultF64 {
        if self.data.is_empty() {
            Err(PyValueError::new_err("cannot compute mean of empty array"))
        } else {
            Ok(self.sum_f64()? / self.data.len() as f64)
        }
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArrayDyn<T>>> {
        let array = PyArrayDyn::<T>::zeros(py, IxDyn(&self.shape), false);
        unsafe {
            let slice = array
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("failed to access NumPy buffer"))?;
            slice.copy_from_slice(&self.data);
        }
        Ok(array.into_py(py))
    }

    fn add(&self, other: &NumericArray<T>) -> PyResult<NumericArray<T>> {
        let broadcast = BroadcastPair::new(self, other)?;
        let data = broadcast
            .rows()
            .map(|result| result.map(|(a, b)| a + b))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(NumericArray {
            data,
            shape: broadcast.output_shape.clone(),
        })
    }

    fn scale(&self, factor: f64) -> PyResult<NumericArray<T>> {
        if !T::SUPPORTS_FRACTIONS && factor.fract() != 0.0 {
            return Err(PyValueError::new_err(format!(
                "dtype {} only supports integer scale factors",
                T::DTYPE_NAME
            )));
        }

        let mut data = Vec::with_capacity(self.data.len());
        for &value in &self.data {
            let scaled = value.try_scale(factor).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "scaling produced values outside {} representable range",
                    T::DTYPE_NAME
                ))
            })?;
            data.push(scaled);
        }
        Ok(NumericArray {
            data,
            shape: self.shape.clone(),
        })
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

        Ok(NumericArray {
            data: vec![converted],
            shape: vec![1],
        })
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
                        acc += self.data[idx].try_to_f64().ok_or_else(|| {
                            PyValueError::new_err("value cannot be represented as float")
                        })?;
                    }
                    if matches!(op, Reduction::Mean) && rows > 0 {
                        acc /= rows as f64;
                    }
                    let converted = Self::convert_from_f64(acc, op)?;
                    out.push(converted);
                }
                Ok(NumericArray {
                    data: out,
                    shape: vec![cols],
                })
            }
            1 => {
                let mut out = Vec::with_capacity(rows);
                for r in 0..rows {
                    let base = r * row_stride;
                    let mut acc = 0.0;
                    for c in 0..cols {
                        let idx = base + c * col_stride;
                        acc += self.data[idx].try_to_f64().ok_or_else(|| {
                            PyValueError::new_err("value cannot be represented as float")
                        })?;
                    }
                    if matches!(op, Reduction::Mean) && cols > 0 {
                        acc /= cols as f64;
                    }
                    let converted = Self::convert_from_f64(acc, op)?;
                    out.push(converted);
                }
                Ok(NumericArray {
                    data: out,
                    shape: vec![rows],
                })
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

struct BroadcastPair<'a, T: NumericElement> {
    left: &'a NumericArray<T>,
    right: &'a NumericArray<T>,
    output_shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
}

impl<'a, T: NumericElement> BroadcastPair<'a, T> {
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
            Ok((self.left.data[left_idx], self.right.data[right_idx]))
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
        #[pyclass(name = $pyname, module = "raptors")]
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
                self.inner.data.clone()
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

            fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArrayDyn<$t>>> {
                self.inner.to_numpy(py)
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
fn from_numpy(_py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let numpy_array: PyReadonlyArrayDyn<'_, f64> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype float64"))?;
    Ok(RustArray::from_inner(NumericArray::from_numpy(
        numpy_array,
    )?))
}

#[pyfunction]
fn from_numpy_f32(_py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArrayF32> {
    let numpy_array: PyReadonlyArrayDyn<'_, f32> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype float32"))?;
    Ok(RustArrayF32::from_inner(NumericArray::from_numpy(
        numpy_array,
    )?))
}

#[pyfunction]
fn from_numpy_i32(_py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArrayI32> {
    let numpy_array: PyReadonlyArrayDyn<'_, i32> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype int32"))?;
    Ok(RustArrayI32::from_inner(NumericArray::from_numpy(
        numpy_array,
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

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Odos Matthews <odosmatthews@gmail.com>")?;
    m.add("__github__", "https://github.com/eddiethedean")?;
    m.add("__doc__", "Rust-backed array core for the Raptors project.")?;

    Ok(())
}
