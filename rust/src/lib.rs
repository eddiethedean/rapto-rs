use numpy::{IxDyn, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyFloat, PyInt, PyModule, PySequence};

#[pyclass(module = "raptors")]
pub struct RustArray {
    data: Vec<f64>,
    shape: Vec<usize>,
}

#[pymethods]
impl RustArray {
    #[new]
    fn new(iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (data, shape) = parse_python_iterable(iterable)?;
        Ok(Self { data, shape })
    }

    #[getter]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn __len__(&self) -> usize {
        // Follow NumPy convention: len returns size along first axis when available.
        self.shape.first().copied().unwrap_or(0)
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn to_list(&self) -> Vec<f64> {
        self.data.clone()
    }

    fn sum(&self) -> f64 {
        self.data.iter().copied().sum()
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArrayDyn<f64>>> {
        let shape_dyn = IxDyn(&self.shape);
        let array = PyArrayDyn::<f64>::zeros(py, shape_dyn, false);
        unsafe {
            let slice = array
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("failed to access NumPy buffer"))?;
            slice.copy_from_slice(&self.data);
        }
        Ok(array.into_py(py))
    }

    fn mean(&self) -> PyResult<f64> {
        if self.data.is_empty() {
            Err(PyValueError::new_err("cannot compute mean of empty array"))
        } else {
            Ok(self.sum() / self.data.len() as f64)
        }
    }

    #[pyo3(signature = (axis))]
    fn sum_axis(&self, axis: usize) -> PyResult<RustArray> {
        reduce_axis(self, axis, Reduction::Sum)
    }

    #[pyo3(signature = (axis))]
    fn mean_axis(&self, axis: usize) -> PyResult<RustArray> {
        reduce_axis(self, axis, Reduction::Mean)
    }

    fn add(&self, other: &RustArray) -> PyResult<RustArray> {
        let broadcast = BroadcastPair::new(self, other)?;
        let data: Vec<f64> = broadcast
            .rows()
            .map(|result| result.map(|(a, b)| a + b))
            .collect::<Result<_, _>>()?;
        Ok(RustArray {
            data,
            shape: broadcast.output_shape.clone(),
        })
    }

    fn scale(&self, factor: f64) -> RustArray {
        let data = self.data.iter().map(|value| value * factor).collect();
        RustArray {
            data,
            shape: self.shape.clone(),
        }
    }
}

#[pyfunction]
fn array(iterable: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    RustArray::new(iterable)
}

#[pyfunction]
fn zeros(shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let dims = parse_shape_argument(shape)?;
    let total = dims.iter().product();
    Ok(RustArray {
        data: vec![0.0; total],
        shape: dims,
    })
}

#[pyfunction]
fn ones(shape: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let dims = parse_shape_argument(shape)?;
    let total = dims.iter().product();
    Ok(RustArray {
        data: vec![1.0; total],
        shape: dims,
    })
}

#[pyfunction]
fn from_numpy(_py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let numpy_array: PyReadonlyArrayDyn<'_, f64> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a NumPy array of dtype float64"))?;
    let shape = numpy_array.shape().to_vec();
    let data = numpy_array.as_array().to_owned().into_raw_vec();
    Ok(RustArray { data, shape })
}

/// Python module initialization for `_raptors`.
#[pymodule]
fn _raptors(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustArray>()?;
    m.add_wrapped(pyo3::wrap_pyfunction!(array))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(zeros))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(ones))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(from_numpy))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Odos Matthews <odosmatthews@gmail.com>")?;
    m.add("__github__", "https://github.com/eddiethedean")?;
    m.add("__doc__", "Rust-backed array core for the Raptors project.")?;

    Ok(())
}

fn parse_python_iterable(iterable: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, Vec<usize>)> {
    let items: Vec<Bound<'_, PyAny>> = iterable.iter()?.collect::<PyResult<Vec<_>>>()?;

    if items.is_empty() {
        return Ok((Vec::new(), vec![0]));
    }

    if items[0].downcast::<PyFloat>().is_ok() || items[0].downcast::<PyInt>().is_ok() {
        let mut data = Vec::with_capacity(items.len());
        for item in items {
            data.push(item.extract::<f64>()?);
        }
        let len = data.len();
        return Ok((data, vec![len]));
    }

    // Attempt 2-D parsing
    let mut data = Vec::new();
    let outer_len = items.len();
    let mut inner_len: Option<usize> = None;

    for row in items {
        let seq = row
            .downcast::<PySequence>()
            .map_err(|_| PyTypeError::new_err("expected a sequence of sequences for 2-D input"))?;
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
            data.push(
                item.extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("expected float-compatible values"))?,
            );
        }
    }

    let cols = inner_len.unwrap_or(0);
    if cols == 0 && !data.is_empty() {
        return Err(PyValueError::new_err(
            "inner sequences cannot be empty for 2-D arrays",
        ));
    }

    Ok((data, vec![outer_len, cols]))
}

#[derive(Clone, Copy)]
enum Reduction {
    Sum,
    Mean,
}

struct BroadcastPair<'a> {
    left: &'a RustArray,
    right: &'a RustArray,
    output_shape: Vec<usize>,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
}

impl<'a> BroadcastPair<'a> {
    fn new(left: &'a RustArray, right: &'a RustArray) -> PyResult<Self> {
        let (shape, left_strides, right_strides) = broadcast_shapes(&left.shape, &right.shape)?;
        Ok(Self {
            left,
            right,
            output_shape: shape,
            left_strides,
            right_strides,
        })
    }

    fn rows(&self) -> impl Iterator<Item = Result<(f64, f64), PyErr>> + '_ {
        let total = self.output_shape.iter().product::<usize>();
        (0..total).map(move |idx| {
            let left_idx = map_index(&self.output_shape, &self.left_strides, idx)?;
            let right_idx = map_index(&self.output_shape, &self.right_strides, idx)?;
            Ok((
                self.left.data[left_idx],
                self.right.data[right_idx],
            ))
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

fn map_index(
    shape: &[usize],
    strides: &[usize],
    flat_index: usize,
) -> PyResult<usize> {
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

fn reduce_axis(array: &RustArray, axis: usize, op: Reduction) -> PyResult<RustArray> {
    match array.shape.len() {
        0 => Err(PyValueError::new_err(
            "cannot reduce an array with no shape information",
        )),
        1 => {
            if axis != 0 {
                return Err(PyValueError::new_err("axis out of bounds for 1-D array"));
            }
            let value = match op {
                Reduction::Sum => array.sum(),
                Reduction::Mean => array.mean()?,
            };
            Ok(RustArray {
                data: vec![value],
                shape: vec![1],
            })
        }
        2 => reduce_axis_2d(array, axis, op),
        _ => Err(PyValueError::new_err(
            "axis reductions are currently supported for up to 2-D arrays",
        )),
    }
}

fn reduce_axis_2d(array: &RustArray, axis: usize, op: Reduction) -> PyResult<RustArray> {
    if axis > 1 {
        return Err(PyValueError::new_err("axis out of bounds for 2-D array"));
    }

    let rows = array.shape[0];
    let cols = array.shape[1];
    let strides = row_major_strides(&array.shape);
    let row_stride = strides[0];
    let col_stride = strides[1];

    match axis {
        0 => {
            let mut out = vec![0.0; cols];
            for r in 0..rows {
                let base = r * row_stride;
                for c in 0..cols {
                    out[c] += array.data[base + c * col_stride];
                }
            }
            if matches!(op, Reduction::Mean) && rows > 0 {
                for value in &mut out {
                    *value /= rows as f64;
                }
            }
            Ok(RustArray {
                data: out,
                shape: vec![cols],
            })
        }
        1 => {
            let mut out = vec![0.0; rows];
            for r in 0..rows {
                let base = r * row_stride;
                let mut acc = 0.0;
                for c in 0..cols {
                    acc += array.data[base + c * col_stride];
                }
                if matches!(op, Reduction::Mean) && cols > 0 {
                    acc /= cols as f64;
                }
                out[r] = acc;
            }
            Ok(RustArray {
                data: out,
                shape: vec![rows],
            })
        }
        _ => unreachable!(),
    }
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
