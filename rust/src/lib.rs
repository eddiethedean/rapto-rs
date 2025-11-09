use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyModule};

#[pyclass(module = "raptors")]
pub struct RustArray {
    data: Vec<f64>,
    len: usize,
}

#[pymethods]
impl RustArray {
    #[new]
    fn new(iterable: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut data = Vec::new();
        for item in iterable.iter()? {
            data.push(item?.extract::<f64>()?);
        }
        let len = data.len();
        Ok(Self { data, len })
    }

    #[getter]
    fn len(&self) -> usize {
        self.len
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn to_list(&self) -> Vec<f64> {
        self.data.clone()
    }

    fn sum(&self) -> f64 {
        self.data.iter().copied().sum()
    }

    fn mean(&self) -> PyResult<f64> {
        let len = self.len();
        if len == 0 {
            Err(PyValueError::new_err("cannot compute mean of empty array"))
        } else {
            Ok(self.sum() / len as f64)
        }
    }

    fn add(&self, other: &RustArray) -> PyResult<RustArray> {
        ensure_same_len(self, other)?;
        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(RustArray {
            len: self.len,
            data,
        })
    }

    fn scale(&self, factor: f64) -> RustArray {
        let data = self.data.iter().map(|value| value * factor).collect();
        RustArray {
            data,
            len: self.len,
        }
    }
}

fn ensure_same_len(lhs: &RustArray, rhs: &RustArray) -> PyResult<()> {
    if lhs.len() != rhs.len() {
        Err(PyValueError::new_err(format!(
            "arrays must share the same length: left={} right={}",
            lhs.len(),
            rhs.len()
        )))
    } else {
        Ok(())
    }
}

#[pyfunction]
fn array(iterable: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    RustArray::new(iterable)
}

#[pyfunction]
fn zeros(length: usize) -> PyResult<RustArray> {
    Ok(RustArray {
        len: length,
        data: vec![0.0; length],
    })
}

#[pyfunction]
fn ones(length: usize) -> PyResult<RustArray> {
    Ok(RustArray {
        len: length,
        data: vec![1.0; length],
    })
}

#[pyfunction]
fn from_numpy(_py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    let numpy_array: PyReadonlyArray1<'_, f64> = array
        .extract()
        .map_err(|_| PyTypeError::new_err("expected a 1-D NumPy array of dtype float64"))?;
    let slice = numpy_array
        .as_slice()
        .map_err(|_| PyTypeError::new_err("expected contiguous NumPy array"))?;
    let len = slice.len();
    Ok(RustArray {
        len,
        data: slice.to_vec(),
    })
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

