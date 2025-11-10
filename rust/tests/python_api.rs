use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use raptors::init_test_module;

/// Integration exercises for the PyO3-facing API.
///
/// These tests require a Python runtime at execute time but do not invoke
/// `cargo test` automatically in this workflow. Run them manually when Python
/// tooling is available: `cd rust && cargo test --tests`.

fn init(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let module = PyModule::new_bound(py, "_raptors_test")?;
    init_test_module(py, &module)?;
    Ok(module)
}

#[test]
fn zeros_constructor_produces_expected_array() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let module = init(py)?;
        let zeros = module.getattr("zeros")?;
        let shape = PyTuple::new_bound(py, [2, 3]);
        let array = zeros.call1((shape,))?;
        let data: Vec<f64> = array.call_method0("to_list")?.extract()?;
        assert_eq!(data, vec![0.0; 6]);
        let mean = array.call_method0("mean")?.extract::<f64>()?;
        assert_eq!(mean, 0.0);
        Ok(())
    })?;

    Ok(())
}

#[test]
fn broadcast_add_handles_row_vector_inputs() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let module = init(py)?;

        let array = module.getattr("array")?;
        let matrix = array.call1((PyList::new_bound(py, [[1.0, 2.0], [3.0, 4.0]]),))?;
        let row = array.call1((PyList::new_bound(py, [10.0, 20.0]),))?;

        let broadcast_add = module.getattr("broadcast_add")?;
        let result = broadcast_add.call1((matrix.clone(), row))?;
        let values: Vec<f64> = result.call_method0("to_list")?.extract()?;
        assert_eq!(values, vec![11.0, 22.0, 13.0, 24.0]);

        let sum = matrix.call_method0("sum")?.extract::<f64>()?;
        assert_eq!(sum, 10.0);
        Ok(())
    })?;

    Ok(())
}

#[test]
fn threading_info_exposes_expected_keys() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| -> PyResult<()> {
        let module = init(py)?;
        let info = module.getattr("threading_info")?.call0()?;
        let info = info.downcast::<PyDict>()?;
        assert!(info.contains("parallel_min_elements")?);
        assert!(info.contains("baseline_cutovers")?);
        Ok(())
    })?;

    Ok(())
}
