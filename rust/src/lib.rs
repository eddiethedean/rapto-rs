use pyo3::prelude::*;

/// Placeholder function that will eventually construct a Raptors array.
#[pyfunction]
fn rustarray_new() -> PyResult<()> {
    Ok(())
}

/// Python module initialization for `_raptors`.
#[pymodule]
fn _raptors(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(pyo3::wrap_pyfunction!(rustarray_new))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Odos Matthews <odosmatthews@gmail.com>")?;
    m.add("__github__", "https://github.com/eddiethedean")?;
    m.add("__doc__", "Rust-backed array core for the Raptors project.")?;

    Ok(())
}

