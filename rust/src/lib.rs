use pyo3::prelude::*;

mod geometry;
mod raycast;
mod spatial;

/// Version for compatibility checking
const VERSION: &str = "0.1.0";

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_class::<geometry::Vec2>()?;
    m.add_function(wrap_pyfunction!(raycast::raycast_vision, m)?)?;
    Ok(())
}
