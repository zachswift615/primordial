use pyo3::prelude::*;

mod batch;
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
    m.add_function(wrap_pyfunction!(batch::batch_raycast_vision, m)?)?;
    Ok(())
}
