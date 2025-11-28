use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Vec2 {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
}

#[pymethods]
impl Vec2 {
    #[new]
    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag > 1e-10 {
            Self { x: self.x / mag, y: self.y / mag }
        } else {
            Self { x: 0.0, y: 0.0 }
        }
    }

    fn dot(&self, other: &Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    fn __repr__(&self) -> String {
        format!("Vec2({}, {})", self.x, self.y)
    }
}
