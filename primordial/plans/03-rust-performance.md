# Rust Performance Optimization Plan

## Overview

This document specifies the implementation of a Rust-based performance layer for the Primordial simulation using PyO3. The primary bottleneck is vision raycasting, which currently takes ~1.7ms per agent in pure Python. The target is <0.1ms per agent to support 100+ agents at 60Hz.

## Goals

- **Primary**: Achieve 100 agents at 60Hz (16.67ms per frame budget)
- **Secondary**: Provide foundation for future physics optimizations
- **Constraint**: Maintain Python API compatibility - no changes to existing Python code

## Current Performance Profile

```
Sensor performance (ms):
  Vision: 1.688    <- BOTTLENECK (32 rays × entity iteration)
  Audio: 0.003
  Touch: 0.003
  Proprioception: 0.0008

Total per agent: ~1.7ms
Target per agent: <0.1ms (17x improvement needed)
```

## Architecture

### File Structure

```
kung-foo-chick-pea-feeble/           # Project root
├── pyproject.toml                    # NEW: Build configuration for maturin
├── Cargo.toml                        # NEW: Rust workspace config
├── rust/                             # NEW: Rust crate source
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                    # PyO3 module definition
│       ├── geometry.rs               # Vec2, Circle, AABB
│       ├── raycast.rs                # Core raycasting implementation
│       ├── spatial.rs                # Spatial acceleration structure
│       └── batch.rs                  # Batched operations for multiple agents
├── primordial/
│   ├── _rust.pyi                     # NEW: Type stubs for Rust module
│   ├── agents/
│   │   └── sensors.py                # MODIFIED: Uses Rust backend
│   └── world/
│       └── helpers_rust.py           # NEW: Rust-accelerated helpers
└── .github/
    └── workflows/
        └── rust-build.yml            # NEW: CI/CD for cross-platform builds
```

### Rust Crate Structure

```toml
# rust/Cargo.toml
[package]
name = "primordial_rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "_rust"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
ndarray = "0.15"
rayon = { version = "1.8", optional = true }

[features]
default = ["parallel"]
parallel = ["rayon"]
```

## Implementation Phases

### Phase 1: Project Setup & Build Configuration (Day 1)

**Goal**: Set up Rust/PyO3 project structure with proper build configuration.

**Tasks**:

1. **Create `pyproject.toml` from scratch** (project has none currently):
   ```toml
   [project]
   name = "primordial"
   version = "0.1.0"
   description = "Primordial life simulation with neural network agents"
   requires-python = ">=3.11"
   dependencies = [
       "numpy>=1.24",
       "torch>=2.0",
   ]

   [project.optional-dependencies]
   dev = [
       "pytest>=7.0",
       "maturin>=1.0",
   ]

   [build-system]
   requires = ["maturin>=1.0,<2.0"]
   build-backend = "maturin"

   [tool.maturin]
   features = ["pyo3/extension-module"]
   python-source = "."
   module-name = "primordial._rust"
   manifest-path = "rust/Cargo.toml"
   ```

2. **Create Rust workspace** at project root:
   ```toml
   # Cargo.toml (at project root)
   [workspace]
   members = ["rust"]
   ```

3. **Initialize Rust crate**:
   ```bash
   mkdir -p rust/src
   ```

4. **Create `rust/Cargo.toml`**:
   ```toml
   [package]
   name = "primordial_rust"
   version = "0.1.0"
   edition = "2021"

   [lib]
   name = "_rust"
   crate-type = ["cdylib"]

   [dependencies]
   pyo3 = { version = "0.20", features = ["extension-module"] }
   numpy = "0.20"
   ndarray = "0.15"
   rayon = { version = "1.8", optional = true }

   [features]
   default = ["parallel"]
   parallel = ["rayon"]
   ```

5. **Create `rust/src/lib.rs`**:
   ```rust
   use pyo3::prelude::*;

   mod geometry;
   mod raycast;

   /// Version for compatibility checking
   const VERSION: &str = "0.1.0";

   #[pymodule]
   fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add("__version__", VERSION)?;
       m.add_class::<geometry::Vec2>()?;
       m.add_function(wrap_pyfunction!(raycast::raycast_vision, m)?)?;
       Ok(())
   }
   ```

6. **Create `rust/src/geometry.rs`**:
   ```rust
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
   ```

7. **Create type stubs** at `primordial/_rust.pyi`:
   ```python
   """Type stubs for Rust extension module."""
   import numpy as np
   from numpy.typing import NDArray

   __version__: str

   class Vec2:
       x: float
       y: float
       def __init__(self, x: float, y: float) -> None: ...
       def magnitude(self) -> float: ...
       def normalized(self) -> Vec2: ...
       def dot(self, other: Vec2) -> float: ...

   def raycast_vision(
       origin: NDArray[np.float32],
       ray_directions: NDArray[np.float32],
       max_distance: float,
       entity_positions: NDArray[np.float32],
       entity_radii: NDArray[np.float32],
       entity_types: NDArray[np.uint8],
       entity_ids: NDArray[np.int32],
       ignore_entity_id: int | None = None,
   ) -> NDArray[np.float32]: ...
   ```

8. **Build and verify**:
   ```bash
   cd /Users/zachswift/projects/kung-foo-chick-pea-feeble
   maturin develop
   python -c "from primordial._rust import Vec2, __version__; print(f'Version: {__version__}'); v = Vec2(3.0, 4.0); print(f'Magnitude: {v.magnitude()}')"
   ```

**Verification**:
- `maturin develop` succeeds without errors
- `from primordial._rust import Vec2, __version__` works
- Vec2.magnitude() returns correct value (5.0 for Vec2(3,4))

---

### Phase 2: Core Raycasting (Day 2)

**Goal**: Implement ray-circle intersection in Rust with correct semantics.

**Tasks**:

1. **Create `rust/src/raycast.rs`**:
   ```rust
   use ndarray::Array2;
   use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
   use pyo3::prelude::*;

   /// Ray-circle intersection with normalized direction.
   ///
   /// IMPORTANT: Direction MUST be normalized (magnitude = 1.0).
   /// Returns distance to intersection or f32::MAX if no hit.
   #[inline]
   fn ray_circle_intersection(
       origin_x: f32, origin_y: f32,
       dir_x: f32, dir_y: f32,  // Must be normalized!
       center_x: f32, center_y: f32,
       radius: f32,
   ) -> f32 {
       // Debug assertion for normalized direction
       debug_assert!(
           (dir_x * dir_x + dir_y * dir_y - 1.0).abs() < 0.01,
           "Ray direction must be normalized"
       );

       // Vector from ray origin to circle center
       let oc_x = origin_x - center_x;
       let oc_y = origin_y - center_y;

       // Since direction is normalized, a = 1.0
       // Simplified quadratic: t^2 + 2bt + c = 0
       let half_b = oc_x * dir_x + oc_y * dir_y;
       let c = oc_x * oc_x + oc_y * oc_y - radius * radius;

       let discriminant = half_b * half_b - c;

       if discriminant < 0.0 {
           return f32::MAX;
       }

       // Get nearest positive intersection
       let sqrt_d = discriminant.sqrt();
       let t1 = -half_b - sqrt_d;
       let t2 = -half_b + sqrt_d;

       if t1 > 0.0 {
           t1
       } else if t2 > 0.0 {
           t2
       } else {
           f32::MAX
       }
   }

   /// Result of a single raycast
   #[derive(Clone, Copy)]
   struct RayHit {
       distance: f32,
       entity_type: u8,
       entity_id: i32,
   }

   /// Cast a single ray against entity list
   fn cast_ray(
       origin_x: f32, origin_y: f32,
       dir_x: f32, dir_y: f32,
       max_distance: f32,
       entities: &[(f32, f32, f32, u8, i32)],  // (x, y, radius, type, id)
       ignore_id: Option<i32>,
   ) -> RayHit {
       let mut closest = RayHit {
           distance: max_distance,
           entity_type: 0,
           entity_id: -1,
       };

       for &(cx, cy, radius, entity_type, entity_id) in entities {
           // Skip ignored entity (self)
           if Some(entity_id) == ignore_id {
               continue;
           }

           let dist = ray_circle_intersection(
               origin_x, origin_y,
               dir_x, dir_y,
               cx, cy, radius
           );

           if dist < closest.distance {
               closest.distance = dist;
               closest.entity_type = entity_type;
               closest.entity_id = entity_id;
           }
       }

       closest
   }

   /// Batch raycast for a single agent's vision.
   ///
   /// Args:
   ///     origin: Agent position (2,)
   ///     ray_directions: Pre-computed normalized ray directions (num_rays, 2)
   ///     max_distance: Maximum ray distance
   ///     entity_positions: Entity centers (num_entities, 2)
   ///     entity_radii: Entity radii (num_entities,)
   ///     entity_types: Entity type codes (num_entities,)
   ///     entity_ids: Entity IDs (num_entities,)
   ///     ignore_entity_id: Optional entity ID to ignore (for self)
   ///
   /// Returns:
   ///     Array (num_rays, 4) with [normalized_distance, entity_type, 0, 0]
   ///     Distance is normalized: 0.0 = max range (far), 1.0 = touching (close)
   #[pyfunction]
   #[pyo3(signature = (origin, ray_directions, max_distance, entity_positions, entity_radii, entity_types, entity_ids, ignore_entity_id=None))]
   pub fn raycast_vision<'py>(
       py: Python<'py>,
       origin: PyReadonlyArray1<'py, f32>,
       ray_directions: PyReadonlyArray2<'py, f32>,
       max_distance: f32,
       entity_positions: PyReadonlyArray2<'py, f32>,
       entity_radii: PyReadonlyArray1<'py, f32>,
       entity_types: PyReadonlyArray1<'py, u8>,
       entity_ids: PyReadonlyArray1<'py, i32>,
       ignore_entity_id: Option<i32>,
   ) -> PyResult<Py<PyArray2<f32>>> {
       let origin = origin.as_slice()?;
       let directions = ray_directions.as_array();
       let positions = entity_positions.as_array();
       let radii = entity_radii.as_slice()?;
       let types = entity_types.as_slice()?;
       let ids = entity_ids.as_slice()?;

       let num_rays = directions.nrows();
       let num_entities = positions.nrows();

       // Build entity list
       let entities: Vec<_> = (0..num_entities)
           .map(|i| (
               positions[[i, 0]],
               positions[[i, 1]],
               radii[i],
               types[i],
               ids[i],
           ))
           .collect();

       // Allocate output array
       let mut result = Array2::<f32>::zeros((num_rays, 4));

       for i in 0..num_rays {
           let dir_x = directions[[i, 0]];
           let dir_y = directions[[i, 1]];

           let hit = cast_ray(
               origin[0], origin[1],
               dir_x, dir_y,
               max_distance,
               &entities,
               ignore_entity_id,
           );

           // Normalize distance: 0 = far (at max_distance), 1 = close (touching)
           // This matches what VisionSensor expects after inversion
           result[[i, 0]] = 1.0 - (hit.distance / max_distance).min(1.0);
           result[[i, 1]] = hit.entity_type as f32;
           result[[i, 2]] = 0.0;  // Reserved for color
           result[[i, 3]] = 0.0;  // Reserved for color
       }

       Ok(result.into_pyarray_bound(py).unbind())
   }
   ```

2. **Update `rust/src/lib.rs`** to export raycast:
   ```rust
   use pyo3::prelude::*;

   mod geometry;
   mod raycast;

   const VERSION: &str = "0.1.0";

   #[pymodule]
   fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add("__version__", VERSION)?;
       m.add_class::<geometry::Vec2>()?;
       m.add_function(wrap_pyfunction!(raycast::raycast_vision, m)?)?;
       Ok(())
   }
   ```

3. **Create unit test** `primordial/tests/test_rust_raycast.py`:
   ```python
   """Tests for Rust raycast implementation."""
   import pytest
   import numpy as np
   import math

   # Skip all tests if Rust extension not available
   try:
       from primordial._rust import raycast_vision
       RUST_AVAILABLE = True
   except ImportError:
       RUST_AVAILABLE = False

   pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")


   class TestRaycastVision:
       """Tests for raycast_vision function."""

       def test_no_entities_returns_zeros(self):
           """Empty entity list returns max distance (0.0 normalized)."""
           origin = np.array([0.0, 0.0], dtype=np.float32)
           directions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

           result = raycast_vision(
               origin, directions, 100.0,
               np.zeros((0, 2), dtype=np.float32),  # No entities
               np.zeros(0, dtype=np.float32),
               np.zeros(0, dtype=np.uint8),
               np.zeros(0, dtype=np.int32),
           )

           assert result.shape == (2, 4)
           assert np.allclose(result[:, 0], 0.0)  # Far = 0.0

       def test_hit_entity_in_front(self):
           """Ray hits entity directly ahead."""
           origin = np.array([0.0, 0.0], dtype=np.float32)
           directions = np.array([[1.0, 0.0]], dtype=np.float32)  # Facing right

           # Entity at (50, 0) with radius 10
           positions = np.array([[50.0, 0.0]], dtype=np.float32)
           radii = np.array([10.0], dtype=np.float32)
           types = np.array([1], dtype=np.uint8)  # Food
           ids = np.array([42], dtype=np.int32)

           result = raycast_vision(
               origin, directions, 100.0,
               positions, radii, types, ids
           )

           # Hit at distance 40 (50 - 10 radius), normalized: 1 - 40/100 = 0.6
           assert result[0, 0] == pytest.approx(0.6, rel=0.01)
           assert result[0, 1] == 1.0  # Food type

       def test_ignore_entity_id(self):
           """Ignored entity is not hit."""
           origin = np.array([0.0, 0.0], dtype=np.float32)
           directions = np.array([[1.0, 0.0]], dtype=np.float32)

           positions = np.array([[30.0, 0.0]], dtype=np.float32)
           radii = np.array([10.0], dtype=np.float32)
           types = np.array([1], dtype=np.uint8)
           ids = np.array([42], dtype=np.int32)

           # Without ignore - should hit
           result = raycast_vision(
               origin, directions, 100.0,
               positions, radii, types, ids
           )
           assert result[0, 0] > 0.5  # Close hit

           # With ignore - should miss
           result = raycast_vision(
               origin, directions, 100.0,
               positions, radii, types, ids,
               ignore_entity_id=42
           )
           assert result[0, 0] == pytest.approx(0.0)  # No hit

       def test_closest_entity_wins(self):
           """Nearest entity is reported when multiple in path."""
           origin = np.array([0.0, 0.0], dtype=np.float32)
           directions = np.array([[1.0, 0.0]], dtype=np.float32)

           # Two entities: food at 30, predator at 60
           positions = np.array([[30.0, 0.0], [60.0, 0.0]], dtype=np.float32)
           radii = np.array([5.0, 5.0], dtype=np.float32)
           types = np.array([1, 2], dtype=np.uint8)  # Food=1, Predator=2
           ids = np.array([1, 2], dtype=np.int32)

           result = raycast_vision(
               origin, directions, 100.0,
               positions, radii, types, ids
           )

           # Should hit food (closer), not predator
           assert result[0, 1] == 1.0  # Food type
   ```

**Verification**:
- `maturin develop` succeeds
- `python -m pytest primordial/tests/test_rust_raycast.py -v` passes
- Rust matches Python behavior for all test cases

---

### Phase 3: Spatial Acceleration with DDA (Day 3)

**Goal**: Add proper grid-based spatial acceleration using DDA ray traversal.

**Tasks**:

1. **Create `rust/src/spatial.rs`** with proper DDA algorithm:
   ```rust
   use std::collections::{HashMap, HashSet};

   /// Spatial grid for accelerated ray queries.
   pub struct SpatialGrid {
       cell_size: f32,
       inv_cell_size: f32,
       cells: HashMap<(i32, i32), Vec<usize>>,
   }

   impl SpatialGrid {
       pub fn new(cell_size: f32) -> Self {
           Self {
               cell_size,
               inv_cell_size: 1.0 / cell_size,
               cells: HashMap::new(),
           }
       }

       #[inline]
       fn cell_coords(&self, x: f32, y: f32) -> (i32, i32) {
           (
               (x * self.inv_cell_size).floor() as i32,
               (y * self.inv_cell_size).floor() as i32,
           )
       }

       /// Build grid from entity positions and radii.
       pub fn build(&mut self, positions: &[(f32, f32)], radii: &[f32]) {
           self.cells.clear();

           for (i, (&(x, y), &r)) in positions.iter().zip(radii).enumerate() {
               let min_cell = self.cell_coords(x - r, y - r);
               let max_cell = self.cell_coords(x + r, y + r);

               for cx in min_cell.0..=max_cell.0 {
                   for cy in min_cell.1..=max_cell.1 {
                       self.cells.entry((cx, cy))
                           .or_insert_with(Vec::new)
                           .push(i);
                   }
               }
           }
       }

       /// Query entities along a ray using DDA algorithm.
       ///
       /// Uses Amanatides & Woo's "A Fast Voxel Traversal Algorithm"
       pub fn query_ray_dda(
           &self,
           origin: (f32, f32),
           dir: (f32, f32),  // Must be normalized
           max_dist: f32,
       ) -> Vec<usize> {
           let mut result = Vec::new();
           let mut visited = HashSet::new();

           // Starting cell
           let mut cell = self.cell_coords(origin.0, origin.1);
           let end_point = (origin.0 + dir.0 * max_dist, origin.1 + dir.1 * max_dist);
           let end_cell = self.cell_coords(end_point.0, end_point.1);

           // Step direction
           let step_x = if dir.0 >= 0.0 { 1 } else { -1 };
           let step_y = if dir.1 >= 0.0 { 1 } else { -1 };

           // Distance to next cell boundary
           let next_x = if dir.0 >= 0.0 {
               (cell.0 + 1) as f32 * self.cell_size
           } else {
               cell.0 as f32 * self.cell_size
           };
           let next_y = if dir.1 >= 0.0 {
               (cell.1 + 1) as f32 * self.cell_size
           } else {
               cell.1 as f32 * self.cell_size
           };

           // t values for crossing cell boundaries
           let mut t_max_x = if dir.0.abs() > 1e-10 {
               (next_x - origin.0) / dir.0
           } else {
               f32::MAX
           };
           let mut t_max_y = if dir.1.abs() > 1e-10 {
               (next_y - origin.1) / dir.1
           } else {
               f32::MAX
           };

           // Delta t for one cell width
           let t_delta_x = if dir.0.abs() > 1e-10 {
               self.cell_size / dir.0.abs()
           } else {
               f32::MAX
           };
           let t_delta_y = if dir.1.abs() > 1e-10 {
               self.cell_size / dir.1.abs()
           } else {
               f32::MAX
           };

           // Traverse grid
           let mut t = 0.0;
           while t < max_dist {
               // Collect entities in current cell
               if let Some(entities) = self.cells.get(&cell) {
                   for &idx in entities {
                       if visited.insert(idx) {
                           result.push(idx);
                       }
                   }
               }

               // Check if we've reached the end
               if cell == end_cell {
                   break;
               }

               // Step to next cell
               if t_max_x < t_max_y {
                   t = t_max_x;
                   t_max_x += t_delta_x;
                   cell.0 += step_x;
               } else {
                   t = t_max_y;
                   t_max_y += t_delta_y;
                   cell.1 += step_y;
               }
           }

           result
       }
   }
   ```

2. **Update raycast.rs to use spatial grid** (optional optimization path)

**Verification**:
- Unit test spatial grid builds correctly
- DDA traversal visits correct cells
- Benchmark shows O(1) vs O(n) improvement for large entity counts

---

### Phase 4: Parallel Batch Processing (Day 4)

**Goal**: Process multiple agents' vision in parallel using rayon with GIL release.

**Tasks**:

1. **Create `rust/src/batch.rs`**:
   ```rust
   use ndarray::Array3;
   use numpy::{PyArray3, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
   use pyo3::prelude::*;

   #[cfg(feature = "parallel")]
   use rayon::prelude::*;

   use crate::raycast::ray_circle_intersection;

   /// Batch vision for multiple agents in parallel.
   ///
   /// Processes all agents' vision rays in parallel, releasing the GIL.
   #[pyfunction]
   pub fn batch_raycast_vision<'py>(
       py: Python<'py>,
       agent_positions: PyReadonlyArray2<'py, f32>,    // (num_agents, 2)
       agent_angles: PyReadonlyArray1<'py, f32>,       // (num_agents,)
       agent_ids: PyReadonlyArray1<'py, i32>,          // (num_agents,) for self-ignore
       num_rays: usize,
       fov: f32,
       max_distance: f32,
       entity_positions: PyReadonlyArray2<'py, f32>,
       entity_radii: PyReadonlyArray1<'py, f32>,
       entity_types: PyReadonlyArray1<'py, u8>,
       entity_ids: PyReadonlyArray1<'py, i32>,
   ) -> PyResult<Py<PyArray3<f32>>> {
       // Extract all data before releasing GIL
       let agent_pos: Vec<_> = agent_positions.as_array()
           .rows().into_iter()
           .map(|r| (r[0], r[1]))
           .collect();
       let agent_ang: Vec<f32> = agent_angles.as_slice()?.to_vec();
       let agent_id: Vec<i32> = agent_ids.as_slice()?.to_vec();

       let ent_pos: Vec<_> = entity_positions.as_array()
           .rows().into_iter()
           .map(|r| (r[0], r[1]))
           .collect();
       let ent_rad: Vec<f32> = entity_radii.as_slice()?.to_vec();
       let ent_type: Vec<u8> = entity_types.as_slice()?.to_vec();
       let ent_id: Vec<i32> = entity_ids.as_slice()?.to_vec();

       let num_agents = agent_pos.len();
       let num_entities = ent_pos.len();

       // Pre-compute ray angle offsets
       let ray_offsets: Vec<f32> = (0..num_rays)
           .map(|i| {
               if num_rays == 1 {
                   0.0
               } else {
                   (i as f32 / (num_rays - 1) as f32 - 0.5) * fov
               }
           })
           .collect();

       // Build entity tuple list
       let entities: Vec<_> = (0..num_entities)
           .map(|i| (ent_pos[i].0, ent_pos[i].1, ent_rad[i], ent_type[i], ent_id[i]))
           .collect();

       // Release GIL and do parallel work
       let results: Vec<f32> = py.allow_threads(|| {
           #[cfg(feature = "parallel")]
           let iter = (0..num_agents).into_par_iter();

           #[cfg(not(feature = "parallel"))]
           let iter = (0..num_agents).into_iter();

           iter.flat_map(|agent_idx| {
               let (ax, ay) = agent_pos[agent_idx];
               let angle = agent_ang[agent_idx];
               let self_id = agent_id[agent_idx];

               let mut agent_result = Vec::with_capacity(num_rays * 4);

               for &offset in &ray_offsets {
                   let ray_angle = angle + offset;
                   let dir_x = ray_angle.cos();
                   let dir_y = ray_angle.sin();

                   let mut closest_dist = max_distance;
                   let mut closest_type = 0u8;

                   for &(ex, ey, er, et, eid) in &entities {
                       // Skip self
                       if eid == self_id {
                           continue;
                       }

                       let dist = ray_circle_intersection(
                           ax, ay, dir_x, dir_y, ex, ey, er
                       );
                       if dist < closest_dist {
                           closest_dist = dist;
                           closest_type = et;
                       }
                   }

                   agent_result.push(1.0 - (closest_dist / max_distance).min(1.0));
                   agent_result.push(closest_type as f32);
                   agent_result.push(0.0);
                   agent_result.push(0.0);
               }

               agent_result
           }).collect()
       });

       // Convert to 3D array
       let result = Array3::from_shape_vec((num_agents, num_rays, 4), results)
           .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

       Ok(result.into_pyarray_bound(py).unbind())
   }
   ```

2. **Make `ray_circle_intersection` public** in raycast.rs:
   ```rust
   pub fn ray_circle_intersection(...) -> f32 { ... }
   ```

3. **Update lib.rs**:
   ```rust
   mod batch;
   // In pymodule:
   m.add_function(wrap_pyfunction!(batch::batch_raycast_vision, m)?)?;
   ```

**Verification**:
- Batch function returns correct shape (num_agents, num_rays, 4)
- Parallel speedup scales with CPU cores
- GIL is released during heavy computation

---

### Phase 5: Python Integration (Day 5)

**Goal**: Integrate Rust backend into existing sensors with fallback.

**Tasks**:

1. **Create `primordial/world/helpers_rust.py`**:
   ```python
   """Rust-accelerated helper functions.

   Drop-in replacements for helpers.py functions using Rust backend.
   Falls back gracefully if Rust extension unavailable.
   """
   from __future__ import annotations

   import warnings
   from typing import TYPE_CHECKING

   import numpy as np

   if TYPE_CHECKING:
       from primordial.world.geometry import Vec2
       from primordial.world.world import World

   # Try to import Rust extension
   try:
       from primordial._rust import raycast_vision as _rust_raycast
       from primordial._rust import __version__ as _rust_version
       RUST_AVAILABLE = True
   except ImportError:
       RUST_AVAILABLE = False
       _rust_version = None

   # Version compatibility check
   REQUIRED_RUST_VERSION = "0.1.0"


   def check_rust_compatibility() -> bool:
       """Check if Rust extension is available and compatible."""
       if not RUST_AVAILABLE:
           return False
       if _rust_version != REQUIRED_RUST_VERSION:
           warnings.warn(
               f"Rust extension version {_rust_version} doesn't match "
               f"required {REQUIRED_RUST_VERSION}"
           )
           return False
       return True


   def _entity_type_code(entity) -> int:
       """Convert entity type to numeric code."""
       from primordial.world.entities import EntityType
       return {
           EntityType.FOOD: 1,
           EntityType.PREDATOR: 2,
           EntityType.VEGETATION: 3,
           EntityType.WATER: 4,
           EntityType.AGENT: 5,
       }.get(entity.entity_type, 0)


   def get_vision_input_fast(
       world: World,
       agent_position: Vec2,
       agent_facing: Vec2,
       vision_range: float = 200.0,
       vision_fov: float = 2.094,  # 120 degrees
       num_rays: int = 32,
       ignore_entity_id: int | None = None,
   ) -> np.ndarray:
       """Rust-accelerated vision input.

       Returns array matching helpers.get_vision_input format but faster.
       Falls back to Python implementation on error.

       Args:
           world: World instance to query.
           agent_position: Agent's position.
           agent_facing: Agent's facing direction (unit vector).
           vision_range: Maximum ray distance.
           vision_fov: Field of view in radians.
           num_rays: Number of rays to cast.
           ignore_entity_id: Entity ID to ignore (typically self).

       Returns:
           Array (num_rays, 4) with [distance, type, 0, 0].
           Distance: 0.0 = far, 1.0 = close.
       """
       if not RUST_AVAILABLE:
           raise ImportError("Rust extension not available")

       # Extract entity data
       entities = [e for e in world.entities.values() if e.is_active]

       if not entities:
           return np.zeros((num_rays, 4), dtype=np.float32)

       try:
           positions = np.array(
               [[e.position.x, e.position.y] for e in entities],
               dtype=np.float32
           )
           radii = np.array([e.radius for e in entities], dtype=np.float32)
           types = np.array([_entity_type_code(e) for e in entities], dtype=np.uint8)
           ids = np.array([e.id for e in entities], dtype=np.int32)

           # Pre-compute normalized ray directions
           facing_angle = np.arctan2(agent_facing.y, agent_facing.x)
           angles = np.linspace(-vision_fov / 2, vision_fov / 2, num_rays) + facing_angle
           directions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)

           origin = np.array([agent_position.x, agent_position.y], dtype=np.float32)

           result = _rust_raycast(
               origin, directions, vision_range,
               positions, radii, types, ids,
               ignore_entity_id
           )

           # Validate output
           if result.shape != (num_rays, 4):
               raise ValueError(f"Unexpected output shape: {result.shape}")

           return result

       except Exception as e:
           warnings.warn(f"Rust raycast failed: {e}, falling back to Python")
           from primordial.world import helpers
           result = helpers.get_vision_input(
               world, agent_position, agent_facing,
               vision_range, vision_fov, num_rays
           )
           # Invert distance (Python helper gives 0=near, we need 0=far)
           result[:, 0] = 1.0 - result[:, 0]
           return result
   ```

2. **Update `primordial/agents/sensors.py`** VisionSensor:
   ```python
   # At top of file, after imports
   try:
       from primordial.world.helpers_rust import get_vision_input_fast, RUST_AVAILABLE
       _USE_RUST = RUST_AVAILABLE
   except ImportError:
       _USE_RUST = False

   class VisionSensor:
       # ... existing code ...

       def sense(
           self,
           position: Vec2,
           facing: Vec2,
           world: World,
           ignore_entity_id: int | None = None,
       ) -> np.ndarray:
           """Cast vision rays and return sensory data."""
           if _USE_RUST:
               try:
                   return get_vision_input_fast(
                       world=world,
                       agent_position=position,
                       agent_facing=facing,
                       vision_range=self.max_range,
                       vision_fov=self.fov,
                       num_rays=self.num_rays,
                       ignore_entity_id=ignore_entity_id,
                   )
               except Exception:
                   pass  # Fall through to Python

           # Python fallback
           result = helpers.get_vision_input(
               world=world,
               agent_position=position,
               agent_facing=facing,
               vision_range=self.max_range,
               vision_fov=self.fov,
               num_rays=self.num_rays,
           )
           result[:, 0] = 1.0 - result[:, 0]
           return result
   ```

**Verification**:
- All existing sensor tests pass
- VisionSensor uses Rust when available
- Graceful fallback to Python

---

### Phase 6: Testing, Benchmarking & CI (Day 6)

**Goal**: Comprehensive testing, performance validation, and CI setup.

**Tasks**:

1. **Create benchmark comparison** `primordial/tests/test_rust_performance.py`:
   ```python
   """Performance comparison tests for Rust vs Python."""
   import time
   import pytest
   import numpy as np

   from primordial.agents import AgentBody
   from primordial.world import World
   from primordial.world.geometry import Vec2
   from primordial.world import helpers

   try:
       from primordial.world.helpers_rust import get_vision_input_fast, RUST_AVAILABLE
   except ImportError:
       RUST_AVAILABLE = False


   @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
   class TestRustPerformance:
       """Performance benchmarks comparing Rust vs Python."""

       @pytest.fixture
       def populated_world(self):
           world = World(width=1000, height=1000)
           world.setup_default_world()
           return world

       def test_rust_faster_than_python(self, populated_world):
           """Rust should be at least 5x faster than Python."""
           world = populated_world
           position = Vec2(500, 500)
           facing = Vec2(1, 0)

           num_iterations = 50

           # Benchmark Python
           start = time.perf_counter()
           for _ in range(num_iterations):
               helpers.get_vision_input(world, position, facing)
           python_time = (time.perf_counter() - start) / num_iterations

           # Benchmark Rust
           start = time.perf_counter()
           for _ in range(num_iterations):
               get_vision_input_fast(world, position, facing)
           rust_time = (time.perf_counter() - start) / num_iterations

           speedup = python_time / rust_time
           print(f"\nPython: {python_time*1000:.3f}ms, Rust: {rust_time*1000:.3f}ms")
           print(f"Speedup: {speedup:.1f}x")

           assert speedup > 5.0, f"Rust only {speedup:.1f}x faster, expected >5x"

       def test_rust_python_equivalence(self, populated_world):
           """Rust and Python produce equivalent results."""
           world = populated_world
           position = Vec2(500, 500)
           facing = Vec2(1, 0)

           python_result = helpers.get_vision_input(world, position, facing)
           python_result[:, 0] = 1.0 - python_result[:, 0]  # Match Rust normalization

           rust_result = get_vision_input_fast(world, position, facing)

           # Allow small floating point differences
           np.testing.assert_allclose(
               rust_result[:, :2],  # distance and type
               python_result[:, :2],
               rtol=0.01,
               atol=0.01
           )
   ```

2. **Create CI workflow** `.github/workflows/rust-build.yml`:
   ```yaml
   name: Build and Test Rust Extension

   on:
     push:
       branches: [main]
     pull_request:
       branches: [main]

   jobs:
     test:
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, macos-latest, windows-latest]
           python-version: ['3.11', '3.12']
         fail-fast: false

       steps:
         - uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python-version }}

         - name: Install Rust
           uses: dtolnay/rust-toolchain@stable

         - name: Install dependencies
           run: |
             python -m pip install --upgrade pip
             pip install maturin pytest numpy torch

         - name: Build Rust extension
           run: maturin develop --release

         - name: Run tests
           run: python -m pytest primordial/tests/ -v

     build-wheels:
       needs: test
       runs-on: ${{ matrix.os }}
       strategy:
         matrix:
           os: [ubuntu-latest, macos-latest, windows-latest]

       steps:
         - uses: actions/checkout@v4
         - uses: dtolnay/rust-toolchain@stable
         - uses: actions/setup-python@v5
           with:
             python-version: '3.11'

         - name: Build wheels
           run: |
             pip install maturin
             maturin build --release

         - name: Upload wheels
           uses: actions/upload-artifact@v4
           with:
             name: wheels-${{ matrix.os }}
             path: target/wheels/*.whl
   ```

**Verification**:
- All tests pass on CI
- Wheels build for all platforms
- Performance regression detection works

---

## Performance Targets

| Metric | Current (Python) | Target (Rust) | Improvement |
|--------|------------------|---------------|-------------|
| Single agent vision | 1.7ms | <0.1ms | 17x |
| 10 agents batch | N/A | <5ms | - |
| 100 agents @ 60Hz | Impossible | Achievable | - |

## Safety Considerations

1. **Panic handling**: All Rust code uses `PyResult` for error propagation
2. **GIL safety**: `py.allow_threads()` used for parallel work
3. **Memory safety**: All arrays copied or borrowed correctly
4. **Edge cases**: Empty arrays, NaN values, infinite distances handled

## Commands Reference

```bash
# Development build (fast compile, debug symbols)
maturin develop

# Release build (optimized)
maturin develop --release

# Build wheels for distribution
maturin build --release

# Run all tests
python -m pytest primordial/tests/ -v

# Run only Rust tests
python -m pytest primordial/tests/test_rust_*.py -v

# Benchmark
python -c "
from primordial.world import World
from primordial.world.geometry import Vec2
from primordial.world.helpers_rust import get_vision_input_fast
import time

world = World()
world.setup_default_world()

start = time.perf_counter()
for _ in range(1000):
    get_vision_input_fast(world, Vec2(500, 500), Vec2(1, 0))
elapsed = (time.perf_counter() - start) / 1000 * 1000
print(f'Rust vision: {elapsed:.3f}ms per call')
"
```
