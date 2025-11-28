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
