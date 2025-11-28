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
