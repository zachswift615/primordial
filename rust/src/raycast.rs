use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;

/// Ray-circle intersection with normalized direction.
///
/// IMPORTANT: Direction MUST be normalized (magnitude = 1.0).
/// Returns distance to intersection or f32::MAX if no hit.
#[inline]
pub fn ray_circle_intersection(
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
