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
    from primordial.world.entities.base import EntityType
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
            vision_range, vision_fov, num_rays,
            ignore_entity_id
        )
        # Invert distance (Python helper gives 0=near, we need 0=far)
        result[:, 0] = 1.0 - result[:, 0]
        return result
