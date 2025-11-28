"""Integration helpers for agent sensors and renderer.

These helpers bridge the World system to the Agent and Renderer systems,
providing convenient functions for extracting sensory data and render information.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from primordial.world.entities import EntityType
from primordial.world.geometry import Vec2

if TYPE_CHECKING:
    from primordial.world.world import World


def get_vision_input(
    world: World,
    agent_position: Vec2,
    agent_facing: Vec2,
    vision_range: float = 200.0,
    vision_fov: float = 2.094,  # 120 degrees in radians
    num_rays: int = 32,
) -> np.ndarray:
    """Cast vision rays from agent perspective.

    Uses raycasting to simulate vision, casting rays spread across the field of view.
    Each ray returns distance and entity type information.

    Args:
        world: World instance to query.
        agent_position: Position of the agent in world coordinates.
        agent_facing: Unit vector indicating agent's facing direction.
        vision_range: Maximum vision distance (default 200.0).
        vision_fov: Field of view in radians (default 2.094 = 120 degrees).
        num_rays: Number of rays to cast (default 32).

    Returns:
        Array of shape (num_rays, 4) containing:
        - Column 0: distance (normalized 0-1 by vision_range)
        - Column 1: entity_type encoded as:
            - 0 = nothing hit
            - 1 = food
            - 2 = predator
            - 3 = vegetation
            - 4 = water
            - 5 = agent
        - Column 2: entity_r (placeholder for future color info, set to 0)
        - Column 3: entity_g (placeholder for future color info, set to 0)
    """
    result = np.zeros((num_rays, 4), dtype=np.float32)

    # Calculate facing angle
    facing_angle = math.atan2(agent_facing.y, agent_facing.x)

    # Get all active entities for raycasting
    all_entities = [e for e in world.entities.values() if e.is_active]

    # Cast rays across FOV
    for i in range(num_rays):
        # Calculate ray angle relative to facing direction
        # Center rays around facing direction
        if num_rays == 1:
            angle_offset = 0.0
        else:
            angle_offset = (i / (num_rays - 1) - 0.5) * vision_fov

        ray_angle = facing_angle + angle_offset
        ray_direction = Vec2(math.cos(ray_angle), math.sin(ray_angle))

        # Perform raycast
        hit_entity, hit_distance = world.physics.raycast(
            origin=agent_position,
            direction=ray_direction,
            max_distance=vision_range,
            entities=all_entities,
            ignore_entity_id=None,  # Could add agent ID here if needed
        )

        # Normalize distance to [0, 1]
        normalized_distance = hit_distance / vision_range

        # Encode entity type
        if hit_entity is None:
            entity_type_code = 0.0
        else:
            entity_type_map = {
                EntityType.FOOD: 1.0,
                EntityType.PREDATOR: 2.0,
                EntityType.VEGETATION: 3.0,
                EntityType.WATER: 4.0,
                EntityType.AGENT: 5.0,
            }
            entity_type_code = entity_type_map.get(hit_entity.entity_type, 0.0)

        # Store results
        result[i, 0] = normalized_distance
        result[i, 1] = entity_type_code
        result[i, 2] = 0.0  # Placeholder for entity_r
        result[i, 3] = 0.0  # Placeholder for entity_g

    return result


def get_sound_input(
    world: World,
    agent_position: Vec2,
    agent_facing: Vec2,
    num_frequency_bins: int = 32,
) -> dict:
    """Get sound sensory data for agent.

    Computes stereo audio and frequency spectrum at the agent's position.

    Args:
        world: World instance to query.
        agent_position: Position of the agent in world coordinates.
        agent_facing: Unit vector indicating agent's facing direction.
        num_frequency_bins: Number of frequency bins for spectrum (default 32).

    Returns:
        Dictionary with keys:
        - 'left_ear': float (0-1) - sound intensity at left ear
        - 'right_ear': float (0-1) - sound intensity at right ear
        - 'spectrum': np.ndarray of shape (num_frequency_bins,) - frequency spectrum
    """
    # Get stereo sound levels
    left_ear, right_ear = world.sound_system.compute_sound_at_position(
        listener_pos=agent_position,
        listener_facing=agent_facing,
    )

    # Get frequency spectrum
    spectrum = world.sound_system.get_frequency_spectrum(
        listener_pos=agent_position,
        num_frequency_bins=num_frequency_bins,
    )

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "spectrum": spectrum,
    }


def get_render_data(world: World) -> dict:
    """Get all data needed for rendering.

    Extracts current world state in a renderer-friendly format.

    Args:
        world: World instance to query.

    Returns:
        Dictionary with keys:
        - 'entities': List[dict] - all active entities with:
            - id: int - entity ID
            - type: str - entity type name
            - position: tuple (x, y) - entity position
            - radius: float - entity radius
            - velocity: tuple (x, y) or None - entity velocity (None if static)
            - is_static: bool - whether entity is static
        - 'brightness': float - current environmental brightness (0.1 to 0.5)
        - 'world_bounds': dict - world boundaries with min_x, min_y, max_x, max_y
        - 'sound_sources': List[dict] - active sound sources with:
            - position: tuple (x, y) - source position
            - intensity: float - sound intensity
            - frequency: float - sound frequency in Hz
        - 'time_of_day': float - normalized time of day (0-1)
    """
    # Extract entity data
    entities = []
    for entity in world.entities.values():
        if entity.is_active:
            entity_data = {
                "id": entity.id,
                "type": entity.entity_type.value,
                "position": (entity.position.x, entity.position.y),
                "radius": entity.radius,
                "velocity": (entity.velocity.x, entity.velocity.y)
                if not entity.is_static
                else None,
                "is_static": entity.is_static,
            }
            entities.append(entity_data)

    # Extract sound sources
    sound_sources = []
    for source in world.sound_system.sources:
        if source.is_active:
            sound_data = {
                "position": (source.position.x, source.position.y),
                "intensity": source.intensity,
                "frequency": source.frequency,
            }
            sound_sources.append(sound_data)

    # Calculate normalized time of day
    time_of_day = (
        world.environment.time_of_day / world.environment.day_length
        if world.environment.day_length > 0
        else 0.0
    )

    return {
        "entities": entities,
        "brightness": world.brightness,
        "world_bounds": {
            "min_x": world.bounds.min_x,
            "min_y": world.bounds.min_y,
            "max_x": world.bounds.max_x,
            "max_y": world.bounds.max_y,
        },
        "sound_sources": sound_sources,
        "time_of_day": time_of_day,
    }


def get_touch_input(
    world: World,
    agent_position: Vec2,
    agent_radius: float,
    touch_range: float = 5.0,
) -> np.ndarray:
    """Get touch sensor data for nearby entities.

    Detects entities in 8 directions around the agent within touch range.

    Args:
        world: World instance to query.
        agent_position: Position of the agent in world coordinates.
        agent_radius: Radius of the agent.
        touch_range: Detection range beyond agent radius (default 5.0).

    Returns:
        Array of 8 float values (one per direction):
        - Each value is 1.0 if something within touch_range in that direction, else 0.0
        - Directions: N (0°), NE (45°), E (90°), SE (135°), S (180°), SW (225°), W (270°), NW (315°)
        - Index 0: N, 1: NE, 2: E, 3: SE, 4: S, 5: SW, 6: W, 7: NW
    """
    result = np.zeros(8, dtype=np.float32)

    # Define 8 directions (in radians)
    # Using standard math coordinates: 0° = East (+X), 90° = North (+Y)
    directions = [
        math.pi / 2,  # N (90°) - positive Y
        math.pi / 4,  # NE (45°)
        0.0,  # E (0°) - positive X
        7 * math.pi / 4,  # SE (315°)
        3 * math.pi / 2,  # S (270°) - negative Y
        5 * math.pi / 4,  # SW (225°)
        math.pi,  # W (180°) - negative X
        3 * math.pi / 4,  # NW (135°)
    ]

    # Query entities within touch range
    total_range = agent_radius + touch_range
    nearby_entities = world.get_entities_in_radius(
        position=agent_position,
        radius=total_range,
        entity_type=None,
    )

    # Check each direction
    for i, angle in enumerate(directions):
        direction_vec = Vec2(math.cos(angle), math.sin(angle))

        # Check if any entity is in this direction
        for entity in nearby_entities:
            # Calculate vector to entity
            to_entity = entity.position - agent_position
            distance = to_entity.magnitude()

            if distance < 0.001:
                # Entity at same position
                continue

            # Normalize
            to_entity_normalized = to_entity * (1.0 / distance)

            # Check if entity is within angular sector (45 degree cone per direction)
            dot_product = to_entity_normalized.dot(direction_vec)
            angular_threshold = math.cos(
                math.pi / 8
            )  # 22.5 degrees on each side = 45 degree cone

            if dot_product >= angular_threshold:
                # Check if within touch range
                # Distance between surfaces = center_distance - agent_radius - entity_radius
                surface_distance = distance - agent_radius - entity.radius
                if surface_distance <= touch_range:
                    result[i] = 1.0
                    break  # Found entity in this direction

    return result
