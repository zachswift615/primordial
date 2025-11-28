"""World system for the Primordial simulation.

Provides 2D physics simulation, entity management, spatial partitioning,
and sound propagation.
"""

from primordial.world.entities import (
    Entity,
    EntityType,
    Food,
    Predator,
    PredatorState,
    Vegetation,
    Water,
)
from primordial.world.environment import Environment
from primordial.world.geometry import AABB, Circle, Vec2
from primordial.world.helpers import (
    get_render_data,
    get_sound_input,
    get_touch_input,
    get_vision_input,
)
from primordial.world.physics import PhysicsEngine
from primordial.world.sound import SoundSource, SoundSystem
from primordial.world.spatial_grid import SpatialGrid
from primordial.world.world import World

__all__ = [
    # Core World
    "World",
    # Geometry
    "Vec2",
    "Circle",
    "AABB",
    # Entities
    "Entity",
    "EntityType",
    "Food",
    "Predator",
    "PredatorState",
    "Vegetation",
    "Water",
    # Systems
    "PhysicsEngine",
    "SoundSystem",
    "SoundSource",
    "SpatialGrid",
    "Environment",
    # Integration Helpers
    "get_vision_input",
    "get_sound_input",
    "get_touch_input",
    "get_render_data",
]
