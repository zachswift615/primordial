"""Base entity class for the world system.

Provides the abstract Entity class that all world entities inherit from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from primordial.world.geometry import Circle, Vec2

if TYPE_CHECKING:
    from primordial.world.world import World


class EntityType(Enum):
    """Entity type enumeration."""

    AGENT = "agent"
    FOOD = "food"
    PREDATOR = "predator"
    VEGETATION = "vegetation"
    WATER = "water"


class Entity(ABC):
    """Abstract base class for all world entities.

    Entities have position, velocity, and physics properties. They can be
    static (immovable) or dynamic (affected by forces).

    Attributes:
        id: Unique entity identifier assigned by the world.
        position: Current position in world coordinates.
        entity_type: Type of entity (for filtering and behavior).
        radius: Collision radius.
        is_static: Whether the entity is immovable.
        is_active: Whether the entity is alive/active in the world.
        velocity: Current velocity (ignored if is_static).
        acceleration: Current acceleration (reset each tick).
        mass: Mass for force calculations.
        friction: Velocity multiplier per tick (0-1, higher = less friction).
    """

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        entity_type: EntityType,
        radius: float,
        is_static: bool = False,
    ) -> None:
        """Initialize entity.

        Args:
            entity_id: Unique identifier (typically assigned by World).
            position: Initial position in world coordinates.
            entity_type: Type of entity.
            radius: Collision radius.
            is_static: If True, entity ignores forces and doesn't move.
        """
        self.id = entity_id
        self.position = position
        self.entity_type = entity_type
        self.radius = radius
        self.is_static = is_static
        self.is_active = True

        # Physics properties (ignored if is_static)
        self.velocity = Vec2(0.0, 0.0)
        self.acceleration = Vec2(0.0, 0.0)
        self.mass = 1.0
        self.friction = 0.95  # Velocity multiplier per tick

    @abstractmethod
    def update(self, world: World, dt: float) -> None:
        """Update entity state. Called each tick before physics.

        Override this to implement entity-specific behavior like AI,
        state changes, or interactions.

        Args:
            world: The world instance for queries and interactions.
            dt: Time step in seconds.
        """
        pass

    @abstractmethod
    def get_collision_shape(self) -> Circle:
        """Return collision shape for physics.

        Returns:
            Circle representing the entity's collision boundary.
        """
        pass

    def apply_force(self, force: Vec2) -> None:
        """Apply force to entity (F = ma).

        Static entities ignore forces.

        Args:
            force: Force vector to apply.
        """
        if not self.is_static:
            # a = F / m
            self.acceleration = self.acceleration + (force * (1.0 / self.mass))

    def apply_impulse(self, impulse: Vec2) -> None:
        """Apply impulse (immediate velocity change).

        Static entities ignore impulses.

        Args:
            impulse: Impulse vector to apply (direct velocity change).
        """
        if not self.is_static:
            self.velocity = self.velocity + impulse

    def set_position(self, position: Vec2) -> None:
        """Set entity position directly.

        Args:
            position: New position.
        """
        self.position = position

    def set_velocity(self, velocity: Vec2) -> None:
        """Set entity velocity directly.

        Args:
            velocity: New velocity.
        """
        if not self.is_static:
            self.velocity = velocity

    def deactivate(self) -> None:
        """Mark entity as inactive (will be removed from world)."""
        self.is_active = False

    def distance_to(self, other: Entity) -> float:
        """Return distance to another entity's position.

        Args:
            other: Other entity.

        Returns:
            Distance between positions.
        """
        return self.position.distance_to(other.position)

    def direction_to(self, other: Entity) -> Vec2:
        """Return normalized direction vector to another entity.

        Args:
            other: Other entity.

        Returns:
            Unit vector pointing from this entity to other.
        """
        return (other.position - self.position).normalized()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id}, "
            f"pos=({self.position.x:.1f}, {self.position.y:.1f}), "
            f"active={self.is_active})"
        )
