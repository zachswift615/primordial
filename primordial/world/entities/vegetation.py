"""Vegetation entity for the world system.

Static obstacles that block movement and vision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import Circle, Vec2

if TYPE_CHECKING:
    from primordial.world.world import World


class Vegetation(Entity):
    """Static vegetation that blocks movement and vision.

    Vegetation serves as obstacles in the world. They don't move or
    have any behavior, but they block agent movement and vision rays.

    Attributes:
        blocks_vision: Whether this vegetation blocks vision rays.
    """

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        radius: float = 20.0,
    ) -> None:
        """Initialize vegetation.

        Args:
            entity_id: Unique identifier.
            position: Position in world coordinates.
            radius: Collision radius (default 20.0).
        """
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.VEGETATION,
            radius=radius,
            is_static=True,
        )
        self._blocks_vision = True

    def update(self, world: World, dt: float) -> None:
        """Vegetation is static, no update needed."""
        pass

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)

    def blocks_vision(self) -> bool:
        """Returns True if this blocks vision rays."""
        return self._blocks_vision
