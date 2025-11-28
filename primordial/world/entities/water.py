"""Water entity for the world system.

Water barriers with ambient sound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import Circle, Vec2

if TYPE_CHECKING:
    from primordial.world.world import World


class Water(Entity):
    """Water barrier with ambient sound.

    Water bodies are static obstacles that also emit sound,
    allowing agents to hear them from a distance.

    Attributes:
        sound_intensity: Volume of water sound (0.0-1.0).
        sound_frequency: Frequency of water sound in Hz.
    """

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        radius: float = 30.0,
        sound_intensity: float = 0.3,
        sound_frequency: float = 300.0,
    ) -> None:
        """Initialize water body.

        Args:
            entity_id: Unique identifier.
            position: Position in world coordinates.
            radius: Collision radius (default 30.0).
            sound_intensity: Volume of ambient sound (default 0.3).
            sound_frequency: Frequency of sound in Hz (default 300.0).
        """
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.WATER,
            radius=radius,
            is_static=True,
        )
        self.sound_intensity = sound_intensity
        self.sound_frequency = sound_frequency

    def update(self, world: World, dt: float) -> None:
        """Water is static, no update needed."""
        pass

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)

    def get_sound_properties(self) -> tuple[float, float]:
        """Return sound properties for the sound system.

        Returns:
            Tuple of (intensity, frequency).
        """
        return (self.sound_intensity, self.sound_frequency)
