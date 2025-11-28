"""Food entity for the world system.

Food items that spawn randomly and give energy when consumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import Circle, Vec2

if TYPE_CHECKING:
    from primordial.world.world import World


class Food(Entity):
    """Food entity that spawns randomly and gives energy.

    Food is a static entity that agents can consume to gain energy.
    Food emits a subtle ambient sound that agents can use to locate it.

    Attributes:
        energy_value: Energy gained when consumed.
        sound_intensity: Volume of food sound (0.0-1.0).
        sound_frequency: Frequency of food sound in Hz.
    """

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        energy_value: float = 50.0,
        sound_intensity: float = 0.1,
        sound_frequency: float = 200.0,
        radius: float = 5.0,
    ) -> None:
        """Initialize food item.

        Args:
            entity_id: Unique identifier.
            position: Position in world coordinates.
            energy_value: Energy given when consumed (default 50.0).
            sound_intensity: Volume of ambient sound (default 0.1).
            sound_frequency: Frequency of sound in Hz (default 200.0).
            radius: Collision radius (default 5.0).
        """
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.FOOD,
            radius=radius,
            is_static=True,
        )
        self.energy_value = energy_value
        self.sound_intensity = sound_intensity
        self.sound_frequency = sound_frequency

    def update(self, world: World, dt: float) -> None:
        """Food is static, no update needed."""
        pass

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)

    def consume(self) -> float:
        """Called when eaten by an agent.

        Deactivates the food and returns its energy value.

        Returns:
            Energy value of this food item.
        """
        self.is_active = False
        return self.energy_value

    def get_sound_properties(self) -> tuple[float, float]:
        """Return sound properties for the sound system.

        Returns:
            Tuple of (intensity, frequency).
        """
        return (self.sound_intensity, self.sound_frequency)
