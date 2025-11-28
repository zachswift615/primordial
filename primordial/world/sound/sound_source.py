"""Sound source for the world system."""

from __future__ import annotations

from dataclasses import dataclass

from primordial.world.geometry import Vec2


@dataclass
class SoundSource:
    """A sound source in the world.

    Represents a point source of sound with position, frequency, and intensity.

    Attributes:
        position: Position of the sound source in world coordinates.
        frequency: Frequency of the sound in Hz.
        intensity: Volume/intensity of the sound (0.0 to 1.0).
        is_active: Whether the sound source is currently emitting.
    """

    position: Vec2
    frequency: float
    intensity: float
    is_active: bool = True

    def copy(self) -> SoundSource:
        """Create a copy of this sound source.

        Returns:
            New SoundSource with same properties.
        """
        return SoundSource(
            position=self.position.copy(),
            frequency=self.frequency,
            intensity=self.intensity,
            is_active=self.is_active,
        )
