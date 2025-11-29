"""Sound system for the world.

Manages sound propagation and mixing for agent audio perception.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from primordial.world.geometry import Vec2
from primordial.world.sound.sound_source import SoundSource


class SoundSystem:
    """Manages sound propagation and mixing.

    Provides spatial audio for agents, including:
    - Distance attenuation (exponential decay)
    - Stereo positioning based on listener orientation
    - Frequency spectrum for neural network input

    Attributes:
        sources: List of active sound sources.
        attenuation_coefficient: Rate of sound decay over distance.
    """

    def __init__(self, attenuation_coefficient: float = 0.002) -> None:
        """Initialize sound system.

        Args:
            attenuation_coefficient: Rate of exponential decay over distance.
                Default 0.002 gives reasonable falloff over 100-500 units.
        """
        self.sources: List[SoundSource] = []
        self.attenuation_coefficient = attenuation_coefficient

    def clear_sources(self) -> None:
        """Clear all sound sources."""
        self.sources.clear()

    def add_source(self, source: SoundSource) -> None:
        """Add a sound source.

        Args:
            source: Sound source to add.
        """
        self.sources.append(source)

    def remove_source(self, source: SoundSource) -> None:
        """Remove a sound source.

        Args:
            source: Sound source to remove.
        """
        self.sources = [s for s in self.sources if s is not source]

    def compute_sound_at_position(
        self,
        listener_pos: Vec2,
        listener_facing: Vec2,
    ) -> Tuple[float, float]:
        """Compute stereo sound levels at listener position.

        Mixes all active sound sources with distance attenuation and
        stereo positioning based on listener facing direction.

        Args:
            listener_pos: Position of the listener in world coordinates.
            listener_facing: Unit vector of listener's facing direction.

        Returns:
            Tuple of (left_ear_intensity, right_ear_intensity), each in [0, 1].
        """
        left_total = 0.0
        right_total = 0.0

        for source in self.sources:
            if not source.is_active:
                continue

            # Calculate distance attenuation
            distance = listener_pos.distance_to(source.position)
            attenuation = np.exp(-self.attenuation_coefficient * distance)
            attenuated_intensity = source.intensity * attenuation

            if attenuated_intensity < 0.001:
                continue  # Too quiet to matter

            # Calculate stereo positioning
            to_source = source.position - listener_pos
            to_source_mag = to_source.magnitude()

            if to_source_mag < 0.001:
                # Source at listener position - equal to both ears
                left_total += attenuated_intensity * 0.5
                right_total += attenuated_intensity * 0.5
                continue

            to_source_normalized = to_source * (1.0 / to_source_mag)

            # Rear attenuation: sounds behind are quieter (head shadow effect)
            # dot product with facing: +1 = directly in front, -1 = directly behind
            forward_dot = to_source_normalized.dot(listener_facing)
            # Map from [-1, 1] to [0.3, 1.0] - sounds behind attenuated to 30%
            directional_attenuation = 0.3 + 0.7 * (forward_dot + 1.0) / 2.0

            # Right vector (perpendicular to facing, rotated 90 degrees clockwise)
            right = Vec2(listener_facing.y, -listener_facing.x)

            # Stereo pan: -1 (left) to +1 (right)
            pan = to_source_normalized.dot(right)

            # Convert pan to left/right intensities
            # pan = -1: all left, pan = 0: equal, pan = +1: all right
            left_gain = (1.0 - pan) / 2.0
            right_gain = (1.0 + pan) / 2.0

            # Apply directional attenuation to both ears
            left_total += attenuated_intensity * left_gain * directional_attenuation
            right_total += attenuated_intensity * right_gain * directional_attenuation

        # Clamp to [0, 1]
        left_total = min(1.0, max(0.0, left_total))
        right_total = min(1.0, max(0.0, right_total))

        return (left_total, right_total)

    def get_frequency_spectrum(
        self,
        listener_pos: Vec2,
        num_frequency_bins: int = 32,
        max_frequency: float = 2000.0,
    ) -> np.ndarray:
        """Get frequency spectrum at listener position.

        Useful for neural network input. Creates a histogram of sound
        intensities across frequency bins.

        Args:
            listener_pos: Position of the listener.
            num_frequency_bins: Number of frequency bins (default 32).
            max_frequency: Maximum frequency in Hz (default 2000.0).

        Returns:
            Array of shape (num_frequency_bins,) with intensities in [0, 1].
        """
        spectrum = np.zeros(num_frequency_bins, dtype=np.float32)
        bin_width = max_frequency / num_frequency_bins

        for source in self.sources:
            if not source.is_active:
                continue

            # Calculate distance attenuation
            distance = listener_pos.distance_to(source.position)
            attenuation = np.exp(-self.attenuation_coefficient * distance)
            attenuated_intensity = source.intensity * attenuation

            if attenuated_intensity < 0.001:
                continue

            # Find frequency bin
            bin_index = int(source.frequency / bin_width)
            if 0 <= bin_index < num_frequency_bins:
                spectrum[bin_index] += attenuated_intensity

        # Clamp to [0, 1]
        spectrum = np.clip(spectrum, 0.0, 1.0)

        return spectrum

    def get_loudest_direction(
        self,
        listener_pos: Vec2,
    ) -> Tuple[Vec2 | None, float]:
        """Get direction to the loudest sound source.

        Useful for simple audio-based navigation.

        Args:
            listener_pos: Position of the listener.

        Returns:
            Tuple of (direction_vector, intensity) or (None, 0.0) if no sounds.
        """
        loudest_intensity = 0.0
        loudest_direction: Vec2 | None = None

        for source in self.sources:
            if not source.is_active:
                continue

            distance = listener_pos.distance_to(source.position)
            attenuation = np.exp(-self.attenuation_coefficient * distance)
            attenuated_intensity = source.intensity * attenuation

            if attenuated_intensity > loudest_intensity:
                loudest_intensity = attenuated_intensity
                to_source = source.position - listener_pos
                if to_source.magnitude() > 0.001:
                    loudest_direction = to_source.normalized()
                else:
                    loudest_direction = Vec2(0.0, 0.0)

        return (loudest_direction, loudest_intensity)

    def get_total_intensity(self, listener_pos: Vec2) -> float:
        """Get total sound intensity at listener position.

        Args:
            listener_pos: Position of the listener.

        Returns:
            Total intensity (sum of all attenuated sources), clamped to [0, 1].
        """
        total = 0.0

        for source in self.sources:
            if not source.is_active:
                continue

            distance = listener_pos.distance_to(source.position)
            attenuation = np.exp(-self.attenuation_coefficient * distance)
            total += source.intensity * attenuation

        return min(1.0, total)

    def get_source_count(self) -> int:
        """Get number of sound sources.

        Returns:
            Number of sources (active and inactive).
        """
        return len(self.sources)

    def get_active_source_count(self) -> int:
        """Get number of active sound sources.

        Returns:
            Number of active sources.
        """
        return sum(1 for s in self.sources if s.is_active)
