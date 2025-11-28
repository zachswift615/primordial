"""Environment system for the world.

Manages environmental properties like day/night cycle and ambient conditions.
"""

from __future__ import annotations

import math


class Environment:
    """Manages environmental properties like day/night cycle.

    Provides a sinusoidal day/night cycle that affects ambient brightness.
    This can be used to modulate vision range, predator behavior, etc.

    Attributes:
        day_length: Length of a full day/night cycle in seconds.
        time_of_day: Current time within the cycle (0.0 to day_length).
        min_brightness: Minimum brightness during night (default 0.1).
        max_brightness: Maximum brightness during day (default 0.5).
    """

    def __init__(
        self,
        day_length: float = 120.0,
        min_brightness: float = 0.1,
        max_brightness: float = 0.5,
    ) -> None:
        """Initialize environment.

        Args:
            day_length: Length of full day/night cycle in seconds (default 120.0).
            min_brightness: Minimum brightness at night (default 0.1).
            max_brightness: Maximum brightness at day (default 0.5).
        """
        self.day_length = day_length
        self.time_of_day = 0.0  # Start at dawn
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def update(self, dt: float) -> None:
        """Update environment state.

        Args:
            dt: Time step in seconds.
        """
        self.time_of_day = (self.time_of_day + dt) % self.day_length

    def get_brightness(self) -> float:
        """Get current brightness level.

        Uses a sinusoidal curve for smooth day/night transitions.
        Brightness is highest at midday (time = day_length/4) and
        lowest at midnight (time = 3*day_length/4).

        Returns:
            Brightness value between min_brightness and max_brightness.
        """
        # Sinusoidal cycle: starts at dawn (rising)
        cycle_position = (self.time_of_day / self.day_length) * 2 * math.pi

        # Use cosine so we start at dawn (halfway between min and max, rising)
        # Shifted so peak is at 1/4 cycle (midday) and trough at 3/4 (midnight)
        sine_value = math.sin(cycle_position)  # -1 to 1

        # Map from [-1, 1] to [min_brightness, max_brightness]
        brightness_range = self.max_brightness - self.min_brightness
        brightness = self.min_brightness + (sine_value + 1.0) / 2.0 * brightness_range

        return brightness

    def is_daytime(self) -> bool:
        """Check if it's currently daytime.

        Daytime is when brightness is above the midpoint.

        Returns:
            True if currently daytime.
        """
        midpoint = (self.min_brightness + self.max_brightness) / 2.0
        return self.get_brightness() > midpoint

    def is_nighttime(self) -> bool:
        """Check if it's currently nighttime.

        Returns:
            True if currently nighttime.
        """
        return not self.is_daytime()

    def get_time_of_day_normalized(self) -> float:
        """Get current time of day as a value from 0.0 to 1.0.

        Returns:
            Normalized time (0.0 = start of cycle, 1.0 = end of cycle).
        """
        return self.time_of_day / self.day_length

    def get_time_until_dawn(self) -> float:
        """Get time until next dawn.

        Dawn occurs when brightness starts rising (time = 0).

        Returns:
            Seconds until next dawn.
        """
        if self.time_of_day == 0.0:
            return 0.0
        return self.day_length - self.time_of_day

    def get_time_until_dusk(self) -> float:
        """Get time until next dusk.

        Dusk occurs when brightness starts falling (time = day_length/2).

        Returns:
            Seconds until next dusk.
        """
        dusk_time = self.day_length / 2.0
        if self.time_of_day < dusk_time:
            return dusk_time - self.time_of_day
        else:
            return self.day_length - self.time_of_day + dusk_time

    def set_time_of_day(self, time: float) -> None:
        """Set the current time of day.

        Args:
            time: Time in seconds (will be wrapped to day_length).
        """
        self.time_of_day = time % self.day_length

    def set_time_normalized(self, normalized_time: float) -> None:
        """Set time of day using normalized value.

        Args:
            normalized_time: Value from 0.0 to 1.0.
        """
        self.time_of_day = (normalized_time % 1.0) * self.day_length

    def skip_to_dawn(self) -> None:
        """Skip time to the next dawn."""
        self.time_of_day = 0.0

    def skip_to_noon(self) -> None:
        """Skip time to noon (brightest point)."""
        self.time_of_day = self.day_length / 4.0

    def skip_to_dusk(self) -> None:
        """Skip time to dusk."""
        self.time_of_day = self.day_length / 2.0

    def skip_to_midnight(self) -> None:
        """Skip time to midnight (darkest point)."""
        self.time_of_day = 3.0 * self.day_length / 4.0
