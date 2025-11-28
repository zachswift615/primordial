"""Tests for Environment (day/night cycle)."""

import math

import pytest

from primordial.world.environment import Environment


class TestEnvironmentCreation:
    """Tests for Environment initialization."""

    def test_creation_defaults(self):
        """Test creating environment with defaults."""
        env = Environment()
        assert env.day_length == 120.0
        assert env.time_of_day == 0.0
        assert env.min_brightness == 0.1
        assert env.max_brightness == 0.5

    def test_creation_custom(self):
        """Test creating environment with custom values."""
        env = Environment(
            day_length=60.0, min_brightness=0.2, max_brightness=0.8
        )
        assert env.day_length == 60.0
        assert env.min_brightness == 0.2
        assert env.max_brightness == 0.8


class TestEnvironmentUpdate:
    """Tests for environment time updates."""

    def test_update_increments_time(self):
        """Test that update increments time."""
        env = Environment(day_length=100.0)

        env.update(10.0)

        assert env.time_of_day == 10.0

    def test_update_wraps_at_day_length(self):
        """Test that time wraps at day_length."""
        env = Environment(day_length=100.0)

        env.update(150.0)

        assert abs(env.time_of_day - 50.0) < 0.001

    def test_update_multiple_times(self):
        """Test multiple updates."""
        env = Environment(day_length=100.0)

        env.update(30.0)
        env.update(30.0)
        env.update(30.0)

        assert abs(env.time_of_day - 90.0) < 0.001


class TestEnvironmentBrightness:
    """Tests for brightness calculation."""

    def test_brightness_at_dawn(self):
        """Test brightness at dawn (time = 0)."""
        env = Environment(day_length=100.0, min_brightness=0.1, max_brightness=0.5)
        env.time_of_day = 0.0

        brightness = env.get_brightness()

        # At dawn (sin(0) = 0), brightness should be midpoint
        midpoint = (0.1 + 0.5) / 2.0
        assert abs(brightness - midpoint) < 0.01

    def test_brightness_at_noon(self):
        """Test brightness at noon (time = day_length/4)."""
        env = Environment(day_length=100.0, min_brightness=0.1, max_brightness=0.5)
        env.time_of_day = 25.0  # 1/4 of day

        brightness = env.get_brightness()

        # At noon (sin(pi/2) = 1), brightness should be max
        assert abs(brightness - 0.5) < 0.01

    def test_brightness_at_dusk(self):
        """Test brightness at dusk (time = day_length/2)."""
        env = Environment(day_length=100.0, min_brightness=0.1, max_brightness=0.5)
        env.time_of_day = 50.0  # 1/2 of day

        brightness = env.get_brightness()

        # At dusk (sin(pi) = 0), brightness should be midpoint
        midpoint = (0.1 + 0.5) / 2.0
        assert abs(brightness - midpoint) < 0.01

    def test_brightness_at_midnight(self):
        """Test brightness at midnight (time = 3*day_length/4)."""
        env = Environment(day_length=100.0, min_brightness=0.1, max_brightness=0.5)
        env.time_of_day = 75.0  # 3/4 of day

        brightness = env.get_brightness()

        # At midnight (sin(3*pi/2) = -1), brightness should be min
        assert abs(brightness - 0.1) < 0.01

    def test_brightness_within_range(self):
        """Test brightness stays within min/max range."""
        env = Environment(day_length=100.0, min_brightness=0.2, max_brightness=0.8)

        for t in range(101):
            env.time_of_day = float(t)
            brightness = env.get_brightness()
            assert 0.2 <= brightness <= 0.8


class TestEnvironmentDayNight:
    """Tests for day/night detection."""

    def test_is_daytime_at_noon(self):
        """Test daytime detection at noon."""
        env = Environment(day_length=100.0)
        env.skip_to_noon()

        assert env.is_daytime() is True
        assert env.is_nighttime() is False

    def test_is_nighttime_at_midnight(self):
        """Test nighttime detection at midnight."""
        env = Environment(day_length=100.0)
        env.skip_to_midnight()

        assert env.is_nighttime() is True
        assert env.is_daytime() is False

    def test_dawn_is_daytime(self):
        """Test that dawn is considered daytime (rising brightness)."""
        env = Environment(day_length=100.0)
        env.time_of_day = 5.0  # Just after dawn

        # Slightly after dawn should be daytime
        assert env.is_daytime() is True

    def test_dusk_is_nighttime(self):
        """Test that after dusk is nighttime."""
        env = Environment(day_length=100.0)
        env.time_of_day = 55.0  # Just after dusk

        # Slightly after dusk should be nighttime
        assert env.is_nighttime() is True


class TestEnvironmentTimeHelpers:
    """Tests for time helper methods."""

    def test_get_time_normalized(self):
        """Test normalized time calculation."""
        env = Environment(day_length=100.0)
        env.time_of_day = 50.0

        assert env.get_time_of_day_normalized() == 0.5

    def test_get_time_until_dawn_at_start(self):
        """Test time until dawn when at dawn."""
        env = Environment(day_length=100.0)
        env.time_of_day = 0.0

        assert env.get_time_until_dawn() == 0.0

    def test_get_time_until_dawn_mid_cycle(self):
        """Test time until dawn from middle of cycle."""
        env = Environment(day_length=100.0)
        env.time_of_day = 75.0

        assert env.get_time_until_dawn() == 25.0

    def test_get_time_until_dusk_before_dusk(self):
        """Test time until dusk before dusk."""
        env = Environment(day_length=100.0)
        env.time_of_day = 25.0  # Noon

        assert env.get_time_until_dusk() == 25.0

    def test_get_time_until_dusk_after_dusk(self):
        """Test time until dusk after dusk."""
        env = Environment(day_length=100.0)
        env.time_of_day = 75.0  # Midnight

        # Should wrap around to next dusk
        assert env.get_time_until_dusk() == 75.0


class TestEnvironmentSetTime:
    """Tests for setting time."""

    def test_set_time_of_day(self):
        """Test setting time directly."""
        env = Environment(day_length=100.0)
        env.set_time_of_day(45.0)

        assert env.time_of_day == 45.0

    def test_set_time_wraps(self):
        """Test setting time wraps at day_length."""
        env = Environment(day_length=100.0)
        env.set_time_of_day(150.0)

        assert env.time_of_day == 50.0

    def test_set_time_normalized(self):
        """Test setting time with normalized value."""
        env = Environment(day_length=100.0)
        env.set_time_normalized(0.5)

        assert env.time_of_day == 50.0

    def test_set_time_normalized_wraps(self):
        """Test normalized time wraps."""
        env = Environment(day_length=100.0)
        env.set_time_normalized(1.5)

        assert env.time_of_day == 50.0


class TestEnvironmentSkip:
    """Tests for skip methods."""

    def test_skip_to_dawn(self):
        """Test skipping to dawn."""
        env = Environment(day_length=100.0)
        env.time_of_day = 50.0
        env.skip_to_dawn()

        assert env.time_of_day == 0.0

    def test_skip_to_noon(self):
        """Test skipping to noon."""
        env = Environment(day_length=100.0)
        env.skip_to_noon()

        assert env.time_of_day == 25.0

    def test_skip_to_dusk(self):
        """Test skipping to dusk."""
        env = Environment(day_length=100.0)
        env.skip_to_dusk()

        assert env.time_of_day == 50.0

    def test_skip_to_midnight(self):
        """Test skipping to midnight."""
        env = Environment(day_length=100.0)
        env.skip_to_midnight()

        assert env.time_of_day == 75.0


class TestEnvironmentCycle:
    """Tests for full day/night cycle."""

    def test_full_cycle_smooth(self):
        """Test that a full cycle has smooth brightness transitions."""
        env = Environment(day_length=100.0, min_brightness=0.1, max_brightness=0.5)

        brightnesses = []
        for i in range(100):
            env.time_of_day = float(i)
            brightnesses.append(env.get_brightness())

        # Check for smooth transitions (no sudden jumps)
        for i in range(1, 100):
            diff = abs(brightnesses[i] - brightnesses[i - 1])
            assert diff < 0.05  # Max change per 1% of cycle

    def test_cycle_returns_to_start(self):
        """Test that brightness returns to starting value after full cycle."""
        env = Environment(day_length=100.0)

        start_brightness = env.get_brightness()
        env.update(100.0)  # Full cycle
        end_brightness = env.get_brightness()

        assert abs(start_brightness - end_brightness) < 0.001
