"""Tests for SoundSource and SoundSystem."""

import math

import numpy as np
import pytest

from primordial.world.geometry import Vec2
from primordial.world.sound import SoundSource, SoundSystem


class TestSoundSource:
    """Tests for SoundSource."""

    def test_creation(self):
        """Test creating a sound source."""
        source = SoundSource(
            position=Vec2(100.0, 200.0),
            frequency=440.0,
            intensity=0.5,
        )
        assert source.position.x == 100.0
        assert source.position.y == 200.0
        assert source.frequency == 440.0
        assert source.intensity == 0.5
        assert source.is_active is True

    def test_creation_inactive(self):
        """Test creating an inactive sound source."""
        source = SoundSource(
            position=Vec2(0.0, 0.0),
            frequency=440.0,
            intensity=0.5,
            is_active=False,
        )
        assert source.is_active is False

    def test_copy(self):
        """Test copying a sound source."""
        source = SoundSource(
            position=Vec2(100.0, 200.0),
            frequency=440.0,
            intensity=0.5,
        )
        copy = source.copy()

        assert copy.position.x == source.position.x
        assert copy.position.y == source.position.y
        assert copy.frequency == source.frequency
        assert copy.intensity == source.intensity
        assert copy is not source
        assert copy.position is not source.position


class TestSoundSystemCreation:
    """Tests for SoundSystem initialization."""

    def test_creation(self):
        """Test creating sound system."""
        system = SoundSystem()
        assert system.attenuation_coefficient == 0.002
        assert len(system.sources) == 0

    def test_creation_custom_attenuation(self):
        """Test creating sound system with custom attenuation."""
        system = SoundSystem(attenuation_coefficient=0.01)
        assert system.attenuation_coefficient == 0.01


class TestSoundSystemSources:
    """Tests for managing sound sources."""

    def test_add_source(self):
        """Test adding a sound source."""
        system = SoundSystem()
        source = SoundSource(Vec2(0.0, 0.0), 440.0, 0.5)

        system.add_source(source)

        assert system.get_source_count() == 1

    def test_add_multiple_sources(self):
        """Test adding multiple sources."""
        system = SoundSystem()

        for i in range(5):
            source = SoundSource(Vec2(i * 10.0, 0.0), 440.0, 0.5)
            system.add_source(source)

        assert system.get_source_count() == 5

    def test_remove_source(self):
        """Test removing a sound source."""
        system = SoundSystem()
        source = SoundSource(Vec2(0.0, 0.0), 440.0, 0.5)

        system.add_source(source)
        system.remove_source(source)

        assert system.get_source_count() == 0

    def test_clear_sources(self):
        """Test clearing all sources."""
        system = SoundSystem()

        for i in range(5):
            system.add_source(SoundSource(Vec2(i * 10.0, 0.0), 440.0, 0.5))

        system.clear_sources()

        assert system.get_source_count() == 0

    def test_active_source_count(self):
        """Test counting active sources."""
        system = SoundSystem()

        system.add_source(SoundSource(Vec2(0.0, 0.0), 440.0, 0.5, is_active=True))
        system.add_source(SoundSource(Vec2(10.0, 0.0), 440.0, 0.5, is_active=False))
        system.add_source(SoundSource(Vec2(20.0, 0.0), 440.0, 0.5, is_active=True))

        assert system.get_source_count() == 3
        assert system.get_active_source_count() == 2


class TestSoundSystemAttenuation:
    """Tests for distance attenuation."""

    def test_no_attenuation_at_source(self):
        """Test intensity at source position."""
        system = SoundSystem(attenuation_coefficient=0.002)
        source = SoundSource(Vec2(100.0, 100.0), 440.0, 1.0)
        system.add_source(source)

        left, right = system.compute_sound_at_position(
            Vec2(100.0, 100.0), Vec2(1.0, 0.0)
        )

        # At source, should be full intensity (split between ears)
        assert abs(left + right - 1.0) < 0.01

    def test_attenuation_with_distance(self):
        """Test that sound decreases with distance."""
        system = SoundSystem(attenuation_coefficient=0.002)
        source = SoundSource(Vec2(0.0, 0.0), 440.0, 1.0)
        system.add_source(source)

        # Close to source
        close_left, close_right = system.compute_sound_at_position(
            Vec2(10.0, 0.0), Vec2(1.0, 0.0)
        )
        close_total = close_left + close_right

        # Far from source
        far_left, far_right = system.compute_sound_at_position(
            Vec2(500.0, 0.0), Vec2(1.0, 0.0)
        )
        far_total = far_left + far_right

        assert far_total < close_total

    def test_very_far_source_silent(self):
        """Test that very distant sources are effectively silent."""
        system = SoundSystem(attenuation_coefficient=0.01)  # Faster falloff
        source = SoundSource(Vec2(0.0, 0.0), 440.0, 1.0)
        system.add_source(source)

        left, right = system.compute_sound_at_position(
            Vec2(1000.0, 0.0), Vec2(1.0, 0.0)
        )

        # Should be essentially silent
        assert left + right < 0.01


class TestSoundSystemStereo:
    """Tests for stereo positioning."""

    def test_sound_to_right(self):
        """Test sound source to the right."""
        system = SoundSystem(attenuation_coefficient=0.0)  # No attenuation
        source = SoundSource(Vec2(100.0, 0.0), 440.0, 1.0)
        system.add_source(source)

        # Listener at origin, facing up (+Y)
        left, right = system.compute_sound_at_position(
            Vec2(0.0, 0.0), Vec2(0.0, 1.0)
        )

        # Sound to the right should be louder in right ear
        assert right > left

    def test_sound_to_left(self):
        """Test sound source to the left."""
        system = SoundSystem(attenuation_coefficient=0.0)
        source = SoundSource(Vec2(-100.0, 0.0), 440.0, 1.0)
        system.add_source(source)

        # Listener at origin, facing up (+Y)
        left, right = system.compute_sound_at_position(
            Vec2(0.0, 0.0), Vec2(0.0, 1.0)
        )

        # Sound to the left should be louder in left ear
        assert left > right

    def test_sound_in_front(self):
        """Test sound source directly in front."""
        system = SoundSystem(attenuation_coefficient=0.0)
        source = SoundSource(Vec2(0.0, 100.0), 440.0, 1.0)
        system.add_source(source)

        # Listener at origin, facing up (+Y)
        left, right = system.compute_sound_at_position(
            Vec2(0.0, 0.0), Vec2(0.0, 1.0)
        )

        # Sound in front should be equal in both ears
        assert abs(left - right) < 0.01

    def test_sound_behind(self):
        """Test sound source directly behind."""
        system = SoundSystem(attenuation_coefficient=0.0)
        source = SoundSource(Vec2(0.0, -100.0), 440.0, 1.0)
        system.add_source(source)

        # Listener at origin, facing up (+Y)
        left, right = system.compute_sound_at_position(
            Vec2(0.0, 0.0), Vec2(0.0, 1.0)
        )

        # Sound behind should be equal in both ears
        assert abs(left - right) < 0.01


class TestSoundSystemFrequencySpectrum:
    """Tests for frequency spectrum."""

    def test_spectrum_shape(self):
        """Test spectrum output shape."""
        system = SoundSystem()
        source = SoundSource(Vec2(0.0, 0.0), 440.0, 1.0)
        system.add_source(source)

        spectrum = system.get_frequency_spectrum(Vec2(0.0, 0.0), num_frequency_bins=32)

        assert spectrum.shape == (32,)
        assert spectrum.dtype == np.float32

    def test_spectrum_frequency_binning(self):
        """Test that frequencies are binned correctly."""
        system = SoundSystem(attenuation_coefficient=0.0)

        # Add source at 500 Hz
        source = SoundSource(Vec2(0.0, 0.0), 500.0, 1.0)
        system.add_source(source)

        spectrum = system.get_frequency_spectrum(
            Vec2(0.0, 0.0), num_frequency_bins=20, max_frequency=2000.0
        )

        # 500 Hz should be in bin 5 (500 / (2000/20) = 5)
        assert spectrum[5] > 0
        # Other bins should be 0
        for i in range(20):
            if i != 5:
                assert spectrum[i] == 0

    def test_spectrum_multiple_frequencies(self):
        """Test spectrum with multiple frequency sources."""
        system = SoundSystem(attenuation_coefficient=0.0)

        system.add_source(SoundSource(Vec2(0.0, 0.0), 200.0, 0.5))
        system.add_source(SoundSource(Vec2(0.0, 0.0), 800.0, 0.3))

        spectrum = system.get_frequency_spectrum(
            Vec2(0.0, 0.0), num_frequency_bins=20, max_frequency=2000.0
        )

        # 200 Hz -> bin 2, 800 Hz -> bin 8
        assert spectrum[2] > 0
        assert spectrum[8] > 0

    def test_spectrum_clamps_to_one(self):
        """Test spectrum values are clamped to [0, 1]."""
        system = SoundSystem(attenuation_coefficient=0.0)

        # Add multiple loud sources at same frequency
        for _ in range(10):
            system.add_source(SoundSource(Vec2(0.0, 0.0), 500.0, 1.0))

        spectrum = system.get_frequency_spectrum(Vec2(0.0, 0.0))

        assert np.all(spectrum <= 1.0)
        assert np.all(spectrum >= 0.0)

    def test_spectrum_inactive_sources_excluded(self):
        """Test that inactive sources don't contribute to spectrum."""
        system = SoundSystem(attenuation_coefficient=0.0)

        source = SoundSource(Vec2(0.0, 0.0), 500.0, 1.0, is_active=False)
        system.add_source(source)

        spectrum = system.get_frequency_spectrum(Vec2(0.0, 0.0))

        assert np.all(spectrum == 0)


class TestSoundSystemLoudestDirection:
    """Tests for loudest direction."""

    def test_loudest_direction(self):
        """Test finding direction to loudest sound."""
        system = SoundSystem(attenuation_coefficient=0.0)

        # Quiet source to left
        system.add_source(SoundSource(Vec2(-100.0, 0.0), 440.0, 0.3))
        # Loud source to right
        system.add_source(SoundSource(Vec2(100.0, 0.0), 440.0, 1.0))

        direction, intensity = system.get_loudest_direction(Vec2(0.0, 0.0))

        assert direction is not None
        assert direction.x > 0  # Should point right
        assert abs(direction.y) < 0.01

    def test_loudest_direction_no_sources(self):
        """Test loudest direction with no sources."""
        system = SoundSystem()

        direction, intensity = system.get_loudest_direction(Vec2(0.0, 0.0))

        assert direction is None
        assert intensity == 0.0

    def test_loudest_direction_considers_distance(self):
        """Test that distance affects loudest determination."""
        system = SoundSystem(attenuation_coefficient=0.01)

        # Loud but far source
        system.add_source(SoundSource(Vec2(500.0, 0.0), 440.0, 1.0))
        # Quieter but close source
        system.add_source(SoundSource(Vec2(10.0, 0.0), 440.0, 0.5))

        direction, intensity = system.get_loudest_direction(Vec2(0.0, 0.0))

        # Close source should be loudest due to attenuation
        assert direction is not None
        assert direction.x > 0


class TestSoundSystemTotalIntensity:
    """Tests for total intensity."""

    def test_total_intensity_single_source(self):
        """Test total intensity with single source."""
        system = SoundSystem(attenuation_coefficient=0.0)
        system.add_source(SoundSource(Vec2(0.0, 0.0), 440.0, 0.7))

        intensity = system.get_total_intensity(Vec2(0.0, 0.0))

        assert abs(intensity - 0.7) < 0.01

    def test_total_intensity_multiple_sources(self):
        """Test total intensity with multiple sources."""
        system = SoundSystem(attenuation_coefficient=0.0)
        system.add_source(SoundSource(Vec2(0.0, 0.0), 440.0, 0.3))
        system.add_source(SoundSource(Vec2(0.0, 0.0), 440.0, 0.4))

        intensity = system.get_total_intensity(Vec2(0.0, 0.0))

        assert abs(intensity - 0.7) < 0.01

    def test_total_intensity_clamped(self):
        """Test total intensity is clamped to 1.0."""
        system = SoundSystem(attenuation_coefficient=0.0)

        for _ in range(10):
            system.add_source(SoundSource(Vec2(0.0, 0.0), 440.0, 1.0))

        intensity = system.get_total_intensity(Vec2(0.0, 0.0))

        assert intensity == 1.0


class TestSoundSystemMixing:
    """Tests for mixing multiple sources."""

    def test_multiple_sources_mixed(self):
        """Test that multiple sources are mixed together."""
        system = SoundSystem(attenuation_coefficient=0.0)

        # Source to the left
        system.add_source(SoundSource(Vec2(-50.0, 0.0), 440.0, 0.5))
        # Source to the right
        system.add_source(SoundSource(Vec2(50.0, 0.0), 440.0, 0.5))

        # Listener facing up
        left, right = system.compute_sound_at_position(
            Vec2(0.0, 0.0), Vec2(0.0, 1.0)
        )

        # Both ears should hear something
        assert left > 0
        assert right > 0

    def test_inactive_sources_not_mixed(self):
        """Test that inactive sources are not mixed."""
        system = SoundSystem(attenuation_coefficient=0.0)

        system.add_source(SoundSource(Vec2(0.0, 0.0), 440.0, 1.0, is_active=False))

        left, right = system.compute_sound_at_position(
            Vec2(0.0, 0.0), Vec2(1.0, 0.0)
        )

        assert left == 0.0
        assert right == 0.0
