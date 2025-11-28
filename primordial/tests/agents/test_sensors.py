"""Tests for agent sensory systems."""

import pytest
import math
import numpy as np

from primordial.agents.genome import AgentGenome
from primordial.agents.sensors import (
    VisionSensor,
    AudioSensor,
    ProprioceptionSensor,
    TouchSensor,
)
from primordial.world import World
from primordial.world.geometry import Vec2
from primordial.world.entities import Food


class TestVisionSensor:
    """Tests for VisionSensor."""

    @pytest.fixture
    def sensor(self):
        """Create vision sensor with default genome."""
        return VisionSensor(AgentGenome())

    @pytest.fixture
    def world(self):
        """Create empty world for testing."""
        return World(width=500, height=500)

    def test_creation(self, sensor):
        """Test sensor creation from genome."""
        assert sensor.num_rays == 32
        assert sensor.max_range == 200.0
        assert sensor.fov == pytest.approx(math.radians(120.0))

    def test_output_shape(self, sensor, world):
        """Test that sense returns correct shape."""
        position = Vec2(250, 250)
        facing = Vec2(1, 0)

        result = sensor.sense(position, facing, world)

        assert result.shape == (32, 4)
        assert result.dtype == np.float32

    def test_empty_world_returns_zeros(self, sensor, world):
        """Test that empty world returns low distance values."""
        position = Vec2(250, 250)
        facing = Vec2(1, 0)

        result = sensor.sense(position, facing, world)

        # With no entities, distance column should reflect hitting nothing
        # Values should be 0-1 normalized
        assert np.all(result[:, 0] >= 0.0)
        assert np.all(result[:, 0] <= 1.0)

    def test_detects_food(self, sensor, world):
        """Test that sensor detects food entity."""
        # Place agent at center
        position = Vec2(250, 250)
        facing = Vec2(1, 0)  # Facing right

        # Place food directly ahead
        food = Food(
            entity_id=0,
            position=Vec2(350, 250),  # 100 units ahead
            energy_value=50.0,
        )
        world.add_entity(food)
        world._rebuild_spatial_grid()

        result = sensor.sense(position, facing, world)

        # The center ray should detect food
        center_ray_idx = 16  # Middle of 32 rays
        # Food should be at normalized distance ~0.5 (100/200 = 0.5 from max)
        # But we invert: 1 - 0.5 = 0.5 means "close"
        # Entity type 1 = food
        assert result[center_ray_idx, 1] == 1.0  # Food type code


class TestAudioSensor:
    """Tests for AudioSensor."""

    @pytest.fixture
    def sensor(self):
        """Create audio sensor with default genome."""
        return AudioSensor(AgentGenome())

    @pytest.fixture
    def world(self):
        """Create world for testing."""
        return World(width=500, height=500)

    def test_creation(self, sensor):
        """Test sensor creation from genome."""
        assert sensor.max_range == 300.0

    def test_output_shape(self, sensor, world):
        """Test that sense returns correct shape."""
        position = Vec2(250, 250)
        facing = Vec2(1, 0)

        result = sensor.sense(position, facing, world)

        assert result.shape == (2,)
        assert result.dtype == np.float32

    def test_output_range(self, sensor, world):
        """Test that output is in valid range."""
        position = Vec2(250, 250)
        facing = Vec2(1, 0)

        result = sensor.sense(position, facing, world)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestProprioceptionSensor:
    """Tests for ProprioceptionSensor."""

    @pytest.fixture
    def sensor(self):
        """Create proprioception sensor with default genome."""
        return ProprioceptionSensor(AgentGenome())

    def test_creation(self, sensor):
        """Test sensor creation from genome."""
        assert sensor.max_speed == 150.0
        assert sensor.max_energy == 100.0

    def test_output_shape(self, sensor):
        """Test that sense returns correct shape."""
        result = sensor.sense(
            energy=50.0,
            health=75.0,
            velocity=Vec2(10, 20),
            angular_velocity=0.5,
            angle=math.pi / 4,
        )

        assert result.shape == (8,)
        assert result.dtype == np.float32

    def test_energy_normalization(self, sensor):
        """Test that energy is normalized to 0-1."""
        result = sensor.sense(
            energy=50.0,  # Half of max 100
            health=100.0,
            velocity=Vec2(0, 0),
            angular_velocity=0.0,
            angle=0.0,
        )

        assert result[0] == pytest.approx(0.5)  # energy_norm

    def test_health_normalization(self, sensor):
        """Test that health is normalized to 0-1."""
        result = sensor.sense(
            energy=100.0,
            health=25.0,  # Quarter of max 100
            velocity=Vec2(0, 0),
            angular_velocity=0.0,
            angle=0.0,
        )

        assert result[1] == pytest.approx(0.25)  # health_norm

    def test_speed_normalization(self, sensor):
        """Test that speed is normalized to 0-1."""
        # Max speed is 150, so velocity with magnitude 75 should be 0.5
        result = sensor.sense(
            energy=100.0,
            health=100.0,
            velocity=Vec2(75, 0),  # magnitude = 75
            angular_velocity=0.0,
            angle=0.0,
        )

        assert result[2] == pytest.approx(0.5)  # speed_norm

    def test_angle_encoding(self, sensor):
        """Test that angle is encoded as sin/cos."""
        angle = math.pi / 4  # 45 degrees

        result = sensor.sense(
            energy=100.0,
            health=100.0,
            velocity=Vec2(0, 0),
            angular_velocity=0.0,
            angle=angle,
        )

        expected_sin = math.sin(angle)
        expected_cos = math.cos(angle)

        assert result[6] == pytest.approx(expected_sin)  # angle_sin
        assert result[7] == pytest.approx(expected_cos)  # angle_cos

    def test_velocity_components(self, sensor):
        """Test velocity component normalization."""
        # Max speed is 150
        result = sensor.sense(
            energy=100.0,
            health=100.0,
            velocity=Vec2(75, -30),
            angular_velocity=0.0,
            angle=0.0,
        )

        assert result[4] == pytest.approx(75 / 150)  # vel_x_norm
        assert result[5] == pytest.approx(-30 / 150)  # vel_y_norm


class TestTouchSensor:
    """Tests for TouchSensor."""

    @pytest.fixture
    def sensor(self):
        """Create touch sensor with default genome."""
        return TouchSensor(AgentGenome())

    @pytest.fixture
    def world(self):
        """Create world for testing."""
        return World(width=500, height=500)

    def test_creation(self, sensor):
        """Test sensor creation from genome."""
        assert sensor.num_sensors == 8
        assert sensor.touch_range == 15.0
        assert sensor.agent_radius == 8.0

    def test_output_shape(self, sensor, world):
        """Test that sense returns correct shape."""
        position = Vec2(250, 250)

        result = sensor.sense(position, world)

        assert result.shape == (8,)
        assert result.dtype == np.float32

    def test_output_range(self, sensor, world):
        """Test that output values are 0 or 1."""
        position = Vec2(250, 250)

        result = sensor.sense(position, world)

        # Each sensor should be 0 (nothing) or 1 (something)
        assert np.all((result == 0.0) | (result == 1.0))

    def test_detects_nearby_entity(self, sensor, world):
        """Test that sensor detects nearby food."""
        position = Vec2(250, 250)

        # Place food very close (within touch range)
        food = Food(
            entity_id=0,
            position=Vec2(270, 250),  # 20 units to the right
            energy_value=50.0,
        )
        world.add_entity(food)
        world._rebuild_spatial_grid()

        result = sensor.sense(position, world)

        # East direction (index 2) should detect the food
        # The food is at 20 units, agent radius is 8, food radius is ~5
        # Surface distance = 20 - 8 - 5 = 7, which is within touch_range of 15
        assert result[2] == 1.0  # East direction
