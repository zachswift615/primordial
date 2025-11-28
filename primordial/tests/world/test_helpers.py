"""Tests for world integration helpers.

Tests cover:
- Vision raycasting with correct distances and entity detection
- Vision FOV behavior
- Sound input structure and values
- Render data completeness
- Touch sensor direction detection
"""

import math

import numpy as np
import pytest

from primordial.world import EntityType, Food, Predator, Vegetation, Water, World
from primordial.world.geometry import Vec2
from primordial.world.helpers import (
    get_render_data,
    get_sound_input,
    get_touch_input,
    get_vision_input,
)


class TestVisionHelper:
    """Tests for get_vision_input function."""

    def test_vision_returns_correct_shape(self):
        """Vision input should return array of shape (num_rays, 4)."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        result = get_vision_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
            num_rays=16,
        )

        assert result.shape == (16, 4)
        assert result.dtype == np.float32

    def test_vision_detects_food_at_correct_distance(self):
        """Vision should detect food entity at correct distance."""
        world = World(width=500.0, height=500.0)

        # Place food directly in front of agent
        agent_pos = Vec2(100.0, 100.0)
        agent_facing = Vec2(1.0, 0.0)

        food = Food(
            entity_id=0,
            position=Vec2(200.0, 100.0),  # 100 units away
            energy_value=50.0,
        )
        world.add_entity(food)

        # Cast single ray directly at food
        result = get_vision_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
            vision_range=200.0,
            num_rays=1,
        )

        # Check distance (should be ~100 / 200 = 0.5)
        # Account for food radius (5.0)
        expected_distance = (100.0 - 5.0) / 200.0
        assert abs(result[0, 0] - expected_distance) < 0.05

        # Check entity type (1 = food)
        assert result[0, 1] == 1.0

    def test_vision_detects_different_entity_types(self):
        """Vision should correctly encode different entity types."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)

        # Add different entity types at various positions
        food = Food(entity_id=0, position=Vec2(350.0, 250.0))
        predator = Predator(
            entity_id=1,
            position=Vec2(250.0, 350.0),
            patrol_center=Vec2(250.0, 350.0),
        )
        vegetation = Vegetation(entity_id=2, position=Vec2(150.0, 250.0))
        water = Water(entity_id=3, position=Vec2(250.0, 150.0))

        world.add_entity(food)
        world.add_entity(predator)
        world.add_entity(vegetation)
        world.add_entity(water)

        # Cast rays in 4 cardinal directions
        for direction, expected_type in [
            (Vec2(1.0, 0.0), 1.0),  # East: food
            (Vec2(0.0, 1.0), 2.0),  # North: predator
            (Vec2(-1.0, 0.0), 3.0),  # West: vegetation
            (Vec2(0.0, -1.0), 4.0),  # South: water
        ]:
            result = get_vision_input(
                world=world,
                agent_position=agent_pos,
                agent_facing=direction,
                num_rays=1,
            )

            assert result[0, 1] == expected_type, f"Wrong type for direction {direction}"

    def test_vision_respects_fov(self):
        """Vision rays should be spread across field of view."""
        world = World(width=500.0, height=500.0)

        # Place entities at different angles
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)  # Facing east

        # Food at +60 degrees (within 120 degree FOV)
        angle_60 = math.radians(60)
        food_pos = Vec2(
            250.0 + 100.0 * math.cos(angle_60),
            250.0 + 100.0 * math.sin(angle_60),
        )
        food = Food(entity_id=0, position=food_pos)
        world.add_entity(food)

        # Cast rays with 120 degree FOV
        result = get_vision_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
            vision_range=200.0,
            vision_fov=math.radians(120),
            num_rays=32,
        )

        # At least one ray should detect the food
        food_detected = np.any(result[:, 1] == 1.0)
        assert food_detected, "Food within FOV should be detected"

    def test_vision_nothing_hit_returns_max_distance(self):
        """Vision should return max distance and type 0 when nothing is hit."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        # Empty world
        result = get_vision_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
            vision_range=200.0,
            num_rays=8,
        )

        # All rays should return max distance (1.0) and type 0
        assert np.all(result[:, 0] == 1.0), "Empty rays should return max distance"
        assert np.all(result[:, 1] == 0.0), "Empty rays should return type 0"

    def test_vision_placeholder_channels_are_zero(self):
        """Vision placeholder color channels should be zero."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        result = get_vision_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
            num_rays=8,
        )

        # Channels 2 and 3 (entity_r, entity_g) should be zero
        assert np.all(result[:, 2] == 0.0)
        assert np.all(result[:, 3] == 0.0)


class TestSoundHelper:
    """Tests for get_sound_input function."""

    def test_sound_input_returns_correct_structure(self):
        """Sound input should return dict with correct keys."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        result = get_sound_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
        )

        assert isinstance(result, dict)
        assert "left_ear" in result
        assert "right_ear" in result
        assert "spectrum" in result

        assert isinstance(result["left_ear"], (float, np.floating))
        assert isinstance(result["right_ear"], (float, np.floating))
        assert isinstance(result["spectrum"], np.ndarray)

    def test_sound_input_spectrum_shape(self):
        """Sound spectrum should have correct shape."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        result = get_sound_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
            num_frequency_bins=64,
        )

        assert result["spectrum"].shape == (64,)

    def test_sound_input_with_food_source(self):
        """Sound input should detect food sound source."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        # Add food (which has sound)
        food = Food(
            entity_id=0,
            position=Vec2(300.0, 250.0),  # 50 units to the right
            sound_intensity=0.8,
        )
        world.add_entity(food)
        world._update_sound_sources()

        result = get_sound_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
        )

        # Sound should be detected (right ear should be louder since food is to the right)
        assert result["right_ear"] > 0.0
        assert result["right_ear"] >= result["left_ear"]

    def test_sound_input_values_in_valid_range(self):
        """Sound input values should be in [0, 1] range."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        # Add multiple sound sources
        for i in range(5):
            food = Food(
                entity_id=i,
                position=Vec2(
                    250.0 + (i - 2) * 50.0,
                    250.0 + (i - 2) * 30.0,
                ),
                sound_intensity=0.5,
            )
            world.add_entity(food)
        world._update_sound_sources()

        result = get_sound_input(
            world=world,
            agent_position=agent_pos,
            agent_facing=agent_facing,
        )

        assert 0.0 <= result["left_ear"] <= 1.0
        assert 0.0 <= result["right_ear"] <= 1.0
        assert np.all((result["spectrum"] >= 0.0) & (result["spectrum"] <= 1.0))


class TestRenderDataHelper:
    """Tests for get_render_data function."""

    def test_render_data_has_all_required_fields(self):
        """Render data should contain all required fields."""
        world = World(width=500.0, height=500.0)

        result = get_render_data(world)

        assert "entities" in result
        assert "brightness" in result
        assert "world_bounds" in result
        assert "sound_sources" in result
        assert "time_of_day" in result

    def test_render_data_world_bounds_correct(self):
        """World bounds should match world dimensions."""
        world = World(width=800.0, height=600.0)

        result = get_render_data(world)

        bounds = result["world_bounds"]
        assert bounds["min_x"] == 0.0
        assert bounds["min_y"] == 0.0
        assert bounds["max_x"] == 800.0
        assert bounds["max_y"] == 600.0

    def test_render_data_includes_all_entities(self):
        """Render data should include all active entities."""
        world = World(width=500.0, height=500.0)

        # Add various entities
        food = Food(entity_id=0, position=Vec2(100.0, 100.0))
        predator = Predator(
            entity_id=1,
            position=Vec2(200.0, 200.0),
            patrol_center=Vec2(200.0, 200.0),
        )
        vegetation = Vegetation(entity_id=2, position=Vec2(300.0, 300.0))

        world.add_entity(food)
        world.add_entity(predator)
        world.add_entity(vegetation)

        result = get_render_data(world)

        assert len(result["entities"]) == 3

        # Check entity data structure
        for entity_data in result["entities"]:
            assert "id" in entity_data
            assert "type" in entity_data
            assert "position" in entity_data
            assert "radius" in entity_data
            assert "velocity" in entity_data
            assert "is_static" in entity_data

            # Position should be tuple
            assert isinstance(entity_data["position"], tuple)
            assert len(entity_data["position"]) == 2

    def test_render_data_static_entities_have_no_velocity(self):
        """Static entities should have None velocity in render data."""
        world = World(width=500.0, height=500.0)

        food = Food(entity_id=0, position=Vec2(100.0, 100.0))
        vegetation = Vegetation(entity_id=1, position=Vec2(200.0, 200.0))

        world.add_entity(food)
        world.add_entity(vegetation)

        result = get_render_data(world)

        for entity_data in result["entities"]:
            if entity_data["is_static"]:
                assert entity_data["velocity"] is None

    def test_render_data_dynamic_entities_have_velocity(self):
        """Dynamic entities should have velocity tuple in render data."""
        world = World(width=500.0, height=500.0)

        predator = Predator(
            entity_id=0,
            position=Vec2(200.0, 200.0),
            patrol_center=Vec2(200.0, 200.0),
        )
        world.add_entity(predator)

        result = get_render_data(world)

        predator_data = result["entities"][0]
        assert predator_data["velocity"] is not None
        assert isinstance(predator_data["velocity"], tuple)
        assert len(predator_data["velocity"]) == 2

    def test_render_data_includes_sound_sources(self):
        """Render data should include active sound sources."""
        world = World(width=500.0, height=500.0)

        food = Food(entity_id=0, position=Vec2(100.0, 100.0), sound_intensity=0.5)
        world.add_entity(food)
        world._update_sound_sources()

        result = get_render_data(world)

        assert len(result["sound_sources"]) > 0

        # Check sound source structure
        for source_data in result["sound_sources"]:
            assert "position" in source_data
            assert "intensity" in source_data
            assert "frequency" in source_data
            assert isinstance(source_data["position"], tuple)

    def test_render_data_brightness_in_valid_range(self):
        """Brightness should be in valid range (0.1 to 0.5)."""
        world = World(width=500.0, height=500.0)

        result = get_render_data(world)

        brightness = result["brightness"]
        assert 0.1 <= brightness <= 0.5

    def test_render_data_time_of_day_normalized(self):
        """Time of day should be normalized to [0, 1] range."""
        world = World(width=500.0, height=500.0)

        # Advance time
        for _ in range(100):
            world.environment.update(world.dt)

        result = get_render_data(world)

        time_of_day = result["time_of_day"]
        assert 0.0 <= time_of_day <= 1.0


class TestTouchHelper:
    """Tests for get_touch_input function."""

    def test_touch_input_returns_correct_shape(self):
        """Touch input should return array of 8 values."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=10.0,
            touch_range=5.0,
        )

        assert result.shape == (8,)
        assert result.dtype == np.float32

    def test_touch_detects_entity_in_correct_direction(self):
        """Touch should detect entity in correct cardinal direction."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_radius = 10.0

        # Place food to the north (0 degrees)
        food_north = Food(entity_id=0, position=Vec2(250.0, 265.0))
        world.add_entity(food_north)
        world._rebuild_spatial_grid()  # Rebuild grid after adding entities

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=agent_radius,
            touch_range=10.0,
        )

        # Direction 0 = North
        assert result[0] == 1.0, "Should detect entity to the north"

    def test_touch_detects_multiple_directions(self):
        """Touch should detect entities in multiple directions."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_radius = 10.0

        # Place entities in multiple directions
        # North
        food_n = Food(entity_id=0, position=Vec2(250.0, 265.0))
        # East
        food_e = Food(entity_id=1, position=Vec2(265.0, 250.0))
        # South
        food_s = Food(entity_id=2, position=Vec2(250.0, 235.0))

        world.add_entity(food_n)
        world.add_entity(food_e)
        world.add_entity(food_s)
        world._rebuild_spatial_grid()  # Rebuild grid after adding entities

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=agent_radius,
            touch_range=10.0,
        )

        # Check N (0), E (2), S (4)
        assert result[0] == 1.0, "Should detect north"
        assert result[2] == 1.0, "Should detect east"
        assert result[4] == 1.0, "Should detect south"

    def test_touch_respects_range_limit(self):
        """Touch should not detect entities beyond touch range."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_radius = 10.0

        # Place food beyond touch range
        food_far = Food(entity_id=0, position=Vec2(250.0, 300.0))  # 50 units away
        world.add_entity(food_far)

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=agent_radius,
            touch_range=5.0,  # Only 15 units total range
        )

        # Should not detect anything
        assert np.all(result == 0.0), "Should not detect entity beyond range"

    def test_touch_all_directions_indexed_correctly(self):
        """Touch directions should be indexed correctly (N, NE, E, SE, S, SW, W, NW)."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)
        agent_radius = 10.0
        touch_range = 20.0

        # Place entities at all 8 cardinal/ordinal directions
        distance = 20.0
        directions = [
            (0.0, 1.0),  # N
            (0.707, 0.707),  # NE
            (1.0, 0.0),  # E
            (0.707, -0.707),  # SE
            (0.0, -1.0),  # S
            (-0.707, -0.707),  # SW
            (-1.0, 0.0),  # W
            (-0.707, 0.707),  # NW
        ]

        for i, (dx, dy) in enumerate(directions):
            pos = Vec2(agent_pos.x + dx * distance, agent_pos.y + dy * distance)
            food = Food(entity_id=i, position=pos)
            world.add_entity(food)

        world._rebuild_spatial_grid()  # Rebuild grid after adding entities

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=agent_radius,
            touch_range=touch_range,
        )

        # All 8 directions should detect something
        assert np.all(result == 1.0), "Should detect entities in all 8 directions"

    def test_touch_empty_world_returns_zeros(self):
        """Touch input should return all zeros in empty world."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=10.0,
            touch_range=5.0,
        )

        assert np.all(result == 0.0), "Empty world should return all zeros"

    def test_touch_values_are_binary(self):
        """Touch input values should be either 0.0 or 1.0."""
        world = World(width=500.0, height=500.0)
        agent_pos = Vec2(250.0, 250.0)

        # Add some entities
        for i in range(3):
            food = Food(
                entity_id=i,
                position=Vec2(250.0 + i * 15.0, 250.0 + i * 10.0),
            )
            world.add_entity(food)

        world._rebuild_spatial_grid()  # Rebuild grid after adding entities

        result = get_touch_input(
            world=world,
            agent_position=agent_pos,
            agent_radius=10.0,
            touch_range=15.0,
        )

        # All values should be 0.0 or 1.0
        assert np.all((result == 0.0) | (result == 1.0))


class TestIntegration:
    """Integration tests combining multiple helpers."""

    def test_helpers_work_together(self):
        """All helpers should work with same world instance."""
        world = World(width=500.0, height=500.0)

        # Setup world
        food = Food(entity_id=0, position=Vec2(300.0, 250.0))
        predator = Predator(
            entity_id=1,
            position=Vec2(200.0, 300.0),
            patrol_center=Vec2(200.0, 300.0),
        )
        world.add_entity(food)
        world.add_entity(predator)
        world._update_sound_sources()

        # Agent state
        agent_pos = Vec2(250.0, 250.0)
        agent_facing = Vec2(1.0, 0.0)

        # Get all sensory inputs
        vision = get_vision_input(world, agent_pos, agent_facing)
        sound = get_sound_input(world, agent_pos, agent_facing)
        touch = get_touch_input(world, agent_pos, agent_radius=10.0)
        render = get_render_data(world)

        # All should succeed without errors
        assert vision.shape == (32, 4)
        assert "left_ear" in sound
        assert touch.shape == (8,)
        assert "entities" in render
        assert len(render["entities"]) == 2

    def test_helpers_with_default_world(self):
        """Helpers should work with default world setup."""
        world = World()
        world.setup_default_world()

        agent_pos = Vec2(500.0, 500.0)
        agent_facing = Vec2(1.0, 0.0)

        # Get all inputs
        vision = get_vision_input(world, agent_pos, agent_facing)
        sound = get_sound_input(world, agent_pos, agent_facing)
        touch = get_touch_input(world, agent_pos, agent_radius=10.0)
        render = get_render_data(world)

        # Should handle complex world without errors
        assert vision is not None
        assert sound is not None
        assert touch is not None
        assert render is not None
        assert len(render["entities"]) > 0  # Default world has entities
