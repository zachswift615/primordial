"""Tests for concrete entity classes (Food, Predator, Vegetation, Water)."""

import math
from unittest.mock import MagicMock

import pytest

from primordial.world.entities import (
    Entity,
    EntityType,
    Food,
    Predator,
    PredatorState,
    Vegetation,
    Water,
)
from primordial.world.geometry import Circle, Vec2


class TestVegetation:
    """Tests for Vegetation entity."""

    def test_creation(self):
        """Test creating vegetation."""
        veg = Vegetation(
            entity_id=1,
            position=Vec2(100.0, 200.0),
        )
        assert veg.id == 1
        assert veg.position.x == 100.0
        assert veg.position.y == 200.0
        assert veg.entity_type == EntityType.VEGETATION
        assert veg.is_static is True
        assert veg.radius == 20.0  # default

    def test_custom_radius(self):
        """Test vegetation with custom radius."""
        veg = Vegetation(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            radius=35.0,
        )
        assert veg.radius == 35.0

    def test_blocks_vision(self):
        """Test that vegetation blocks vision."""
        veg = Vegetation(entity_id=1, position=Vec2(0.0, 0.0))
        assert veg.blocks_vision() is True

    def test_collision_shape(self):
        """Test collision shape."""
        veg = Vegetation(
            entity_id=1,
            position=Vec2(50.0, 75.0),
            radius=25.0,
        )
        shape = veg.get_collision_shape()
        assert isinstance(shape, Circle)
        assert shape.center.x == 50.0
        assert shape.center.y == 75.0
        assert shape.radius == 25.0

    def test_update_does_nothing(self):
        """Test that update doesn't change position."""
        veg = Vegetation(entity_id=1, position=Vec2(100.0, 100.0))
        original_pos = veg.position.copy()
        veg.update(None, 1.0)
        assert veg.position == original_pos


class TestWater:
    """Tests for Water entity."""

    def test_creation(self):
        """Test creating water."""
        water = Water(
            entity_id=1,
            position=Vec2(100.0, 200.0),
        )
        assert water.id == 1
        assert water.position.x == 100.0
        assert water.position.y == 200.0
        assert water.entity_type == EntityType.WATER
        assert water.is_static is True
        assert water.radius == 30.0  # default

    def test_custom_properties(self):
        """Test water with custom properties."""
        water = Water(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            radius=50.0,
            sound_intensity=0.5,
            sound_frequency=400.0,
        )
        assert water.radius == 50.0
        assert water.sound_intensity == 0.5
        assert water.sound_frequency == 400.0

    def test_sound_properties(self):
        """Test getting sound properties."""
        water = Water(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            sound_intensity=0.3,
            sound_frequency=300.0,
        )
        intensity, frequency = water.get_sound_properties()
        assert intensity == 0.3
        assert frequency == 300.0

    def test_collision_shape(self):
        """Test collision shape."""
        water = Water(
            entity_id=1,
            position=Vec2(50.0, 75.0),
            radius=40.0,
        )
        shape = water.get_collision_shape()
        assert isinstance(shape, Circle)
        assert shape.center.x == 50.0
        assert shape.center.y == 75.0
        assert shape.radius == 40.0


class TestFood:
    """Tests for Food entity."""

    def test_creation(self):
        """Test creating food."""
        food = Food(
            entity_id=1,
            position=Vec2(100.0, 200.0),
        )
        assert food.id == 1
        assert food.position.x == 100.0
        assert food.position.y == 200.0
        assert food.entity_type == EntityType.FOOD
        assert food.is_static is True
        assert food.radius == 5.0  # default
        assert food.energy_value == 50.0  # default

    def test_custom_properties(self):
        """Test food with custom properties."""
        food = Food(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            energy_value=100.0,
            sound_intensity=0.2,
            sound_frequency=250.0,
            radius=8.0,
        )
        assert food.energy_value == 100.0
        assert food.sound_intensity == 0.2
        assert food.sound_frequency == 250.0
        assert food.radius == 8.0

    def test_consume(self):
        """Test consuming food."""
        food = Food(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            energy_value=75.0,
        )
        assert food.is_active is True

        energy = food.consume()

        assert energy == 75.0
        assert food.is_active is False

    def test_sound_properties(self):
        """Test getting sound properties."""
        food = Food(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            sound_intensity=0.1,
            sound_frequency=200.0,
        )
        intensity, frequency = food.get_sound_properties()
        assert intensity == 0.1
        assert frequency == 200.0

    def test_collision_shape(self):
        """Test collision shape."""
        food = Food(
            entity_id=1,
            position=Vec2(50.0, 75.0),
            radius=10.0,
        )
        shape = food.get_collision_shape()
        assert isinstance(shape, Circle)
        assert shape.center.x == 50.0
        assert shape.center.y == 75.0
        assert shape.radius == 10.0


class TestPredatorCreation:
    """Tests for Predator creation."""

    def test_creation(self):
        """Test creating predator."""
        pred = Predator(
            entity_id=1,
            position=Vec2(100.0, 200.0),
            patrol_center=Vec2(100.0, 200.0),
        )
        assert pred.id == 1
        assert pred.position.x == 100.0
        assert pred.position.y == 200.0
        assert pred.entity_type == EntityType.PREDATOR
        assert pred.is_static is False
        assert pred.radius == 15.0
        assert pred.state == PredatorState.PATROLLING

    def test_custom_patrol_radius(self):
        """Test predator with custom patrol radius."""
        pred = Predator(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            patrol_center=Vec2(0.0, 0.0),
            patrol_radius=200.0,
        )
        assert pred.patrol_radius == 200.0

    def test_default_properties(self):
        """Test default predator properties."""
        pred = Predator(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            patrol_center=Vec2(0.0, 0.0),
        )
        assert pred.mass == 2.0
        assert pred.friction == 0.90
        assert pred.detection_radius == 200.0
        assert pred.chase_speed == 80.0
        assert pred.patrol_speed == 30.0
        assert pred.damage == 20.0
        assert pred.attack_cooldown_max == 1.0

    def test_collision_shape(self):
        """Test collision shape."""
        pred = Predator(
            entity_id=1,
            position=Vec2(50.0, 75.0),
            patrol_center=Vec2(50.0, 75.0),
        )
        shape = pred.get_collision_shape()
        assert isinstance(shape, Circle)
        assert shape.center.x == 50.0
        assert shape.center.y == 75.0
        assert shape.radius == 15.0


class TestPredatorPatrol:
    """Tests for Predator patrol behavior."""

    def test_patrol_generates_targets(self):
        """Test that patrol generates random targets."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
            patrol_radius=100.0,
        )

        # Generate multiple targets
        targets = [pred._generate_patrol_target() for _ in range(10)]

        # All targets should be within patrol radius
        for target in targets:
            distance = pred.patrol_center.distance_to(target)
            assert distance <= pred.patrol_radius

    def test_patrol_applies_force(self):
        """Test that patrolling applies force toward target."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.patrol_target = Vec2(600.0, 500.0)  # Target to the right

        # Create mock world with no agents nearby
        mock_world = MagicMock()
        mock_world.get_entities_in_radius.return_value = []

        pred.update(mock_world, 0.1)

        # Should have applied force toward target
        assert pred.acceleration.x > 0

    def test_patrol_reaches_target(self):
        """Test that predator picks new target when reaching current one."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.patrol_target = Vec2(505.0, 500.0)  # Very close

        original_target = pred.patrol_target

        mock_world = MagicMock()
        mock_world.get_entities_in_radius.return_value = []

        pred.update(mock_world, 0.1)

        # Target should have changed
        assert pred.patrol_target != original_target


class TestPredatorChase:
    """Tests for Predator chase behavior."""

    def test_detects_agent(self):
        """Test that predator detects nearby agents."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.position = Vec2(550.0, 500.0)  # Within detection radius
        mock_agent.is_active = True

        mock_world = MagicMock()
        mock_world.get_entities_in_radius.return_value = [mock_agent]

        pred.update(mock_world, 0.1)

        assert pred.state == PredatorState.CHASING
        assert pred.target_entity == mock_agent

    def test_chase_applies_force(self):
        """Test that chasing applies force toward target."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.CHASING

        mock_target = MagicMock()
        mock_target.position = Vec2(600.0, 500.0)
        mock_target.is_active = True
        mock_target.radius = 10.0
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        # Should have applied force toward target
        assert pred.acceleration.x > 0

    def test_chase_is_growling(self):
        """Test that predator growls while chasing."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.CHASING

        mock_target = MagicMock()
        mock_target.position = Vec2(600.0, 500.0)
        mock_target.is_active = True
        mock_target.radius = 10.0
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        assert pred.is_growling is True

    def test_chase_abandons_when_too_far(self):
        """Test that predator abandons chase when too far from patrol."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(100.0, 100.0),  # Far from position
            patrol_radius=50.0,
        )
        pred.chase_abandon_distance = 200.0
        pred.state = PredatorState.CHASING

        mock_target = MagicMock()
        mock_target.position = Vec2(600.0, 500.0)
        mock_target.is_active = True
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        # Should have abandoned chase
        assert pred.state == PredatorState.RETURNING
        assert pred.target_entity is None

    def test_chase_ends_when_target_inactive(self):
        """Test that chase ends when target becomes inactive."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.CHASING

        mock_target = MagicMock()
        mock_target.is_active = False  # Target died
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        assert pred.state == PredatorState.RETURNING
        assert pred.target_entity is None


class TestPredatorAttack:
    """Tests for Predator attack behavior."""

    def test_attack_when_in_range(self):
        """Test that predator attacks when close enough."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.CHASING
        pred.attack_cooldown = 0.0

        mock_target = MagicMock()
        mock_target.position = Vec2(520.0, 500.0)  # 20 units away
        mock_target.is_active = True
        mock_target.radius = 10.0  # Combined radius = 25, so in range
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        # Should have called take_damage
        mock_target.take_damage.assert_called_once_with(20.0)

    def test_attack_cooldown(self):
        """Test that attack has cooldown."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.CHASING
        pred.attack_cooldown = 0.5  # On cooldown

        mock_target = MagicMock()
        mock_target.position = Vec2(510.0, 500.0)  # In range
        mock_target.is_active = True
        mock_target.radius = 10.0
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        # Should NOT have attacked
        mock_target.take_damage.assert_not_called()

    def test_attack_sets_cooldown(self):
        """Test that attacking sets cooldown."""
        pred = Predator(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.CHASING
        pred.attack_cooldown = 0.0

        mock_target = MagicMock()
        mock_target.position = Vec2(510.0, 500.0)
        mock_target.is_active = True
        mock_target.radius = 10.0
        pred.target_entity = mock_target

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        assert pred.attack_cooldown == pred.attack_cooldown_max


class TestPredatorReturn:
    """Tests for Predator return behavior."""

    def test_return_moves_to_patrol_center(self):
        """Test that returning moves toward patrol center."""
        pred = Predator(
            entity_id=1,
            position=Vec2(700.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )
        pred.state = PredatorState.RETURNING

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        # Should have applied force toward patrol center (left)
        assert pred.acceleration.x < 0

    def test_return_ends_at_patrol_area(self):
        """Test that return ends when reaching patrol area."""
        pred = Predator(
            entity_id=1,
            position=Vec2(550.0, 500.0),  # Within patrol radius
            patrol_center=Vec2(500.0, 500.0),
            patrol_radius=100.0,
        )
        pred.state = PredatorState.RETURNING

        mock_world = MagicMock()

        pred.update(mock_world, 0.1)

        assert pred.state == PredatorState.PATROLLING


class TestPredatorSound:
    """Tests for Predator sound properties."""

    def test_sound_when_growling(self):
        """Test sound properties when growling."""
        pred = Predator(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            patrol_center=Vec2(0.0, 0.0),
        )
        pred.is_growling = True

        sound = pred.get_sound_properties()

        assert sound is not None
        intensity, frequency = sound
        assert intensity == 0.5
        assert frequency == 100.0

    def test_no_sound_when_not_growling(self):
        """Test no sound when not growling."""
        pred = Predator(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            patrol_center=Vec2(0.0, 0.0),
        )
        pred.is_growling = False

        sound = pred.get_sound_properties()

        assert sound is None

    def test_set_patrol_center(self):
        """Test setting patrol center."""
        pred = Predator(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            patrol_center=Vec2(0.0, 0.0),
        )

        pred.set_patrol_center(Vec2(100.0, 200.0))

        assert pred.patrol_center.x == 100.0
        assert pred.patrol_center.y == 200.0
