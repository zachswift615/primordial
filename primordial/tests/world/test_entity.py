"""Tests for Entity base class."""

import pytest

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import Circle, Vec2


class DummyEntity(Entity):
    """Concrete dummy entity for testing the abstract base class."""

    def update(self, world, dt: float) -> None:
        """No-op update for testing."""
        pass

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)


class TestEntityCreation:
    """Tests for Entity creation and properties."""

    def test_basic_creation(self):
        """Test creating a basic entity."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 200.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        assert entity.id == 1
        assert entity.position.x == 100.0
        assert entity.position.y == 200.0
        assert entity.entity_type == EntityType.FOOD
        assert entity.radius == 10.0
        assert entity.is_active is True
        assert entity.is_static is False

    def test_static_entity(self):
        """Test creating a static entity."""
        entity = DummyEntity(
            entity_id=2,
            position=Vec2(50.0, 50.0),
            entity_type=EntityType.VEGETATION,
            radius=20.0,
            is_static=True,
        )
        assert entity.is_static is True

    def test_default_physics_properties(self):
        """Test default physics properties."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        assert entity.velocity.x == 0.0
        assert entity.velocity.y == 0.0
        assert entity.acceleration.x == 0.0
        assert entity.acceleration.y == 0.0
        assert entity.mass == 1.0
        assert entity.friction == 0.95


class TestEntityPhysics:
    """Tests for Entity physics methods."""

    def test_apply_force(self):
        """Test applying force to entity."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.mass = 2.0
        entity.apply_force(Vec2(10.0, 20.0))
        # a = F / m = (10, 20) / 2 = (5, 10)
        assert entity.acceleration.x == 5.0
        assert entity.acceleration.y == 10.0

    def test_apply_force_accumulates(self):
        """Test that forces accumulate."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.apply_force(Vec2(10.0, 0.0))
        entity.apply_force(Vec2(5.0, 0.0))
        assert entity.acceleration.x == 15.0

    def test_apply_force_static_ignored(self):
        """Test that static entities ignore forces."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )
        entity.apply_force(Vec2(100.0, 100.0))
        assert entity.acceleration.x == 0.0
        assert entity.acceleration.y == 0.0

    def test_apply_impulse(self):
        """Test applying impulse to entity."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.apply_impulse(Vec2(5.0, 10.0))
        assert entity.velocity.x == 5.0
        assert entity.velocity.y == 10.0

    def test_apply_impulse_static_ignored(self):
        """Test that static entities ignore impulses."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )
        entity.apply_impulse(Vec2(100.0, 100.0))
        assert entity.velocity.x == 0.0
        assert entity.velocity.y == 0.0

    def test_set_position(self):
        """Test setting position directly."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.set_position(Vec2(100.0, 200.0))
        assert entity.position.x == 100.0
        assert entity.position.y == 200.0

    def test_set_velocity(self):
        """Test setting velocity directly."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.set_velocity(Vec2(50.0, 60.0))
        assert entity.velocity.x == 50.0
        assert entity.velocity.y == 60.0

    def test_set_velocity_static_ignored(self):
        """Test that static entities ignore velocity setting."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )
        entity.set_velocity(Vec2(50.0, 60.0))
        assert entity.velocity.x == 0.0
        assert entity.velocity.y == 0.0


class TestEntityMethods:
    """Tests for Entity utility methods."""

    def test_deactivate(self):
        """Test deactivating entity."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        assert entity.is_active is True
        entity.deactivate()
        assert entity.is_active is False

    def test_distance_to(self):
        """Test distance calculation between entities."""
        entity1 = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity2 = DummyEntity(
            entity_id=2,
            position=Vec2(3.0, 4.0),
            entity_type=EntityType.FOOD,
            radius=5.0,
        )
        assert entity1.distance_to(entity2) == 5.0  # 3-4-5 triangle

    def test_direction_to(self):
        """Test direction calculation between entities."""
        entity1 = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity2 = DummyEntity(
            entity_id=2,
            position=Vec2(10.0, 0.0),
            entity_type=EntityType.FOOD,
            radius=5.0,
        )
        direction = entity1.direction_to(entity2)
        assert abs(direction.x - 1.0) < 1e-6
        assert abs(direction.y) < 1e-6

    def test_get_collision_shape(self):
        """Test getting collision shape."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 200.0),
            entity_type=EntityType.AGENT,
            radius=15.0,
        )
        shape = entity.get_collision_shape()
        assert isinstance(shape, Circle)
        assert shape.center.x == 100.0
        assert shape.center.y == 200.0
        assert shape.radius == 15.0

    def test_repr(self):
        """Test string representation."""
        entity = DummyEntity(
            entity_id=42,
            position=Vec2(100.5, 200.5),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        repr_str = repr(entity)
        assert "DummyEntity" in repr_str
        assert "42" in repr_str
        assert "100.5" in repr_str
        assert "200.5" in repr_str


class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_types_exist(self):
        """Test all expected entity types exist."""
        assert EntityType.AGENT.value == "agent"
        assert EntityType.FOOD.value == "food"
        assert EntityType.PREDATOR.value == "predator"
        assert EntityType.VEGETATION.value == "vegetation"
        assert EntityType.WATER.value == "water"

    def test_type_comparison(self):
        """Test entity type comparison."""
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        assert entity.entity_type == EntityType.FOOD
        assert entity.entity_type != EntityType.AGENT
