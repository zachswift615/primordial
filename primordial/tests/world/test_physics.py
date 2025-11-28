"""Tests for PhysicsEngine."""

import math

import pytest

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import AABB, Circle, Vec2
from primordial.world.physics import PhysicsEngine


class DummyEntity(Entity):
    """Concrete entity for testing."""

    def update(self, world, dt: float) -> None:
        pass

    def get_collision_shape(self) -> Circle:
        return Circle(self.position, self.radius)


class TestPhysicsEngineCreation:
    """Tests for PhysicsEngine initialization."""

    def test_creation(self):
        """Test creating physics engine."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)
        assert physics.world_bounds == bounds
        assert physics.restitution == 0.8

    def test_creation_with_restitution(self):
        """Test creating physics engine with custom restitution."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds, restitution=0.5)
        assert physics.restitution == 0.5


class TestMotionIntegration:
    """Tests for physics motion integration."""

    def test_velocity_update(self):
        """Test that velocity updates from acceleration."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.acceleration = Vec2(100.0, 0.0)
        entity.friction = 1.0  # No friction for this test

        physics.step([entity], dt=0.1)

        # v = v + a * dt = 0 + 100 * 0.1 = 10
        assert abs(entity.velocity.x - 10.0) < 0.01

    def test_position_update(self):
        """Test that position updates from velocity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(100.0, 0.0)
        entity.friction = 1.0

        physics.step([entity], dt=0.1)

        # p = p + v * dt = 500 + 100 * 0.1 = 510
        assert abs(entity.position.x - 510.0) < 0.01

    def test_friction_application(self):
        """Test that friction reduces velocity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(100.0, 0.0)
        entity.friction = 0.9

        physics.step([entity], dt=0.1)

        # v = v * friction = 100 * 0.9 = 90 (before position update)
        # After many steps, velocity should decrease
        for _ in range(10):
            physics.step([entity], dt=0.1)

        assert entity.velocity.x < 50.0

    def test_acceleration_reset(self):
        """Test that acceleration resets after step."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.acceleration = Vec2(100.0, 50.0)

        physics.step([entity], dt=0.1)

        assert entity.acceleration.x == 0.0
        assert entity.acceleration.y == 0.0

    def test_static_entity_not_moved(self):
        """Test that static entities don't move."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )
        entity.velocity = Vec2(100.0, 100.0)  # Shouldn't matter

        physics.step([entity], dt=0.1)

        # Position should not change
        assert entity.position.x == 500.0
        assert entity.position.y == 500.0

    def test_inactive_entity_not_moved(self):
        """Test that inactive entities don't move."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 500.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(100.0, 100.0)
        entity.is_active = False

        physics.step([entity], dt=0.1)

        # Position should not change
        assert entity.position.x == 500.0
        assert entity.position.y == 500.0


class TestBoundaryEnforcement:
    """Tests for world boundary enforcement."""

    def test_left_boundary(self):
        """Test entity bounces off left boundary."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(-5.0, 500.0),  # Outside left boundary
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(-100.0, 0.0)

        physics.step([entity], dt=0.0)

        # Position should be clamped
        assert entity.position.x >= entity.radius
        # Velocity should reverse (bounce)
        assert entity.velocity.x > 0

    def test_right_boundary(self):
        """Test entity bounces off right boundary."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(1005.0, 500.0),  # Outside right boundary
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(100.0, 0.0)

        physics.step([entity], dt=0.0)

        # Position should be clamped
        assert entity.position.x <= 1000.0 - entity.radius
        # Velocity should reverse
        assert entity.velocity.x < 0

    def test_top_boundary(self):
        """Test entity bounces off top boundary."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, 1005.0),  # Outside top boundary
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(0.0, 100.0)

        physics.step([entity], dt=0.0)

        # Position should be clamped
        assert entity.position.y <= 1000.0 - entity.radius
        # Velocity should reverse
        assert entity.velocity.y < 0

    def test_bottom_boundary(self):
        """Test entity bounces off bottom boundary."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(500.0, -5.0),  # Outside bottom boundary
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity.velocity = Vec2(0.0, -100.0)

        physics.step([entity], dt=0.0)

        # Position should be clamped
        assert entity.position.y >= entity.radius
        # Velocity should reverse
        assert entity.velocity.y > 0


class TestCollisionDetection:
    """Tests for collision detection."""

    def test_collision_detected(self):
        """Test that overlapping entities are detected as colliding."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_a = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity_b = DummyEntity(
            entity_id=2,
            position=Vec2(115.0, 100.0),  # 15 units apart, overlapping
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        assert physics._check_collision(entity_a, entity_b)

    def test_no_collision_detected(self):
        """Test that non-overlapping entities don't collide."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_a = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity_b = DummyEntity(
            entity_id=2,
            position=Vec2(125.0, 100.0),  # 25 units apart, not overlapping
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        assert not physics._check_collision(entity_a, entity_b)


class TestCollisionResolution:
    """Tests for collision resolution."""

    def test_dynamic_dynamic_separation(self):
        """Test that two dynamic entities are separated equally."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_a = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity_b = DummyEntity(
            entity_id=2,
            position=Vec2(110.0, 100.0),  # 10 units apart, overlap of 10
            entity_type=EntityType.AGENT,
            radius=10.0,
        )

        physics._resolve_collision(entity_a, entity_b)

        # After resolution, they should not overlap
        distance = entity_a.position.distance_to(entity_b.position)
        assert distance >= 19.9  # Should be at least 20 (radii sum)

    def test_static_dynamic_separation(self):
        """Test that only the dynamic entity moves when colliding with static."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        static_entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )
        dynamic_entity = DummyEntity(
            entity_id=2,
            position=Vec2(110.0, 100.0),  # Overlapping
            entity_type=EntityType.AGENT,
            radius=10.0,
        )

        physics._resolve_collision(static_entity, dynamic_entity)

        # Static entity should not move
        assert static_entity.position.x == 100.0
        assert static_entity.position.y == 100.0

        # Dynamic entity should be pushed away
        assert dynamic_entity.position.x > 110.0

    def test_collision_impulse(self):
        """Test that collision applies velocity impulse."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_a = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity_a.velocity = Vec2(50.0, 0.0)  # Moving right

        entity_b = DummyEntity(
            entity_id=2,
            position=Vec2(115.0, 100.0),  # In front
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity_b.velocity = Vec2(0.0, 0.0)  # Stationary

        physics._resolve_collision(entity_a, entity_b)

        # After collision, entity_b should gain velocity
        assert entity_b.velocity.x > 0

    def test_static_static_no_resolution(self):
        """Test that two static entities don't resolve collision."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_a = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )
        entity_b = DummyEntity(
            entity_id=2,
            position=Vec2(110.0, 100.0),  # Overlapping
            entity_type=EntityType.VEGETATION,
            radius=10.0,
            is_static=True,
        )

        original_a_pos = entity_a.position.copy()
        original_b_pos = entity_b.position.copy()

        physics._resolve_collision(entity_a, entity_b)

        # Neither should move
        assert entity_a.position == original_a_pos
        assert entity_b.position == original_b_pos


class TestRaycast:
    """Tests for raycasting."""

    def test_raycast_hit(self):
        """Test raycast hits an entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 50.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),  # Pointing right
            max_distance=500.0,
            entities=[entity],
        )

        assert hit_entity == entity
        assert abs(distance - 90.0) < 0.1  # 100 - 10 = 90

    def test_raycast_miss(self):
        """Test raycast misses when no entity in path."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),  # Above ray
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),  # Pointing right
            max_distance=500.0,
            entities=[entity],
        )

        assert hit_entity is None
        assert distance == 500.0

    def test_raycast_max_distance(self):
        """Test raycast respects max distance."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(200.0, 50.0),  # Far away
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),
            max_distance=100.0,  # Entity is beyond this
            entities=[entity],
        )

        assert hit_entity is None
        assert distance == 100.0

    def test_raycast_closest(self):
        """Test raycast returns closest entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_near = DummyEntity(
            entity_id=1,
            position=Vec2(50.0, 50.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        entity_far = DummyEntity(
            entity_id=2,
            position=Vec2(150.0, 50.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),
            max_distance=500.0,
            entities=[entity_far, entity_near],  # Far listed first
        )

        assert hit_entity == entity_near

    def test_raycast_ignore_entity(self):
        """Test raycast ignores specified entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity_self = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 50.0),  # At origin
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity_target = DummyEntity(
            entity_id=2,
            position=Vec2(100.0, 50.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),
            max_distance=500.0,
            entities=[entity_self, entity_target],
            ignore_entity_id=1,  # Ignore self
        )

        assert hit_entity == entity_target

    def test_raycast_inactive_entity(self):
        """Test raycast ignores inactive entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(50.0, 50.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        entity.is_active = False

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),
            max_distance=500.0,
            entities=[entity],
        )

        assert hit_entity is None

    def test_raycast_origin_inside_circle(self):
        """Test raycast when origin is inside an entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(5.0, 50.0),  # Very close to origin
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),  # Inside entity
            direction=Vec2(1.0, 0.0),
            max_distance=500.0,
            entities=[entity],
        )

        assert hit_entity == entity
        assert distance == 0.0

    def test_raycast_behind_origin(self):
        """Test raycast doesn't hit entities behind origin."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(-50.0, 50.0),  # Behind origin
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        hit_entity, distance = physics.raycast(
            origin=Vec2(0.0, 50.0),
            direction=Vec2(1.0, 0.0),  # Pointing right (away from entity)
            max_distance=500.0,
            entities=[entity],
        )

        assert hit_entity is None


class TestFullPhysicsStep:
    """Integration tests for full physics step."""

    def test_multiple_entities(self):
        """Test physics step with multiple entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        physics = PhysicsEngine(bounds)

        entities = [
            DummyEntity(
                entity_id=i,
                position=Vec2(200.0 + i * 100, 500.0),
                entity_type=EntityType.AGENT,
                radius=10.0,
            )
            for i in range(5)
        ]

        for entity in entities:
            entity.velocity = Vec2(10.0, 0.0)

        # Run several steps
        for _ in range(10):
            physics.step(entities, dt=1.0 / 60.0)

        # All entities should have moved to the right
        for entity in entities:
            assert entity.position.x > 200.0

    def test_entities_stay_in_bounds(self):
        """Test that entities stay within world bounds over time."""
        bounds = AABB(0.0, 0.0, 100.0, 100.0)
        physics = PhysicsEngine(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(50.0, 50.0),
            entity_type=EntityType.AGENT,
            radius=5.0,
        )
        entity.velocity = Vec2(1000.0, 1000.0)  # Very fast

        # Run many steps
        for _ in range(100):
            physics.step([entity], dt=1.0 / 60.0)

        # Entity should still be within bounds
        assert entity.position.x >= entity.radius
        assert entity.position.x <= 100.0 - entity.radius
        assert entity.position.y >= entity.radius
        assert entity.position.y <= 100.0 - entity.radius
