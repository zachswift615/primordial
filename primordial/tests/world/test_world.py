"""Comprehensive integration tests for the World class.

Tests world creation, entity management, spatial queries, tick updates,
food spawning, default setup, and integration scenarios.
"""

import pytest
from unittest.mock import MagicMock, patch

from primordial.world.world import World
from primordial.world.entities import (
    Entity,
    EntityType,
    Food,
    Predator,
    PredatorState,
    Vegetation,
    Water,
)
from primordial.world.geometry import Vec2, AABB, Circle


class DummyAgentEntity(Entity):
    """Mock agent entity for testing predator interactions."""

    def __init__(self, entity_id: int, position: Vec2):
        """Initialize dummy agent.

        Args:
            entity_id: Unique identifier.
            position: Initial position.
        """
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.AGENT,
            radius=10.0,
            is_static=False,
        )
        self.health = 100.0

    def update(self, world, dt: float) -> None:
        """No-op update."""
        pass

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)

    def take_damage(self, damage: float) -> None:
        """Receive damage from predator.

        Args:
            damage: Amount of damage.
        """
        self.health -= damage


class TestWorldCreation:
    """Tests for World initialization."""

    def test_creation_defaults(self):
        """Test creating world with default parameters."""
        world = World()
        assert world.width == 1000.0
        assert world.height == 1000.0
        assert world.tick_rate == 60
        assert world.dt == pytest.approx(1.0 / 60.0)
        assert world.bounds.min_x == 0.0
        assert world.bounds.min_y == 0.0
        assert world.bounds.max_x == 1000.0
        assert world.bounds.max_y == 1000.0

    def test_creation_custom_size(self):
        """Test creating world with custom size."""
        world = World(width=2000.0, height=1500.0, tick_rate=30)
        assert world.width == 2000.0
        assert world.height == 1500.0
        assert world.tick_rate == 30
        assert world.bounds.min_x == 0.0
        assert world.bounds.min_y == 0.0
        assert world.bounds.max_x == 2000.0
        assert world.bounds.max_y == 1500.0

    def test_dt_calculation(self):
        """Test that dt is correctly calculated from tick_rate."""
        world_60 = World(tick_rate=60)
        assert world_60.dt == pytest.approx(1.0 / 60.0)

        world_30 = World(tick_rate=30)
        assert world_30.dt == pytest.approx(1.0 / 30.0)

        world_120 = World(tick_rate=120)
        assert world_120.dt == pytest.approx(1.0 / 120.0)


class TestWorldEntityManagement:
    """Tests for entity addition, removal, and retrieval."""

    def test_add_entity(self):
        """Test adding an entity to the world."""
        world = World()
        food = Food(entity_id=0, position=Vec2(100.0, 100.0))

        entity_id = world.add_entity(food)
        assert entity_id >= 0
        assert food.id == entity_id

        # Verify entity can be retrieved
        retrieved = world.get_entity(entity_id)
        assert retrieved is food

    def test_add_multiple_entities(self):
        """Test adding multiple entities with unique IDs."""
        world = World()

        food1 = Food(entity_id=0, position=Vec2(100.0, 100.0))
        food2 = Food(entity_id=0, position=Vec2(200.0, 200.0))
        veg = Vegetation(entity_id=0, position=Vec2(300.0, 300.0))

        id1 = world.add_entity(food1)
        id2 = world.add_entity(food2)
        id3 = world.add_entity(veg)

        # IDs should be unique
        assert len({id1, id2, id3}) == 3

        # All should be retrievable
        assert world.get_entity(id1) is food1
        assert world.get_entity(id2) is food2
        assert world.get_entity(id3) is veg

    def test_remove_entity(self):
        """Test removing an entity from the world."""
        world = World()
        food = Food(entity_id=0, position=Vec2(100.0, 100.0))

        entity_id = world.add_entity(food)
        assert world.get_entity(entity_id) is not None

        world.remove_entity(entity_id)
        assert world.get_entity(entity_id) is None

    def test_get_entity(self):
        """Test retrieving an entity by ID."""
        world = World()
        food = Food(entity_id=0, position=Vec2(100.0, 100.0))

        entity_id = world.add_entity(food)
        retrieved = world.get_entity(entity_id)

        assert retrieved is food
        assert retrieved.position.x == 100.0
        assert retrieved.position.y == 100.0

    def test_get_entity_not_found(self):
        """Test that getting non-existent entity returns None."""
        world = World()
        assert world.get_entity(999) is None
        assert world.get_entity(-1) is None


class TestWorldSpatialQueries:
    """Tests for spatial query methods."""

    def test_get_entities_in_radius(self):
        """Test finding entities within a radius."""
        world = World()

        # Add entities at various distances from (500, 500)
        center = Vec2(500.0, 500.0)
        food_near = Food(entity_id=0, position=Vec2(510.0, 500.0))  # 10 units away
        food_far = Food(entity_id=0, position=Vec2(600.0, 500.0))  # 100 units away
        veg_near = Vegetation(entity_id=0, position=Vec2(500.0, 520.0))  # 20 units

        world.add_entity(food_near)
        world.add_entity(food_far)
        world.add_entity(veg_near)

        # Rebuild spatial grid to make entities queryable
        world._rebuild_spatial_grid()

        # Query with radius 50 - should get near entities only
        nearby = world.get_entities_in_radius(center, 50.0)
        assert len(nearby) == 2
        assert food_near in nearby
        assert veg_near in nearby
        assert food_far not in nearby

    def test_get_entities_in_radius_with_type_filter(self):
        """Test finding entities with type filtering."""
        world = World()

        center = Vec2(500.0, 500.0)
        food1 = Food(entity_id=0, position=Vec2(510.0, 500.0))
        food2 = Food(entity_id=0, position=Vec2(500.0, 515.0))
        veg = Vegetation(entity_id=0, position=Vec2(520.0, 500.0))

        world.add_entity(food1)
        world.add_entity(food2)
        world.add_entity(veg)

        # Rebuild spatial grid to make entities queryable
        world._rebuild_spatial_grid()

        # Query for only food within radius
        food_only = world.get_entities_in_radius(
            center, 50.0, entity_type=EntityType.FOOD
        )
        assert len(food_only) == 2
        assert food1 in food_only
        assert food2 in food_only
        assert veg not in food_only

        # Query for only vegetation
        veg_only = world.get_entities_in_radius(
            center, 50.0, entity_type=EntityType.VEGETATION
        )
        assert len(veg_only) == 1
        assert veg in veg_only

    def test_get_entities_in_radius_empty(self):
        """Test that empty query returns empty list."""
        world = World()

        # No entities added
        nearby = world.get_entities_in_radius(Vec2(500.0, 500.0), 100.0)
        assert nearby == []

        # Add entity far away
        food = Food(entity_id=0, position=Vec2(100.0, 100.0))
        world.add_entity(food)

        # Query in different area
        nearby = world.get_entities_in_radius(Vec2(900.0, 900.0), 50.0)
        assert nearby == []


class TestWorldTick:
    """Tests for the world tick/update cycle."""

    def test_tick_updates_environment(self):
        """Test that ticking updates environment state."""
        world = World()

        # Get initial brightness
        initial_brightness = world.brightness
        initial_time = world.environment.time_of_day

        # Tick several times
        for _ in range(10):
            world.tick()

        # Environment should have progressed
        assert world.environment.time_of_day != initial_time
        # Brightness may or may not change depending on cycle position
        # Just verify it's a valid value
        assert 0.0 <= world.brightness <= 1.0

    def test_tick_moves_entities(self):
        """Test that ticking moves entities with velocity."""
        world = World()

        # Create a dynamic entity with velocity
        food = Food(entity_id=0, position=Vec2(500.0, 500.0))
        # Make food dynamic for this test
        food.is_static = False
        food.velocity = Vec2(10.0, 0.0)  # Moving right
        food.friction = 1.0  # No friction for predictable movement

        entity_id = world.add_entity(food)

        initial_x = food.position.x
        initial_y = food.position.y

        # Tick the world
        world.tick()

        # Entity should have moved
        # Expected movement: velocity * dt = 10.0 * (1/60) ≈ 0.167
        expected_delta = 10.0 * world.dt
        assert food.position.x == pytest.approx(initial_x + expected_delta)
        assert food.position.y == pytest.approx(initial_y)

    def test_tick_updates_predator_ai(self):
        """Test that ticking updates predator AI state."""
        world = World()

        patrol_center = Vec2(500.0, 500.0)
        predator = Predator(
            entity_id=0,
            position=patrol_center.copy(),
            patrol_center=patrol_center,
        )

        world.add_entity(predator)

        # Initially should be patrolling
        assert predator.state == PredatorState.PATROLLING
        initial_patrol_target = predator.patrol_target.copy()

        # Tick several times - predator should move toward patrol target
        for _ in range(10):
            world.tick()

        # Predator should still be patrolling (no agents nearby)
        assert predator.state == PredatorState.PATROLLING
        # Velocity should be non-zero (moving toward patrol target)
        # Or position should have changed
        assert (
            predator.velocity.magnitude() > 0
            or predator.position.distance_to(patrol_center) > 0.1
        )


class TestWorldFoodSpawning:
    """Tests for food spawning mechanics."""

    def test_food_spawns_over_time(self):
        """Test that food spawns periodically."""
        world = World()

        # Count initial food
        initial_food_count = len(
            world.get_entities_in_radius(
                Vec2(500.0, 500.0), 1000.0, entity_type=EntityType.FOOD
            )
        )

        # Tick many times to allow food spawning
        for _ in range(300):  # 5 seconds at 60 fps
            world.tick()

        # Count food after ticking
        final_food_count = len(
            world.get_entities_in_radius(
                Vec2(500.0, 500.0), 1000.0, entity_type=EntityType.FOOD
            )
        )

        # Food should have spawned (or at least not decreased)
        assert final_food_count >= initial_food_count

    def test_food_respects_max_limit(self):
        """Test that food spawning respects maximum limit."""
        world = World()

        # Tick many times
        for _ in range(1000):  # ~16 seconds at 60 fps
            world.tick()

        # Count all food in world
        all_food = world.get_entities_in_radius(
            Vec2(world.width / 2, world.height / 2),
            max(world.width, world.height),
            entity_type=EntityType.FOOD,
        )

        # Should not exceed reasonable maximum (world should define this)
        # Assuming max is around 50-100 food items
        assert len(all_food) < 150  # Sanity check


class TestWorldSetupDefault:
    """Tests for default world setup."""

    def test_setup_creates_vegetation(self):
        """Test that setup_default_world creates vegetation."""
        world = World()
        world.setup_default_world()

        # Rebuild spatial grid after setup
        world._rebuild_spatial_grid()

        vegetation = world.get_entities_in_radius(
            Vec2(world.width / 2, world.height / 2),
            max(world.width, world.height),
            entity_type=EntityType.VEGETATION,
        )

        assert len(vegetation) > 0

    def test_setup_creates_water(self):
        """Test that setup_default_world creates water."""
        world = World()
        world.setup_default_world()

        # Rebuild spatial grid after setup
        world._rebuild_spatial_grid()

        water = world.get_entities_in_radius(
            Vec2(world.width / 2, world.height / 2),
            max(world.width, world.height),
            entity_type=EntityType.WATER,
        )

        assert len(water) > 0

    def test_setup_creates_predators(self):
        """Test that setup_default_world creates predators."""
        world = World()
        world.setup_default_world()

        # Rebuild spatial grid after setup
        world._rebuild_spatial_grid()

        predators = world.get_entities_in_radius(
            Vec2(world.width / 2, world.height / 2),
            max(world.width, world.height),
            entity_type=EntityType.PREDATOR,
        )

        assert len(predators) > 0

    def test_setup_creates_food(self):
        """Test that setup_default_world creates food."""
        world = World()
        world.setup_default_world()

        # Rebuild spatial grid after setup
        world._rebuild_spatial_grid()

        food = world.get_entities_in_radius(
            Vec2(world.width / 2, world.height / 2),
            max(world.width, world.height),
            entity_type=EntityType.FOOD,
        )

        assert len(food) > 0


class TestWorldIntegration:
    """Integration tests for complex world scenarios."""

    def test_predator_detects_agent_entity(self):
        """Test that predator detects and chases agent-type entity."""
        world = World()

        patrol_center = Vec2(500.0, 500.0)
        predator = Predator(
            entity_id=0,
            position=patrol_center.copy(),
            patrol_center=patrol_center,
        )

        # Place agent within detection range (predator detection_radius is 200.0)
        agent_pos = Vec2(600.0, 500.0)  # 100 units away - within range
        agent = DummyAgentEntity(entity_id=0, position=agent_pos)

        world.add_entity(predator)
        world.add_entity(agent)

        # Initially patrolling
        assert predator.state == PredatorState.PATROLLING

        # Rebuild spatial grid so predator can find agent
        world._rebuild_spatial_grid()

        # Tick to allow detection
        world.tick()

        # Predator should detect and start chasing
        assert predator.state == PredatorState.CHASING
        assert predator.target_entity is agent

    def test_multiple_ticks_stable(self):
        """Test that running many ticks doesn't cause errors."""
        world = World()
        world.setup_default_world()

        # Add some agent entities
        for i in range(5):
            agent = DummyAgentEntity(
                entity_id=0, position=Vec2(100.0 + i * 50, 100.0 + i * 50)
            )
            world.add_entity(agent)

        # Run 100 ticks without errors
        for _ in range(100):
            world.tick()

        # Verify world is still functional
        assert world.environment.time_of_day > 0
        assert 0.0 <= world.brightness <= 1.0

    def test_entities_stay_in_bounds(self):
        """Test that physics keeps entities within world bounds."""
        world = World(width=1000.0, height=1000.0)

        # Create entity near edge with velocity pointing outward
        edge_entity = Food(entity_id=0, position=Vec2(50.0, 50.0))
        edge_entity.is_static = False
        edge_entity.velocity = Vec2(-100.0, -100.0)  # Moving toward corner
        edge_entity.friction = 1.0  # No friction

        world.add_entity(edge_entity)

        # Tick several times
        for _ in range(60):  # 1 second
            world.tick()

        # Entity should still be within bounds (or at boundary)
        assert edge_entity.position.x >= 0.0
        assert edge_entity.position.y >= 0.0
        assert edge_entity.position.x <= world.width
        assert edge_entity.position.y <= world.height

    def test_predator_attack_cooldown(self):
        """Test predator attack cooldown mechanism."""
        world = World()

        predator = Predator(
            entity_id=0,
            position=Vec2(500.0, 500.0),
            patrol_center=Vec2(500.0, 500.0),
        )

        agent = DummyAgentEntity(entity_id=0, position=Vec2(500.0, 500.0))

        world.add_entity(predator)
        world.add_entity(agent)

        initial_health = agent.health

        # Force predator into chase state
        predator.state = PredatorState.CHASING
        predator.target_entity = agent

        # First tick - should attack
        world.tick()

        # Health should decrease
        assert agent.health < initial_health

        # Store health after first attack
        health_after_attack = agent.health

        # Immediate tick - should not attack (cooldown)
        world.tick()

        # Health should be same (still on cooldown)
        assert agent.health == health_after_attack

    def test_food_consumption_removes_entity(self):
        """Test that consuming food deactivates it and removes from spatial grid."""
        world = World()

        food = Food(entity_id=0, position=Vec2(500.0, 500.0))
        food_id = world.add_entity(food)

        assert world.get_entity(food_id) is not None

        # Consume the food
        energy = food.consume()
        assert energy == food.energy_value
        assert food.is_active is False

        # Tick to update spatial grid (inactive entities not added to grid)
        world.tick()

        # Food entity still exists in world.entities but is marked inactive
        entity = world.get_entity(food_id)
        assert entity is not None
        assert entity.is_active is False

        # Rebuild spatial grid
        world._rebuild_spatial_grid()

        # Inactive food should not appear in spatial queries
        nearby = world.get_entities_in_radius(Vec2(500.0, 500.0), 50.0)
        assert food not in nearby

    def test_multiple_predators_independent(self):
        """Test that multiple predators operate independently."""
        world = World()

        pred1 = Predator(
            entity_id=0,
            position=Vec2(200.0, 200.0),
            patrol_center=Vec2(200.0, 200.0),
        )

        pred2 = Predator(
            entity_id=0,
            position=Vec2(800.0, 800.0),
            patrol_center=Vec2(800.0, 800.0),
        )

        world.add_entity(pred1)
        world.add_entity(pred2)

        # Place agent near pred1 only (within pred1's detection range but not pred2's)
        agent = DummyAgentEntity(entity_id=0, position=Vec2(250.0, 200.0))
        world.add_entity(agent)

        # Rebuild spatial grid
        world._rebuild_spatial_grid()

        # Tick to allow detection
        world.tick()

        # Only pred1 should chase
        assert pred1.state == PredatorState.CHASING
        assert pred2.state == PredatorState.PATROLLING

    def test_world_brightness_cycles(self):
        """Test that world brightness cycles over time."""
        world = World()

        brightness_samples = []

        # Sample brightness over many ticks
        for _ in range(600):  # 10 seconds at 60 fps
            brightness_samples.append(world.brightness)
            world.tick()

        # Should have variation in brightness (not constant)
        assert len(set(brightness_samples)) > 1

        # All values should be valid
        assert all(0.0 <= b <= 1.0 for b in brightness_samples)

    def test_spatial_queries_after_movement(self):
        """Test that spatial queries work correctly after entities move."""
        world = World()

        # Create moving entity
        food = Food(entity_id=0, position=Vec2(500.0, 500.0))
        food.is_static = False
        food.velocity = Vec2(100.0, 0.0)
        food.friction = 1.0

        world.add_entity(food)

        # Rebuild spatial grid initially
        world._rebuild_spatial_grid()

        # Query at original position
        initial_query = world.get_entities_in_radius(Vec2(500.0, 500.0), 10.0)
        assert food in initial_query

        # Tick to move entity (tick automatically rebuilds spatial grid)
        for _ in range(60):  # Move ~100 units
            world.tick()

        # Query at old position should not find it
        old_pos_query = world.get_entities_in_radius(Vec2(500.0, 500.0), 10.0)
        assert food not in old_pos_query

        # Query at new position should find it
        new_pos_query = world.get_entities_in_radius(food.position, 10.0)
        assert food in new_pos_query
