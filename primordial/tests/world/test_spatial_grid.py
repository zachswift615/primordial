"""Tests for SpatialGrid."""

import pytest

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import AABB, Circle, Vec2
from primordial.world.spatial_grid import SpatialGrid


class DummyEntity(Entity):
    """Concrete entity for testing."""

    def update(self, world, dt: float) -> None:
        pass

    def get_collision_shape(self) -> Circle:
        return Circle(self.position, self.radius)


class TestSpatialGridCreation:
    """Tests for SpatialGrid initialization."""

    def test_creation(self):
        """Test creating spatial grid."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=100.0)

        assert grid.bounds == bounds
        assert grid.cell_size == 100.0
        assert grid.cols == 11  # 1000/100 + 1
        assert grid.rows == 11

    def test_creation_default_cell_size(self):
        """Test default cell size."""
        bounds = AABB(0.0, 0.0, 500.0, 500.0)
        grid = SpatialGrid(bounds)

        assert grid.cell_size == 100.0

    def test_empty_grid(self):
        """Test empty grid has no cells."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        assert grid.get_cell_count() == 0
        assert grid.get_entity_count() == 0


class TestSpatialGridInsert:
    """Tests for inserting entities."""

    def test_insert_single_entity(self):
        """Test inserting a single entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=100.0)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(150.0, 150.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        grid.insert(entity)

        assert grid.get_cell_count() >= 1

    def test_insert_entity_spanning_cells(self):
        """Test entity spanning multiple cells."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=100.0)

        # Large entity at cell boundary
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),  # At corner of 4 cells
            entity_type=EntityType.VEGETATION,
            radius=50.0,  # Spans into adjacent cells
        )

        grid.insert(entity)

        # Should be in multiple cells
        assert grid.get_entity_count() > 1

    def test_insert_multiple_entities(self):
        """Test inserting multiple entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=100.0)

        for i in range(10):
            entity = DummyEntity(
                entity_id=i,
                position=Vec2(i * 100.0 + 50.0, 50.0),
                entity_type=EntityType.FOOD,
                radius=5.0,
            )
            grid.insert(entity)

        assert grid.get_cell_count() >= 10


class TestSpatialGridClear:
    """Tests for clearing the grid."""

    def test_clear(self):
        """Test clearing all entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        for i in range(5):
            entity = DummyEntity(
                entity_id=i,
                position=Vec2(i * 100.0, 50.0),
                entity_type=EntityType.FOOD,
                radius=10.0,
            )
            grid.insert(entity)

        assert grid.get_cell_count() > 0

        grid.clear()

        assert grid.get_cell_count() == 0
        assert grid.get_entity_count() == 0


class TestSpatialGridRemove:
    """Tests for removing entities."""

    def test_remove_entity(self):
        """Test removing an entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(150.0, 150.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        grid.insert(entity)
        initial_count = grid.get_entity_count()

        grid.remove(entity)

        assert grid.get_entity_count() < initial_count


class TestSpatialGridQueryRadius:
    """Tests for radius queries."""

    def test_query_finds_entity_in_range(self):
        """Test query finds entity within radius."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        grid.insert(entity)

        results = grid.query_radius(Vec2(120.0, 100.0), radius=50.0)

        assert len(results) == 1
        assert results[0].id == 1

    def test_query_excludes_entity_out_of_range(self):
        """Test query excludes entity outside radius."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        grid.insert(entity)

        results = grid.query_radius(Vec2(500.0, 500.0), radius=50.0)

        assert len(results) == 0

    def test_query_filters_by_type(self):
        """Test query can filter by entity type."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        food = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        vegetation = DummyEntity(
            entity_id=2,
            position=Vec2(110.0, 100.0),
            entity_type=EntityType.VEGETATION,
            radius=10.0,
        )
        grid.insert(food)
        grid.insert(vegetation)

        # Query for food only
        results = grid.query_radius(
            Vec2(100.0, 100.0), radius=50.0, entity_type=EntityType.FOOD
        )

        assert len(results) == 1
        assert results[0].entity_type == EntityType.FOOD

    def test_query_excludes_inactive(self):
        """Test query excludes inactive entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        entity.is_active = False
        grid.insert(entity)

        results = grid.query_radius(Vec2(100.0, 100.0), radius=50.0)

        assert len(results) == 0

    def test_query_no_duplicates(self):
        """Test query doesn't return duplicates for multi-cell entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=50.0)

        # Large entity spanning multiple cells
        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.VEGETATION,
            radius=40.0,
        )
        grid.insert(entity)

        # Query that covers multiple cells
        results = grid.query_radius(Vec2(100.0, 100.0), radius=100.0)

        assert len(results) == 1

    def test_query_multiple_entities(self):
        """Test query returns multiple entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        for i in range(5):
            entity = DummyEntity(
                entity_id=i,
                position=Vec2(100.0 + i * 10.0, 100.0),
                entity_type=EntityType.FOOD,
                radius=5.0,
            )
            grid.insert(entity)

        results = grid.query_radius(Vec2(120.0, 100.0), radius=100.0)

        assert len(results) == 5


class TestSpatialGridQueryPoint:
    """Tests for point queries."""

    def test_query_point_inside_entity(self):
        """Test point query finds entity containing point."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=20.0,
        )
        grid.insert(entity)

        results = grid.query_point(Vec2(105.0, 100.0))

        assert len(results) == 1
        assert results[0].id == 1

    def test_query_point_outside_entity(self):
        """Test point query returns empty when point is outside."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=20.0,
        )
        grid.insert(entity)

        results = grid.query_point(Vec2(200.0, 200.0))

        assert len(results) == 0

    def test_query_point_filters_by_type(self):
        """Test point query can filter by type."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        food = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=30.0,
        )
        vegetation = DummyEntity(
            entity_id=2,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.VEGETATION,
            radius=30.0,
        )
        grid.insert(food)
        grid.insert(vegetation)

        results = grid.query_point(Vec2(100.0, 100.0), entity_type=EntityType.FOOD)

        assert len(results) == 1
        assert results[0].entity_type == EntityType.FOOD


class TestSpatialGridQueryAABB:
    """Tests for AABB queries."""

    def test_query_aabb_finds_entity(self):
        """Test AABB query finds intersecting entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=20.0,
        )
        grid.insert(entity)

        query_box = AABB(90.0, 90.0, 150.0, 150.0)
        results = grid.query_aabb(query_box)

        assert len(results) == 1
        assert results[0].id == 1

    def test_query_aabb_excludes_non_intersecting(self):
        """Test AABB query excludes non-intersecting entity."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(100.0, 100.0),
            entity_type=EntityType.FOOD,
            radius=20.0,
        )
        grid.insert(entity)

        query_box = AABB(500.0, 500.0, 600.0, 600.0)
        results = grid.query_aabb(query_box)

        assert len(results) == 0


class TestSpatialGridPotentialCollisions:
    """Tests for potential collision queries."""

    def test_get_potential_collisions(self):
        """Test getting potential collisions."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=100.0)

        entity1 = DummyEntity(
            entity_id=1,
            position=Vec2(50.0, 50.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity2 = DummyEntity(
            entity_id=2,
            position=Vec2(60.0, 50.0),  # Same cell
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        entity3 = DummyEntity(
            entity_id=3,
            position=Vec2(500.0, 500.0),  # Different cell
            entity_type=EntityType.FOOD,
            radius=10.0,
        )

        grid.insert(entity1)
        grid.insert(entity2)
        grid.insert(entity3)

        potentials = grid.get_potential_collisions(entity1)

        # Should include entity2 but not entity3
        ids = [e.id for e in potentials]
        assert 2 in ids
        assert 3 not in ids
        assert 1 not in ids  # Should not include self

    def test_potential_collisions_excludes_inactive(self):
        """Test potential collisions excludes inactive entities."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity1 = DummyEntity(
            entity_id=1,
            position=Vec2(50.0, 50.0),
            entity_type=EntityType.AGENT,
            radius=10.0,
        )
        entity2 = DummyEntity(
            entity_id=2,
            position=Vec2(60.0, 50.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        entity2.is_active = False

        grid.insert(entity1)
        grid.insert(entity2)

        potentials = grid.get_potential_collisions(entity1)

        assert len(potentials) == 0


class TestSpatialGridEdgeCases:
    """Tests for edge cases."""

    def test_entity_at_boundary(self):
        """Test entity at world boundary."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(0.0, 0.0),
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        grid.insert(entity)

        results = grid.query_radius(Vec2(5.0, 5.0), radius=20.0)

        assert len(results) == 1

    def test_entity_outside_boundary(self):
        """Test entity slightly outside boundary (clamped)."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(-10.0, -10.0),  # Outside bounds
            entity_type=EntityType.FOOD,
            radius=10.0,
        )
        grid.insert(entity)

        # Should still be queryable (cell coords clamped)
        results = grid.query_radius(Vec2(0.0, 0.0), radius=50.0)

        assert len(results) == 1

    def test_very_large_entity(self):
        """Test entity larger than cell size."""
        bounds = AABB(0.0, 0.0, 1000.0, 1000.0)
        grid = SpatialGrid(bounds, cell_size=50.0)

        entity = DummyEntity(
            entity_id=1,
            position=Vec2(200.0, 200.0),
            entity_type=EntityType.VEGETATION,
            radius=100.0,  # Larger than cell
        )
        grid.insert(entity)

        # Should be in many cells
        assert grid.get_entity_count() > 4

        # Should be findable from entity center
        # query_radius checks distance from query center to entity position
        assert len(grid.query_radius(Vec2(200.0, 200.0), radius=10.0)) == 1
        # From nearby point within range (distance ~70, need radius > 70)
        assert len(grid.query_radius(Vec2(150.0, 150.0), radius=100.0)) == 1
