"""Spatial partitioning grid for efficient collision detection and queries.

Divides the world into cells for O(1) average-case spatial lookups instead
of O(n) brute force.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from primordial.world.geometry import AABB, Vec2

if TYPE_CHECKING:
    from primordial.world.entities.base import Entity, EntityType


class SpatialGrid:
    """Spatial partitioning grid for efficient collision detection and queries.

    Divides the world into a grid of cells. Entities are inserted into all
    cells they overlap, allowing fast spatial queries by only checking
    entities in nearby cells.

    Attributes:
        bounds: World boundaries.
        cell_size: Size of each grid cell.
        cols: Number of columns in the grid.
        rows: Number of rows in the grid.
        grid: Dictionary mapping cell coordinates to entity lists.
    """

    def __init__(self, bounds: AABB, cell_size: float = 100.0) -> None:
        """Initialize spatial grid.

        Args:
            bounds: World boundaries (AABB).
            cell_size: Size of each grid cell (default 100.0).
        """
        self.bounds = bounds
        self.cell_size = cell_size
        self.cols = int((bounds.max_x - bounds.min_x) / cell_size) + 1
        self.rows = int((bounds.max_y - bounds.min_y) / cell_size) + 1
        self.grid: Dict[Tuple[int, int], List[Entity]] = {}

    def clear(self) -> None:
        """Clear all entities from grid."""
        self.grid.clear()

    def insert(self, entity: Entity) -> None:
        """Insert entity into appropriate grid cells.

        Entities are inserted into all cells they overlap based on their
        collision shape.

        Args:
            entity: Entity to insert.
        """
        shape = entity.get_collision_shape()

        # Get grid cell range that entity spans
        min_cell = self._get_cell(
            Vec2(shape.center.x - shape.radius, shape.center.y - shape.radius)
        )
        max_cell = self._get_cell(
            Vec2(shape.center.x + shape.radius, shape.center.y + shape.radius)
        )

        # Insert into all spanned cells
        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key not in self.grid:
                    self.grid[cell_key] = []
                self.grid[cell_key].append(entity)

    def remove(self, entity: Entity) -> None:
        """Remove entity from all grid cells.

        Args:
            entity: Entity to remove.
        """
        shape = entity.get_collision_shape()

        # Get grid cell range that entity spans
        min_cell = self._get_cell(
            Vec2(shape.center.x - shape.radius, shape.center.y - shape.radius)
        )
        max_cell = self._get_cell(
            Vec2(shape.center.x + shape.radius, shape.center.y + shape.radius)
        )

        # Remove from all spanned cells
        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key in self.grid:
                    self.grid[cell_key] = [
                        e for e in self.grid[cell_key] if e.id != entity.id
                    ]

    def query_radius(
        self,
        center: Vec2,
        radius: float,
        entity_type: Optional[EntityType] = None,
    ) -> List[Entity]:
        """Get all entities within radius of center.

        Args:
            center: Center point of query.
            radius: Search radius.
            entity_type: Optional filter by entity type.

        Returns:
            List of entities within radius.
        """
        # Get grid cells in range
        min_cell = self._get_cell(Vec2(center.x - radius, center.y - radius))
        max_cell = self._get_cell(Vec2(center.x + radius, center.y + radius))

        # Collect candidates from cells (use set to avoid duplicates)
        candidates: Set[int] = set()
        candidate_entities: Dict[int, Entity] = {}

        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key in self.grid:
                    for entity in self.grid[cell_key]:
                        if entity.id not in candidates:
                            candidates.add(entity.id)
                            candidate_entities[entity.id] = entity

        # Filter by actual distance and type
        results = []
        radius_sq = radius * radius

        for entity in candidate_entities.values():
            if not entity.is_active:
                continue

            # Check distance (use squared to avoid sqrt)
            dist_sq = center.distance_squared_to(entity.position)
            if dist_sq <= radius_sq:
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)

        return results

    def query_point(
        self,
        point: Vec2,
        entity_type: Optional[EntityType] = None,
    ) -> List[Entity]:
        """Get all entities containing a point.

        Args:
            point: Point to query.
            entity_type: Optional filter by entity type.

        Returns:
            List of entities containing the point.
        """
        cell = self._get_cell(point)

        if cell not in self.grid:
            return []

        results = []
        for entity in self.grid[cell]:
            if not entity.is_active:
                continue

            # Check if point is inside entity
            shape = entity.get_collision_shape()
            if shape.contains_point(point):
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)

        return results

    def query_aabb(
        self,
        aabb: AABB,
        entity_type: Optional[EntityType] = None,
    ) -> List[Entity]:
        """Get all entities intersecting an AABB.

        Args:
            aabb: Axis-aligned bounding box to query.
            entity_type: Optional filter by entity type.

        Returns:
            List of entities intersecting the AABB.
        """
        min_cell = self._get_cell(Vec2(aabb.min_x, aabb.min_y))
        max_cell = self._get_cell(Vec2(aabb.max_x, aabb.max_y))

        # Collect candidates
        candidates: Set[int] = set()
        candidate_entities: Dict[int, Entity] = {}

        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key in self.grid:
                    for entity in self.grid[cell_key]:
                        if entity.id not in candidates:
                            candidates.add(entity.id)
                            candidate_entities[entity.id] = entity

        # Filter by actual intersection
        results = []
        for entity in candidate_entities.values():
            if not entity.is_active:
                continue

            shape = entity.get_collision_shape()
            if aabb.intersects_circle(shape):
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)

        return results

    def get_potential_collisions(self, entity: Entity) -> List[Entity]:
        """Get entities that might be colliding with the given entity.

        Args:
            entity: Entity to check collisions for.

        Returns:
            List of entities in the same grid cells (potential collisions).
        """
        shape = entity.get_collision_shape()

        min_cell = self._get_cell(
            Vec2(shape.center.x - shape.radius, shape.center.y - shape.radius)
        )
        max_cell = self._get_cell(
            Vec2(shape.center.x + shape.radius, shape.center.y + shape.radius)
        )

        candidates: Set[int] = set()
        results: List[Entity] = []

        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key in self.grid:
                    for other in self.grid[cell_key]:
                        if other.id != entity.id and other.id not in candidates:
                            candidates.add(other.id)
                            if other.is_active:
                                results.append(other)

        return results

    def _get_cell(self, position: Vec2) -> Tuple[int, int]:
        """Convert world position to grid cell coordinates.

        Args:
            position: World position.

        Returns:
            Tuple of (col, row) grid coordinates.
        """
        col = int((position.x - self.bounds.min_x) / self.cell_size)
        row = int((position.y - self.bounds.min_y) / self.cell_size)

        # Clamp to grid bounds
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        return (col, row)

    def get_cell_count(self) -> int:
        """Get number of non-empty cells.

        Returns:
            Number of cells containing entities.
        """
        return len(self.grid)

    def get_entity_count(self) -> int:
        """Get total entity references in grid.

        Note: This may be larger than actual entity count since entities
        can span multiple cells.

        Returns:
            Total entity references across all cells.
        """
        return sum(len(entities) for entities in self.grid.values())
