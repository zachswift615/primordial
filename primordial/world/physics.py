"""Physics engine for the world system.

Provides motion integration, collision detection, and boundary enforcement.
Uses Semi-Implicit Euler integration for stability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from primordial.world.geometry import AABB, Vec2

if TYPE_CHECKING:
    from primordial.world.entities.base import Entity


class PhysicsEngine:
    """Handles physics integration and collision detection.

    Uses Semi-Implicit Euler integration:
    1. v(t+dt) = v(t) + a(t) * dt
    2. v(t+dt) = v(t+dt) * friction
    3. p(t+dt) = p(t) + v(t+dt) * dt
    4. a(t+dt) = 0

    Attributes:
        world_bounds: The boundaries of the world.
        restitution: Collision restitution coefficient (bounciness).
    """

    def __init__(self, world_bounds: AABB, restitution: float = 0.8) -> None:
        """Initialize physics engine.

        Args:
            world_bounds: Axis-aligned bounding box defining world limits.
            restitution: Coefficient of restitution for collisions (0-1).
        """
        self.world_bounds = world_bounds
        self.restitution = restitution

    def step(self, entities: List[Entity], dt: float) -> None:
        """Perform one physics step.

        1. Apply forces and integrate motion for dynamic entities
        2. Detect and resolve collisions
        3. Enforce world boundaries

        Args:
            entities: All entities in the world.
            dt: Time step in seconds.
        """
        # 1. Integrate motion for all dynamic entities
        for entity in entities:
            if not entity.is_static and entity.is_active:
                self._integrate_motion(entity, dt)

        # 2. Detect and resolve collisions
        self._handle_collisions(entities)

        # 3. Enforce world boundaries
        for entity in entities:
            if not entity.is_static and entity.is_active:
                self._enforce_boundaries(entity)

    def _integrate_motion(self, entity: Entity, dt: float) -> None:
        """Semi-implicit Euler integration.

        Updates velocity from acceleration, applies friction,
        then updates position from velocity.

        Args:
            entity: Entity to update.
            dt: Time step in seconds.
        """
        # v = v + a * dt
        entity.velocity = entity.velocity + entity.acceleration * dt

        # Apply friction (velocity damping)
        entity.velocity = entity.velocity * entity.friction

        # p = p + v * dt
        entity.position = entity.position + entity.velocity * dt

        # Reset acceleration for next frame
        entity.acceleration = Vec2(0.0, 0.0)

    def _handle_collisions(self, entities: List[Entity]) -> None:
        """Detect and resolve collisions between entities.

        Uses O(n^2) brute force approach. For better performance with
        many entities, use SpatialGrid for broad-phase detection.

        Args:
            entities: All entities to check for collisions.
        """
        active_entities = [e for e in entities if e.is_active]

        for i, entity_a in enumerate(active_entities):
            for entity_b in active_entities[i + 1:]:
                if self._check_collision(entity_a, entity_b):
                    self._resolve_collision(entity_a, entity_b)

    def _check_collision(self, a: Entity, b: Entity) -> bool:
        """Check if two entities are colliding.

        Uses circle-circle intersection based on entity collision shapes.

        Args:
            a: First entity.
            b: Second entity.

        Returns:
            True if entities are colliding.
        """
        shape_a = a.get_collision_shape()
        shape_b = b.get_collision_shape()
        return shape_a.intersects(shape_b)

    def _resolve_collision(self, a: Entity, b: Entity) -> None:
        """Resolve collision between two entities.

        Separates overlapping entities and applies impulse-based
        velocity changes for dynamic entities.

        Args:
            a: First entity.
            b: Second entity.
        """
        # Calculate overlap
        distance = a.position.distance_to(b.position)
        min_distance = a.radius + b.radius

        if distance == 0:
            # Entities exactly overlapping, push apart arbitrarily
            direction = Vec2(1.0, 0.0)
            overlap = min_distance
        else:
            direction = (a.position - b.position).normalized()
            overlap = min_distance - distance

        # Separate entities based on whether they're static
        if a.is_static and b.is_static:
            return  # Both static, no resolution needed
        elif a.is_static:
            # Only move b
            b.position = b.position - direction * overlap
        elif b.is_static:
            # Only move a
            a.position = a.position + direction * overlap
        else:
            # Move both (half each)
            a.position = a.position + direction * (overlap / 2)
            b.position = b.position - direction * (overlap / 2)

            # Apply collision impulse (elastic collision)
            relative_velocity = a.velocity - b.velocity
            impulse_magnitude = relative_velocity.dot(direction)

            if impulse_magnitude > 0:
                # Objects moving apart, no impulse needed
                return

            # Apply impulse with restitution
            impulse = direction * impulse_magnitude * self.restitution
            a.velocity = a.velocity - impulse
            b.velocity = b.velocity + impulse

    def _enforce_boundaries(self, entity: Entity) -> None:
        """Keep entity within world bounds.

        Bounces entity off walls with damping.

        Args:
            entity: Entity to constrain.
        """
        damping = 0.5  # Energy loss on boundary collision

        # X boundaries
        if entity.position.x - entity.radius < self.world_bounds.min_x:
            entity.position = Vec2(
                self.world_bounds.min_x + entity.radius,
                entity.position.y,
            )
            entity.velocity = Vec2(
                abs(entity.velocity.x) * damping,
                entity.velocity.y,
            )
        elif entity.position.x + entity.radius > self.world_bounds.max_x:
            entity.position = Vec2(
                self.world_bounds.max_x - entity.radius,
                entity.position.y,
            )
            entity.velocity = Vec2(
                -abs(entity.velocity.x) * damping,
                entity.velocity.y,
            )

        # Y boundaries
        if entity.position.y - entity.radius < self.world_bounds.min_y:
            entity.position = Vec2(
                entity.position.x,
                self.world_bounds.min_y + entity.radius,
            )
            entity.velocity = Vec2(
                entity.velocity.x,
                abs(entity.velocity.y) * damping,
            )
        elif entity.position.y + entity.radius > self.world_bounds.max_y:
            entity.position = Vec2(
                entity.position.x,
                self.world_bounds.max_y - entity.radius,
            )
            entity.velocity = Vec2(
                entity.velocity.x,
                -abs(entity.velocity.y) * damping,
            )

    def raycast(
        self,
        origin: Vec2,
        direction: Vec2,
        max_distance: float,
        entities: List[Entity],
        ignore_entity_id: int | None = None,
    ) -> Tuple[Entity | None, float]:
        """Cast a ray and find the closest entity hit.

        Args:
            origin: Ray origin point.
            direction: Normalized ray direction.
            max_distance: Maximum ray distance.
            entities: Entities to check against.
            ignore_entity_id: Optional entity ID to ignore (e.g., self).

        Returns:
            Tuple of (hit_entity, distance) or (None, max_distance) if no hit.
        """
        closest_entity = None
        closest_distance = max_distance

        for entity in entities:
            if not entity.is_active:
                continue
            if ignore_entity_id is not None and entity.id == ignore_entity_id:
                continue

            # Ray-circle intersection
            hit_distance = self._ray_circle_intersection(
                origin, direction, entity.position, entity.radius
            )

            if hit_distance is not None and hit_distance < closest_distance:
                closest_distance = hit_distance
                closest_entity = entity

        return closest_entity, closest_distance

    def _ray_circle_intersection(
        self,
        origin: Vec2,
        direction: Vec2,
        circle_center: Vec2,
        circle_radius: float,
    ) -> float | None:
        """Calculate ray-circle intersection distance.

        Args:
            origin: Ray origin.
            direction: Normalized ray direction.
            circle_center: Center of the circle.
            circle_radius: Radius of the circle.

        Returns:
            Distance to intersection or None if no intersection.
        """
        # Vector from ray origin to circle center
        to_center = circle_center - origin

        # Project onto ray direction
        t_closest = to_center.dot(direction)

        if t_closest < 0:
            # Circle is behind ray
            return None

        # Find closest point on ray to circle center
        closest_point = origin + direction * t_closest

        # Distance from closest point to circle center
        dist_sq = closest_point.distance_squared_to(circle_center)
        radius_sq = circle_radius * circle_radius

        if dist_sq > radius_sq:
            # Ray misses circle
            return None

        # Calculate intersection distance
        half_chord = (radius_sq - dist_sq) ** 0.5
        t_hit = t_closest - half_chord

        if t_hit < 0:
            # Origin is inside circle
            return 0.0

        return t_hit
