"""World class - main orchestrator for the 2D physics simulation.

The World class ties together all subsystems:
- PhysicsEngine for movement and collision
- SpatialGrid for efficient spatial queries
- SoundSystem for audio propagation
- Environment for day/night cycles
- Entity management and spawning
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from primordial.world.entities import (
    Entity,
    EntityType,
    Food,
    Predator,
    Vegetation,
    Water,
)
from primordial.world.environment import Environment
from primordial.world.geometry import AABB, Vec2
from primordial.world.physics import PhysicsEngine
from primordial.world.sound import SoundSource, SoundSystem
from primordial.world.spatial_grid import SpatialGrid


class World:
    """Main world simulation class.

    Orchestrates all world systems and manages the simulation loop.
    Handles entity lifecycle, physics, sound, and environmental updates.

    Attributes:
        width: World width in units.
        height: World height in units.
        tick_rate: Simulation ticks per second.
        dt: Time step per tick (1.0 / tick_rate).
        bounds: World boundaries as AABB.
        physics: Physics engine instance.
        sound_system: Sound system instance.
        environment: Environment instance.
        spatial_grid: Spatial partitioning grid.
        entities: Dictionary of all entities by ID.
        next_entity_id: Next available entity ID.
        agents: List of agent entities.
        food_items: List of food entities.
        predators: List of predator entities.
        static_entities: List of static entities (vegetation, water).
        food_spawn_rate: Average food spawns per second.
        max_food: Maximum food items in world.
        food_spawn_timer: Internal timer for food spawning.
    """

    def __init__(
        self,
        width: float = 1000.0,
        height: float = 1000.0,
        tick_rate: int = 60,
    ) -> None:
        """Initialize world.

        Args:
            width: World width in units (default 1000.0).
            height: World height in units (default 1000.0).
            tick_rate: Simulation ticks per second (default 60).
        """
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        self.dt = 1.0 / tick_rate

        # World bounds
        self.bounds = AABB(0.0, 0.0, width, height)

        # Initialize subsystems
        self.physics = PhysicsEngine(self.bounds)
        self.sound_system = SoundSystem()
        self.environment = Environment()
        self.spatial_grid = SpatialGrid(self.bounds, cell_size=100.0)

        # Entity storage
        self.entities: Dict[int, Entity] = {}
        self.next_entity_id = 0

        # Entity tracking by type
        self.agents: List[Entity] = []
        self.food_items: List[Food] = []
        self.predators: List[Predator] = []
        self.static_entities: List[Entity] = []

        # Food spawning configuration with natural variation
        self.base_food_spawn_rate = 2.0  # Much higher rate (was 0.5)
        self.food_spawn_rate_variation = 0.4  # How much rate varies (0-1)
        self.food_spawn_cycle_period = 60.0  # Seconds for full abundance cycle
        self.max_food = 100  # Much higher cap (was 30)
        self.food_spawn_timer = 0.0
        self.food_cycle_time = 0.0  # Tracks where we are in the cycle

        # Predator population configuration
        self.max_predators = 10  # Cap to prevent runaway growth
        self.min_predators = 1  # Always keep at least one predator

    def add_entity(self, entity: Entity) -> int:
        """Add entity to world.

        Assigns a unique ID to the entity and adds it to appropriate tracking lists.

        Args:
            entity: Entity instance to add.

        Returns:
            Assigned entity ID.
        """
        entity.id = self.next_entity_id
        self.entities[entity.id] = entity
        self.next_entity_id += 1

        # Add to type-specific tracking
        if entity.entity_type == EntityType.AGENT:
            self.agents.append(entity)
        elif entity.entity_type == EntityType.FOOD:
            self.food_items.append(entity)
        elif entity.entity_type == EntityType.PREDATOR:
            self.predators.append(entity)
        elif entity.is_static:
            self.static_entities.append(entity)

        return entity.id

    def remove_entity(self, entity_id: int) -> None:
        """Remove entity from world.

        Removes entity from all tracking structures.

        Args:
            entity_id: ID of entity to remove.
        """
        if entity_id not in self.entities:
            return

        entity = self.entities[entity_id]
        entity.is_active = False

        # Remove from type-specific tracking
        if entity.entity_type == EntityType.AGENT:
            self.agents = [e for e in self.agents if e.id != entity_id]
        elif entity.entity_type == EntityType.FOOD:
            self.food_items = [e for e in self.food_items if e.id != entity_id]
        elif entity.entity_type == EntityType.PREDATOR:
            self.predators = [e for e in self.predators if e.id != entity_id]
        elif entity.is_static:
            self.static_entities = [e for e in self.static_entities if e.id != entity_id]

        del self.entities[entity_id]

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """Get entity by ID.

        Args:
            entity_id: Entity ID to look up.

        Returns:
            Entity instance or None if not found.
        """
        return self.entities.get(entity_id)

    def get_entities_in_radius(
        self,
        position: Vec2,
        radius: float,
        entity_type: Optional[EntityType] = None,
    ) -> List[Entity]:
        """Get all entities within radius of position.

        Uses spatial grid for efficient lookup.

        Args:
            position: Center of search.
            radius: Search radius.
            entity_type: Optional filter by entity type (None = all types).

        Returns:
            List of entities within radius.
        """
        return self.spatial_grid.query_radius(position, radius, entity_type)

    def tick(self, dt: float = None) -> None:
        """Advance simulation by elapsed time.

        Args:
            dt: Elapsed time in seconds. If None, uses fixed timestep (1/tick_rate).

        Main simulation loop that:
        1. Updates environment
        2. Updates food spawning
        3. Updates all active entities
        4. Manages predator population
        5. Runs physics step
        6. Rebuilds spatial grid
        7. Updates sound sources
        """
        # Use provided dt or fall back to fixed timestep
        step_dt = dt if dt is not None else self.dt

        # Cap dt to prevent instability from large time jumps
        step_dt = min(step_dt, 0.1)  # Max 100ms per tick

        # 1. Update environment
        self.environment.update(step_dt)

        # 2. Spawn food
        self._update_food_spawning(step_dt)

        # 3. Update all entities
        for entity in list(self.entities.values()):
            if entity.is_active:
                entity.update(self, step_dt)

        # 4. Manage predator population (reproduction, death, cleanup)
        self._update_predator_population()

        # 5. Physics step
        self.physics.step(list(self.entities.values()), step_dt)

        # 6. Update spatial grid
        self._rebuild_spatial_grid()

        # 7. Update sound system
        self._update_sound_sources()

    def _update_food_spawning(self, dt: float) -> None:
        """Spawn food items at variable rate.

        Uses a timer-based approach with natural variation - food abundance
        cycles like seasons, sometimes plentiful, sometimes scarce.

        Args:
            dt: Elapsed time in seconds.
        """
        self.food_spawn_timer += dt
        self.food_cycle_time += dt

        # Calculate current spawn rate with sinusoidal variation
        # Creates natural "seasons" of abundance and scarcity
        cycle_phase = (self.food_cycle_time / self.food_spawn_cycle_period) * 2 * np.pi
        variation = np.sin(cycle_phase) * self.food_spawn_rate_variation
        current_rate = self.base_food_spawn_rate * (1.0 + variation)

        # Add some random noise for unpredictability
        current_rate *= np.random.uniform(0.8, 1.2)

        # Clamp to reasonable bounds
        current_rate = max(0.1, min(2.0, current_rate))

        # Count only active food items
        active_food_count = sum(1 for f in self.food_items if f.is_active)

        # Check if time to spawn
        if current_rate > 0:
            spawn_interval = 1.0 / current_rate
            while self.food_spawn_timer >= spawn_interval and active_food_count < self.max_food:
                self._spawn_food()
                self.food_spawn_timer -= spawn_interval
                active_food_count += 1

    def _spawn_food(self) -> None:
        """Spawn a food item at random location.

        Creates food with margin from world edges and avoids spawning
        on top of vegetation, water, or other entities.
        """
        margin = 50.0
        max_attempts = 20

        for _ in range(max_attempts):
            x = np.random.uniform(margin, self.width - margin)
            y = np.random.uniform(margin, self.height - margin)
            pos = Vec2(x, y)

            # Check if position overlaps with any vegetation
            overlaps = False
            for veg in self.vegetation:
                if pos.distance_to(veg.position) < veg.radius + 10:  # 10 unit buffer
                    overlaps = True
                    break

            # Check if position overlaps with water or other static entities
            if not overlaps:
                for static in self.static_entities:
                    if hasattr(static, 'radius') and pos.distance_to(static.position) < static.radius + 10:
                        overlaps = True
                        break

            # Check if position overlaps with existing food
            if not overlaps:
                for food_item in self.food_items:
                    if food_item.is_active and pos.distance_to(food_item.position) < 15:
                        overlaps = True
                        break

            # Check if position overlaps with predators
            if not overlaps:
                for pred in self.predators:
                    if pred.is_active and pos.distance_to(pred.position) < pred.radius + 10:
                        overlaps = True
                        break

            if not overlaps:
                food = Food(
                    entity_id=self.next_entity_id,
                    position=pos,
                    energy_value=50.0,
                    sound_intensity=0.1,
                )
                self.add_entity(food)
                return

        # If all attempts failed, spawn anyway (rare edge case)
        x = np.random.uniform(margin, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)
        food = Food(
            entity_id=self.next_entity_id,
            position=Vec2(x, y),
            energy_value=50.0,
            sound_intensity=0.1,
        )
        self.add_entity(food)

    def _update_predator_population(self) -> None:
        """Manage predator population: remove dead, check reproduction.

        Handles:
        - Removing predators that died from starvation
        - Spawning offspring from well-fed predators
        - Enforcing population caps
        """
        # Count active predators and collect dead ones
        active_predators = []
        dead_predators = []

        for predator in self.predators:
            if predator.is_active:
                active_predators.append(predator)
            else:
                dead_predators.append(predator)

        # Remove dead predators from world
        for predator in dead_predators:
            self.remove_entity(predator.id)

        # Check for reproduction (only if below max)
        if len(active_predators) < self.max_predators:
            for predator in active_predators:
                if predator.can_reproduce and predator.try_reproduce():
                    self._spawn_predator_offspring(predator)
                    # Only spawn one per tick to prevent rapid growth
                    break

        # Note: No automatic respawn - predators must reproduce to survive

    def _spawn_predator_offspring(self, parent: Predator) -> None:
        """Spawn a predator offspring near parent.

        Args:
            parent: Parent predator.
        """
        # Spawn near parent with some offset
        offset = Vec2(
            np.random.uniform(-50, 50),
            np.random.uniform(-50, 50)
        )
        new_pos = parent.position + offset

        # Clamp to world bounds
        margin = 50.0
        new_pos = Vec2(
            max(margin, min(self.width - margin, new_pos.x)),
            max(margin, min(self.height - margin, new_pos.y))
        )

        # Create offspring with patrol center near spawn point
        offspring = Predator(
            entity_id=self.next_entity_id,
            position=new_pos,
            patrol_center=new_pos,
            patrol_radius=150.0,
        )
        # Offspring starts with moderate energy
        offspring.energy = 80.0

        self.add_entity(offspring)

    def _spawn_predator_random(self) -> None:
        """Spawn a predator at random location (for minimum population)."""
        margin = 100.0
        pos = Vec2(
            np.random.uniform(margin, self.width - margin),
            np.random.uniform(margin, self.height - margin)
        )

        predator = Predator(
            entity_id=self.next_entity_id,
            position=pos,
            patrol_center=pos,
            patrol_radius=150.0,
        )

        self.add_entity(predator)

    def _rebuild_spatial_grid(self) -> None:
        """Rebuild spatial grid with current entity positions.

        Clears the grid and re-inserts all active entities for
        efficient spatial queries.
        """
        self.spatial_grid.clear()
        for entity in self.entities.values():
            if entity.is_active:
                self.spatial_grid.insert(entity)

    def _update_sound_sources(self) -> None:
        """Update sound system with current sound sources.

        Clears existing sources and adds sounds from:
        - All active food items
        - Water bodies
        - Growling predators (when chasing)
        """
        self.sound_system.clear_sources()

        # Add food sounds
        for food in self.food_items:
            if food.is_active:
                self.sound_system.add_source(
                    SoundSource(
                        position=food.position,
                        frequency=food.sound_frequency,
                        intensity=food.sound_intensity,
                        is_active=True,
                    )
                )

        # Add predator sounds (growls when chasing, footsteps when patrolling)
        for predator in self.predators:
            if predator.is_active:
                sound_props = predator.get_sound_properties()
                if sound_props is not None:
                    intensity, frequency = sound_props
                    self.sound_system.add_source(
                        SoundSource(
                            position=predator.position,
                            frequency=frequency,
                            intensity=intensity,
                            is_active=True,
                        )
                    )

        # Add water sounds
        for entity in self.static_entities:
            if isinstance(entity, Water) and entity.is_active:
                self.sound_system.add_source(
                    SoundSource(
                        position=entity.position,
                        frequency=entity.sound_frequency,
                        intensity=entity.sound_intensity,
                        is_active=True,
                    )
                )

    def setup_default_world(self) -> None:
        """Set up a default world with entities.

        Procedurally generates:
        - 15 vegetation clusters (3-8 plants each)
        - 5 water bodies
        - 3 predators
        - 10 initial food items
        """
        # Add vegetation clusters
        for _ in range(15):
            cluster_center = Vec2(
                np.random.uniform(100, self.width - 100),
                np.random.uniform(100, self.height - 100),
            )
            cluster_size = np.random.randint(3, 8)

            for _ in range(cluster_size):
                offset = Vec2(
                    np.random.uniform(-50, 50),
                    np.random.uniform(-50, 50),
                )
                veg = Vegetation(
                    entity_id=self.next_entity_id,
                    position=cluster_center + offset,
                    radius=np.random.uniform(15, 25),
                )
                self.add_entity(veg)

        # Add water bodies
        for _ in range(5):
            water = Water(
                entity_id=self.next_entity_id,
                position=Vec2(
                    np.random.uniform(100, self.width - 100),
                    np.random.uniform(100, self.height - 100),
                ),
                radius=np.random.uniform(25, 40),
            )
            self.add_entity(water)

        # Add predators
        for _ in range(3):
            patrol_center = Vec2(
                np.random.uniform(200, self.width - 200),
                np.random.uniform(200, self.height - 200),
            )
            predator = Predator(
                entity_id=self.next_entity_id,
                position=patrol_center,
                patrol_center=patrol_center,
                patrol_radius=150.0,
            )
            self.add_entity(predator)

        # Initial food spawn
        for _ in range(10):
            self._spawn_food()

    @property
    def brightness(self) -> float:
        """Current environmental brightness (0.1 to 0.5).

        Returns:
            Current brightness level from environment.
        """
        return self.environment.get_brightness()

    @property
    def vegetation(self) -> List[Vegetation]:
        """Get all vegetation entities.

        Returns:
            List of Vegetation entities.
        """
        return [e for e in self.static_entities if isinstance(e, Vegetation)]

    def has_line_of_sight(self, from_pos: Vec2, to_pos: Vec2) -> bool:
        """Check if there's clear line of sight between two positions.

        Tests if the ray from from_pos to to_pos intersects any vegetation.
        Used for predator detection - agents can hide behind vegetation.

        Args:
            from_pos: Starting position of the line of sight check.
            to_pos: Target position to check visibility to.

        Returns:
            True if line of sight is clear, False if blocked by vegetation.
        """
        for veg in self.vegetation:
            if self._ray_intersects_circle(from_pos, to_pos, veg.position, veg.radius):
                return False
        return True

    def _ray_intersects_circle(
        self,
        ray_start: Vec2,
        ray_end: Vec2,
        circle_center: Vec2,
        circle_radius: float,
    ) -> bool:
        """Check if a ray segment intersects a circle.

        Uses the quadratic formula to find intersection points.

        Args:
            ray_start: Start of ray segment.
            ray_end: End of ray segment.
            circle_center: Center of the circle.
            circle_radius: Radius of the circle.

        Returns:
            True if the ray segment intersects the circle.
        """
        # Direction vector of ray
        dx = ray_end.x - ray_start.x
        dy = ray_end.y - ray_start.y

        # Vector from ray start to circle center
        fx = ray_start.x - circle_center.x
        fy = ray_start.y - circle_center.y

        # Quadratic coefficients: a*t^2 + b*t + c = 0
        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - circle_radius * circle_radius

        discriminant = b * b - 4.0 * a * c

        if discriminant < 0:
            return False

        # Check if intersection is within ray segment [0, 1]
        discriminant = discriminant ** 0.5

        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        # Check if either intersection point is on the segment
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)
