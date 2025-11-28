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

        # Food spawning configuration
        self.food_spawn_rate = 0.5  # Average food items spawned per second
        self.max_food = 30
        self.food_spawn_timer = 0.0

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

    def tick(self) -> None:
        """Advance simulation by one tick (1/60th second).

        Main simulation loop that:
        1. Updates environment
        2. Updates food spawning
        3. Updates all active entities
        4. Runs physics step
        5. Rebuilds spatial grid
        6. Updates sound sources
        """
        # 1. Update environment
        self.environment.update(self.dt)

        # 2. Spawn food
        self._update_food_spawning()

        # 3. Update all entities
        for entity in list(self.entities.values()):
            if entity.is_active:
                entity.update(self, self.dt)

        # 4. Physics step
        self.physics.step(list(self.entities.values()), self.dt)

        # 5. Update spatial grid
        self._rebuild_spatial_grid()

        # 6. Update sound system
        self._update_sound_sources()

    def _update_food_spawning(self) -> None:
        """Spawn food items at configured rate.

        Uses a timer-based approach to spawn food at regular intervals
        up to the maximum food limit.
        """
        self.food_spawn_timer += self.dt

        # Check if time to spawn
        spawn_interval = 1.0 / self.food_spawn_rate
        while self.food_spawn_timer >= spawn_interval and len(self.food_items) < self.max_food:
            self._spawn_food()
            self.food_spawn_timer -= spawn_interval

    def _spawn_food(self) -> None:
        """Spawn a food item at random location.

        Creates food with margin from world edges to avoid spawning
        on boundaries.
        """
        # Random position with margin from edges
        margin = 50.0
        x = np.random.uniform(margin, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)

        food = Food(
            entity_id=self.next_entity_id,
            position=Vec2(x, y),
            energy_value=50.0,
            sound_intensity=0.1,
        )

        self.add_entity(food)

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

        # Add predator growls (only when chasing)
        for predator in self.predators:
            if predator.is_active and predator.is_growling:
                self.sound_system.add_source(
                    SoundSource(
                        position=predator.position,
                        frequency=predator.growl_frequency,
                        intensity=predator.growl_intensity,
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
