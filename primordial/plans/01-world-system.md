# World & Physics System Implementation Plan

## Overview

The World & Physics System provides the foundational simulation environment for the Primordial survival simulation. It manages a continuous 2D space (1000x1000 units) containing multiple entity types (food, predators, vegetation, water), implements realistic physics (movement, collision), handles environmental cycles (day/night), and provides a sound propagation system for agent sensory input.

This system runs at 60 ticks/second and must efficiently handle multiple agents, dozens of food items, several predators, and static environment elements. All coordinates use floating-point precision for smooth, continuous movement.

**Key Design Principles:**
- Separation of concerns: Physics, entities, and sound are modular
- Performance: Spatial partitioning for collision detection
- Extensibility: Easy to add new entity types
- Deterministic: Same inputs produce same outputs (for debugging/replay)

---

## File Structure

```
primordial/
├── world/
│   ├── __init__.py                 # Exports World, Entity types
│   ├── world.py                    # Main World class, tick loop
│   ├── physics.py                  # Physics engine (movement, collision)
│   ├── spatial_grid.py             # Spatial partitioning for performance
│   ├── entities/
│   │   ├── __init__.py             # Entity base class exports
│   │   ├── base.py                 # Abstract Entity class
│   │   ├── food.py                 # Food entity
│   │   ├── predator.py             # Predator entity with AI
│   │   ├── vegetation.py           # Static vegetation obstacles
│   │   └── water.py                # Water barrier entity
│   ├── sound/
│   │   ├── __init__.py             # Sound system exports
│   │   ├── sound_system.py         # Sound propagation and mixing
│   │   └── sound_source.py         # Individual sound sources
│   └── environment.py              # Day/night cycle, boundaries
└── tests/
    └── world/
        ├── test_physics.py         # Physics unit tests
        ├── test_collision.py       # Collision detection tests
        ├── test_predator_ai.py     # Predator behavior tests
        ├── test_sound.py           # Sound propagation tests
        └── test_world.py           # Integration tests
```

---

## Data Structures

### Core Types

```python
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Vec2:
    """2D vector for positions, velocities, etc."""
    x: float
    y: float

    def __add__(self, other: 'Vec2') -> 'Vec2': ...
    def __sub__(self, other: 'Vec2') -> 'Vec2': ...
    def __mul__(self, scalar: float) -> 'Vec2': ...
    def magnitude(self) -> float: ...
    def normalized(self) -> 'Vec2': ...
    def distance_to(self, other: 'Vec2') -> float: ...
    def dot(self, other: 'Vec2') -> float: ...
    def to_numpy(self) -> np.ndarray: ...

@dataclass
class Circle:
    """Circular collision shape."""
    center: Vec2
    radius: float

    def contains_point(self, point: Vec2) -> bool: ...
    def intersects(self, other: 'Circle') -> bool: ...
    def distance_to(self, other: 'Circle') -> float: ...

@dataclass
class AABB:
    """Axis-aligned bounding box for spatial partitioning."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains_point(self, point: Vec2) -> bool: ...
    def intersects(self, other: 'AABB') -> bool: ...
    def contains_circle(self, circle: Circle) -> bool: ...
```

### Entity Base Class

```python
from abc import ABC, abstractmethod
from enum import Enum

class EntityType(Enum):
    """Entity type enumeration."""
    AGENT = "agent"
    FOOD = "food"
    PREDATOR = "predator"
    VEGETATION = "vegetation"
    WATER = "water"

class Entity(ABC):
    """Abstract base class for all world entities."""

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        entity_type: EntityType,
        radius: float,
        is_static: bool = False
    ):
        self.id = entity_id
        self.position = position
        self.type = entity_type
        self.radius = radius
        self.is_static = is_static
        self.is_active = True

        # Physics properties (ignored if is_static)
        self.velocity = Vec2(0.0, 0.0)
        self.acceleration = Vec2(0.0, 0.0)
        self.mass = 1.0
        self.friction = 0.95  # Velocity multiplier per tick

    @abstractmethod
    def update(self, world: 'World', dt: float) -> None:
        """Update entity state. Called each tick."""
        pass

    @abstractmethod
    def get_collision_shape(self) -> Circle:
        """Return collision shape for physics."""
        pass

    def apply_force(self, force: Vec2) -> None:
        """Apply force to entity (F = ma)."""
        if not self.is_static:
            self.acceleration += force / self.mass
```

### Food Entity

```python
class Food(Entity):
    """Food entity that spawns randomly and gives energy."""

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        energy_value: float = 50.0,
        sound_intensity: float = 0.1
    ):
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.FOOD,
            radius=5.0,
            is_static=True
        )
        self.energy_value = energy_value
        self.sound_intensity = sound_intensity
        self.sound_frequency = 200.0  # Hz, subtle ambient sound

    def update(self, world: 'World', dt: float) -> None:
        """Food is static, no update needed."""
        pass

    def get_collision_shape(self) -> Circle:
        return Circle(self.position, self.radius)

    def consume(self) -> float:
        """Called when eaten. Returns energy value."""
        self.is_active = False
        return self.energy_value
```

### Predator Entity

```python
from enum import Enum

class PredatorState(Enum):
    """Predator AI states."""
    PATROLLING = "patrolling"
    CHASING = "chasing"
    RETURNING = "returning"

class Predator(Entity):
    """Predator that patrols and chases agents."""

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        patrol_center: Vec2,
        patrol_radius: float = 150.0
    ):
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.PREDATOR,
            radius=15.0,
            is_static=False
        )
        self.mass = 2.0
        self.friction = 0.90

        # AI properties
        self.state = PredatorState.PATROLLING
        self.patrol_center = patrol_center
        self.patrol_radius = patrol_radius
        self.patrol_target = self._generate_patrol_target()

        # Detection and chase
        self.detection_radius = 200.0
        self.chase_speed = 80.0  # units/second
        self.patrol_speed = 30.0
        self.target_entity: Optional[Entity] = None
        self.chase_abandon_distance = 300.0

        # Combat
        self.damage = 20.0
        self.attack_cooldown = 0.0
        self.attack_cooldown_max = 1.0  # seconds

        # Sound
        self.growl_intensity = 0.5
        self.growl_frequency = 100.0  # Hz, low growl
        self.is_growling = False

    def update(self, world: 'World', dt: float) -> None:
        """Update predator AI and movement."""
        # Update attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt

        # State machine
        if self.state == PredatorState.PATROLLING:
            self._update_patrol(world, dt)
        elif self.state == PredatorState.CHASING:
            self._update_chase(world, dt)
        elif self.state == PredatorState.RETURNING:
            self._update_return(world, dt)

        # Update growling state
        self.is_growling = (self.state == PredatorState.CHASING)

    def _update_patrol(self, world: 'World', dt: float) -> None:
        """Patrol around patrol_center."""
        # Check for nearby agents
        nearby_agents = world.get_entities_in_radius(
            self.position,
            self.detection_radius,
            entity_type=EntityType.AGENT
        )

        if nearby_agents:
            # Start chasing closest agent
            closest = min(nearby_agents, key=lambda a: a.position.distance_to(self.position))
            self.target_entity = closest
            self.state = PredatorState.CHASING
            return

        # Move toward patrol target
        direction = (self.patrol_target - self.position).normalized()
        force = direction * self.patrol_speed * self.mass
        self.apply_force(force)

        # If reached patrol target, pick new one
        if self.position.distance_to(self.patrol_target) < 10.0:
            self.patrol_target = self._generate_patrol_target()

    def _update_chase(self, world: 'World', dt: float) -> None:
        """Chase target entity."""
        if self.target_entity is None or not self.target_entity.is_active:
            self.target_entity = None
            self.state = PredatorState.RETURNING
            return

        distance = self.position.distance_to(self.target_entity.position)

        # Abandon chase if too far from patrol center
        if self.position.distance_to(self.patrol_center) > self.chase_abandon_distance:
            self.target_entity = None
            self.state = PredatorState.RETURNING
            return

        # Move toward target
        direction = (self.target_entity.position - self.position).normalized()
        force = direction * self.chase_speed * self.mass
        self.apply_force(force)

        # Attack if in range
        if distance < (self.radius + self.target_entity.radius) and self.attack_cooldown <= 0:
            self._attack(self.target_entity)

    def _update_return(self, world: 'World', dt: float) -> None:
        """Return to patrol center."""
        if self.position.distance_to(self.patrol_center) < self.patrol_radius:
            self.state = PredatorState.PATROLLING
            self.patrol_target = self._generate_patrol_target()
            return

        # Move toward patrol center
        direction = (self.patrol_center - self.position).normalized()
        force = direction * self.patrol_speed * self.mass
        self.apply_force(force)

    def _generate_patrol_target(self) -> Vec2:
        """Generate random patrol target within patrol radius."""
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, self.patrol_radius)
        offset = Vec2(np.cos(angle) * radius, np.sin(angle) * radius)
        return self.patrol_center + offset

    def _attack(self, target: Entity) -> None:
        """Attack target entity."""
        if hasattr(target, 'take_damage'):
            target.take_damage(self.damage)
        self.attack_cooldown = self.attack_cooldown_max

    def get_collision_shape(self) -> Circle:
        return Circle(self.position, self.radius)
```

### Vegetation Entity

```python
class Vegetation(Entity):
    """Static vegetation that blocks movement and vision."""

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        radius: float = 20.0
    ):
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.VEGETATION,
            radius=radius,
            is_static=True
        )

    def update(self, world: 'World', dt: float) -> None:
        """Vegetation is static."""
        pass

    def get_collision_shape(self) -> Circle:
        return Circle(self.position, self.radius)

    def blocks_vision(self) -> bool:
        """Returns True if this blocks vision rays."""
        return True
```

### Water Entity

```python
class Water(Entity):
    """Water barrier with ambient sound."""

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        radius: float = 30.0
    ):
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.WATER,
            radius=radius,
            is_static=True
        )
        self.sound_intensity = 0.3
        self.sound_frequency = 300.0  # Hz, flowing water sound

    def update(self, world: 'World', dt: float) -> None:
        """Water is static."""
        pass

    def get_collision_shape(self) -> Circle:
        return Circle(self.position, self.radius)
```

### Physics Engine

```python
class PhysicsEngine:
    """Handles physics integration and collision detection."""

    def __init__(self, world_bounds: AABB):
        self.world_bounds = world_bounds
        self.gravity = Vec2(0.0, 0.0)  # No gravity in this simulation

    def step(self, entities: List[Entity], dt: float) -> None:
        """Perform one physics step."""
        # 1. Apply forces and integrate motion
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
        """Semi-implicit Euler integration."""
        # v = v + a * dt
        entity.velocity += entity.acceleration * dt

        # Apply friction
        entity.velocity *= entity.friction

        # p = p + v * dt
        entity.position += entity.velocity * dt

        # Reset acceleration
        entity.acceleration = Vec2(0.0, 0.0)

    def _handle_collisions(self, entities: List[Entity]) -> None:
        """Detect and resolve collisions between entities."""
        active_entities = [e for e in entities if e.is_active]

        for i, entity_a in enumerate(active_entities):
            for entity_b in active_entities[i + 1:]:
                if self._check_collision(entity_a, entity_b):
                    self._resolve_collision(entity_a, entity_b)

    def _check_collision(self, a: Entity, b: Entity) -> bool:
        """Check if two entities are colliding."""
        shape_a = a.get_collision_shape()
        shape_b = b.get_collision_shape()
        return shape_a.intersects(shape_b)

    def _resolve_collision(self, a: Entity, b: Entity) -> None:
        """Resolve collision between two entities."""
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
            b.position -= direction * overlap
        elif b.is_static:
            # Only move a
            a.position += direction * overlap
        else:
            # Move both (half each)
            a.position += direction * (overlap / 2)
            b.position -= direction * (overlap / 2)

            # Apply collision impulse (elastic collision)
            relative_velocity = a.velocity - b.velocity
            impulse_magnitude = relative_velocity.dot(direction)

            if impulse_magnitude > 0:
                # Objects moving apart, no impulse needed
                return

            # Apply impulse
            impulse = direction * impulse_magnitude * 0.8  # 0.8 = restitution
            a.velocity -= impulse
            b.velocity += impulse

    def _enforce_boundaries(self, entity: Entity) -> None:
        """Keep entity within world bounds."""
        # X boundaries
        if entity.position.x - entity.radius < self.world_bounds.min_x:
            entity.position.x = self.world_bounds.min_x + entity.radius
            entity.velocity.x = abs(entity.velocity.x) * 0.5  # Bounce with damping
        elif entity.position.x + entity.radius > self.world_bounds.max_x:
            entity.position.x = self.world_bounds.max_x - entity.radius
            entity.velocity.x = -abs(entity.velocity.x) * 0.5

        # Y boundaries
        if entity.position.y - entity.radius < self.world_bounds.min_y:
            entity.position.y = self.world_bounds.min_y + entity.radius
            entity.velocity.y = abs(entity.velocity.y) * 0.5
        elif entity.position.y + entity.radius > self.world_bounds.max_y:
            entity.position.y = self.world_bounds.max_y - entity.radius
            entity.velocity.y = -abs(entity.velocity.y) * 0.5
```

### Spatial Grid (for performance)

```python
class SpatialGrid:
    """Spatial partitioning grid for efficient collision detection and queries."""

    def __init__(self, bounds: AABB, cell_size: float):
        self.bounds = bounds
        self.cell_size = cell_size
        self.cols = int((bounds.max_x - bounds.min_x) / cell_size) + 1
        self.rows = int((bounds.max_y - bounds.min_y) / cell_size) + 1
        self.grid: Dict[Tuple[int, int], List[Entity]] = {}

    def clear(self) -> None:
        """Clear all entities from grid."""
        self.grid.clear()

    def insert(self, entity: Entity) -> None:
        """Insert entity into appropriate grid cells."""
        shape = entity.get_collision_shape()

        # Get grid cell range that entity spans
        min_cell = self._get_cell(Vec2(shape.center.x - shape.radius, shape.center.y - shape.radius))
        max_cell = self._get_cell(Vec2(shape.center.x + shape.radius, shape.center.y + shape.radius))

        # Insert into all spanned cells
        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key not in self.grid:
                    self.grid[cell_key] = []
                self.grid[cell_key].append(entity)

    def query_radius(
        self,
        center: Vec2,
        radius: float,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """Get all entities within radius of center."""
        # Get grid cells in range
        min_cell = self._get_cell(Vec2(center.x - radius, center.y - radius))
        max_cell = self._get_cell(Vec2(center.x + radius, center.y + radius))

        # Collect candidates from cells
        candidates = set()
        for row in range(min_cell[1], max_cell[1] + 1):
            for col in range(min_cell[0], max_cell[0] + 1):
                cell_key = (col, row)
                if cell_key in self.grid:
                    candidates.update(self.grid[cell_key])

        # Filter by actual distance and type
        results = []
        for entity in candidates:
            if entity.position.distance_to(center) <= radius:
                if entity_type is None or entity.type == entity_type:
                    results.append(entity)

        return results

    def _get_cell(self, position: Vec2) -> Tuple[int, int]:
        """Convert world position to grid cell coordinates."""
        col = int((position.x - self.bounds.min_x) / self.cell_size)
        row = int((position.y - self.bounds.min_y) / self.cell_size)

        # Clamp to grid bounds
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        return (col, row)
```

### Sound System

```python
@dataclass
class SoundSource:
    """A sound source in the world."""
    position: Vec2
    frequency: float  # Hz
    intensity: float  # 0.0 to 1.0
    is_active: bool = True

class SoundSystem:
    """Manages sound propagation and mixing."""

    def __init__(self, attenuation_coefficient: float = 0.002):
        self.sources: List[SoundSource] = []
        self.attenuation_coefficient = attenuation_coefficient

    def clear_sources(self) -> None:
        """Clear all sound sources."""
        self.sources.clear()

    def add_source(self, source: SoundSource) -> None:
        """Add a sound source."""
        self.sources.append(source)

    def compute_sound_at_position(
        self,
        listener_pos: Vec2,
        listener_facing: Vec2
    ) -> Tuple[float, float]:
        """
        Compute stereo sound levels at listener position.

        Returns:
            (left_ear_intensity, right_ear_intensity): Tuple of sound levels
        """
        left_total = 0.0
        right_total = 0.0

        for source in self.sources:
            if not source.is_active:
                continue

            # Calculate distance attenuation
            distance = listener_pos.distance_to(source.position)
            attenuation = np.exp(-self.attenuation_coefficient * distance)
            attenuated_intensity = source.intensity * attenuation

            if attenuated_intensity < 0.001:
                continue  # Too quiet to matter

            # Calculate stereo positioning
            to_source = (source.position - listener_pos).normalized()

            # Right vector (perpendicular to facing)
            right = Vec2(-listener_facing.y, listener_facing.x)

            # Stereo pan: -1 (left) to +1 (right)
            pan = to_source.dot(right)

            # Convert pan to left/right intensities
            # pan = -1: all left, pan = 0: equal, pan = +1: all right
            left_gain = (1.0 - pan) / 2.0
            right_gain = (1.0 + pan) / 2.0

            left_total += attenuated_intensity * left_gain
            right_total += attenuated_intensity * right_gain

        # Clamp to [0, 1]
        left_total = min(1.0, left_total)
        right_total = min(1.0, right_total)

        return (left_total, right_total)

    def get_frequency_spectrum(
        self,
        listener_pos: Vec2,
        num_frequency_bins: int = 32
    ) -> np.ndarray:
        """
        Get frequency spectrum at listener position.

        Returns:
            Array of intensities for each frequency bin (0-2000 Hz range)
        """
        spectrum = np.zeros(num_frequency_bins)
        max_frequency = 2000.0
        bin_width = max_frequency / num_frequency_bins

        for source in self.sources:
            if not source.is_active:
                continue

            # Calculate distance attenuation
            distance = listener_pos.distance_to(source.position)
            attenuation = np.exp(-self.attenuation_coefficient * distance)
            attenuated_intensity = source.intensity * attenuation

            if attenuated_intensity < 0.001:
                continue

            # Find frequency bin
            bin_index = int(source.frequency / bin_width)
            if 0 <= bin_index < num_frequency_bins:
                spectrum[bin_index] += attenuated_intensity

        # Clamp
        spectrum = np.clip(spectrum, 0.0, 1.0)

        return spectrum
```

### Environment

```python
class Environment:
    """Manages environmental properties like day/night cycle."""

    def __init__(self, day_length: float = 120.0):
        """
        Args:
            day_length: Length of full day/night cycle in seconds
        """
        self.day_length = day_length
        self.time_of_day = 0.0  # 0.0 to day_length

    def update(self, dt: float) -> None:
        """Update environment state."""
        self.time_of_day = (self.time_of_day + dt) % self.day_length

    def get_brightness(self) -> float:
        """
        Get current brightness level (0.0 to 1.0).
        0.5 = full day, 0.1 = full night
        """
        # Sinusoidal day/night cycle
        cycle_position = (self.time_of_day / self.day_length) * 2 * np.pi
        brightness = 0.3 + 0.2 * np.sin(cycle_position)  # Range: 0.1 to 0.5
        return brightness

    def is_daytime(self) -> bool:
        """Returns True if currently daytime."""
        return self.get_brightness() > 0.3
```

### World Class

```python
class World:
    """Main world simulation class."""

    def __init__(
        self,
        width: float = 1000.0,
        height: float = 1000.0,
        tick_rate: int = 60
    ):
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        self.dt = 1.0 / tick_rate

        # World bounds
        self.bounds = AABB(0.0, 0.0, width, height)

        # Systems
        self.physics = PhysicsEngine(self.bounds)
        self.sound_system = SoundSystem()
        self.environment = Environment()
        self.spatial_grid = SpatialGrid(self.bounds, cell_size=100.0)

        # Entities
        self.entities: Dict[int, Entity] = {}
        self.next_entity_id = 0

        # Entity tracking by type
        self.agents: List[Entity] = []
        self.food_items: List[Food] = []
        self.predators: List[Predator] = []
        self.static_entities: List[Entity] = []

        # Configuration
        self.food_spawn_rate = 0.5  # Average food items spawned per second
        self.max_food = 30
        self.food_spawn_timer = 0.0

    def add_entity(self, entity: Entity) -> int:
        """Add entity to world. Returns entity ID."""
        entity.id = self.next_entity_id
        self.entities[entity.id] = entity
        self.next_entity_id += 1

        # Add to type-specific tracking
        if entity.type == EntityType.AGENT:
            self.agents.append(entity)
        elif entity.type == EntityType.FOOD:
            self.food_items.append(entity)
        elif entity.type == EntityType.PREDATOR:
            self.predators.append(entity)
        elif entity.is_static:
            self.static_entities.append(entity)

        return entity.id

    def remove_entity(self, entity_id: int) -> None:
        """Remove entity from world."""
        if entity_id not in self.entities:
            return

        entity = self.entities[entity_id]
        entity.is_active = False

        # Remove from type-specific tracking
        if entity.type == EntityType.AGENT:
            self.agents = [e for e in self.agents if e.id != entity_id]
        elif entity.type == EntityType.FOOD:
            self.food_items = [e for e in self.food_items if e.id != entity_id]
        elif entity.type == EntityType.PREDATOR:
            self.predators = [e for e in self.predators if e.id != entity_id]
        elif entity.is_static:
            self.static_entities = [e for e in self.static_entities if e.id != entity_id]

        del self.entities[entity_id]

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_in_radius(
        self,
        position: Vec2,
        radius: float,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """Get all entities within radius of position."""
        return self.spatial_grid.query_radius(position, radius, entity_type)

    def tick(self) -> None:
        """Advance simulation by one tick (1/60th second)."""
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
        """Spawn food items at configured rate."""
        self.food_spawn_timer += self.dt

        # Check if time to spawn
        spawn_interval = 1.0 / self.food_spawn_rate
        while self.food_spawn_timer >= spawn_interval and len(self.food_items) < self.max_food:
            self._spawn_food()
            self.food_spawn_timer -= spawn_interval

    def _spawn_food(self) -> None:
        """Spawn a food item at random location."""
        # Random position with margin from edges
        margin = 50.0
        x = np.random.uniform(margin, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)

        food = Food(
            entity_id=self.next_entity_id,
            position=Vec2(x, y),
            energy_value=50.0,
            sound_intensity=0.1
        )

        self.add_entity(food)

    def _rebuild_spatial_grid(self) -> None:
        """Rebuild spatial grid with current entity positions."""
        self.spatial_grid.clear()
        for entity in self.entities.values():
            if entity.is_active:
                self.spatial_grid.insert(entity)

    def _update_sound_sources(self) -> None:
        """Update sound system with current sound sources."""
        self.sound_system.clear_sources()

        # Add food sounds
        for food in self.food_items:
            if food.is_active:
                self.sound_system.add_source(SoundSource(
                    position=food.position,
                    frequency=food.sound_frequency,
                    intensity=food.sound_intensity,
                    is_active=True
                ))

        # Add predator growls
        for predator in self.predators:
            if predator.is_active and predator.is_growling:
                self.sound_system.add_source(SoundSource(
                    position=predator.position,
                    frequency=predator.growl_frequency,
                    intensity=predator.growl_intensity,
                    is_active=True
                ))

        # Add water sounds
        for entity in self.static_entities:
            if isinstance(entity, Water) and entity.is_active:
                self.sound_system.add_source(SoundSource(
                    position=entity.position,
                    frequency=entity.sound_frequency,
                    intensity=entity.sound_intensity,
                    is_active=True
                ))

    def setup_default_world(self) -> None:
        """Set up a default world with entities."""
        # Add vegetation clusters
        for _ in range(15):
            cluster_center = Vec2(
                np.random.uniform(100, self.width - 100),
                np.random.uniform(100, self.height - 100)
            )
            cluster_size = np.random.randint(3, 8)

            for _ in range(cluster_size):
                offset = Vec2(
                    np.random.uniform(-50, 50),
                    np.random.uniform(-50, 50)
                )
                veg = Vegetation(
                    entity_id=self.next_entity_id,
                    position=cluster_center + offset,
                    radius=np.random.uniform(15, 25)
                )
                self.add_entity(veg)

        # Add water bodies
        for _ in range(5):
            water = Water(
                entity_id=self.next_entity_id,
                position=Vec2(
                    np.random.uniform(100, self.width - 100),
                    np.random.uniform(100, self.height - 100)
                ),
                radius=np.random.uniform(25, 40)
            )
            self.add_entity(water)

        # Add predators
        for _ in range(3):
            patrol_center = Vec2(
                np.random.uniform(200, self.width - 200),
                np.random.uniform(200, self.height - 200)
            )
            predator = Predator(
                entity_id=self.next_entity_id,
                position=patrol_center,
                patrol_center=patrol_center,
                patrol_radius=150.0
            )
            self.add_entity(predator)

        # Initial food spawn
        for _ in range(10):
            self._spawn_food()
```

---

## Core Algorithms

### 1. Physics Integration (Semi-Implicit Euler)

Semi-implicit Euler is more stable than explicit Euler for velocity-based physics:

```
1. Update velocity: v(t+dt) = v(t) + a(t) * dt
2. Apply friction: v(t+dt) = v(t+dt) * friction_coefficient
3. Update position: p(t+dt) = p(t) + v(t+dt) * dt
4. Reset acceleration: a(t+dt) = 0
```

**Why this method:**
- Stable for interactive simulations
- Energy dissipation through friction prevents runaway velocities
- Simple to implement and debug

### 2. Collision Detection & Resolution

**Broad Phase (Spatial Grid):**
```
1. Divide world into grid cells (100x100 units each)
2. For each entity, determine which cells it overlaps
3. Insert entity reference into those cells
4. For queries, check only entities in nearby cells
```

**Narrow Phase (Circle-Circle):**
```
For each pair of entities in same grid cell:
    1. Calculate distance = ||pos_a - pos_b||
    2. If distance < (radius_a + radius_b):
        3. Collision detected
        4. Calculate overlap = (radius_a + radius_b) - distance
        5. Calculate separation_direction = normalize(pos_a - pos_b)
        6. Separate entities based on static/dynamic flags
        7. Apply impulse for dynamic-dynamic collisions
```

**Impulse-Based Resolution:**
```
relative_velocity = v_a - v_b
impulse_magnitude = dot(relative_velocity, separation_direction)
if impulse_magnitude < 0:  # Objects moving toward each other
    impulse = separation_direction * impulse_magnitude * restitution
    v_a -= impulse
    v_b += impulse
```

### 3. Predator AI State Machine

```
State: PATROLLING
    - Move toward random point within patrol_radius of patrol_center
    - When reaching point, generate new random point
    - Continuously check for agents within detection_radius
    - If agent detected -> transition to CHASING

State: CHASING
    - Set target to closest detected agent
    - Move toward target at chase_speed
    - If distance to target < attack_range:
        - Deal damage if attack_cooldown == 0
        - Reset attack_cooldown
    - If distance to patrol_center > chase_abandon_distance:
        - Transition to RETURNING
    - If target becomes inactive:
        - Transition to RETURNING

State: RETURNING
    - Move toward patrol_center at patrol_speed
    - If distance to patrol_center < patrol_radius:
        - Transition to PATROLLING
```

**Steering Behavior:**
```
desired_velocity = normalize(target - position) * max_speed
steering_force = desired_velocity - current_velocity
apply_force(steering_force * mass)
```

### 4. Sound Propagation

**Distance Attenuation:**
```
attenuated_intensity = base_intensity * exp(-attenuation_coefficient * distance)
```
- attenuation_coefficient = 0.002 gives reasonable falloff over 100-500 units
- Exponential decay is physically plausible

**Stereo Positioning:**
```
1. Calculate vector from listener to source: to_source = normalize(source_pos - listener_pos)
2. Calculate listener's right vector: right = perpendicular(listener_facing)
3. Calculate stereo pan: pan = dot(to_source, right)  # Range: -1 (left) to +1 (right)
4. Convert to gains:
   left_gain = (1 - pan) / 2
   right_gain = (1 + pan) / 2
5. Apply gains:
   left_intensity = attenuated_intensity * left_gain
   right_intensity = attenuated_intensity * right_gain
```

**Frequency Spectrum:**
```
For neural network input, provide spectrum across frequency bins:
1. Divide 0-2000 Hz into N bins (e.g., 32 bins)
2. For each sound source:
   - Find bin index = floor(source_frequency / bin_width)
   - Add attenuated_intensity to that bin
3. Clamp bins to [0, 1]
4. Return as NumPy array for neural input
```

---

## API Specification

### World API

```python
class World:
    """Main world simulation interface."""

    def __init__(
        self,
        width: float = 1000.0,
        height: float = 1000.0,
        tick_rate: int = 60
    ) -> None:
        """
        Initialize world.

        Args:
            width: World width in units
            height: World height in units
            tick_rate: Simulation ticks per second
        """
        ...

    def tick(self) -> None:
        """
        Advance simulation by one tick (1/tick_rate seconds).

        Updates all entities, runs physics, handles collisions,
        updates spatial grid, and refreshes sound sources.
        """
        ...

    def add_entity(self, entity: Entity) -> int:
        """
        Add entity to world.

        Args:
            entity: Entity instance to add

        Returns:
            Assigned entity ID
        """
        ...

    def remove_entity(self, entity_id: int) -> None:
        """
        Remove entity from world.

        Args:
            entity_id: ID of entity to remove
        """
        ...

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """
        Get entity by ID.

        Args:
            entity_id: Entity ID to look up

        Returns:
            Entity instance or None if not found
        """
        ...

    def get_entities_in_radius(
        self,
        position: Vec2,
        radius: float,
        entity_type: Optional[EntityType] = None
    ) -> List[Entity]:
        """
        Query entities within radius of position.

        Uses spatial grid for efficient lookup.

        Args:
            position: Center of search
            radius: Search radius
            entity_type: Filter by type (None = all types)

        Returns:
            List of entities within radius
        """
        ...

    def setup_default_world(self) -> None:
        """
        Populate world with default entities:
        - 15 vegetation clusters (3-8 plants each)
        - 5 water bodies
        - 3 predators
        - 10 initial food items
        """
        ...

    @property
    def brightness(self) -> float:
        """Current environmental brightness (0.1 to 0.5)."""
        return self.environment.get_brightness()
```

### Sound System API

```python
class SoundSystem:
    """Sound propagation and mixing."""

    def compute_sound_at_position(
        self,
        listener_pos: Vec2,
        listener_facing: Vec2
    ) -> Tuple[float, float]:
        """
        Compute stereo sound at listener position.

        Mixes all active sound sources with distance attenuation
        and stereo positioning based on listener facing direction.

        Args:
            listener_pos: Listener position in world
            listener_facing: Unit vector of listener facing direction

        Returns:
            (left_ear_intensity, right_ear_intensity): Each in [0, 1]
        """
        ...

    def get_frequency_spectrum(
        self,
        listener_pos: Vec2,
        num_frequency_bins: int = 32
    ) -> np.ndarray:
        """
        Get frequency spectrum at listener position.

        Useful for neural network input. Bins cover 0-2000 Hz.

        Args:
            listener_pos: Listener position in world
            num_frequency_bins: Number of frequency bins

        Returns:
            Array of shape (num_frequency_bins,) with intensities
        """
        ...
```

### Physics API

```python
class PhysicsEngine:
    """Physics simulation."""

    def step(self, entities: List[Entity], dt: float) -> None:
        """
        Perform one physics step.

        1. Integrate motion for all dynamic entities
        2. Detect and resolve collisions
        3. Enforce world boundaries

        Args:
            entities: All entities in world
            dt: Time step in seconds
        """
        ...
```

### Entity API

```python
class Entity(ABC):
    """Base entity interface."""

    def apply_force(self, force: Vec2) -> None:
        """
        Apply force to entity.

        For dynamic entities, adds to acceleration (F = ma).
        Static entities ignore forces.

        Args:
            force: Force vector in world units
        """
        ...

    @abstractmethod
    def update(self, world: 'World', dt: float) -> None:
        """
        Update entity state.

        Called once per tick before physics step. Implement
        entity-specific behavior (AI, state updates, etc.).

        Args:
            world: World instance for queries
            dt: Time step in seconds
        """
        ...

    @abstractmethod
    def get_collision_shape(self) -> Circle:
        """
        Get collision shape for physics.

        Returns:
            Circle representing entity's collision boundary
        """
        ...
```

---

## Integration Points

### Agent Sensory Input

The world system provides inputs for agent senses:

**1. Vision Input:**
```python
def get_vision_input(
    agent_position: Vec2,
    agent_facing: Vec2,
    vision_range: float,
    vision_fov: float,
    vision_rays: int
) -> np.ndarray:
    """
    Cast vision rays from agent.

    Returns array of (distance, entity_type) tuples for each ray.
    Checks for vegetation blocking line of sight.
    """
    ...
```

**Implementation approach:**
- Cast N rays in arc defined by FOV
- For each ray, iterate through distance checking for collisions
- Return nearest entity hit (or max range if nothing)
- Vegetation blocks rays (early termination)

**2. Sound Input:**
```python
def get_sound_input(agent_position: Vec2, agent_facing: Vec2) -> Dict[str, float]:
    """
    Get sound sensory data.

    Returns:
        {
            'left_ear': float,      # 0-1
            'right_ear': float,     # 0-1
            'spectrum': np.ndarray  # (32,) frequency bins
        }
    """
    left, right = world.sound_system.compute_sound_at_position(
        agent_position, agent_facing
    )
    spectrum = world.sound_system.get_frequency_spectrum(agent_position)

    return {
        'left_ear': left,
        'right_ear': right,
        'spectrum': spectrum
    }
```

**3. Proximity/Touch Input:**
```python
def get_nearby_entities(agent_position: Vec2, radius: float) -> List[Entity]:
    """Get entities within radius (for collision detection)."""
    return world.get_entities_in_radius(agent_position, radius)
```

**4. Internal State:**
```python
# Agent tracks its own state
agent.energy  # Decreases over time, increased by eating food
agent.health  # Decreased by predator attacks
agent.velocity  # For proprioception
```

### Renderer Integration

The renderer needs to visualize world state:

```python
def get_render_data() -> Dict[str, Any]:
    """
    Get data for rendering.

    Returns:
        {
            'entities': List[Dict],  # All active entities with pos, type, radius
            'brightness': float,      # Current environmental brightness
            'world_bounds': AABB,     # World boundaries
            'sound_sources': List[Dict]  # Active sound sources (for debug vis)
        }
    """
    return {
        'entities': [
            {
                'id': e.id,
                'type': e.type.value,
                'position': (e.position.x, e.position.y),
                'radius': e.radius,
                'velocity': (e.velocity.x, e.velocity.y) if not e.is_static else None,
                'is_static': e.is_static
            }
            for e in world.entities.values() if e.is_active
        ],
        'brightness': world.environment.get_brightness(),
        'world_bounds': world.bounds,
        'sound_sources': [
            {
                'position': (s.position.x, s.position.y),
                'intensity': s.intensity,
                'frequency': s.frequency
            }
            for s in world.sound_system.sources if s.is_active
        ]
    }
```

Renderer uses this data to draw:
- Circles for entities (color-coded by type)
- Vegetation (green)
- Water (blue with wave effect)
- Food (yellow)
- Predators (red, larger when growling)
- Agents (color based on energy/health)
- Background brightness based on day/night

---

## Testing Strategy

### Unit Tests

**1. Physics Tests (`test_physics.py`):**
```python
def test_motion_integration():
    """Verify velocity and position update correctly."""

def test_friction_application():
    """Verify friction slows entities over time."""

def test_boundary_enforcement():
    """Verify entities bounce at world edges."""

def test_force_application():
    """Verify F=ma correctly updates acceleration."""
```

**2. Collision Tests (`test_collision.py`):**
```python
def test_circle_circle_collision():
    """Verify circle intersection detection."""

def test_collision_separation():
    """Verify entities pushed apart when overlapping."""

def test_static_dynamic_collision():
    """Verify static entities don't move."""

def test_collision_impulse():
    """Verify velocity changes after collision."""
```

**3. Predator AI Tests (`test_predator_ai.py`):**
```python
def test_patrol_behavior():
    """Verify predator moves to patrol points."""

def test_agent_detection():
    """Verify predator detects nearby agents."""

def test_chase_behavior():
    """Verify predator chases target."""

def test_chase_abandonment():
    """Verify predator returns when too far from patrol."""

def test_attack_timing():
    """Verify attack cooldown prevents rapid attacks."""
```

**4. Sound Tests (`test_sound.py`):**
```python
def test_distance_attenuation():
    """Verify sound decreases with distance."""

def test_stereo_positioning():
    """Verify left/right ear differences based on facing."""

def test_frequency_spectrum():
    """Verify frequency binning works correctly."""

def test_multiple_sources():
    """Verify multiple sounds mix correctly."""
```

### Integration Tests

**5. World Tests (`test_world.py`):**
```python
def test_world_tick():
    """Verify full tick completes without errors."""

def test_food_spawning():
    """Verify food spawns at configured rate."""

def test_entity_lifecycle():
    """Verify add/remove/query entity operations."""

def test_spatial_grid_queries():
    """Verify spatial queries return correct entities."""

def test_default_world_setup():
    """Verify default world creates expected entities."""
```

### Performance Tests

```python
def test_60_fps_performance():
    """Verify world.tick() completes in < 16.67ms."""
    world = World()
    world.setup_default_world()

    # Add 10 agents
    for _ in range(10):
        agent = Agent(...)
        world.add_entity(agent)

    # Measure 1000 ticks
    start = time.perf_counter()
    for _ in range(1000):
        world.tick()
    elapsed = time.perf_counter() - start

    avg_tick_time = elapsed / 1000
    assert avg_tick_time < 0.01667  # 60 FPS = 16.67ms per tick
```

### Manual Testing Scenarios

1. **Predator Chase:** Place agent near predator, verify chase and attack
2. **Sound Direction:** Move agent around sound source, verify stereo changes
3. **Food Consumption:** Agent eats food, verify energy increase and food removal
4. **Boundary Behavior:** Drive entity into wall, verify bounce
5. **Day/Night Cycle:** Run for 2 minutes, verify brightness changes

---

## Implementation Order

### Phase 1: Core Foundation (Day 1)

1. **Create file structure** (30 min)
   - Create all directories and `__init__.py` files
   - Set up imports

2. **Implement Vec2 and geometry primitives** (1 hour)
   - `Vec2` class with all operations
   - `Circle` and `AABB` classes
   - Unit tests for geometry

3. **Implement Entity base class** (30 min)
   - Abstract `Entity` class
   - Basic properties (position, velocity, etc.)

4. **Implement PhysicsEngine** (2 hours)
   - Motion integration
   - Boundary enforcement
   - Unit tests for physics

### Phase 2: Entities (Day 2)

5. **Implement static entities** (1 hour)
   - `Vegetation` class
   - `Water` class
   - Test instantiation

6. **Implement Food entity** (30 min)
   - `Food` class with consumption logic
   - Test food lifecycle

7. **Implement Predator entity** (3 hours)
   - Predator class skeleton
   - State machine (patrol, chase, return)
   - Steering behaviors
   - Attack logic
   - Unit tests for AI states

### Phase 3: Collision & Spatial (Day 3)

8. **Implement collision detection** (2 hours)
   - Circle-circle intersection
   - Collision resolution (separation + impulse)
   - Tests for various collision scenarios

9. **Implement SpatialGrid** (2 hours)
   - Grid structure
   - Insert/query operations
   - Test query correctness and performance

10. **Integrate collision into PhysicsEngine** (1 hour)
    - Broad phase using spatial grid
    - Narrow phase collision checks
    - Integration tests

### Phase 4: Sound System (Day 4)

11. **Implement SoundSource and SoundSystem** (2 hours)
    - Sound source data structure
    - Distance attenuation
    - Stereo positioning
    - Unit tests for sound calculations

12. **Implement frequency spectrum** (1 hour)
    - Frequency binning
    - Multi-source mixing
    - Tests for spectrum accuracy

### Phase 5: World Integration (Day 5)

13. **Implement Environment** (1 hour)
    - Day/night cycle
    - Brightness calculation
    - Test cycle timing

14. **Implement World class** (3 hours)
    - Entity management (add/remove/query)
    - Tick loop integration
    - Food spawning logic
    - Sound source updates
    - Integration tests

15. **Implement setup_default_world** (1 hour)
    - Procedural entity placement
    - Test default world creation

### Phase 6: Testing & Polish (Day 6)

16. **Write comprehensive tests** (3 hours)
    - Complete unit test coverage
    - Integration test scenarios
    - Performance benchmarking

17. **Performance optimization** (2 hours)
    - Profile tick loop
    - Optimize spatial grid cell size
    - Reduce allocation overhead

18. **Documentation** (1 hour)
    - Add docstrings to all public APIs
    - Create usage examples
    - Document configuration parameters

### Phase 7: Integration Preparation (Day 7)

19. **Create integration helpers** (2 hours)
    - `get_vision_input()` helper for agent vision
    - `get_sound_input()` helper for agent hearing
    - `get_render_data()` helper for renderer

20. **End-to-end testing** (2 hours)
    - Manual testing of all systems
    - Verify 60 FPS performance
    - Test with multiple agents

---

## Performance Considerations

### Target Performance

- **60 FPS** = 16.67ms per tick maximum
- **Budget allocation:**
  - Entity updates: 5ms
  - Physics integration: 3ms
  - Collision detection: 5ms
  - Spatial grid rebuild: 2ms
  - Sound updates: 1ms
  - Buffer: 0.67ms

### Optimization Strategies

1. **Spatial Grid Cell Size:**
   - Too small: Overhead of many cells
   - Too large: Too many entities per cell
   - Recommend: 100x100 units (10x10 grid for 1000x1000 world)

2. **Collision Detection:**
   - Use spatial grid to check only nearby entities
   - Early exit for static-static pairs
   - Cache collision shapes

3. **Sound System:**
   - Skip sources with intensity < 0.001 after attenuation
   - Limit to ~50 active sound sources max

4. **NumPy Usage:**
   - Use NumPy for batch operations where possible
   - Avoid unnecessary array allocations

5. **Entity Pooling:**
   - Reuse Food entity objects instead of delete/create
   - Mark inactive rather than removing from list

### Profiling Points

Add timing measurements at:
- Start/end of `world.tick()`
- Each major subsystem (physics, collision, sound)
- Individual predator AI updates (if slow)

---

## Configuration Parameters

Expose these as World constructor parameters or config file:

```python
@dataclass
class WorldConfig:
    """World configuration parameters."""

    # World size
    width: float = 1000.0
    height: float = 1000.0
    tick_rate: int = 60

    # Food spawning
    food_spawn_rate: float = 0.5  # per second
    max_food: int = 30
    food_energy_value: float = 50.0
    food_radius: float = 5.0

    # Predators
    num_predators: int = 3
    predator_detection_radius: float = 200.0
    predator_chase_speed: float = 80.0
    predator_patrol_speed: float = 30.0
    predator_damage: float = 20.0

    # Vegetation
    num_vegetation_clusters: int = 15
    vegetation_cluster_size_min: int = 3
    vegetation_cluster_size_max: int = 8
    vegetation_radius_min: float = 15.0
    vegetation_radius_max: float = 25.0

    # Water
    num_water_bodies: int = 5
    water_radius_min: float = 25.0
    water_radius_max: float = 40.0

    # Physics
    friction: float = 0.95
    restitution: float = 0.8

    # Sound
    sound_attenuation: float = 0.002

    # Environment
    day_length: float = 120.0  # seconds

    # Spatial grid
    spatial_grid_cell_size: float = 100.0
```

Usage:
```python
config = WorldConfig(
    num_predators=5,
    predator_chase_speed=100.0
)
world = World(config=config)
```

---

## Future Extensions

### Features Not in Initial Implementation

1. **Agent-Agent Interaction:**
   - Breeding (combine neural networks)
   - Competition for food
   - Social behaviors

2. **Advanced Predator AI:**
   - Pack hunting coordination
   - Learning from failed chases
   - Ambush behaviors

3. **Environmental Hazards:**
   - Poisonous plants
   - Quicksand zones
   - Temperature zones

4. **Resource Scarcity:**
   - Food depletion zones
   - Migration patterns

5. **Sound Occlusion:**
   - Vegetation dampens sound
   - Echo/reverb effects

### API Design for Extensions

Keep these interfaces extensible:

- `Entity.update()` allows arbitrary behavior
- `EntityType` enum easily extended
- `World.add_entity()` accepts any Entity subclass
- Sound system supports arbitrary frequencies/intensities

---

## Dependencies

```
# requirements.txt
numpy>=1.24.0
pygame>=2.5.0
pytest>=7.4.0
pytest-benchmark>=4.0.0
```

---

## Summary

This implementation plan provides:

1. **Modular architecture** - Clean separation between physics, entities, sound
2. **Performance** - Spatial grid ensures 60 FPS even with many entities
3. **Extensibility** - Easy to add new entity types and behaviors
4. **Testability** - Clear unit/integration test boundaries
5. **Integration-ready** - Well-defined APIs for agent senses and rendering

**Total estimated implementation time:** 6-7 days for a single developer, resulting in a robust, performant world simulation ready for agent integration.

**Next steps after completion:**
1. Implement Agent entity with sensory inputs
2. Implement Living Resonance Network (Fourier-based neural architecture)
3. Implement Pygame renderer
4. Connect all systems and run full simulation
