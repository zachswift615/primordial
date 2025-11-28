"""Predator entity for the world system.

Predators patrol areas and chase agents that come within detection range.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

from primordial.world.entities.base import Entity, EntityType
from primordial.world.geometry import Circle, Vec2

if TYPE_CHECKING:
    from primordial.world.world import World


class PredatorState(Enum):
    """Predator AI states."""

    PATROLLING = "patrolling"
    CHASING = "chasing"
    RETURNING = "returning"


class Predator(Entity):
    """Predator that patrols and chases agents.

    Predators have a state machine with three states:
    - PATROLLING: Wander around patrol_center within patrol_radius
    - CHASING: Pursue detected agent until caught or too far from patrol area
    - RETURNING: Return to patrol area after abandoning chase

    Attributes:
        state: Current AI state.
        patrol_center: Center of patrol area.
        patrol_radius: Radius of patrol area.
        patrol_target: Current target point for patrol.
        detection_radius: Range at which agents are detected.
        chase_speed: Movement speed when chasing.
        patrol_speed: Movement speed when patrolling.
        target_entity: Currently targeted entity (when chasing).
        chase_abandon_distance: Distance from patrol center to abandon chase.
        damage: Damage dealt per attack.
        attack_cooldown: Current cooldown timer.
        attack_cooldown_max: Time between attacks.
        growl_intensity: Volume of growl sound.
        growl_frequency: Frequency of growl sound in Hz.
        is_growling: Whether currently growling (while chasing).
    """

    def __init__(
        self,
        entity_id: int,
        position: Vec2,
        patrol_center: Vec2,
        patrol_radius: float = 150.0,
    ) -> None:
        """Initialize predator.

        Args:
            entity_id: Unique identifier.
            position: Initial position in world coordinates.
            patrol_center: Center of patrol area.
            patrol_radius: Radius of patrol area (default 150.0).
        """
        super().__init__(
            entity_id=entity_id,
            position=position,
            entity_type=EntityType.PREDATOR,
            radius=15.0,
            is_static=False,
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

    def update(self, world: World, dt: float) -> None:
        """Update predator AI and movement.

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
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
        self.is_growling = self.state == PredatorState.CHASING

    def _update_patrol(self, world: World, dt: float) -> None:
        """Patrol around patrol_center.

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
        # Check for nearby agents
        nearby_agents = world.get_entities_in_radius(
            self.position,
            self.detection_radius,
            entity_type=EntityType.AGENT,
        )

        if nearby_agents:
            # Start chasing closest agent
            closest = min(
                nearby_agents,
                key=lambda a: a.position.distance_to(self.position),
            )
            self.target_entity = closest
            self.state = PredatorState.CHASING
            return

        # Move toward patrol target
        to_target = self.patrol_target - self.position
        distance = to_target.magnitude()

        if distance > 0.1:
            direction = to_target.normalized()
            force = direction * self.patrol_speed * self.mass
            self.apply_force(force)

        # If reached patrol target, pick new one
        if distance < 10.0:
            self.patrol_target = self._generate_patrol_target()

    def _update_chase(self, world: World, dt: float) -> None:
        """Chase target entity.

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
        if self.target_entity is None or not self.target_entity.is_active:
            self.target_entity = None
            self.state = PredatorState.RETURNING
            return

        distance_to_target = self.position.distance_to(self.target_entity.position)

        # Abandon chase if too far from patrol center
        distance_to_patrol = self.position.distance_to(self.patrol_center)
        if distance_to_patrol > self.chase_abandon_distance:
            self.target_entity = None
            self.state = PredatorState.RETURNING
            return

        # Move toward target
        to_target = self.target_entity.position - self.position
        if to_target.magnitude() > 0.1:
            direction = to_target.normalized()
            force = direction * self.chase_speed * self.mass
            self.apply_force(force)

        # Attack if in range
        attack_range = self.radius + self.target_entity.radius
        if distance_to_target < attack_range and self.attack_cooldown <= 0:
            self._attack(self.target_entity)

    def _update_return(self, world: World, dt: float) -> None:
        """Return to patrol center.

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
        distance_to_patrol = self.position.distance_to(self.patrol_center)

        if distance_to_patrol < self.patrol_radius:
            self.state = PredatorState.PATROLLING
            self.patrol_target = self._generate_patrol_target()
            return

        # Move toward patrol center
        to_center = self.patrol_center - self.position
        if to_center.magnitude() > 0.1:
            direction = to_center.normalized()
            force = direction * self.patrol_speed * self.mass
            self.apply_force(force)

    def _generate_patrol_target(self) -> Vec2:
        """Generate random patrol target within patrol radius.

        Returns:
            Random position within patrol area.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, self.patrol_radius)
        offset = Vec2(np.cos(angle) * radius, np.sin(angle) * radius)
        return self.patrol_center + offset

    def _attack(self, target: Entity) -> None:
        """Attack target entity.

        Args:
            target: Entity to attack.
        """
        if hasattr(target, "take_damage"):
            target.take_damage(self.damage)
        self.attack_cooldown = self.attack_cooldown_max

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)

    def get_sound_properties(self) -> tuple[float, float] | None:
        """Return sound properties if currently growling.

        Returns:
            Tuple of (intensity, frequency) if growling, else None.
        """
        if self.is_growling:
            return (self.growl_intensity, self.growl_frequency)
        return None

    def set_patrol_center(self, center: Vec2) -> None:
        """Update patrol center.

        Args:
            center: New patrol center position.
        """
        self.patrol_center = center
        self.patrol_target = self._generate_patrol_target()
