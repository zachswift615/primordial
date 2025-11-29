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
        self.detection_radius = 250.0  # Increased detection range
        self.chase_speed = 120.0  # Faster than agents (was 80)
        self.patrol_speed = 40.0  # Slightly faster patrol
        self.target_entity: Optional[Entity] = None
        self.chase_abandon_distance = 350.0  # Chase further before giving up

        # Combat
        self.damage = 20.0  # Keep original damage
        self.attack_cooldown = 0.0
        self.attack_cooldown_max = 1.0  # Keep original attack speed

        # Sound
        self.growl_intensity = 0.5
        self.growl_frequency = 100.0  # Hz, low growl
        self.is_growling = False

        # Footstep sound (pulsing while patrolling/returning)
        self.footstep_intensity = 0.3
        self.pulse_phase = 0.0  # For footstep pulse modulation

        # Line-of-sight tracking for chase
        self.los_lost_time = 0.0  # Time since LOS was lost
        self.los_grace_period = 1.0  # Seconds to wait before abandoning chase
        self.last_known_target_pos: Optional[Vec2] = None

        # Energy system for population dynamics
        self.energy = 80.0  # Start lower - must hunt to reproduce
        self.max_energy = 200.0
        self.energy_drain_rate = 1.0  # Slower starvation - lasts ~80s
        self.kill_energy_gain = 80.0  # Energy gained per kill
        self.reproduction_threshold = 150.0  # Need 150 energy to reproduce (requires ~1 kill)
        self.reproduction_cost = 70.0  # Cost of reproduction
        self.can_reproduce = False  # Flag set when ready to spawn offspring

    def update(self, world: World, dt: float) -> None:
        """Update predator AI and movement.

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
        # Update energy (drain over time)
        self.energy -= self.energy_drain_rate * dt
        if self.energy <= 0:
            # Starvation death
            self.is_active = False
            return

        # Check if ready to reproduce
        self.can_reproduce = self.energy >= self.reproduction_threshold

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

        # Update growling state (continuous when chasing)
        self.is_growling = self.state == PredatorState.CHASING

        # Update footstep pulse phase based on speed
        speed = self.velocity.magnitude()
        if speed > 0.1:
            # Pulse rate increases with speed: speed/15 Hz
            pulse_rate = speed / 15.0
            self.pulse_phase += pulse_rate * dt * 2 * np.pi
            # Keep phase in reasonable bounds
            if self.pulse_phase > 2 * np.pi:
                self.pulse_phase -= 2 * np.pi

    def _update_patrol(self, world: World, dt: float) -> None:
        """Patrol around patrol_center.

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
        # Check for nearby agents with line-of-sight
        nearby_agents = world.get_entities_in_radius(
            self.position,
            self.detection_radius,
            entity_type=EntityType.AGENT,
        )

        # Filter to agents we can actually see (not blocked by vegetation)
        visible_agents = [
            agent for agent in nearby_agents
            if world.has_line_of_sight(self.position, agent.position)
        ]

        if visible_agents:
            # Start chasing closest visible agent
            closest = min(
                visible_agents,
                key=lambda a: a.position.distance_to(self.position),
            )
            self.target_entity = closest
            self.last_known_target_pos = closest.position
            self.los_lost_time = 0.0
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
            self.last_known_target_pos = None
            self.state = PredatorState.RETURNING
            return

        # Abandon chase if too far from patrol center
        distance_to_patrol = self.position.distance_to(self.patrol_center)
        if distance_to_patrol > self.chase_abandon_distance:
            self.target_entity = None
            self.last_known_target_pos = None
            self.state = PredatorState.RETURNING
            return

        # Check line-of-sight to target
        has_los = world.has_line_of_sight(self.position, self.target_entity.position)

        if has_los:
            # Can see target - reset LOS timer and update last known position
            self.los_lost_time = 0.0
            self.last_known_target_pos = self.target_entity.position
        else:
            # Lost line of sight - target is hiding!
            self.los_lost_time += dt
            if self.los_lost_time >= self.los_grace_period:
                # Target has been hidden too long, give up
                self.target_entity = None
                self.last_known_target_pos = None
                self.state = PredatorState.RETURNING
                return

        # Determine where to move: actual target if visible, else last known position
        if has_los:
            move_target = self.target_entity.position
        elif self.last_known_target_pos is not None:
            move_target = self.last_known_target_pos
        else:
            # Shouldn't happen, but fallback
            self.state = PredatorState.RETURNING
            return

        # Move toward target/last known position
        to_target = move_target - self.position
        if to_target.magnitude() > 0.1:
            direction = to_target.normalized()
            force = direction * self.chase_speed * self.mass
            self.apply_force(force)

        # Attack only if we have LOS and in range
        if has_los:
            distance_to_target = self.position.distance_to(self.target_entity.position)
            attack_range = self.radius + self.target_entity.radius
            if distance_to_target < attack_range and self.attack_cooldown <= 0:
                self._attack(self.target_entity, world)

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

    def _attack(self, target: Entity, world: World) -> None:
        """Attack target entity.

        Args:
            target: Entity to attack.
            world: World instance for group protection check.
        """
        if hasattr(target, "take_damage"):
            # Pass world so target can check for group protection
            target.take_damage(self.damage, world)

            # Check if this attack killed the target
            if hasattr(target, "is_alive") and not target.is_alive:
                self.on_kill()

        self.attack_cooldown = self.attack_cooldown_max

    def on_kill(self) -> None:
        """Called when predator successfully kills an agent."""
        self.energy = min(self.max_energy, self.energy + self.kill_energy_gain)

    def try_reproduce(self) -> bool:
        """Attempt reproduction if energy is high enough.

        Returns:
            True if reproduction successful (caller should spawn offspring).
        """
        if self.energy >= self.reproduction_threshold:
            self.energy -= self.reproduction_cost
            self.can_reproduce = False
            return True
        return False

    def get_collision_shape(self) -> Circle:
        """Return collision shape."""
        return Circle(self.position, self.radius)

    def get_sound_properties(self) -> tuple[float, float] | None:
        """Return sound properties for current state.

        When chasing: continuous growl at full intensity.
        When patrolling/returning: pulsing footsteps at lower intensity.

        Returns:
            Tuple of (intensity, frequency) or None if silent.
        """
        if self.is_growling:
            # Chasing: continuous growl
            return (self.growl_intensity, self.growl_frequency)

        # Patrolling/returning: pulsing footsteps based on movement
        speed = self.velocity.magnitude()
        if speed > 0.1:
            # Pulse intensity based on phase: 0.5 + 0.5 * sin(phase)
            pulse_mod = 0.5 + 0.5 * np.sin(self.pulse_phase)
            intensity = self.footstep_intensity * pulse_mod
            # Same frequency as growl but pulsing and quieter
            return (intensity, self.growl_frequency)

        return None

    def set_patrol_center(self, center: Vec2) -> None:
        """Update patrol center.

        Args:
            center: New patrol center position.
        """
        self.patrol_center = center
        self.patrol_target = self._generate_patrol_target()
