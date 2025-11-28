"""Agent body implementation.

The AgentBody class extends Entity to provide a complete physical
embodiment with sensors, actuators, and survival mechanics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Any

import numpy as np
import torch

from primordial.world.entities import Entity, EntityType
from primordial.world.geometry import Circle, Vec2
from primordial.agents.genome import AgentGenome, create_default_genome
from primordial.agents.actions import AgentAction
from primordial.agents.sensors import (
    VisionSensor,
    AudioSensor,
    ProprioceptionSensor,
    TouchSensor,
)

if TYPE_CHECKING:
    from primordial.world.world import World


class AgentBody(Entity):
    """Physical body of an agent in the simulation.

    Extends Entity with:
    - Genome-defined capabilities
    - Sensory systems (vision, audio, proprioception, touch)
    - Survival mechanics (energy, health)
    - Action application (thrust, torque, eating, vocalization)

    Attributes:
        genome: Heritable parameters defining agent capabilities.
        energy: Current energy level (0 to max_energy).
        health: Current health level (0 to max_health).
        age: Time since birth in seconds.
        angle: Facing direction in radians (0 = right, increases CCW).
        angular_velocity: Current rotation rate in radians/sec.
        is_alive: Whether the agent is alive.
        is_eating: Whether currently consuming food.
        last_damage_time: Age when last damage was received.
        last_action: Most recent action (used for vocalization detection).

        Sensors:
            vision: VisionSensor instance.
            audio: AudioSensor instance.
            proprioception: ProprioceptionSensor instance.
            touch: TouchSensor instance.
    """

    # Observation dimensions (for neural network input)
    VISION_DIM = 32 * 4  # 128 values (32 rays × 4)
    AUDIO_DIM = 2  # stereo
    PROPRIO_DIM = 8  # internal state
    TOUCH_DIM = 8  # 8 directions
    OBSERVATION_DIM = VISION_DIM + AUDIO_DIM + PROPRIO_DIM + TOUCH_DIM  # 146

    # Action dimensions
    ACTION_DIM = 5  # thrust, torque, vocalize_freq, vocalize_amp, eat

    def __init__(
        self,
        agent_id: str,
        genome: Optional[AgentGenome] = None,
        initial_position: Optional[Vec2] = None,
        initial_angle: float = 0.0,
    ) -> None:
        """Initialize agent body with genome and starting state.

        Args:
            agent_id: Unique string identifier for the agent.
            genome: Agent genome (uses default if None).
            initial_position: Starting position (defaults to origin).
            initial_angle: Starting facing angle in radians.
        """
        self.genome = genome if genome is not None else create_default_genome()
        pos = initial_position if initial_position is not None else Vec2(0.0, 0.0)

        # Initialize base Entity
        super().__init__(
            entity_id=0,  # Will be assigned by World
            position=pos,
            entity_type=EntityType.AGENT,
            radius=self.genome.radius,
            is_static=False,
        )

        # Store string agent ID separately (Entity.id is int)
        self.agent_id = agent_id
        self.mass = self.genome.mass

        # Facing direction
        self.angle = initial_angle
        self.angular_velocity = 0.0

        # Survival metrics
        self.energy = self.genome.max_energy
        self.health = self.genome.max_health
        self.age = 0.0

        # Status flags
        self.is_alive = True
        self.is_eating = False
        self.last_damage_time = -1.0
        self.death_cause: Optional[str] = None

        # Most recent action (for vocalization detection by other agents)
        self.last_action: Optional[AgentAction] = None

        # Initialize sensors
        self.vision = VisionSensor(self.genome)
        self.audio = AudioSensor(self.genome)
        self.proprioception = ProprioceptionSensor(self.genome)
        self.touch = TouchSensor(self.genome)

    @property
    def facing(self) -> Vec2:
        """Get facing direction as unit vector.

        Returns:
            Unit vector in the direction the agent is facing.
        """
        return Vec2(math.cos(self.angle), math.sin(self.angle))

    def update(self, world: World, dt: float) -> None:
        """Complete update cycle called by World.tick().

        This is called before physics. The agent should apply its
        last_action here (if any) and update survival mechanics.

        Note: The actual action application is done via apply_action()
        called from external code (the learning loop).

        Args:
            world: World instance for queries.
            dt: Time step in seconds.
        """
        if not self.is_alive:
            return

        # Update age
        self.age += dt

        # Apply drag to angular velocity
        drag_coefficient = 0.95
        self.angular_velocity *= drag_coefficient ** (dt * 60)  # Normalize to 60fps

    def apply_action(self, action: AgentAction, dt: float, world: World) -> None:
        """Apply action and update physics.

        This should be called from the learning loop before world.tick().

        Args:
            action: Action outputs from neural network.
            dt: Time step in seconds.
            world: World instance for eating queries.
        """
        if not self.is_alive:
            return

        self.last_action = action

        # 1. Apply thrust (forward/backward movement)
        if abs(action.thrust) > 0.001:
            thrust_force = action.thrust * self.genome.thrust_force
            direction = self.facing
            # F = ma, so a = F/m
            acceleration = direction * (thrust_force / self.mass)
            self.velocity = self.velocity + (acceleration * dt)

        # 2. Apply torque (rotation)
        if abs(action.torque) > 0.001:
            torque = action.torque * self.genome.torque_force
            # Angular acceleration = torque / moment of inertia
            # For a disk: I = 0.5 * m * r^2
            moment_of_inertia = 0.5 * self.mass * self.radius**2
            angular_accel = torque / moment_of_inertia
            self.angular_velocity += angular_accel * dt

        # 3. Limit velocities
        speed = self.velocity.magnitude()
        if speed > self.genome.max_speed:
            self.velocity = self.velocity * (self.genome.max_speed / speed)

        self.angular_velocity = max(
            -self.genome.max_angular_speed,
            min(self.genome.max_angular_speed, self.angular_velocity),
        )

        # 4. Update angle
        self.angle += self.angular_velocity * dt
        self.angle = self.angle % (2 * math.pi)

        # 5. Handle eating
        if action.eat > 0.1:
            self._try_eat(action.eat, world, dt)
        else:
            self.is_eating = False

        # 6. Update energy (depletion from actions)
        self._update_energy(action, dt)

        # 7. Update health
        self._update_health(dt)

    def _try_eat(self, effort: float, world: World, dt: float) -> None:
        """Attempt to consume nearby food.

        Args:
            effort: Eating effort (0-1).
            world: World instance.
            dt: Time step.
        """
        eating_radius = self.radius + 5.0  # Must be very close

        for food in world.food_items:
            if not food.is_active:
                continue

            distance = self.position.distance_to(food.position)

            if distance < eating_radius + food.radius:
                # Consume food (consumes entire food item)
                consumed = food.consume()

                # Gain energy based on efficiency
                energy_gain = consumed * self.genome.eating_efficiency
                self.energy = min(self.genome.max_energy, self.energy + energy_gain)

                self.is_eating = True
                return

        self.is_eating = False

    def _update_energy(self, action: AgentAction, dt: float) -> None:
        """Update energy based on activity.

        Args:
            action: Current action being performed.
            dt: Time step.
        """
        # Base metabolic cost
        energy_cost = self.genome.base_energy_cost * dt

        # Movement costs
        thrust_cost = abs(action.thrust) * self.genome.movement_energy_mult * dt
        torque_cost = abs(action.torque) * self.genome.movement_energy_mult * dt * 0.5

        # Vocalization cost
        vocalize_cost = action.vocalize[1] * self.genome.vocalize_energy_mult * dt

        # Total depletion
        total_cost = energy_cost + thrust_cost + torque_cost + vocalize_cost
        self.energy = max(0.0, self.energy - total_cost)

    def _update_health(self, dt: float) -> None:
        """Update health (healing or starvation damage).

        Args:
            dt: Time step.
        """
        # Healing when energy is high
        if self.energy > self.genome.max_energy * 0.5:
            if self.health < self.genome.max_health:
                self.health = min(
                    self.genome.max_health,
                    self.health + self.genome.healing_rate * dt,
                )

        # Damage from starvation
        elif self.energy <= 0:
            starvation_damage = 5.0 * dt  # 5 health/sec when starving
            self.take_damage(starvation_damage)

        # Check death from health
        if self.health <= 0:
            self.die("health_depleted")

    def take_damage(self, amount: float) -> None:
        """Apply damage with resistance.

        Args:
            amount: Raw damage amount.
        """
        actual_damage = amount / self.genome.damage_resistance
        self.health = max(0.0, self.health - actual_damage)
        self.last_damage_time = self.age

        if self.health <= 0:
            self.die("damage")

    def die(self, cause: str) -> None:
        """Mark agent as dead.

        Args:
            cause: String describing cause of death.
        """
        self.is_alive = False
        self.is_active = False
        self.death_cause = cause

    def get_observations(self, world: World) -> Dict[str, np.ndarray]:
        """Collect all sensory observations.

        Args:
            world: World instance to query.

        Returns:
            Dictionary with keys:
                'vision': (num_rays, 4) array
                'audio': (2,) array
                'proprioception': (8,) array
                'touch': (8,) array
        """
        return {
            "vision": self.vision.sense(self.position, self.facing, world),
            "audio": self.audio.sense(self.position, self.facing, world),
            "proprioception": self.proprioception.sense(
                energy=self.energy,
                health=self.health,
                velocity=self.velocity,
                angular_velocity=self.angular_velocity,
                angle=self.angle,
            ),
            "touch": self.touch.sense(self.position, world),
        }

    def get_observation_tensor(self, world: World) -> torch.Tensor:
        """Get all observations as a single flattened tensor.

        Layout:
            [0:128]   - Vision (32 rays × 4)
            [128:130] - Audio (2)
            [130:138] - Proprioception (8)
            [138:146] - Touch (8)

        Args:
            world: World instance to query.

        Returns:
            torch.Tensor of shape (146,) float32.
        """
        obs = self.get_observations(world)

        vision_flat = obs["vision"].flatten()  # (128,)
        audio_flat = obs["audio"]  # (2,)
        proprio_flat = obs["proprioception"]  # (8,)
        touch_flat = obs["touch"]  # (8,)

        combined = np.concatenate(
            [vision_flat, audio_flat, proprio_flat, touch_flat]
        )

        return torch.from_numpy(combined).float()

    def get_collision_shape(self) -> Circle:
        """Return collision shape for physics.

        Returns:
            Circle at current position with agent radius.
        """
        return Circle(self.position, self.radius)

    def save(self) -> Dict[str, Any]:
        """Serialize agent state for checkpointing.

        Returns:
            Dictionary with complete agent state.
        """
        return {
            "agent_id": self.agent_id,
            "genome": self.genome.to_dict(),
            "position": (self.position.x, self.position.y),
            "velocity": (self.velocity.x, self.velocity.y),
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "energy": self.energy,
            "health": self.health,
            "age": self.age,
            "is_alive": self.is_alive,
            "death_cause": self.death_cause,
        }

    @classmethod
    def load(cls, data: Dict[str, Any]) -> AgentBody:
        """Deserialize agent from saved state.

        Args:
            data: Dictionary from save().

        Returns:
            Restored AgentBody instance.
        """
        genome = AgentGenome.from_dict(data["genome"])
        agent = cls(
            agent_id=data["agent_id"],
            genome=genome,
            initial_position=Vec2(data["position"][0], data["position"][1]),
            initial_angle=data["angle"],
        )

        agent.velocity = Vec2(data["velocity"][0], data["velocity"][1])
        agent.angular_velocity = data["angular_velocity"]
        agent.energy = data["energy"]
        agent.health = data["health"]
        agent.age = data["age"]
        agent.is_alive = data["is_alive"]
        agent.death_cause = data.get("death_cause")

        if not agent.is_alive:
            agent.is_active = False

        return agent

    def __repr__(self) -> str:
        status = "alive" if self.is_alive else f"dead({self.death_cause})"
        return (
            f"AgentBody("
            f"id={self.agent_id}, "
            f"pos=({self.position.x:.1f}, {self.position.y:.1f}), "
            f"energy={self.energy:.1f}, "
            f"health={self.health:.1f}, "
            f"{status})"
        )
