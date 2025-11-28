"""Agent sensory systems.

Sensors provide the agent's perception of the world through:
- Vision: Raycasting-based sight with color information
- Audio: Stereo hearing for directional sound
- Proprioception: Internal state awareness (energy, health, velocity)
- Touch: 8-directional contact sensors

All sensors output normalized float arrays suitable for neural network input.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from primordial.world.geometry import Vec2
from primordial.world import helpers

if TYPE_CHECKING:
    from primordial.agents.genome import AgentGenome
    from primordial.world.world import World

# Try to import Rust-accelerated helpers
try:
    from primordial.world.helpers_rust import get_vision_input_fast, RUST_AVAILABLE
    _USE_RUST = RUST_AVAILABLE
except ImportError:
    _USE_RUST = False


class VisionSensor:
    """Ray-based vision system.

    Casts multiple rays across the field of view to detect
    entities and their distances.

    Attributes:
        num_rays: Number of vision rays to cast.
        max_range: Maximum vision distance.
        fov: Field of view in radians.
    """

    def __init__(self, genome: AgentGenome) -> None:
        """Initialize vision sensor from genome.

        Args:
            genome: Agent genome with vision parameters.
        """
        self.num_rays = genome.vision_rays
        self.max_range = genome.vision_range
        self.fov = math.radians(genome.vision_fov)

    def sense(
        self,
        position: Vec2,
        facing: Vec2,
        world: World,
        ignore_entity_id: int | None = None,
    ) -> np.ndarray:
        """Cast vision rays and return sensory data.

        Uses the world's helper function for raycasting.

        Args:
            position: Agent position in world coordinates.
            facing: Unit vector indicating facing direction.
            world: World instance to query.
            ignore_entity_id: Entity ID to ignore (typically self).

        Returns:
            np.ndarray of shape (num_rays, 4) float32:
                - Column 0: distance (normalized 0-1, 0=far, 1=near)
                - Column 1: entity_type code (0-5)
                - Column 2-3: reserved for color (currently 0)
        """
        # Try Rust-accelerated path first
        if _USE_RUST:
            try:
                return get_vision_input_fast(
                    world=world,
                    agent_position=position,
                    agent_facing=facing,
                    vision_range=self.max_range,
                    vision_fov=self.fov,
                    num_rays=self.num_rays,
                    ignore_entity_id=ignore_entity_id,
                )
            except Exception:
                pass  # Fall through to Python

        # Python fallback
        result = helpers.get_vision_input(
            world=world,
            agent_position=position,
            agent_facing=facing,
            vision_range=self.max_range,
            vision_fov=self.fov,
            num_rays=self.num_rays,
        )

        # Invert distance normalization (helper gives 0=near, we want 0=far)
        # Actually the helper already does hit_distance / vision_range
        # where 0 = nothing hit, 1 = at max range
        # We want 0 = far, 1 = close, so we invert
        result[:, 0] = 1.0 - result[:, 0]

        return result


class AudioSensor:
    """Stereo hearing system.

    Provides directional sound information through two "ears"
    positioned relative to the agent's facing direction.

    Attributes:
        max_range: Maximum hearing distance.
    """

    def __init__(self, genome: AgentGenome) -> None:
        """Initialize audio sensor from genome.

        Args:
            genome: Agent genome with audio parameters.
        """
        self.max_range = genome.audio_range

    def sense(
        self,
        position: Vec2,
        facing: Vec2,
        world: World,
    ) -> np.ndarray:
        """Mix sounds from all sound sources.

        Uses the world's sound system for stereo computation.

        Args:
            position: Agent position in world coordinates.
            facing: Unit vector indicating facing direction.
            world: World instance to query.

        Returns:
            np.ndarray of shape (2,) float32:
                [left_ear_intensity, right_ear_intensity]
                Values clamped to 0.0-1.0.
        """
        sound_data = helpers.get_sound_input(
            world=world,
            agent_position=position,
            agent_facing=facing,
            num_frequency_bins=1,  # We just need stereo, not spectrum
        )

        return np.array(
            [sound_data["left_ear"], sound_data["right_ear"]],
            dtype=np.float32,
        )


class ProprioceptionSensor:
    """Internal state awareness.

    Provides normalized information about the agent's own
    physical and metabolic state.

    This sensor doesn't need world access - it reads from agent state.
    """

    def __init__(self, genome: AgentGenome) -> None:
        """Initialize proprioception sensor from genome.

        Args:
            genome: Agent genome for normalization bounds.
        """
        self.max_speed = genome.max_speed
        self.max_angular_speed = genome.max_angular_speed
        self.max_energy = genome.max_energy
        self.max_health = genome.max_health

    def sense(
        self,
        energy: float,
        health: float,
        velocity: Vec2,
        angular_velocity: float,
        angle: float,
    ) -> np.ndarray:
        """Return normalized internal state information.

        All values are normalized to approximately 0-1 range
        (velocity components may be negative).

        Args:
            energy: Current energy level.
            health: Current health level.
            velocity: Current velocity vector.
            angular_velocity: Current rotation rate.
            angle: Current facing angle in radians.

        Returns:
            np.ndarray of shape (8,) float32:
                [energy_norm, health_norm, speed_norm, ang_vel_norm,
                 vel_x_norm, vel_y_norm, angle_sin, angle_cos]
        """
        # Normalize energy and health to 0-1
        energy_norm = energy / self.max_energy
        health_norm = health / self.max_health

        # Normalize speed
        speed = velocity.magnitude()
        speed_norm = min(1.0, speed / self.max_speed)

        # Normalize angular velocity (-1 to 1)
        ang_vel_norm = angular_velocity / self.max_angular_speed
        ang_vel_norm = max(-1.0, min(1.0, ang_vel_norm))

        # Velocity components (normalized by max_speed, can be negative)
        vel_x_norm = velocity.x / self.max_speed
        vel_y_norm = velocity.y / self.max_speed

        # Angle as sin/cos (handles wraparound naturally)
        angle_sin = math.sin(angle)
        angle_cos = math.cos(angle)

        return np.array(
            [
                energy_norm,
                health_norm,
                speed_norm,
                ang_vel_norm,
                vel_x_norm,
                vel_y_norm,
                angle_sin,
                angle_cos,
            ],
            dtype=np.float32,
        )


class TouchSensor:
    """8-directional contact sensors.

    Detects nearby objects in 8 directions around the agent,
    providing distance and object type information.

    Attributes:
        num_sensors: Number of touch sensors (always 8).
        touch_range: Detection range beyond agent radius.
        agent_radius: Agent's collision radius.
    """

    def __init__(self, genome: AgentGenome) -> None:
        """Initialize touch sensor from genome.

        Args:
            genome: Agent genome with touch parameters.
        """
        self.num_sensors = 8
        self.touch_range = genome.touch_range
        self.agent_radius = genome.radius

    def sense(
        self,
        position: Vec2,
        world: World,
    ) -> np.ndarray:
        """Detect contact in 8 directions.

        Uses the world's helper function for proximity detection.

        Args:
            position: Agent position in world coordinates.
            world: World instance to query.

        Returns:
            np.ndarray of shape (8,) float32:
                Each value is 1.0 if something within touch_range
                in that direction, else 0.0.
                Directions: N, NE, E, SE, S, SW, W, NW
        """
        return helpers.get_touch_input(
            world=world,
            agent_position=position,
            agent_radius=self.agent_radius,
            touch_range=self.touch_range,
        )
