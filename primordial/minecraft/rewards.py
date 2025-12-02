"""Reward computation for Minecraft environments."""
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardInfo:
    """Breakdown of reward components."""
    total: float
    health_delta: float
    food_delta: float
    navigation: float
    movement: float
    death: float


class MinecraftRewardComputer:
    """Computes rewards from Minecraft observations."""

    def __init__(
        self,
        health_scale: float = 2.0,
        food_scale: float = 0.5,
        navigation_scale: float = 1.0,
        movement_bonus: float = 0.01,
        idle_penalty: float = -0.01,
        death_penalty: float = -10.0,
    ):
        self.health_scale = health_scale
        self.food_scale = food_scale
        self.navigation_scale = navigation_scale
        self.movement_bonus = movement_bonus
        self.idle_penalty = idle_penalty
        self.death_penalty = death_penalty

        # Previous state for computing deltas
        self.prev_health: Optional[float] = None
        self.prev_food: Optional[float] = None
        self.prev_distance: Optional[float] = None
        self.prev_position: Optional[np.ndarray] = None

    def reset(self):
        """Reset state for new episode."""
        self.prev_health = None
        self.prev_food = None
        self.prev_distance = None
        self.prev_position = None

    def compute(
        self,
        obs: Dict[str, Any],
        env_reward: float,
        done: bool,
        info: Dict[str, Any]
    ) -> RewardInfo:
        """Compute shaped reward from observation.

        Args:
            obs: Current observation
            env_reward: Reward from environment (usually sparse)
            done: Whether episode ended
            info: Additional info from environment

        Returns:
            RewardInfo with breakdown of reward components
        """
        reward = 0.0
        health_delta_reward = 0.0
        food_delta_reward = 0.0
        navigation_reward = 0.0
        movement_reward = 0.0
        death_reward = 0.0

        # Extract current stats
        current_health = self._get_health(obs)
        current_food = self._get_food(obs)
        current_position = self._get_position(obs)

        # Health delta reward
        if self.prev_health is not None:
            health_delta = current_health - self.prev_health
            health_delta_reward = health_delta * self.health_scale
            reward += health_delta_reward

        # Food delta reward
        if self.prev_food is not None:
            food_delta = current_food - self.prev_food
            food_delta_reward = food_delta * self.food_scale
            reward += food_delta_reward

        # Navigation reward (for NavigateDense task)
        # Uses compass angle or distance to goal
        if 'compassAngle' in obs:
            # Reward for facing toward goal (angle near 0)
            compass = abs(obs['compassAngle'])
            # Closer to 0 = better, reward decreases with angle
            navigation_reward = (1.0 - compass / 180.0) * 0.01 * self.navigation_scale
            reward += navigation_reward

        # Distance-based navigation reward
        if self.prev_distance is not None and 'distance' in info:
            current_distance = info['distance']
            distance_delta = self.prev_distance - current_distance  # Positive = closer
            navigation_reward += distance_delta * self.navigation_scale
            reward += distance_delta * self.navigation_scale
            self.prev_distance = current_distance
        elif 'distance' in info:
            self.prev_distance = info['distance']

        # Movement reward/penalty
        if current_position is not None and self.prev_position is not None:
            movement = np.linalg.norm(current_position - self.prev_position)
            if movement > 0.1:  # Moved
                movement_reward = self.movement_bonus
            else:  # Idle
                movement_reward = self.idle_penalty
            reward += movement_reward

        # Death penalty
        if done and current_health <= 0:
            death_reward = self.death_penalty
            reward += death_reward

        # Include environment reward (usually sparse navigation success)
        reward += env_reward

        # Update previous state
        self.prev_health = current_health
        self.prev_food = current_food
        self.prev_position = current_position

        return RewardInfo(
            total=reward,
            health_delta=health_delta_reward,
            food_delta=food_delta_reward,
            navigation=navigation_reward,
            movement=movement_reward,
            death=death_reward,
        )

    def _get_health(self, obs: Dict[str, Any]) -> float:
        """Extract health from observation."""
        if 'life_stats' in obs:
            return obs['life_stats'].get('life', 20) / 20.0
        return 1.0

    def _get_food(self, obs: Dict[str, Any]) -> float:
        """Extract food from observation."""
        if 'life_stats' in obs:
            return obs['life_stats'].get('food', 20) / 20.0
        return 1.0

    def _get_position(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract position from observation."""
        if 'location_stats' in obs:
            pos = obs['location_stats'].get('pos', None)
            if pos is not None:
                return np.array(pos)
        return None
