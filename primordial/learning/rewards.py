"""
Reward computation and modulation for online learning.

Implements:
- RewardHistoryBuffer: Tracks reward history for multi-task learning
- SurvivalRewards: Intrinsic rewards from survival events
- HumanTeaching: Human-provided reward signals
- RewardCombiner: Combines survival and teaching rewards
"""

from typing import Dict, List, Tuple
import torch


class RewardHistoryBuffer:
    """
    Tracks reward history for multi-task reward prediction.

    When agent makes a prediction at time t, it predicts rewards for
    t+1, t+2, ..., t+horizon. We need to store actual rewards to compute
    the prediction loss once those timesteps occur.

    Uses dict for O(1) reward lookup instead of O(n) deque iteration.
    Handles stale predictions that exceed max age.
    """

    def __init__(
        self,
        horizon: int = 5,
        max_pending: int = 100,
        max_stale_steps: int = 50  # Discard predictions older than this
    ):
        """
        Initialize reward history buffer.

        Args:
            horizon: Number of future steps to predict rewards for
            max_pending: Maximum number of pending predictions to track
            max_stale_steps: Discard predictions older than this many steps
        """
        self.horizon = horizon
        self.max_pending = max_pending
        self.max_stale_steps = max_stale_steps

        # Dict for O(1) reward lookup: step -> reward
        self.reward_history: Dict[int, float] = {}

        # Pending predictions awaiting actual rewards
        self.pending_predictions: List[Dict] = []

        # Track oldest step for cleanup
        self._oldest_step = 0

    def record_prediction(self, step: int, reward_preds: torch.Tensor):
        """
        Record a reward prediction for later loss computation.

        Args:
            step: Current timestep
            reward_preds: (horizon,) predicted rewards for next H steps
        """
        self.pending_predictions.append({
            'step': step,
            'predictions': reward_preds.detach().clone(),
            'steps_remaining': self.horizon
        })

        # Enforce max_pending limit
        if len(self.pending_predictions) > self.max_pending:
            self.pending_predictions.pop(0)

    def record_actual_reward(self, step: int, reward: float):
        """
        Record an actual reward that occurred. O(1) insertion.

        Args:
            step: Current timestep
            reward: Actual reward value
        """
        self.reward_history[step] = reward

        # Cleanup old entries to prevent unbounded growth
        self._cleanup_old_entries(step)

    def _cleanup_old_entries(self, current_step: int):
        """Remove reward history entries older than needed."""
        # Keep rewards from (current_step - horizon - max_stale_steps) onwards
        cutoff = current_step - self.horizon - self.max_stale_steps

        if cutoff > self._oldest_step:
            # Remove old entries
            old_keys = [k for k in self.reward_history if k < cutoff]
            for k in old_keys:
                del self.reward_history[k]
            self._oldest_step = cutoff

    def get_ready_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get prediction/actual pairs ready for loss computation.

        Returns predictions that now have enough actual reward history
        to compute loss against. Discards stale predictions.

        Returns:
            List of (predicted_rewards, actual_rewards) tensors
        """
        ready_pairs = []
        remaining = []

        for pending in self.pending_predictions:
            pending['steps_remaining'] -= 1
            age = self.horizon - pending['steps_remaining']

            # Check for stale predictions (too old, discard)
            if age > self.max_stale_steps:
                # Prediction is stale, skip without computing loss
                continue

            if pending['steps_remaining'] <= 0:
                # This prediction has waited long enough
                # Gather actual rewards using O(1) dict lookup
                pred_step = pending['step']
                actual_rewards = []

                for i in range(1, self.horizon + 1):
                    target_step = pred_step + i
                    # O(1) lookup instead of O(n) iteration
                    reward = self.reward_history.get(target_step, 0.0)
                    actual_rewards.append(reward)

                ready_pairs.append((
                    pending['predictions'],
                    torch.tensor(actual_rewards)
                ))
            else:
                remaining.append(pending)

        self.pending_predictions = remaining
        return ready_pairs

    def on_death(self):
        """
        Clear buffer state on agent death.

        Stale predictions from before death should not affect
        learning after respawn.
        """
        self.pending_predictions.clear()
        self.reward_history.clear()
        self._oldest_step = 0


class SurvivalRewards:
    """
    Intrinsic reward signals from survival events.

    Provides both event-based rewards (eating, damage, death) and
    continuous rewards (health/energy status).
    """

    # Event-based rewards
    EATING_FOOD = +1.0
    TAKING_DAMAGE = -2.0
    DEATH = -10.0

    # Continuous rewards (per step)
    STARVING = -0.1        # When energy < 30%
    LOW_HEALTH = -0.05     # When health < 50%
    HEALTHY = +0.01        # When health > 80% and energy > 50%

    # Exploration/curiosity rewards
    MOVEMENT_BONUS = +0.02  # Small reward for moving (encourages exploration)
    IDLE_PENALTY = -0.01    # Small penalty for staying still

    # Breeding drive discomfort
    HIGH_BREEDING_DRIVE = -0.03  # Periodic discomfort when drive > 0.7

    # Social connection
    LONELINESS_PENALTY = -0.02  # Discomfort when social connection is low
    SOCIAL_BONUS = +0.01  # Small bonus for being with others

    @staticmethod
    def compute_reward(prev_state, current_state, events):
        """
        Compute survival reward from state transition and events.

        Args:
            prev_state: AgentState before action
            current_state: AgentState after action
            events: List of events that occurred

        Returns:
            float: total survival reward
        """
        reward = 0.0

        # Event-based
        if 'ate_food' in events:
            reward += SurvivalRewards.EATING_FOOD
        if 'took_damage' in events:
            reward += SurvivalRewards.TAKING_DAMAGE
        if 'died' in events:
            reward += SurvivalRewards.DEATH

        # Continuous
        if current_state.energy < 0.3 * current_state.max_energy:
            reward += SurvivalRewards.STARVING
        if current_state.health < 0.5 * current_state.max_health:
            reward += SurvivalRewards.LOW_HEALTH
        if (current_state.health > 0.8 * current_state.max_health and
            current_state.energy > 0.5 * current_state.max_energy):
            reward += SurvivalRewards.HEALTHY

        # Movement/exploration reward - encourage agent to move around
        if hasattr(current_state, 'speed') and current_state.speed > 5.0:
            reward += SurvivalRewards.MOVEMENT_BONUS
        elif hasattr(current_state, 'speed') and current_state.speed < 1.0:
            reward += SurvivalRewards.IDLE_PENALTY

        # Breeding drive discomfort - high drive causes periodic negative sensation
        if hasattr(current_state, 'breeding_drive') and current_state.breeding_drive > 0.7:
            reward += SurvivalRewards.HIGH_BREEDING_DRIVE

        # Social connection - loneliness hurts, being together feels good
        if hasattr(current_state, 'social_connection'):
            if current_state.social_connection < 0.3:
                reward += SurvivalRewards.LONELINESS_PENALTY
            elif current_state.social_connection > 0.7:
                reward += SurvivalRewards.SOCIAL_BONUS

        return reward


class HumanTeaching:
    """
    Human-provided reward signals.

    Human feedback is distributed over a time window to credit-assign
    the teaching signal to recent behaviors.
    """

    REWARD_BUTTON = +1.0
    PUNISH_BUTTON = -1.0

    def __init__(self, window_size=10):
        """
        Initialize human teaching system.

        Args:
            window_size: Number of steps to apply human feedback over
        """
        self.pending_rewards = []
        self.window_size = window_size

    def add_teaching_signal(self, reward_value):
        """
        Called when human presses reward/punish button.

        Args:
            reward_value: Reward value (typically REWARD_BUTTON or PUNISH_BUTTON)
        """
        self.pending_rewards.append({
            'value': reward_value,
            'steps_remaining': self.window_size
        })

    def get_current_reward(self):
        """
        Get aggregated human teaching reward for current step.

        Distributes each teaching signal evenly over its window.

        Returns:
            float: Total teaching reward for this step
        """
        total = 0.0

        # Update and sum all pending rewards
        active_rewards = []
        for r in self.pending_rewards:
            if r['steps_remaining'] > 0:
                total += r['value'] / self.window_size
                r['steps_remaining'] -= 1
                active_rewards.append(r)

        self.pending_rewards = active_rewards
        return total


class RewardCombiner:
    """
    Combines survival and human teaching rewards.

    Allows different weighting of intrinsic survival rewards vs
    extrinsic human teaching signals.
    """

    def __init__(self, survival_weight=1.0, teaching_weight=1.5):
        """
        Initialize reward combiner.

        Args:
            survival_weight: Scaling factor for survival rewards
            teaching_weight: Scaling factor for human teaching
                           (higher = human feedback more important)
        """
        self.survival_weight = survival_weight
        self.teaching_weight = teaching_weight
        self.human_teaching = HumanTeaching()

    def compute_total_reward(self, prev_state, current_state, events):
        """
        Compute combined reward signal.

        Args:
            prev_state: AgentState before action
            current_state: AgentState after action
            events: List of events that occurred

        Returns:
            Tuple[float, float, float]: (total_reward, survival_reward, teaching_reward)
        """
        survival = SurvivalRewards.compute_reward(
            prev_state, current_state, events
        )
        teaching = self.human_teaching.get_current_reward()

        total = (
            self.survival_weight * survival +
            self.teaching_weight * teaching
        )

        return total, survival, teaching  # Return components for logging
