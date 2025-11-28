"""Online learning utilities for LRN architecture.

This module provides dependency-free utilities for online learning:
- RewardHistoryBuffer: Track reward predictions and actual rewards
- GradientClipper: Clip gradients for training stability
- ExponentialMovingAverage: EMA of model weights
- GradientMonitor: Track gradient statistics
- OnlineLRScheduler: Learning rate scheduling with warmup

These utilities support Phase 7 of the learning system implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional


class RewardHistoryBuffer:
    """Tracks reward history for multi-task reward prediction.

    Uses dict for O(1) reward lookup instead of O(n) deque iteration.
    Handles stale predictions that exceed max age.

    When agent makes a prediction at time t, it predicts rewards for
    t+1, t+2, ..., t+horizon. We need to store actual rewards to compute
    the prediction loss once those timesteps occur.
    """

    def __init__(
        self,
        horizon: int = 5,
        max_pending: int = 100,
        max_stale_steps: int = 50,
    ):
        """Initialize reward history buffer.

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
        """Record a reward prediction for later loss computation.

        Args:
            step: Current timestep
            reward_preds: (horizon,) predicted rewards for next H steps
        """
        self.pending_predictions.append(
            {
                "step": step,
                "predictions": reward_preds.detach().clone(),
                "steps_remaining": self.horizon,
            }
        )

        # Enforce max_pending limit
        if len(self.pending_predictions) > self.max_pending:
            self.pending_predictions.pop(0)

    def record_actual_reward(self, step: int, reward: float):
        """Record an actual reward that occurred. O(1) insertion.

        Args:
            step: Current timestep
            reward: Actual reward value
        """
        self.reward_history[step] = reward

        # Cleanup old entries to prevent unbounded growth
        self._cleanup_old_entries(step)

    def _cleanup_old_entries(self, current_step: int):
        """Remove reward history entries older than needed.

        Args:
            current_step: Current timestep
        """
        # Keep rewards from (current_step - horizon - max_stale_steps) onwards
        cutoff = current_step - self.horizon - self.max_stale_steps

        if cutoff > self._oldest_step:
            # Remove old entries
            old_keys = [k for k in self.reward_history if k < cutoff]
            for k in old_keys:
                del self.reward_history[k]
            self._oldest_step = cutoff

    def get_ready_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get prediction/actual pairs ready for loss computation.

        Returns predictions that now have enough actual reward history
        to compute loss against. Discards stale predictions.

        Returns:
            List of (predicted_rewards, actual_rewards) tensors
        """
        ready_pairs = []
        remaining = []

        for pending in self.pending_predictions:
            pending["steps_remaining"] -= 1
            age = self.horizon - pending["steps_remaining"]

            # Check for stale predictions (too old, discard)
            if age > self.max_stale_steps:
                # Prediction is stale, skip without computing loss
                continue

            if pending["steps_remaining"] <= 0:
                # This prediction has waited long enough
                # Gather actual rewards using O(1) dict lookup
                pred_step = pending["step"]
                actual_rewards = []

                for i in range(1, self.horizon + 1):
                    target_step = pred_step + i
                    # O(1) lookup instead of O(n) iteration
                    reward = self.reward_history.get(target_step, 0.0)
                    actual_rewards.append(reward)

                ready_pairs.append(
                    (pending["predictions"], torch.tensor(actual_rewards))
                )
            else:
                remaining.append(pending)

        self.pending_predictions = remaining
        return ready_pairs

    def on_death(self):
        """Clear buffer state on agent death.

        Stale predictions from before death should not affect
        learning after respawn.
        """
        self.pending_predictions.clear()
        self.reward_history.clear()
        self._oldest_step = 0


class GradientClipper:
    """Gradient clipping for training stability.

    Essential for single-sample updates in online learning.
    Supports both norm-based and value-based clipping.
    """

    def __init__(
        self,
        clip_type: str = "norm",
        max_norm: float = 1.0,
        max_value: float = 10.0,
    ):
        """Initialize gradient clipper.

        Args:
            clip_type: 'norm' or 'value'
            max_norm: Maximum gradient norm (for clip_type='norm')
            max_value: Maximum gradient value (for clip_type='value')
        """
        self.clip_type = clip_type
        self.max_norm = max_norm
        self.max_value = max_value

    def clip(self, model: nn.Module) -> float:
        """Clip gradients in-place.

        Args:
            model: PyTorch model with gradients

        Returns:
            Total gradient norm before/after clipping
        """
        if self.clip_type == "norm":
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_norm
            )
            # clip_grad_norm_ returns a tensor, convert to float
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(model.parameters(), self.max_value)
            grad_norm = self._compute_grad_norm(model)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")

        return grad_norm

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm.

        Args:
            model: PyTorch model

        Returns:
            Total gradient norm (L2 norm)
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm**0.5


class ExponentialMovingAverage:
    """EMA of model weights for stable predictions.

    Maintains shadow weights that are a moving average of model weights.
    Can temporarily apply shadow weights for inference, then restore.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA.

        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = slower update)
                  shadow = decay * shadow + (1 - decay) * param
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights after an optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights.

        Use this before inference for more stable predictions.
        Must call restore() afterwards to get back original weights.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original weights after apply_shadow().

        Returns model to its pre-apply_shadow state.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()


class GradientMonitor:
    """Track gradient statistics for stability monitoring.

    Records gradient norms and variance over a sliding window
    to detect instability or vanishing/exploding gradients.
    """

    def __init__(self, window_size: int = 100):
        """Initialize gradient monitor.

        Args:
            window_size: Number of recent steps to track
        """
        self.window_size = window_size
        self.grad_norms: List[float] = []
        self.grad_vars: List[float] = []

    def record(self, model: nn.Module):
        """Record gradient statistics from a model.

        Args:
            model: PyTorch model with gradients
        """
        grad_norm = 0.0
        grad_values = []

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
                grad_values.extend(p.grad.data.flatten().cpu().numpy())

        grad_norm = grad_norm**0.5
        grad_var = np.var(grad_values) if len(grad_values) > 0 else 0.0

        self.grad_norms.append(grad_norm)
        self.grad_vars.append(grad_var)

        # Keep only recent history
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)
            self.grad_vars.pop(0)

    def get_statistics(self) -> dict:
        """Get gradient statistics.

        Returns:
            Dictionary with mean/std of gradient norms and variance,
            plus stability indicator
        """
        if len(self.grad_norms) == 0:
            return {
                "grad_norm_mean": 0.0,
                "grad_norm_std": 0.0,
                "grad_var_mean": 0.0,
                "is_stable": True,
            }

        return {
            "grad_norm_mean": np.mean(self.grad_norms),
            "grad_norm_std": np.std(self.grad_norms),
            "grad_var_mean": np.mean(self.grad_vars),
            "is_stable": self._check_stability(),
        }

    def _check_stability(self) -> bool:
        """Check if gradients are stable.

        Returns:
            True if gradients appear stable, False if unstable
        """
        if len(self.grad_norms) < 10:
            return True

        recent_std = np.std(self.grad_norms[-10:])
        overall_mean = np.mean(self.grad_norms)

        # Unstable if recent variance is too high
        return bool(recent_std < 2 * overall_mean)


class OnlineLRScheduler:
    """Learning rate scheduler for online learning.

    Implements linear warmup followed by exponential decay.
    Useful for stabilizing early training and preventing
    overfitting in continual learning scenarios.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        base_lr: float = 1e-4,
        min_lr: float = 1e-6,
        decay_rate: float = 0.9999,
    ):
        """Initialize learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule
            warmup_steps: Number of steps for linear warmup
            base_lr: Base learning rate after warmup
            min_lr: Minimum learning rate (floor)
            decay_rate: Exponential decay rate (very slow, e.g., 0.9999)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.step_count = 0

    def step(self) -> float:
        """Update learning rate and return current value.

        Returns:
            Current learning rate after update
        """
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Exponential decay
            steps_after_warmup = self.step_count - self.warmup_steps
            lr = self.base_lr * (self.decay_rate**steps_after_warmup)

        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr
