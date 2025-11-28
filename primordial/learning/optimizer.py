"""
Custom optimizers and learning rate schedulers for online learning.

Implements:
- RewardModulatedOptimizer: Modulates gradients by reward signal
- OnlineLRScheduler: Learning rate schedule with warmup and decay
"""

import numpy as np
import torch


class RewardModulatedOptimizer:
    """
    Optimizer wrapper that modulates gradients by reward.

    Implements the equation:
        θ_{t+1} = θ_t - η * m(r_t) * ∇L(θ_t)

    Where:
        η = base learning rate
        m(r_t) = modulation factor based on reward r_t
        ∇L(θ_t) = gradient of prediction loss

    Effect:
    - Positive reward (r > 0): m(r) > 1, learn faster
    - Negative reward (r < 0): m(r) < 1, learn slower (or unlearn)
    - Zero reward (r = 0): m(r) = 1, normal learning
    """

    def __init__(
        self,
        optimizer,
        modulation_type='linear',
        reward_scale=0.1,
        min_modulation=0.1,
        max_modulation=3.0
    ):
        """
        Initialize reward-modulated optimizer.

        Args:
            optimizer: Base PyTorch optimizer (e.g., AdamW)
            modulation_type: 'linear', 'sigmoid', or 'exponential'
            reward_scale: Sensitivity to reward signal
            min_modulation: Lower bound on modulation factor
            max_modulation: Upper bound on modulation factor
        """
        self.optimizer = optimizer
        self.modulation_type = modulation_type
        self.reward_scale = reward_scale
        self.min_modulation = min_modulation
        self.max_modulation = max_modulation

    def compute_modulation(self, reward):
        """
        Compute gradient modulation factor from reward.

        Args:
            reward: Scalar reward value

        Returns:
            float: Modulation factor in [min_modulation, max_modulation]
        """
        if self.modulation_type == 'linear':
            # Linear scaling: 1.0 + α * reward
            mod = 1.0 + self.reward_scale * reward
        elif self.modulation_type == 'sigmoid':
            # Sigmoid scaling (bounded to [0,1], then mapped to [0.1, 3.0])
            mod = torch.sigmoid(torch.tensor(self.reward_scale * reward)).item()
            mod = 0.1 + 2.9 * mod  # Map [0,1] to [0.1, 3.0]
        elif self.modulation_type == 'exponential':
            # Exponential scaling
            mod = torch.exp(torch.tensor(self.reward_scale * reward)).item()
        else:
            raise ValueError(f"Unknown modulation type: {self.modulation_type}")

        # Clamp to prevent extreme values
        return np.clip(mod, self.min_modulation, self.max_modulation)

    def step(self, reward):
        """
        Perform optimizer step with reward-modulated gradients.

        Scales all gradients by modulation factor, then performs
        standard optimizer step.

        Args:
            reward: Scalar reward value

        Returns:
            float: Modulation factor used (for logging)
        """
        modulation = self.compute_modulation(reward)

        # Scale all gradients by modulation factor
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad *= modulation

        # Perform standard optimizer step
        self.optimizer.step()

        return modulation  # Return for logging

    def zero_grad(self):
        """Pass through to base optimizer."""
        self.optimizer.zero_grad()


class OnlineLRScheduler:
    """
    Custom learning rate scheduler for online learning.

    Implements:
    1. Linear warmup: Gradually increase LR from 0 to base_lr
    2. Exponential decay: Slowly decrease LR after warmup

    This helps stabilize early learning and prevent divergence.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps=1000,
        base_lr=1e-4,
        min_lr=1e-6,
        decay_rate=0.9999  # Very slow decay
    ):
        """
        Initialize online learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule
            warmup_steps: Number of steps for linear warmup
            base_lr: Target learning rate after warmup
            min_lr: Minimum learning rate (floor)
            decay_rate: Exponential decay rate per step
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.step_count = 0

    def step(self):
        """
        Update learning rate.

        Returns:
            float: Current learning rate
        """
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Exponential decay
            steps_after_warmup = self.step_count - self.warmup_steps
            lr = self.base_lr * (self.decay_rate ** steps_after_warmup)

        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
