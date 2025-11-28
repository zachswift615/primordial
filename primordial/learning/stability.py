"""
Stability measures for online learning.

Includes gradient clipping, normalization, EMA of weights, and gradient monitoring
to prevent catastrophic failures during continual learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class GradientClipper:
    """Gradient clipping for stability during single-sample updates."""

    def __init__(
        self,
        clip_type: str = 'norm',
        max_norm: float = 1.0,
        max_value: float = 10.0
    ):
        """
        Args:
            clip_type: 'norm' or 'value'
            max_norm: Maximum gradient norm (for clip_type='norm')
            max_value: Maximum gradient value (for clip_type='value')
        """
        self.clip_type = clip_type
        self.max_norm = max_norm
        self.max_value = max_value

    def clip(self, model: nn.Module) -> float:
        """
        Clip gradients in-place.

        Args:
            model: PyTorch model with gradients

        Returns:
            Total gradient norm before clipping
        """
        if self.clip_type == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_norm
            )
            return grad_norm.item()
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                self.max_value
            )
            grad_norm = self._compute_grad_norm(model)
            return grad_norm
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


class GradientAccumulator:
    """Accumulate gradients over multiple steps before update."""

    def __init__(self, accumulation_steps: int = 4):
        """
        Args:
            accumulation_steps: Number of steps to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_update(self) -> bool:
        """Check if we should perform optimizer step."""
        self.step_count += 1
        if self.step_count >= self.accumulation_steps:
            self.step_count = 0
            return True
        return False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for accumulation."""
        return loss / self.accumulation_steps


class ExponentialMovingAverage:
    """EMA of model weights for stable predictions."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = slower update)
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Optional[Dict[str, torch.Tensor]] = None

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights."""
        if self.backup is None:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = None


class GradientMonitor:
    """Track gradient statistics for stability monitoring."""

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of steps to keep in history
        """
        self.window_size = window_size
        self.grad_norms = []
        self.grad_vars = []

    def record(self, model: nn.Module):
        """Record gradient statistics."""
        grad_norm = 0.0
        grad_values = []

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
                grad_values.extend(p.grad.data.flatten().cpu().numpy())

        grad_norm = grad_norm ** 0.5
        grad_var = np.var(grad_values) if len(grad_values) > 0 else 0.0

        self.grad_norms.append(grad_norm)
        self.grad_vars.append(grad_var)

        # Keep only recent history
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)
            self.grad_vars.pop(0)

    def get_statistics(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.grad_norms:
            return {
                'grad_norm_mean': 0.0,
                'grad_norm_std': 0.0,
                'grad_var_mean': 0.0,
                'is_stable': True
            }

        return {
            'grad_norm_mean': np.mean(self.grad_norms),
            'grad_norm_std': np.std(self.grad_norms),
            'grad_var_mean': np.mean(self.grad_vars),
            'is_stable': self._check_stability()
        }

    def _check_stability(self) -> bool:
        """Check if gradients are stable."""
        if len(self.grad_norms) < 10:
            return True

        recent_std = np.std(self.grad_norms[-10:])
        overall_mean = np.mean(self.grad_norms)

        # Unstable if recent variance is too high
        return recent_std < 2 * overall_mean
