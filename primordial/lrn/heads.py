"""Output heads for the Fourier prototype."""
import torch
import torch.nn as nn

from .config import PrototypeConfig


class RewardHead(nn.Module):
    """
    Predicts upcoming reward values for multi-task learning.

    This creates a DIRECT gradient toward survival by predicting
    whether current patterns lead to positive or negative outcomes.

    Matches the RewardHead from the full LRN architecture.
    """

    def __init__(self, config: PrototypeConfig):
        super().__init__()

        self.reward_horizon = config.reward_horizon

        # Simple MLP: hidden_dim -> 64 -> reward_horizon
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, config.reward_horizon)
            # No activation - rewards can be any real value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_dim) - pooled features
        Returns:
            (batch, reward_horizon) - predicted rewards for next N steps
        """
        return self.mlp(x)


class SensoryHead(nn.Module):
    """
    Predicts next sensory state (sequence-to-sequence).

    For the prototype, this is a simple linear projection.
    """

    def __init__(self, config: PrototypeConfig, output_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(config.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, output_dim)
        """
        return self.proj(x)
