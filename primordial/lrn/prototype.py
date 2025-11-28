"""Minimal Fourier prototype for validation with multi-task learning."""
import torch
import torch.nn as nn
from typing import Tuple

from .config import PrototypeConfig
from .mixing import FourierMixingLayer
from .heads import SensoryHead, RewardHead


class FourierPrototype(nn.Module):
    """
    Minimal Fourier-based model for validating FFT mixing learns.

    Multi-task architecture matching the full LRN:
        Input (seq_len, input_dim)
        -> Linear projection to hidden_dim
        -> N x FourierMixingLayer
        -> Pooling (mean over sequence)
        -> Dual heads:
           - SensoryHead: predicts next sensory state (seq_len, 1)
           - RewardHead: predicts upcoming rewards (reward_horizon,)

    This validates BOTH:
    1. Fourier mixing learns temporal patterns (sensory prediction)
    2. Fourier mixing learns reward prediction (survival gradient)
    """

    def __init__(
        self,
        config: PrototypeConfig,
        input_dim: int = 1,
    ):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # Fourier mixing layers
        self.mixing_layers = nn.ModuleList([
            FourierMixingLayer(config)
            for _ in range(config.num_mixing_layers)
        ])

        # Output heads (multi-task)
        self.sensory_head = SensoryHead(config, output_dim=input_dim)
        self.reward_head = RewardHead(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-task outputs.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            sensory_pred: (batch, seq_len, input_dim) - predicted next sensory state
            reward_pred: (batch, reward_horizon) - predicted upcoming rewards
        """
        # Project to hidden dim
        x = self.input_proj(x)  # (B, seq_len, hidden_dim)

        # Apply Fourier mixing layers
        for layer in self.mixing_layers:
            x = layer(x)

        # Sensory prediction (sequence-to-sequence)
        sensory_pred = self.sensory_head(x)  # (B, seq_len, input_dim)

        # Pooling for reward head (mean over sequence)
        pooled = x.mean(dim=1)  # (B, hidden_dim)

        # Reward prediction
        reward_pred = self.reward_head(pooled)  # (B, reward_horizon)

        return sensory_pred, reward_pred
