"""Output heads for the full Living Resonance Network (LRN)."""
import torch
import torch.nn as nn
from typing import Dict

from .lrn_config import LRNConfig


class PredictionHead(nn.Module):
    """
    Predicts next sensory state for self-supervised learning.

    Output matches flattened input dimension (343 values).

    Input: (batch, 3*hidden_dim) = (batch, 384) pooled features
    Output: (batch, 343) predicted next sensory state
    """

    def __init__(self, config: LRNConfig):
        super().__init__()

        # Input: pooled features (3 * hidden_dim)
        input_dim = 3 * config.hidden_dim  # 384

        # Total sensory dimension
        output_dim = config.total_sensory_dim  # 343

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.pred_hidden_dim),  # 384 -> 256
            nn.GELU(),
            nn.Linear(config.pred_hidden_dim, output_dim)  # 256 -> 343
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3*hidden_dim) - pooled features
        Returns:
            (batch, output_dim) - predicted next sensory state
        """
        return self.mlp(x)

    def split_prediction(self, pred: torch.Tensor, config: LRNConfig) -> Dict[str, torch.Tensor]:
        """
        Split flat prediction into modality components.

        Args:
            pred: (batch, output_dim)
            config: LRNConfig instance
        Returns:
            Dictionary with vision, audio, proprio, touch predictions
        """
        vision_size = config.vision_shape[0] * config.vision_shape[1]  # 128
        audio_size = config.audio_shape[0] * config.audio_shape[1]    # 200

        idx = 0
        vision_pred = pred[:, idx:idx+vision_size].view(-1, *config.vision_shape)
        idx += vision_size

        audio_pred = pred[:, idx:idx+audio_size].view(-1, *config.audio_shape)
        idx += audio_size

        proprio_pred = pred[:, idx:idx+config.proprio_dim]
        idx += config.proprio_dim

        touch_pred = pred[:, idx:idx+config.touch_dim]

        return {
            'vision': vision_pred,
            'audio': audio_pred,
            'proprio': proprio_pred,
            'touch': touch_pred
        }


class LRNRewardHead(nn.Module):
    """
    Predicts upcoming reward values for multi-task learning.

    This creates a DIRECT gradient toward survival by predicting
    whether current patterns lead to positive or negative outcomes.

    Input: (batch, 3*hidden_dim) = (batch, 384) pooled features
    Output: (batch, reward_horizon) = (batch, 5)
    """

    def __init__(self, config: LRNConfig):
        super().__init__()

        # Input: pooled features (3 * hidden_dim)
        input_dim = 3 * config.hidden_dim  # 384

        self.reward_horizon = config.reward_horizon  # default: 5

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),  # 384 -> 64
            nn.GELU(),
            nn.Linear(64, self.reward_horizon)  # 64 -> 5
            # No activation - rewards can be any real value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3*hidden_dim) - pooled features
        Returns:
            (batch, reward_horizon) - predicted rewards for next N steps

        Example outputs:
            [+0.8, +0.5, +0.2, 0.0, 0.0]  → "good things coming soon"
            [-1.5, -0.5, 0.0, 0.0, 0.0]   → "pain imminent!"
            [0.0, 0.0, 0.0, 0.0, 0.0]     → "nothing special expected"
        """
        return self.mlp(x)


class ActionHead(nn.Module):
    """
    Outputs agent actions from pooled features.

    Actions: (thrust, torque, vocalize, freq, eat)

    Input: (batch, 3*hidden_dim) = (batch, 384)
    Output: (batch, 5)
    """

    def __init__(self, config: LRNConfig):
        super().__init__()

        # Input: pooled features (3 * hidden_dim)
        input_dim = 3 * config.hidden_dim  # 384

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.action_hidden_dim),  # 384 -> 128
            nn.GELU(),
            nn.Linear(config.action_hidden_dim, config.action_dim)  # 128 -> 5
        )

        # Action bounds (applied externally, but documented here)
        # thrust: [-1, 1]
        # torque: [-1, 1]
        # vocalize: [0, 1]
        # freq: [0, 1] (maps to frequency range)
        # eat: [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3*hidden_dim) - pooled features
        Returns:
            (batch, 5) - raw action logits (apply tanh/sigmoid externally)
        """
        return self.mlp(x)
