"""Modality-specific encoders for Living Resonance Network."""
import torch
import torch.nn as nn
from .lrn_config import LRNConfig


class WaveletEncoder(nn.Module):
    """Base class for modality-specific encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, output_seq_len: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape varies by modality
        Returns:
            Encoded tensor of shape (batch, output_seq_len, hidden_dim)
        """
        raise NotImplementedError


class VisionEncoder(WaveletEncoder):
    """Encodes vision rays to sequence."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.vision_shape[1],  # 4: (dist, r, g, b)
            hidden_dim=config.hidden_dim,
            output_seq_len=config.vision_seq_len
        )
        # Simple linear projection per ray
        self.projection = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 32, 4) - vision rays
        Returns:
            (batch, 32, hidden_dim) - encoded rays
        """
        # x: (B, 32, 4) → (B, 32, hidden_dim)
        return self.projection(x)


class AudioEncoder(WaveletEncoder):
    """Encodes audio samples to sequence."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.audio_shape[1],  # 2: stereo
            hidden_dim=config.hidden_dim,
            output_seq_len=config.audio_seq_len
        )
        self.projection = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 100, 2) - audio samples
        Returns:
            (batch, 100, hidden_dim) - encoded samples
        """
        # x: (B, 100, 2) → (B, 100, hidden_dim)
        return self.projection(x)


class ProprioEncoder(WaveletEncoder):
    """Encodes proprioception to sequence by expansion."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.proprio_dim,  # 7
            hidden_dim=config.hidden_dim,
            output_seq_len=config.proprio_seq_len  # 16
        )
        # Project to sequence embedding
        self.projection = nn.Linear(self.input_dim,
                                    self.output_seq_len * self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 7) - proprioceptive state
        Returns:
            (batch, 16, hidden_dim) - expanded sequence
        """
        batch_size = x.shape[0]
        # (B, 7) → (B, 16*hidden_dim) → (B, 16, hidden_dim)
        x = self.projection(x)
        return x.view(batch_size, self.output_seq_len, self.hidden_dim)


class TouchEncoder(WaveletEncoder):
    """Encodes touch sensors to sequence by expansion."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.touch_dim,  # 8
            hidden_dim=config.hidden_dim,
            output_seq_len=config.touch_seq_len  # 16
        )
        self.projection = nn.Linear(self.input_dim,
                                    self.output_seq_len * self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 8) - touch sensors
        Returns:
            (batch, 16, hidden_dim) - expanded sequence
        """
        batch_size = x.shape[0]
        # (B, 8) → (B, 16*hidden_dim) → (B, 16, hidden_dim)
        x = self.projection(x)
        return x.view(batch_size, self.output_seq_len, self.hidden_dim)
