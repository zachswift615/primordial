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


# =============================================================================
# Minecraft-specific encoders
# =============================================================================

class MinecraftVisionEncoder(nn.Module):
    """CNN encoder for Minecraft RGB frames.

    Converts (batch, 3, 64, 64) RGB images to (batch, 32, hidden_dim) sequence
    that matches the output shape of the Primordial VisionEncoder.
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.output_seq_len = config.vision_seq_len  # 32, to match Primordial

        rgb_size = config.mc_rgb_size  # 64 or 128

        # CNN backbone
        self.conv = nn.Sequential(
            # Layer 1: (3, 64, 64) -> (32, 31, 31) for 64x64
            #          (3, 128, 128) -> (32, 63, 63) for 128x128
            nn.Conv2d(config.mc_rgb_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 2: (32, 31, 31) -> (64, 15, 15) for 64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 3: (64, 15, 15) -> (128, 7, 7) for 64x64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Calculate flattened size after convolutions
        # For 64x64: 64 -> 32 -> 16 -> 8, so 128 * 8 * 8 = 8192
        # For 128x128: 128 -> 64 -> 32 -> 16, so 128 * 16 * 16 = 32768
        if rgb_size == 64:
            self.flat_size = 128 * 8 * 8  # 8192
        elif rgb_size == 128:
            self.flat_size = 128 * 16 * 16  # 32768
        else:
            # Compute dynamically for other sizes
            test_input = torch.zeros(1, config.mc_rgb_channels, rgb_size, rgb_size)
            test_output = self.conv(test_input)
            self.flat_size = test_output.numel()

        # Project to sequence
        self.projection = nn.Linear(
            self.flat_size,
            self.output_seq_len * self.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 64, 64) or (batch, 3, 128, 128) - RGB frames

        Returns:
            (batch, 32, hidden_dim) - encoded sequence matching Primordial format
        """
        batch_size = x.shape[0]

        # CNN forward
        x = self.conv(x)  # (B, 128, H, W)

        # Flatten
        x = x.view(batch_size, -1)  # (B, flat_size)

        # Project to sequence
        x = self.projection(x)  # (B, 32 * hidden_dim)

        # Reshape to sequence
        x = x.view(batch_size, self.output_seq_len, self.hidden_dim)

        return x


class MinecraftProprioEncoder(WaveletEncoder):
    """Encodes Minecraft life_stats to sequence.

    Input: 10 values (health, food, oxygen, armor, saturation, xp,
                      yaw, pitch, is_sleeping, compass_angle)
    Output: (batch, 16, hidden_dim) matching Primordial ProprioEncoder
    """

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.mc_proprio_dim,  # 10
            hidden_dim=config.hidden_dim,
            output_seq_len=config.proprio_seq_len  # 16
        )
        self.projection = nn.Linear(
            self.input_dim,
            self.output_seq_len * self.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 10) - Minecraft life stats

        Returns:
            (batch, 16, hidden_dim) - expanded sequence
        """
        batch_size = x.shape[0]
        x = self.projection(x)
        return x.view(batch_size, self.output_seq_len, self.hidden_dim)


class MinecraftTouchEncoder(WaveletEncoder):
    """Encodes Minecraft damage_source to sequence.

    Input: 8 values (damage_amount, damage_yaw, damage_pitch, damage_distance,
                     is_explosion, is_fire, is_magic, is_projectile)
    Output: (batch, 16, hidden_dim) matching Primordial TouchEncoder
    """

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.mc_touch_dim,  # 8
            hidden_dim=config.hidden_dim,
            output_seq_len=config.touch_seq_len  # 16
        )
        self.projection = nn.Linear(
            self.input_dim,
            self.output_seq_len * self.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 8) - damage source info

        Returns:
            (batch, 16, hidden_dim) - expanded sequence
        """
        batch_size = x.shape[0]
        x = self.projection(x)
        return x.view(batch_size, self.output_seq_len, self.hidden_dim)


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
