"""Configuration for Living Resonance Network (LRN)."""
from dataclasses import dataclass, field
from typing import Tuple, Literal


@dataclass
class LRNConfig:
    """Configuration for Living Resonance Network.

    Supports two environments:
    - "primordial": Original 2D simulation with ray-based vision
    - "minecraft": MineDojo 3D environment with RGB vision
    """

    # Environment selection (determines encoder types)
    environment: Literal["primordial", "minecraft"] = "primordial"

    # === Primordial-specific settings (used when environment="primordial") ===
    vision_shape: Tuple[int, int] = (32, 4)      # (rays, features: dist,r,g,b)
    audio_shape: Tuple[int, int] = (100, 2)      # (samples, stereo)
    proprio_dim: int = 7
    touch_dim: int = 8

    # === Minecraft-specific settings (used when environment="minecraft") ===
    mc_rgb_size: int = 64                        # RGB frame size (64x64 or 128x128)
    mc_rgb_channels: int = 3                     # RGB channels
    mc_proprio_dim: int = 10                     # health, food, oxygen, armor, etc.
    mc_touch_dim: int = 8                        # damage_source features
    mc_action_dim: int = 8                       # forward/back/left/right/jump/camera_x/camera_y/attack

    # === Shared architecture settings ===
    hidden_dim: int = 128                         # Embedding dimension
    num_mixing_layers: int = 6                   # Fourier mixing layers

    # Encoder output sequence lengths
    vision_seq_len: int = 32                     # Keep spatial structure
    audio_seq_len: int = 100                     # Keep temporal structure
    proprio_seq_len: int = 16                    # Expand to sequence
    touch_seq_len: int = 16                      # Expand to sequence

    # Heads
    pred_hidden_dim: int = 256
    action_hidden_dim: int = 128
    action_dim: int = 5                          # Output action space (Primordial default)
    reward_horizon: int = 5                      # Steps ahead to predict rewards
    reward_loss_weight: float = 1.0              # Weight for reward loss (1.0-2.0 recommended)

    # FFT settings
    use_real_fft: bool = True                    # Use rfft for efficiency
    spectral_dropout: float = 0.0                # Dropout in frequency domain

    # Normalization
    layer_norm_eps: float = 1e-5

    # Activation
    activation: str = "gelu"                     # or "relu", "swish"

    # Genome modulation (optional)
    genome_dim: int = 100                        # Size of genome vector
    use_genome_modulation: bool = True

    def __post_init__(self):
        """Apply environment-specific overrides."""
        if self.environment == "minecraft":
            # Override action_dim for Minecraft
            self.action_dim = self.mc_action_dim
            # Override proprio/touch dims
            self.proprio_dim = self.mc_proprio_dim
            self.touch_dim = self.mc_touch_dim
            # Disable genome modulation for Minecraft (no evolution yet)
            self.use_genome_modulation = False

    @property
    def total_seq_len(self) -> int:
        """Total sequence length after concatenation."""
        return (self.vision_seq_len + self.audio_seq_len +
                self.proprio_seq_len + self.touch_seq_len)

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins for rfft."""
        return self.total_seq_len // 2 + 1 if self.use_real_fft else self.total_seq_len

    @property
    def total_sensory_dim(self) -> int:
        """Total flattened sensory dimension for prediction head output."""
        return (self.vision_shape[0] * self.vision_shape[1] +  # 128
                self.audio_shape[0] * self.audio_shape[1] +    # 200
                self.proprio_dim +                              # 7
                self.touch_dim)                                 # 8 = 343
