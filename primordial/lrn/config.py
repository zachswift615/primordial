"""Configuration for Fourier mixing prototype."""
from dataclasses import dataclass


@dataclass
class PrototypeConfig:
    """Minimal config for Fourier mixing validation."""

    # Sequence dimensions
    seq_len: int = 64
    hidden_dim: int = 32

    # Architecture
    num_mixing_layers: int = 2

    # FFT settings
    use_real_fft: bool = True

    # Normalization
    layer_norm_eps: float = 1e-5

    # Multi-task learning (matches full LRN)
    reward_horizon: int = 5  # Predict rewards for next N steps
    reward_loss_weight: float = 1.0  # Weight for reward loss vs sensory loss

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins for rfft."""
        return self.seq_len // 2 + 1 if self.use_real_fft else self.seq_len
