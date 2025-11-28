"""Fourier-based mixing layer for full LRN architecture."""
import torch
import torch.nn as nn

from .lrn_config import LRNConfig
from .utils import init_spectral_filter


class LRNFourierMixingLayer(nn.Module):
    """
    Fourier-based mixing layer for full LRN architecture.

    Uses LRNConfig with total_seq_len=164 and supports:
    - Spectral dropout (optional)
    - Configurable activation function (gelu/relu/swish)
    - Learnable spectral filters with low-frequency bias

    Replaces self-attention with O(n log n) FFT operations.
    Based on FNet and FFTNet 2025 research.

    Architecture:
    - Filter shape: (total_seq_len, freq_bins, 2)
    - Spectral bias initialization (favor low frequencies)
    - Slicing logic for hidden_dim/seq_len mismatch
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.total_seq_len

        # Number of frequency bins
        if config.use_real_fft:
            self.freq_bins = self.seq_len // 2 + 1  # rfft output size
        else:
            self.freq_bins = self.seq_len

        # Learnable spectral filter (stored as real tensor, converted to complex)
        # Shape: (seq_len, freq_bins, 2) for (real, imaginary) components
        # Use shared utility function with low-frequency bias
        self.spectral_filter = nn.Parameter(
            init_spectral_filter(self.seq_len, self.freq_bins)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_eps)

        # Configurable activation function
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "swish":
            self.activation = nn.SiLU()  # Swish = SiLU
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        # Optional spectral dropout
        self.dropout = nn.Dropout(config.spectral_dropout) if config.spectral_dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier mixing with learnable spectral filters.

        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        residual = x

        # Transpose to (batch, hidden_dim, seq_len) for FFT along sequence
        x = x.transpose(1, 2)  # (B, hidden_dim, seq_len)

        # Apply FFT along sequence dimension
        if self.config.use_real_fft:
            x_fft = torch.fft.rfft(x, dim=2)  # (B, hidden_dim, freq_bins) complex
        else:
            x_fft = torch.fft.fft(x, dim=2)   # (B, hidden_dim, seq_len) complex

        # Convert spectral filter to complex
        # (seq_len, freq_bins, 2) -> (seq_len, freq_bins) complex
        filter_complex = torch.view_as_complex(self.spectral_filter.contiguous())

        # Slice or repeat filter to match hidden_dim
        # (Parent LRN logic: filter dimension 0 is indexed by hidden_dim)
        if hidden_dim <= seq_len:
            filter_slice = filter_complex[:hidden_dim, :]  # (hidden_dim, freq_bins)
        else:
            # Repeat filter if hidden_dim > seq_len
            repeats = (hidden_dim + seq_len - 1) // seq_len
            filter_slice = filter_complex.repeat(repeats, 1)[:hidden_dim, :]

        # Apply spectral filtering
        x_filtered = x_fft * filter_slice.unsqueeze(0)  # (B, hidden_dim, freq_bins)

        # Optional spectral dropout
        if self.dropout is not None and self.training:
            # Apply dropout to magnitude while preserving phase
            magnitude = torch.abs(x_filtered)
            phase = torch.angle(x_filtered)
            magnitude = self.dropout(magnitude)
            x_filtered = magnitude * torch.exp(1j * phase)

        # Inverse FFT back to time domain
        if self.config.use_real_fft:
            x_out = torch.fft.irfft(x_filtered, n=seq_len, dim=2)  # (B, hidden_dim, seq_len)
        else:
            x_out = torch.fft.ifft(x_filtered, dim=2).real  # (B, hidden_dim, seq_len)

        # Transpose back to (batch, seq_len, hidden_dim)
        x_out = x_out.transpose(1, 2)  # (B, seq_len, hidden_dim)

        # Residual connection
        x_out = x_out + residual

        # Normalization and activation
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)

        return x_out
