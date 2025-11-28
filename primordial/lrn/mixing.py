"""Fourier-based mixing layer with learnable spectral filters."""
import torch
import torch.nn as nn

from .config import PrototypeConfig


class FourierMixingLayer(nn.Module):
    """
    Fourier-based mixing layer with learnable spectral filters.

    Replaces self-attention with O(n log n) FFT operations.
    Based on FNet and FFTNet 2025 research.

    Architecture matches parent LRN spec:
    - Filter shape: (seq_len, freq_bins, 2)
    - Spectral bias initialization (favor low frequencies)
    - Slicing logic for hidden_dim/seq_len mismatch
    """

    def __init__(self, config: PrototypeConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.seq_len
        self.freq_bins = config.freq_bins

        # Learnable spectral filter (stored as real tensor, converted to complex)
        # Shape: (seq_len, freq_bins, 2) - matches parent LRN architecture
        self.spectral_filter = nn.Parameter(
            self._init_spectral_filter()
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_eps)

        # Activation
        self.activation = nn.GELU()

    def _init_spectral_filter(self) -> torch.Tensor:
        """
        Initialize spectral filter with frequency-dependent decay (spectral bias).

        Low frequencies get larger initial values, high frequencies get smaller.
        This matches biological and empirical observations that neural networks
        learn low frequencies first.
        """
        # Frequency decay: exp(-freq / (freq_bins / 4))
        freqs = torch.arange(self.freq_bins, dtype=torch.float32)
        decay = torch.exp(-freqs / (self.freq_bins / 4))

        # Initialize with decay applied
        # Shape: (seq_len, freq_bins, 2) for real and imaginary
        filter_init = torch.randn(self.seq_len, self.freq_bins, 2) * 0.1
        filter_init = filter_init * decay.unsqueeze(0).unsqueeze(-1)

        return filter_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier mixing with learnable spectral filters.

        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        residual = x

        # Transpose to (batch, hidden_dim, seq_len) for FFT along sequence
        x = x.transpose(1, 2)  # (B, hidden_dim, seq_len)

        # Apply FFT along sequence dimension
        if self.config.use_real_fft:
            x_fft = torch.fft.rfft(x, dim=2)  # (B, hidden_dim, freq_bins) complex
        else:
            x_fft = torch.fft.fft(x, dim=2)

        # Convert spectral filter to complex
        # Shape: (seq_len, freq_bins) complex
        filter_complex = torch.view_as_complex(self.spectral_filter.contiguous())

        # Slice or repeat filter to match hidden_dim (parent LRN logic)
        if self.hidden_dim <= self.seq_len:
            filter_slice = filter_complex[:self.hidden_dim, :]  # (hidden_dim, freq_bins)
        else:
            # Repeat filter if hidden_dim > seq_len
            repeats = (self.hidden_dim + self.seq_len - 1) // self.seq_len
            filter_slice = filter_complex.repeat(repeats, 1)[:self.hidden_dim, :]

        # Apply spectral filtering
        x_filtered = x_fft * filter_slice.unsqueeze(0)  # (B, hidden_dim, freq_bins)

        # Inverse FFT back to time domain
        if self.config.use_real_fft:
            x_out = torch.fft.irfft(x_filtered, n=self.seq_len, dim=2)
        else:
            x_out = torch.fft.ifft(x_filtered, dim=2).real

        # Transpose back to (batch, seq_len, hidden_dim)
        x_out = x_out.transpose(1, 2)

        # Residual connection, normalization, activation
        x_out = x_out + residual
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)

        return x_out
