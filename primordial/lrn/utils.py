"""FFT utility functions for LRN architecture."""
import torch


def init_spectral_filter(seq_len: int, freq_bins: int) -> torch.Tensor:
    """Initialize spectral filter with low-frequency bias.

    Low frequencies get larger initial values (~50x larger than high frequencies).
    Formula: decay = exp(-freq / (freq_bins / 4))

    Args:
        seq_len: Sequence length (filter dimension 0)
        freq_bins: Number of frequency bins (filter dimension 1)

    Returns:
        Tensor of shape (seq_len, freq_bins, 2) for real/imaginary components
    """
    freqs = torch.arange(freq_bins, dtype=torch.float32)
    decay = torch.exp(-freqs / (freq_bins / 4))  # Decay high frequencies

    # Real and imaginary components
    filter_init = torch.randn(seq_len, freq_bins, 2) * 0.1
    filter_init = filter_init * decay.unsqueeze(0).unsqueeze(-1)

    return filter_init


def complex_to_real(c: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to real tensor with (*, 2) shape.

    Args:
        c: Complex tensor of any shape

    Returns:
        Real tensor with shape (*c.shape, 2) where last dim is [real, imag]
    """
    return torch.stack([c.real, c.imag], dim=-1)


def real_to_complex(r: torch.Tensor) -> torch.Tensor:
    """Convert (*, 2) real tensor to complex tensor.

    Args:
        r: Real tensor with last dimension 2 for [real, imag]

    Returns:
        Complex tensor with shape r.shape[:-1]
    """
    return torch.view_as_complex(r.contiguous())
