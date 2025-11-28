"""Tests for FourierMixingLayer."""
import torch
import pytest


def test_fourier_mixing_layer_output_shape():
    """Test that output shape matches input shape."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig(seq_len=64, hidden_dim=32)
    layer = FourierMixingLayer(config)

    x = torch.randn(1, 64, 32)  # (batch, seq_len, hidden_dim)
    output = layer(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"


def test_fourier_mixing_layer_batch():
    """Test with larger batch size."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig(seq_len=64, hidden_dim=32)
    layer = FourierMixingLayer(config)

    x = torch.randn(8, 64, 32)
    output = layer(x)

    assert output.shape == (8, 64, 32)
