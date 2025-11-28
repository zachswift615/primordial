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


def test_gradient_flow_through_fft():
    """Test gradients flow through FFT operations."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig()
    layer = FourierMixingLayer(config)

    x = torch.randn(1, 64, 32, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient exists
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf in input gradient"

    # Check spectral filter gradient exists
    assert layer.spectral_filter.grad is not None, "No gradient for spectral filter"
    assert not torch.isnan(layer.spectral_filter.grad).any(), "NaN in filter gradient"


def test_gradient_no_nan_after_many_steps():
    """Test gradients remain stable over multiple forward/backward passes."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig()
    layer = FourierMixingLayer(config)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    for step in range(100):
        x = torch.randn(4, 64, 32)
        output = layer(x)
        loss = output.sum()

        optimizer.zero_grad()
        loss.backward()

        # Check for NaN/Inf
        for name, param in layer.named_parameters():
            assert not torch.isnan(param.grad).any(), f"NaN gradient at step {step}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient at step {step}"

        optimizer.step()

        # Check weights didn't explode
        for name, param in layer.named_parameters():
            assert not torch.isnan(param).any(), f"NaN weights at step {step}"
