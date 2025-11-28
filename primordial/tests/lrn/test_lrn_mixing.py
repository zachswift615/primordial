"""Tests for LRNFourierMixingLayer."""
import torch
import pytest


def test_lrn_mixing_layer_output_shape():
    """Test that output shape matches input shape with LRNConfig."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig()
    layer = LRNFourierMixingLayer(config)

    # LRNConfig: total_seq_len=164, hidden_dim=128
    x = torch.randn(2, 164, 128)  # (batch, seq_len, hidden_dim)
    output = layer(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert output.shape == (2, 164, 128)


def test_lrn_mixing_layer_batch():
    """Test with larger batch size."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig()
    layer = LRNFourierMixingLayer(config)

    x = torch.randn(16, 164, 128)
    output = layer(x)

    assert output.shape == (16, 164, 128)


def test_spectral_dropout_when_enabled():
    """Test that spectral dropout is applied when enabled during training."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(spectral_dropout=0.5)
    layer = LRNFourierMixingLayer(config)
    layer.train()  # Enable training mode

    x = torch.randn(4, 164, 128)

    # Run multiple times and check for variation (dropout should cause differences)
    outputs = []
    for _ in range(5):
        with torch.no_grad():
            output = layer(x)
            outputs.append(output)

    # Outputs should be different due to dropout
    assert not torch.allclose(outputs[0], outputs[1]), "Dropout should cause variation"


def test_spectral_dropout_disabled_in_eval():
    """Test that spectral dropout is disabled in eval mode."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(spectral_dropout=0.5)
    layer = LRNFourierMixingLayer(config)
    layer.eval()  # Disable dropout

    x = torch.randn(4, 164, 128)

    # Run multiple times - outputs should be identical in eval mode
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = layer(x)
            outputs.append(output)

    # Outputs should be identical without dropout
    assert torch.allclose(outputs[0], outputs[1]), "Eval mode should be deterministic"
    assert torch.allclose(outputs[1], outputs[2]), "Eval mode should be deterministic"


def test_no_dropout_when_disabled():
    """Test that no dropout is applied when spectral_dropout=0."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(spectral_dropout=0.0)
    layer = LRNFourierMixingLayer(config)
    layer.train()

    x = torch.randn(4, 164, 128)

    # Run multiple times - should be identical even in training mode
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = layer(x)
            outputs.append(output)

    assert torch.allclose(outputs[0], outputs[1]), "No dropout should be deterministic"


def test_activation_gelu():
    """Test GELU activation function."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(activation="gelu")
    layer = LRNFourierMixingLayer(config)

    assert isinstance(layer.activation, torch.nn.GELU)

    x = torch.randn(2, 164, 128)
    output = layer(x)
    assert output.shape == (2, 164, 128)


def test_activation_relu():
    """Test ReLU activation function."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(activation="relu")
    layer = LRNFourierMixingLayer(config)

    assert isinstance(layer.activation, torch.nn.ReLU)

    x = torch.randn(2, 164, 128)
    output = layer(x)
    assert output.shape == (2, 164, 128)


def test_activation_swish():
    """Test Swish (SiLU) activation function."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(activation="swish")
    layer = LRNFourierMixingLayer(config)

    assert isinstance(layer.activation, torch.nn.SiLU)

    x = torch.randn(2, 164, 128)
    output = layer(x)
    assert output.shape == (2, 164, 128)


def test_invalid_activation_raises():
    """Test that invalid activation raises ValueError."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(activation="invalid")

    with pytest.raises(ValueError, match="Unsupported activation"):
        layer = LRNFourierMixingLayer(config)


def test_gradient_flow():
    """Test gradients flow through FFT operations."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig()
    layer = LRNFourierMixingLayer(config)

    x = torch.randn(2, 164, 128, requires_grad=True)
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


def test_gradient_stability():
    """Test gradients remain stable over multiple forward/backward passes."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig()
    layer = LRNFourierMixingLayer(config)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    for step in range(100):
        x = torch.randn(4, 164, 128)
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


def test_stacked_layers():
    """Test that 6 stacked layers work correctly (full LRN architecture)."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(num_mixing_layers=6)

    # Create stack of 6 layers
    layers = torch.nn.ModuleList([
        LRNFourierMixingLayer(config) for _ in range(config.num_mixing_layers)
    ])

    x = torch.randn(2, 164, 128)

    # Forward through all layers
    for layer in layers:
        x = layer(x)

    assert x.shape == (2, 164, 128), "Output shape should be preserved"
    assert not torch.isnan(x).any(), "Output should not contain NaN"
    assert not torch.isinf(x).any(), "Output should not contain Inf"


def test_stacked_layers_gradient_flow():
    """Test gradients flow through 6 stacked layers."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig(num_mixing_layers=6)

    # Create stack of 6 layers
    layers = torch.nn.Sequential(*[
        LRNFourierMixingLayer(config) for _ in range(config.num_mixing_layers)
    ])

    x = torch.randn(2, 164, 128, requires_grad=True)
    output = layers(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient exists
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    # Check all layers have gradients
    for i, layer in enumerate(layers):
        assert layer.spectral_filter.grad is not None, f"No gradient for layer {i}"
        assert not torch.isnan(layer.spectral_filter.grad).any(), f"NaN in layer {i} gradient"


def test_uses_shared_init_spectral_filter():
    """Test that layer uses shared init_spectral_filter utility."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer
    from primordial.lrn.utils import init_spectral_filter

    config = LRNConfig()
    layer = LRNFourierMixingLayer(config)

    # Check filter shape matches expected from init_spectral_filter
    expected_shape = (config.total_seq_len, config.freq_bins, 2)
    assert layer.spectral_filter.shape == expected_shape

    # Verify low-frequency bias: first frequencies should have larger magnitudes
    filter_magnitudes = torch.norm(layer.spectral_filter, dim=2).mean(dim=0)

    # Low frequencies should be larger than high frequencies
    low_freq_mag = filter_magnitudes[:10].mean()
    high_freq_mag = filter_magnitudes[-10:].mean()

    assert low_freq_mag > high_freq_mag, "Low frequencies should have larger initial values"


def test_total_seq_len_property():
    """Test that layer correctly uses config.total_seq_len."""
    from primordial.lrn.lrn_config import LRNConfig
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer

    config = LRNConfig()
    layer = LRNFourierMixingLayer(config)

    # Check seq_len is total_seq_len
    assert layer.seq_len == config.total_seq_len
    assert layer.seq_len == 164  # 32 + 100 + 16 + 16

    # Check freq_bins calculation
    expected_freq_bins = config.total_seq_len // 2 + 1 if config.use_real_fft else config.total_seq_len
    assert layer.freq_bins == expected_freq_bins
