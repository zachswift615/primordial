"""Tests for modality encoders."""
import torch
import pytest
from primordial.lrn.lrn_config import LRNConfig
from primordial.lrn.encoders import (
    WaveletEncoder,
    VisionEncoder,
    AudioEncoder,
    ProprioEncoder,
    TouchEncoder,
)


# ============================================================================
# Vision Encoder Tests
# ============================================================================


def test_vision_encoder_output_shape():
    """Test VisionEncoder produces correct output shape."""
    config = LRNConfig()
    encoder = VisionEncoder(config)

    # Input: (batch, 32, 4)
    batch_size = 4
    x = torch.randn(batch_size, 32, 4)
    output = encoder(x)

    # Expected: (batch, 32, 128)
    expected_shape = (batch_size, config.vision_seq_len, config.hidden_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_vision_encoder_batch_processing():
    """Test VisionEncoder handles different batch sizes."""
    config = LRNConfig()
    encoder = VisionEncoder(config)

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, 32, 4)
        output = encoder(x)
        assert output.shape == (batch_size, 32, config.hidden_dim)


def test_vision_encoder_gradient_flow():
    """Test gradients flow through VisionEncoder."""
    config = LRNConfig()
    encoder = VisionEncoder(config)

    x = torch.randn(2, 32, 4, requires_grad=True)
    output = encoder(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf in input gradient"

    # Check projection weights gradient
    assert encoder.projection.weight.grad is not None, "No gradient for projection weights"
    assert not torch.isnan(encoder.projection.weight.grad).any(), "NaN in weight gradient"


# ============================================================================
# Audio Encoder Tests
# ============================================================================


def test_audio_encoder_output_shape():
    """Test AudioEncoder produces correct output shape."""
    config = LRNConfig()
    encoder = AudioEncoder(config)

    # Input: (batch, 100, 2)
    batch_size = 4
    x = torch.randn(batch_size, 100, 2)
    output = encoder(x)

    # Expected: (batch, 100, 128)
    expected_shape = (batch_size, config.audio_seq_len, config.hidden_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_audio_encoder_batch_processing():
    """Test AudioEncoder handles different batch sizes."""
    config = LRNConfig()
    encoder = AudioEncoder(config)

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, 100, 2)
        output = encoder(x)
        assert output.shape == (batch_size, 100, config.hidden_dim)


def test_audio_encoder_gradient_flow():
    """Test gradients flow through AudioEncoder."""
    config = LRNConfig()
    encoder = AudioEncoder(config)

    x = torch.randn(2, 100, 2, requires_grad=True)
    output = encoder(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf in input gradient"

    # Check projection weights gradient
    assert encoder.projection.weight.grad is not None, "No gradient for projection weights"
    assert not torch.isnan(encoder.projection.weight.grad).any(), "NaN in weight gradient"


# ============================================================================
# Proprio Encoder Tests
# ============================================================================


def test_proprio_encoder_output_shape():
    """Test ProprioEncoder produces correct output shape."""
    config = LRNConfig()
    encoder = ProprioEncoder(config)

    # Input: (batch, 7)
    batch_size = 4
    x = torch.randn(batch_size, 7)
    output = encoder(x)

    # Expected: (batch, 16, 128)
    expected_shape = (batch_size, config.proprio_seq_len, config.hidden_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_proprio_encoder_batch_processing():
    """Test ProprioEncoder handles different batch sizes."""
    config = LRNConfig()
    encoder = ProprioEncoder(config)

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, 7)
        output = encoder(x)
        assert output.shape == (batch_size, 16, config.hidden_dim)


def test_proprio_encoder_gradient_flow():
    """Test gradients flow through ProprioEncoder."""
    config = LRNConfig()
    encoder = ProprioEncoder(config)

    x = torch.randn(2, 7, requires_grad=True)
    output = encoder(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf in input gradient"

    # Check projection weights gradient
    assert encoder.projection.weight.grad is not None, "No gradient for projection weights"
    assert not torch.isnan(encoder.projection.weight.grad).any(), "NaN in weight gradient"


def test_proprio_encoder_expansion():
    """Test ProprioEncoder correctly expands vector to sequence."""
    config = LRNConfig()
    encoder = ProprioEncoder(config)

    # Verify projection layer creates correct output size
    batch_size = 2
    x = torch.randn(batch_size, 7)
    output = encoder(x)

    # Check sequence dimension is properly created
    assert output.shape[1] == config.proprio_seq_len
    assert output.shape[2] == config.hidden_dim


# ============================================================================
# Touch Encoder Tests
# ============================================================================


def test_touch_encoder_output_shape():
    """Test TouchEncoder produces correct output shape."""
    config = LRNConfig()
    encoder = TouchEncoder(config)

    # Input: (batch, 8)
    batch_size = 4
    x = torch.randn(batch_size, 8)
    output = encoder(x)

    # Expected: (batch, 16, 128)
    expected_shape = (batch_size, config.touch_seq_len, config.hidden_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_touch_encoder_batch_processing():
    """Test TouchEncoder handles different batch sizes."""
    config = LRNConfig()
    encoder = TouchEncoder(config)

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, 8)
        output = encoder(x)
        assert output.shape == (batch_size, 16, config.hidden_dim)


def test_touch_encoder_gradient_flow():
    """Test gradients flow through TouchEncoder."""
    config = LRNConfig()
    encoder = TouchEncoder(config)

    x = torch.randn(2, 8, requires_grad=True)
    output = encoder(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf in input gradient"

    # Check projection weights gradient
    assert encoder.projection.weight.grad is not None, "No gradient for projection weights"
    assert not torch.isnan(encoder.projection.weight.grad).any(), "NaN in weight gradient"


def test_touch_encoder_expansion():
    """Test TouchEncoder correctly expands vector to sequence."""
    config = LRNConfig()
    encoder = TouchEncoder(config)

    # Verify projection layer creates correct output size
    batch_size = 2
    x = torch.randn(batch_size, 8)
    output = encoder(x)

    # Check sequence dimension is properly created
    assert output.shape[1] == config.touch_seq_len
    assert output.shape[2] == config.hidden_dim


# ============================================================================
# Custom Config Tests
# ============================================================================


def test_vision_encoder_custom_config():
    """Test VisionEncoder with custom config values."""
    config = LRNConfig(hidden_dim=256, vision_seq_len=64)
    encoder = VisionEncoder(config)

    x = torch.randn(2, 64, 4)
    output = encoder(x)

    assert output.shape == (2, 64, 256)


def test_audio_encoder_custom_config():
    """Test AudioEncoder with custom config values."""
    config = LRNConfig(hidden_dim=256, audio_seq_len=200)
    encoder = AudioEncoder(config)

    x = torch.randn(2, 200, 2)
    output = encoder(x)

    assert output.shape == (2, 200, 256)


def test_proprio_encoder_custom_config():
    """Test ProprioEncoder with custom config values."""
    config = LRNConfig(hidden_dim=256, proprio_seq_len=32, proprio_dim=10)
    encoder = ProprioEncoder(config)

    x = torch.randn(2, 10)
    output = encoder(x)

    assert output.shape == (2, 32, 256)


def test_touch_encoder_custom_config():
    """Test TouchEncoder with custom config values."""
    config = LRNConfig(hidden_dim=256, touch_seq_len=32, touch_dim=12)
    encoder = TouchEncoder(config)

    x = torch.randn(2, 12)
    output = encoder(x)

    assert output.shape == (2, 32, 256)


# ============================================================================
# Base Class Tests
# ============================================================================


def test_wavelet_encoder_abstract():
    """Test WaveletEncoder base class raises NotImplementedError."""
    encoder = WaveletEncoder(input_dim=4, hidden_dim=128, output_seq_len=32)

    with pytest.raises(NotImplementedError):
        encoder(torch.randn(1, 32, 4))


def test_encoder_inheritance():
    """Test all encoders inherit from WaveletEncoder."""
    assert issubclass(VisionEncoder, WaveletEncoder)
    assert issubclass(AudioEncoder, WaveletEncoder)
    assert issubclass(ProprioEncoder, WaveletEncoder)
    assert issubclass(TouchEncoder, WaveletEncoder)


# ============================================================================
# Integration Tests
# ============================================================================


def test_all_encoders_together():
    """Test all encoders can be created and used together."""
    config = LRNConfig()

    vision_enc = VisionEncoder(config)
    audio_enc = AudioEncoder(config)
    proprio_enc = ProprioEncoder(config)
    touch_enc = TouchEncoder(config)

    batch_size = 4
    vision = torch.randn(batch_size, 32, 4)
    audio = torch.randn(batch_size, 100, 2)
    proprio = torch.randn(batch_size, 7)
    touch = torch.randn(batch_size, 8)

    vision_out = vision_enc(vision)
    audio_out = audio_enc(audio)
    proprio_out = proprio_enc(proprio)
    touch_out = touch_enc(touch)

    # Verify all outputs have correct batch size and hidden dim
    assert vision_out.shape[0] == batch_size
    assert audio_out.shape[0] == batch_size
    assert proprio_out.shape[0] == batch_size
    assert touch_out.shape[0] == batch_size

    assert vision_out.shape[2] == config.hidden_dim
    assert audio_out.shape[2] == config.hidden_dim
    assert proprio_out.shape[2] == config.hidden_dim
    assert touch_out.shape[2] == config.hidden_dim

    # Verify sequence lengths
    assert vision_out.shape[1] == config.vision_seq_len
    assert audio_out.shape[1] == config.audio_seq_len
    assert proprio_out.shape[1] == config.proprio_seq_len
    assert touch_out.shape[1] == config.touch_seq_len


def test_concatenation_of_all_encoders():
    """Test all encoder outputs can be concatenated."""
    config = LRNConfig()

    vision_enc = VisionEncoder(config)
    audio_enc = AudioEncoder(config)
    proprio_enc = ProprioEncoder(config)
    touch_enc = TouchEncoder(config)

    batch_size = 4
    vision = torch.randn(batch_size, 32, 4)
    audio = torch.randn(batch_size, 100, 2)
    proprio = torch.randn(batch_size, 7)
    touch = torch.randn(batch_size, 8)

    vision_out = vision_enc(vision)
    audio_out = audio_enc(audio)
    proprio_out = proprio_enc(proprio)
    touch_out = touch_enc(touch)

    # Concatenate along sequence dimension
    combined = torch.cat([vision_out, audio_out, proprio_out, touch_out], dim=1)

    # Verify combined shape
    expected_seq_len = config.total_seq_len  # 32 + 100 + 16 + 16 = 164
    assert combined.shape == (batch_size, expected_seq_len, config.hidden_dim)
