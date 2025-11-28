"""Tests for LRNConfig."""
import pytest
from primordial.lrn.lrn_config import LRNConfig


def test_default_values():
    """Test that default configuration values are set correctly."""
    config = LRNConfig()

    # Input dimensions
    assert config.vision_shape == (32, 4)
    assert config.audio_shape == (100, 2)
    assert config.proprio_dim == 7
    assert config.touch_dim == 8

    # Architecture
    assert config.hidden_dim == 128
    assert config.num_mixing_layers == 6

    # Encoder sequence lengths
    assert config.vision_seq_len == 32
    assert config.audio_seq_len == 100
    assert config.proprio_seq_len == 16
    assert config.touch_seq_len == 16

    # Heads
    assert config.pred_hidden_dim == 256
    assert config.action_hidden_dim == 128
    assert config.action_dim == 5
    assert config.reward_horizon == 5
    assert config.reward_loss_weight == 1.0

    # FFT settings
    assert config.use_real_fft is True
    assert config.spectral_dropout == 0.0

    # Normalization
    assert config.layer_norm_eps == 1e-5

    # Activation
    assert config.activation == "gelu"

    # Genome modulation
    assert config.genome_dim == 100
    assert config.use_genome_modulation is True


def test_computed_property_total_seq_len():
    """Test total_seq_len computed property."""
    config = LRNConfig()

    # 32 + 100 + 16 + 16 = 164
    assert config.total_seq_len == 164


def test_computed_property_freq_bins():
    """Test freq_bins computed property."""
    # Test with real FFT (default)
    config = LRNConfig()
    # 164 // 2 + 1 = 83
    assert config.freq_bins == 83

    # Test with full FFT
    config_full = LRNConfig(use_real_fft=False)
    assert config_full.freq_bins == 164


def test_computed_property_total_sensory_dim():
    """Test total_sensory_dim computed property."""
    config = LRNConfig()

    # (32*4) + (100*2) + 7 + 8 = 128 + 200 + 7 + 8 = 343
    assert config.total_sensory_dim == 343


def test_custom_values_override_defaults():
    """Test that custom values properly override defaults."""
    config = LRNConfig(
        vision_shape=(64, 8),
        audio_shape=(200, 2),
        proprio_dim=14,
        touch_dim=16,
        hidden_dim=256,
        num_mixing_layers=12,
        vision_seq_len=64,
        audio_seq_len=200,
        proprio_seq_len=32,
        touch_seq_len=32,
        pred_hidden_dim=512,
        action_hidden_dim=256,
        action_dim=10,
        reward_horizon=10,
        reward_loss_weight=2.0,
        use_real_fft=False,
        spectral_dropout=0.1,
        layer_norm_eps=1e-6,
        activation="relu",
        genome_dim=200,
        use_genome_modulation=False
    )

    # Verify all custom values
    assert config.vision_shape == (64, 8)
    assert config.audio_shape == (200, 2)
    assert config.proprio_dim == 14
    assert config.touch_dim == 16
    assert config.hidden_dim == 256
    assert config.num_mixing_layers == 12
    assert config.vision_seq_len == 64
    assert config.audio_seq_len == 200
    assert config.proprio_seq_len == 32
    assert config.touch_seq_len == 32
    assert config.pred_hidden_dim == 512
    assert config.action_hidden_dim == 256
    assert config.action_dim == 10
    assert config.reward_horizon == 10
    assert config.reward_loss_weight == 2.0
    assert config.use_real_fft is False
    assert config.spectral_dropout == 0.1
    assert config.layer_norm_eps == 1e-6
    assert config.activation == "relu"
    assert config.genome_dim == 200
    assert config.use_genome_modulation is False


def test_computed_properties_with_custom_values():
    """Test computed properties update correctly with custom values."""
    config = LRNConfig(
        vision_seq_len=64,
        audio_seq_len=200,
        proprio_seq_len=32,
        touch_seq_len=32,
        vision_shape=(64, 8),
        audio_shape=(200, 2),
        proprio_dim=14,
        touch_dim=16,
        use_real_fft=False
    )

    # Total sequence length: 64 + 200 + 32 + 32 = 328
    assert config.total_seq_len == 328

    # Freq bins (full FFT): 328
    assert config.freq_bins == 328

    # Total sensory dim: (64*8) + (200*2) + 14 + 16 = 512 + 400 + 14 + 16 = 942
    assert config.total_sensory_dim == 942


def test_activation_options():
    """Test different activation function options."""
    config_gelu = LRNConfig(activation="gelu")
    assert config_gelu.activation == "gelu"

    config_relu = LRNConfig(activation="relu")
    assert config_relu.activation == "relu"

    config_swish = LRNConfig(activation="swish")
    assert config_swish.activation == "swish"


def test_reward_horizon_options():
    """Test different reward horizon configurations."""
    # Short horizon
    config_short = LRNConfig(reward_horizon=1)
    assert config_short.reward_horizon == 1

    # Default horizon
    config_default = LRNConfig()
    assert config_default.reward_horizon == 5

    # Long horizon
    config_long = LRNConfig(reward_horizon=10)
    assert config_long.reward_horizon == 10


def test_fft_mode_options():
    """Test FFT mode configurations."""
    # Real FFT (efficient)
    config_real = LRNConfig(use_real_fft=True)
    assert config_real.use_real_fft is True
    assert config_real.freq_bins == 83  # 164 // 2 + 1

    # Full FFT
    config_full = LRNConfig(use_real_fft=False)
    assert config_full.use_real_fft is False
    assert config_full.freq_bins == 164


def test_spectral_dropout_range():
    """Test spectral dropout configurations."""
    # No dropout
    config_no_dropout = LRNConfig(spectral_dropout=0.0)
    assert config_no_dropout.spectral_dropout == 0.0

    # Light dropout
    config_light = LRNConfig(spectral_dropout=0.1)
    assert config_light.spectral_dropout == 0.1

    # Heavy dropout
    config_heavy = LRNConfig(spectral_dropout=0.5)
    assert config_heavy.spectral_dropout == 0.5


def test_genome_modulation_toggle():
    """Test genome modulation on/off."""
    # Enabled (default)
    config_enabled = LRNConfig()
    assert config_enabled.use_genome_modulation is True
    assert config_enabled.genome_dim == 100

    # Disabled
    config_disabled = LRNConfig(use_genome_modulation=False)
    assert config_disabled.use_genome_modulation is False

    # Custom genome dimension
    config_custom = LRNConfig(genome_dim=256)
    assert config_custom.genome_dim == 256
