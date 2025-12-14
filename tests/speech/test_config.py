"""Tests for speech configuration."""
import pytest
from primordial.speech.config import SpeechConfig


def test_sparc_config_defaults():
    """SPARC configuration should have sensible defaults."""
    config = SpeechConfig()

    # SPARC feature dimensions
    assert config.sparc_ema_dim == 12
    assert config.sparc_frame_rate == 50  # Hz

    # Voice embedding
    assert config.sparc_speaker_dim == 64

    # Loss weights
    assert config.ema_loss_weight == 1.0
    assert config.sparc_pitch_loss_weight == 0.5
    assert config.sparc_loudness_loss_weight == 0.3
    assert config.smoothness_loss_weight == 0.1


def test_sparc_output_frames():
    """Should compute correct SPARC frame count for audio duration."""
    config = SpeechConfig(audio_duration=2.0)
    # 2 seconds at 50Hz = 100 frames
    assert config.sparc_n_frames == 100


def test_sparc_mel_to_sparc_ratio():
    """Should compute mel-to-SPARC frame ratio."""
    config = SpeechConfig()
    # Mel at 100Hz (hop=160 @ 16kHz), SPARC at 50Hz
    # Ratio = 100/50 = 2
    assert config.mel_to_sparc_ratio == 2.0
