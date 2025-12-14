"""Tests for speech output heads."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.heads import ArticulatoryHead


class TestArticulatoryHead:
    """Tests for SPARC articulatory output head."""

    @pytest.fixture
    def config(self):
        return SpeechConfig(hidden_dim=128, audio_duration=1.0)

    @pytest.fixture
    def head(self, config):
        return ArticulatoryHead(config, input_dim=384)

    def test_output_shapes(self, head, config):
        """Should output correct SPARC feature shapes."""
        batch_size = 4
        pooled = torch.randn(batch_size, 384)

        output = head(pooled)

        n_frames = config.sparc_n_frames
        assert output['ema'].shape == (batch_size, n_frames, 12)
        assert output['pitch'].shape == (batch_size, n_frames, 1)
        assert output['loudness'].shape == (batch_size, n_frames, 1)

    def test_ema_range(self, head):
        """EMA should be in reasonable range (tanh bounded)."""
        pooled = torch.randn(4, 384)
        output = head(pooled)

        # Tanh output is in [-1, 1]
        assert output['ema'].min() >= -1.0
        assert output['ema'].max() <= 1.0

    def test_pitch_positive(self, head):
        """Pitch should be positive Hz values."""
        pooled = torch.randn(4, 384)
        output = head(pooled)

        # Pitch should be positive (softplus + offset)
        assert output['pitch'].min() > 0

    def test_loudness_range(self, head):
        """Loudness should be in [0, 1]."""
        pooled = torch.randn(4, 384)
        output = head(pooled)

        assert output['loudness'].min() >= 0
        assert output['loudness'].max() <= 1

    def test_differentiable(self, head):
        """Head should support gradient flow."""
        pooled = torch.randn(4, 384, requires_grad=True)
        output = head(pooled)

        loss = output['ema'].sum() + output['pitch'].sum() + output['loudness'].sum()
        loss.backward()

        assert pooled.grad is not None

    def test_from_sequence(self, config):
        """Should work with sequence input (no pooling)."""
        head = ArticulatoryHead(config, input_dim=128, from_sequence=True)

        # Sequence input: (batch, seq_len, hidden_dim)
        seq = torch.randn(4, 100, 128)
        output = head(seq)

        n_frames = config.sparc_n_frames
        assert output['ema'].shape == (4, n_frames, 12)
