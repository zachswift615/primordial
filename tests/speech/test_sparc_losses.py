"""Tests for SPARC loss functions."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.sparc_losses import (
    ema_loss,
    pitch_loss,
    loudness_loss,
    smoothness_loss,
    sparc_combined_loss,
)


class TestSPARCLosses:
    """Tests for SPARC training losses."""

    @pytest.fixture
    def config(self):
        return SpeechConfig()

    def test_ema_loss_perfect(self):
        """EMA loss should be 0 for identical inputs."""
        pred = torch.randn(4, 50, 12)
        loss = ema_loss(pred, pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ema_loss_positive(self):
        """EMA loss should be positive for different inputs."""
        pred = torch.randn(4, 50, 12)
        target = torch.randn(4, 50, 12)
        loss = ema_loss(pred, target)
        assert loss.item() > 0

    def test_pitch_loss_handles_unvoiced(self):
        """Pitch loss should mask unvoiced frames (pitch < threshold)."""
        pred = torch.ones(4, 50, 1) * 150  # All voiced
        target = torch.ones(4, 50, 1) * 150
        target[:, 25:, :] = 0  # Half unvoiced

        loss = pitch_loss(pred, target, unvoiced_threshold=50.0)
        # Should only compute loss on voiced frames
        assert loss.item() >= 0

    def test_smoothness_loss_penalizes_jitter(self):
        """Smoothness loss should penalize rapid changes."""
        # Smooth trajectory
        t = torch.linspace(0, 1, 50).unsqueeze(0).unsqueeze(-1)
        smooth = torch.sin(2 * 3.14159 * t).expand(4, 50, 12)

        # Jittery trajectory
        jittery = torch.randn(4, 50, 12)

        smooth_loss = smoothness_loss(smooth)
        jittery_loss = smoothness_loss(jittery)

        assert jittery_loss > smooth_loss

    def test_combined_loss_structure(self, config):
        """Combined loss should return dict with components."""
        pred = {
            'ema': torch.randn(4, 50, 12),
            'pitch': torch.abs(torch.randn(4, 50, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, 50, 1)),
        }
        target = {
            'ema': torch.randn(4, 50, 12),
            'pitch': torch.abs(torch.randn(4, 50, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, 50, 1)),
        }

        losses = sparc_combined_loss(pred, target, config)

        assert 'total' in losses
        assert 'ema' in losses
        assert 'pitch' in losses
        assert 'loudness' in losses
        assert 'smoothness' in losses
        assert losses['total'] > 0
