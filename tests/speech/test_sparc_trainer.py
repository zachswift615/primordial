"""Tests for SPARC trainer."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.training import SpeechLRN, SPARCTrainer


class TestSPARCTrainer:
    """Tests for SPARC supervised training."""

    @pytest.fixture
    def config(self):
        return SpeechConfig(encoder_type='cnn', audio_duration=1.0)

    @pytest.fixture
    def model(self, config):
        return SpeechLRN(config, output_head='articulatory')

    @pytest.fixture
    def trainer(self, model, config):
        return SPARCTrainer(model, config, lr=1e-3)

    def test_train_step_returns_losses(self, trainer, config):
        """Train step should return loss dictionary."""
        batch_size = 4
        mel = torch.randn(batch_size, config.n_mels, config.n_frames)
        target = {
            'ema': torch.randn(batch_size, config.sparc_n_frames, 12),
            'pitch': torch.abs(torch.randn(batch_size, config.sparc_n_frames, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(batch_size, config.sparc_n_frames, 1)),
        }

        losses = trainer.train_step(mel, target)

        assert 'total' in losses
        assert 'ema' in losses
        assert losses['total'] > 0

    def test_train_step_updates_weights(self, trainer, config):
        """Train step should update model weights."""
        mel = torch.randn(4, config.n_mels, config.n_frames)
        target = {
            'ema': torch.randn(4, config.sparc_n_frames, 12),
            'pitch': torch.abs(torch.randn(4, config.sparc_n_frames, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, config.sparc_n_frames, 1)),
        }

        # Get initial weights
        initial_weights = trainer.model.articulatory_head.ema_head[0].weight.clone()

        # Train step
        trainer.train_step(mel, target)

        # Check weights changed
        final_weights = trainer.model.articulatory_head.ema_head[0].weight
        assert not torch.allclose(initial_weights, final_weights)

    def test_validation_step(self, trainer, config):
        """Validation step should not update weights."""
        mel = torch.randn(4, config.n_mels, config.n_frames)
        target = {
            'ema': torch.randn(4, config.sparc_n_frames, 12),
            'pitch': torch.abs(torch.randn(4, config.sparc_n_frames, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, config.sparc_n_frames, 1)),
        }

        initial_weights = trainer.model.articulatory_head.ema_head[0].weight.clone()

        losses = trainer.validation_step(mel, target)

        final_weights = trainer.model.articulatory_head.ema_head[0].weight
        assert torch.allclose(initial_weights, final_weights)
        assert 'total' in losses
