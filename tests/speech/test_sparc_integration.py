"""Tests for SPARC integration module."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.sparc_integration import SPARCWrapper, VoiceIdentity


class TestSPARCWrapper:
    """Tests for SPARC wrapper (mock mode for CI)."""

    @pytest.fixture
    def config(self):
        return SpeechConfig()

    @pytest.fixture
    def wrapper(self, config):
        # Use mock mode for testing without SPARC installed
        return SPARCWrapper(config, mock=True)

    def test_encode_shape(self, wrapper, config):
        """Encode should return correct feature shapes."""
        batch_size = 4
        audio_len = int(config.sample_rate * config.audio_duration)
        audio = torch.randn(batch_size, audio_len)

        ema, pitch, loudness = wrapper.encode(audio)

        expected_frames = config.sparc_n_frames
        assert ema.shape == (batch_size, expected_frames, 12)
        assert pitch.shape == (batch_size, expected_frames, 1)
        assert loudness.shape == (batch_size, expected_frames, 1)

    def test_decode_shape(self, wrapper, config):
        """Decode should return audio of correct length."""
        batch_size = 2
        n_frames = config.sparc_n_frames

        ema = torch.randn(batch_size, n_frames, 12)
        pitch = torch.randn(batch_size, n_frames, 1)
        loudness = torch.randn(batch_size, n_frames, 1)
        spk_emb = torch.randn(batch_size, 64)

        audio = wrapper.decode(ema, pitch, loudness, spk_emb)

        expected_samples = int(config.sample_rate * config.audio_duration)
        assert audio.shape == (batch_size, expected_samples)

    def test_decode_is_differentiable(self, wrapper, config):
        """Decode should support gradient flow."""
        n_frames = config.sparc_n_frames

        ema = torch.randn(1, n_frames, 12, requires_grad=True)
        pitch = torch.randn(1, n_frames, 1, requires_grad=True)
        loudness = torch.randn(1, n_frames, 1, requires_grad=True)
        spk_emb = torch.randn(1, 64)

        audio = wrapper.decode(ema, pitch, loudness, spk_emb)
        loss = audio.sum()
        loss.backward()

        assert ema.grad is not None
        assert pitch.grad is not None
        assert loudness.grad is not None


class TestVoiceIdentity:
    """Tests for voice identity management."""

    def test_random_embedding(self):
        """Should create random embedding when no file provided."""
        voice = VoiceIdentity(dim=64)
        emb = voice.get_embedding(batch_size=4)

        assert emb.shape == (4, 64)

    def test_embedding_is_fixed(self):
        """Same voice should return same embedding."""
        voice = VoiceIdentity(dim=64)
        emb1 = voice.get_embedding(batch_size=1)
        emb2 = voice.get_embedding(batch_size=1)

        assert torch.allclose(emb1, emb2)
