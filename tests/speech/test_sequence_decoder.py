"""Tests for autoregressive sequence decoder."""
import torch
import pytest
from primordial.speech.sequence_decoder import SequenceDecoder, SpeechSequenceLRN
from primordial.speech.config import SpeechConfig
from primordial.speech.latent import SOS_TOKEN, EOS_TOKEN


@pytest.fixture
def config():
    return SpeechConfig()


@pytest.fixture
def decoder(config):
    return SequenceDecoder(config)


def test_decoder_init(decoder):
    """Decoder should initialize with correct components."""
    assert hasattr(decoder, 'phoneme_embed')
    assert hasattr(decoder, 'pos_encoding')
    assert hasattr(decoder, 'transformer')
    assert hasattr(decoder, 'memory_proj')
    assert hasattr(decoder, 'discrete_head')
    assert hasattr(decoder, 'latent_head')


def test_decoder_forward_shape(decoder):
    """Forward pass should produce correct output shapes."""
    batch_size = 4
    seq_len = 5

    # Inputs
    input_ids = torch.randint(0, 43, (batch_size, seq_len))
    memory = torch.randn(batch_size, 384)  # Pooled audio encoding

    # Forward
    discrete_logits, latent = decoder(input_ids, memory)

    # Check shapes
    assert discrete_logits.shape == (batch_size, seq_len, 43)
    assert latent.shape == (batch_size, seq_len, 6)


def test_decoder_latent_bounded(decoder):
    """Latent output should be bounded to [-1, 1] via tanh."""
    input_ids = torch.randint(0, 43, (2, 3))
    memory = torch.randn(2, 384)

    _, latent = decoder(input_ids, memory)

    assert latent.min() >= -1.0
    assert latent.max() <= 1.0


def test_causal_mask_shape(decoder):
    """Causal mask should be upper triangular."""
    mask = decoder._generate_causal_mask(5, 'cpu')

    assert mask.shape == (5, 5)
    assert mask.dtype == torch.bool
    # Upper triangle (excluding diagonal) should be True (masked)
    assert mask[0, 1] == True
    assert mask[0, 4] == True
    # Diagonal and below should be False (visible)
    assert mask[0, 0] == False
    assert mask[4, 0] == False
    assert mask[4, 4] == False


def test_module_exports():
    """SequenceDecoder should be importable from speech module."""
    from primordial.speech import SequenceDecoder, SOS_TOKEN, EOS_TOKEN, TOTAL_VOCAB

    assert SequenceDecoder is not None
    assert SOS_TOKEN == 41
    assert EOS_TOKEN == 42
    assert TOTAL_VOCAB == 43


@pytest.fixture
def full_model(config):
    return SpeechSequenceLRN(config)


def test_full_model_init(full_model):
    """Full model should have encoder and decoder."""
    assert hasattr(full_model, 'encoder')
    assert hasattr(full_model, 'decoder')


def test_full_model_forward(full_model):
    """Full model forward pass should work end-to-end."""
    batch_size = 2
    seq_len = 4

    # Mel spectrogram input
    mel = torch.randn(batch_size, 80, 64)
    # Target tokens for teacher forcing
    target_tokens = torch.randint(0, 41, (batch_size, seq_len))

    discrete_logits, latent = full_model(mel, target_tokens)

    assert discrete_logits.shape == (batch_size, seq_len, 43)
    assert latent.shape == (batch_size, seq_len, 6)


def test_full_model_generate(full_model):
    """Generate should produce variable-length sequences."""
    mel = torch.randn(1, 80, 64)

    phonemes, latents = full_model.generate(mel, max_length=10)

    assert isinstance(phonemes, list)
    assert len(phonemes) <= 10
    assert all(isinstance(p, str) for p in phonemes)
    # Latents should match phoneme count
    if latents is not None:
        assert latents.shape[0] == len(phonemes)
        assert latents.shape[1] == 6


def test_all_exports():
    """All new components should be importable."""
    from primordial.speech import (
        SequenceDecoder,
        SpeechSequenceLRN,
        SequenceTrainer,
        WordDataset,
        WORD_PHONEMES,
        SOS_TOKEN,
        EOS_TOKEN,
        TOTAL_VOCAB,
    )
    assert all([
        SequenceDecoder, SpeechSequenceLRN, SequenceTrainer,
        WordDataset, WORD_PHONEMES
    ])
