"""Tests for sequence training."""
import torch
import pytest
from primordial.speech.training import SequenceTrainer
from primordial.speech.sequence_decoder import SpeechSequenceLRN
from primordial.speech.config import SpeechConfig


@pytest.fixture
def config():
    return SpeechConfig()


@pytest.fixture
def model(config):
    return SpeechSequenceLRN(config)


@pytest.fixture
def trainer(model, config):
    return SequenceTrainer(model, config)


def test_trainer_init(trainer):
    """Trainer should initialize with optimizer."""
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'step_count')


def test_train_step(trainer):
    """Training step should return loss dict."""
    mel = torch.randn(2, 80, 64)
    input_tokens = torch.randint(0, 41, (2, 5))
    target_tokens = torch.randint(0, 41, (2, 5))

    losses = trainer.train_step(mel, input_tokens, target_tokens)

    assert 'total' in losses
    assert 'discrete' in losses
    assert 'latent' in losses
    assert losses['total'] > 0


def test_compute_accuracy(trainer):
    """Accuracy computation should work."""
    logits = torch.zeros(2, 5, 43)
    logits[:, :, 0] = 10.0  # All predict token 0

    targets = torch.zeros(2, 5, dtype=torch.long)  # All targets are 0

    acc = trainer._compute_accuracy(logits, targets)
    assert acc == 1.0  # Perfect accuracy


def test_compute_acoustic_match_identical_audio():
    """Acoustic match of identical audio should be ~1.0."""
    from primordial.speech.training import compute_acoustic_match
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    tts = create_tts_backend(config)

    # Generate audio for same phonemes twice
    phonemes = ['HH', 'EH', 'L', 'OW']
    target_audio = tts.synthesize_phonemes(phonemes)

    score = compute_acoustic_match(model, tts, config, phonemes, target_audio)

    assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    assert score > 0.9, f"Identical audio should match >0.9, got {score}"


def test_compute_acoustic_match_different_phonemes():
    """Different phonemes should have lower acoustic match than identical."""
    from primordial.speech.training import compute_acoustic_match
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    tts = create_tts_backend(config)

    # Generate audio for different phonemes
    target_phonemes = ['HH', 'EH', 'L', 'OW']  # "hello"
    generated_phonemes = ['B', 'AY']  # "bye"
    target_audio = tts.synthesize_phonemes(target_phonemes)

    # Score for different phonemes
    diff_score = compute_acoustic_match(model, tts, config, generated_phonemes, target_audio)

    # Score for identical phonemes
    same_score = compute_acoustic_match(model, tts, config, target_phonemes, target_audio)

    assert 0.0 <= diff_score <= 1.0, f"Score {diff_score} out of range"
    assert diff_score < same_score, f"Different phonemes ({diff_score:.2f}) should match less than identical ({same_score:.2f})"


def test_compute_acoustic_match_empty_phonemes():
    """Empty phoneme list should return 0.0."""
    from primordial.speech.training import compute_acoustic_match
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    tts = create_tts_backend(config)

    target_audio = tts.synthesize_phonemes(['HH', 'AY'])

    score = compute_acoustic_match(model, tts, config, [], target_audio)

    assert score == 0.0, f"Empty phonemes should return 0.0, got {score}"
