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
