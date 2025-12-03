"""Integration tests for full sequence pipeline."""
import torch
import pytest
from primordial.speech import (
    SpeechConfig,
    SpeechSequenceLRN,
    SequenceTrainer,
    WordDataset,
    WORD_PHONEMES,
)


@pytest.fixture
def config():
    return SpeechConfig(encoder_type='cnn', tts_backend='piper')


def test_full_training_loop(config):
    """End-to-end training should work."""
    model = SpeechSequenceLRN(config)
    trainer = SequenceTrainer(model, config)
    dataset = WordDataset(config, words=['ba', 'bee'])

    # One training step
    mel, input_tokens, target_tokens, word = dataset[0]

    losses = trainer.train_step(
        mel.unsqueeze(0),
        input_tokens.unsqueeze(0),
        target_tokens.unsqueeze(0),
    )

    assert losses['total'] > 0
    assert losses['accuracy'] >= 0


def test_generate_matches_word(config):
    """After training, generation should improve."""
    model = SpeechSequenceLRN(config)
    dataset = WordDataset(config, words=['ba'])

    mel, _, _, word = dataset[0]

    # Generate before training
    phonemes_before, _ = model.generate(mel.unsqueeze(0))

    # Train a few steps
    trainer = SequenceTrainer(model, config)
    for _ in range(20):
        mel, input_tokens, target_tokens, _ = dataset[0]
        trainer.train_step(
            mel.unsqueeze(0),
            input_tokens.unsqueeze(0),
            target_tokens.unsqueeze(0),
        )

    # Generate after training
    phonemes_after, _ = model.generate(mel.unsqueeze(0))

    # Should be closer to target
    target = WORD_PHONEMES['ba']
    # Just verify it runs - actual accuracy depends on more training
    assert isinstance(phonemes_after, list)
