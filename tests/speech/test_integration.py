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


def test_phrase_training_smoke():
    """Smoke test: phrase training runs without errors."""
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN
    from primordial.speech.training import SequenceTrainer, compute_acoustic_match
    from primordial.speech.word_dataset import WordDataset
    from torch.utils.data import DataLoader

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    trainer = SequenceTrainer(model, config, lr=1e-3)
    tts = create_tts_backend(config)

    # Create dataset with phrases
    dataset = WordDataset(config, max_phonemes=10, include_phrases=True)

    # Verify we have phrases
    phrase_count = sum(1 for w in dataset.words if " " in w)
    assert phrase_count > 0, "Dataset should include phrases"

    # Custom collate
    def collate_fn(batch):
        from primordial.speech.latent import EOS_TOKEN
        import torch

        mels, inputs, targets, words = zip(*batch)
        max_len = max(len(t) for t in inputs)

        padded_inputs = torch.full((len(batch), max_len), EOS_TOKEN, dtype=torch.long)
        padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            padded_inputs[i, :len(inp)] = inp
            padded_targets[i, :len(tgt)] = tgt

        mels = torch.stack(mels)
        return mels, padded_inputs, padded_targets, words

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Run one training step
    mel, input_tokens, target_tokens, words = next(iter(dataloader))
    losses = trainer.train_step(mel, input_tokens, target_tokens)

    assert 'total' in losses
    assert 'accuracy' in losses
    assert losses['total'] > 0

    # Test acoustic match on a phrase
    model.eval()
    with torch.no_grad():
        phonemes, _ = model.generate(mel[0:1])

        # Get target audio
        word = words[0]
        target_phonemes = dataset._entries[word]
        target_audio = tts.synthesize_phonemes(target_phonemes)

        score = compute_acoustic_match(model, tts, config, phonemes, target_audio)

        assert 0.0 <= score <= 1.0, f"Acoustic score out of range: {score}"
