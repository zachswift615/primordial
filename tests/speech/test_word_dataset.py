"""Tests for word phoneme dataset."""
import torch
import pytest
from primordial.speech.word_dataset import WORD_PHONEMES, WordDataset
from primordial.speech.config import SpeechConfig
from primordial.speech.latent import SOS_TOKEN, EOS_TOKEN


def test_word_phonemes_dict():
    """Word dictionary should contain expected words."""
    assert 'hello' in WORD_PHONEMES
    assert WORD_PHONEMES['hello'] == ['HH', 'EH', 'L', 'OW']

    assert 'ba' in WORD_PHONEMES
    assert WORD_PHONEMES['ba'] == ['B', 'AA']


def test_word_dataset_length():
    """Dataset length should match word count."""
    config = SpeechConfig()
    dataset = WordDataset(config, words=['ba', 'hello'])

    assert len(dataset) == 2


def test_word_dataset_getitem():
    """Getting item should return mel, input tokens, target tokens."""
    config = SpeechConfig()
    dataset = WordDataset(config, words=['hello'])

    mel, input_tokens, target_tokens, word = dataset[0]

    # Mel spectrogram
    assert mel.shape[0] == config.n_mels

    # Input: [SOS, HH, EH, L, OW]
    assert input_tokens[0] == SOS_TOKEN
    assert len(input_tokens) == 5  # SOS + 4 phonemes

    # Target: [HH, EH, L, OW, EOS]
    assert target_tokens[-1] == EOS_TOKEN
    assert len(target_tokens) == 5  # 4 phonemes + EOS

    assert word == 'hello'


def test_word_dataset_curriculum_filter():
    """Dataset should filter by max phoneme length."""
    config = SpeechConfig()

    # Only short words
    short_dataset = WordDataset(config, max_phonemes=3)
    # Should include 'ba' (2), 'bee' (2), etc. but not 'hello' (4)
    words = [short_dataset[i][3] for i in range(len(short_dataset))]
    assert 'ba' in words
    assert 'hello' not in words


def test_phrase_phonemes_exist():
    """PHRASE_PHONEMES should contain multi-word phrases."""
    from primordial.speech.word_dataset import PHRASE_PHONEMES

    assert len(PHRASE_PHONEMES) > 0, "PHRASE_PHONEMES should not be empty"
    assert "hello world" in PHRASE_PHONEMES, "Should contain 'hello world'"
    assert len(PHRASE_PHONEMES["hello world"]) == 8, "hello world = 8 phonemes"


def test_get_all_entries_includes_phrases():
    """get_all_entries should merge words and phrases."""
    from primordial.speech.word_dataset import (
        get_all_entries, WORD_PHONEMES, PHRASE_PHONEMES
    )

    all_entries = get_all_entries(include_phrases=True)

    assert len(all_entries) == len(WORD_PHONEMES) + len(PHRASE_PHONEMES)
    assert "hello" in all_entries  # word
    assert "hello world" in all_entries  # phrase


def test_get_all_entries_excludes_phrases():
    """get_all_entries(include_phrases=False) should only return words."""
    from primordial.speech.word_dataset import get_all_entries, WORD_PHONEMES

    entries = get_all_entries(include_phrases=False)

    assert len(entries) == len(WORD_PHONEMES)
    assert "hello" in entries
    assert "hello world" not in entries


def test_word_dataset_max_phonemes_filters_phrases():
    """WordDataset with max_phonemes should filter correctly."""
    from primordial.speech.word_dataset import WordDataset, PHRASE_PHONEMES
    from primordial.speech import SpeechConfig

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')

    # max_phonemes=6 should exclude "hello world" (8 phonemes)
    dataset = WordDataset(config, max_phonemes=6, include_phrases=True)

    assert "hello world" not in dataset.words
    assert "bye bye" in dataset.words  # 4 phonemes

    # Verify we got some phrases
    phrase_count = sum(1 for w in dataset.words if " " in w)
    assert phrase_count > 0, "Should include some short phrases"
