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
