"""Phoneme inventory and utilities for speech learning.

Uses ARPABET notation (standard for English TTS systems like Piper).
"""

from typing import Dict, List, Optional

# ARPABET phoneme inventory for English (39 phonemes)
# This matches what Piper and most English TTS systems use

VOWELS = [
    'AA',  # father, hot
    'AE',  # cat, bat
    'AH',  # but, cut
    'AO',  # dog, thought
    'AW',  # cow, how
    'AY',  # my, bye
    'EH',  # bed, red
    'ER',  # bird, her
    'EY',  # say, day
    'IH',  # bit, sit
    'IY',  # bee, see
    'OW',  # go, show
    'OY',  # boy, toy
    'UH',  # book, put
    'UW',  # food, blue
]

CONSONANTS = [
    'B',   # boy
    'CH',  # church
    'D',   # dog
    'DH',  # the, that
    'F',   # fish
    'G',   # go
    'HH',  # hat
    'JH',  # judge
    'K',   # cat
    'L',   # love
    'M',   # mom
    'N',   # no
    'NG',  # sing
    'P',   # pop
    'R',   # red
    'S',   # see
    'SH',  # she
    'T',   # top
    'TH',  # think
    'V',   # very
    'W',   # way
    'Y',   # yes
    'Z',   # zoo
    'ZH',  # measure
]

# Special tokens
SILENCE = 'SIL'  # Silence/pause
UNKNOWN = 'UNK'  # Unknown phoneme

# Complete inventory
PHONEME_INVENTORY: List[str] = [SILENCE] + VOWELS + CONSONANTS + [UNKNOWN]

# Number of phonemes (for model output dimension)
NUM_PHONEMES = len(PHONEME_INVENTORY)  # 41 (15 vowels + 24 consonants + SIL + UNK)

# Mappings
_PHONEME_TO_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PHONEME_INVENTORY)}
_INDEX_TO_PHONEME: Dict[int, str] = {i: p for i, p in enumerate(PHONEME_INVENTORY)}


def phoneme_to_index(phoneme: str) -> int:
    """Convert phoneme string to index.

    Args:
        phoneme: ARPABET phoneme string (e.g., 'AA', 'B', 'SIL')

    Returns:
        Integer index into PHONEME_INVENTORY
    """
    return _PHONEME_TO_INDEX.get(phoneme.upper(), _PHONEME_TO_INDEX[UNKNOWN])


def index_to_phoneme(index: int) -> str:
    """Convert index to phoneme string.

    Args:
        index: Integer index

    Returns:
        ARPABET phoneme string
    """
    return _INDEX_TO_PHONEME.get(index, UNKNOWN)


def phoneme_sequence_to_indices(phonemes: List[str]) -> List[int]:
    """Convert list of phonemes to list of indices."""
    return [phoneme_to_index(p) for p in phonemes]


def indices_to_phoneme_sequence(indices: List[int]) -> List[str]:
    """Convert list of indices to list of phonemes."""
    return [index_to_phoneme(i) for i in indices]


# Simple phoneme examples for each phoneme (for generating training data)
PHONEME_EXAMPLES: Dict[str, List[str]] = {
    'AA': ['father', 'hot', 'body'],
    'AE': ['cat', 'bat', 'hat'],
    'AH': ['but', 'cut', 'love'],
    'AO': ['dog', 'thought', 'law'],
    'AW': ['cow', 'how', 'now'],
    'AY': ['my', 'bye', 'eye'],
    'EH': ['bed', 'red', 'said'],
    'ER': ['bird', 'her', 'word'],
    'EY': ['say', 'day', 'way'],
    'IH': ['bit', 'sit', 'is'],
    'IY': ['bee', 'see', 'key'],
    'OW': ['go', 'show', 'no'],
    'OY': ['boy', 'toy', 'joy'],
    'UH': ['book', 'put', 'good'],
    'UW': ['food', 'blue', 'you'],
    'B': ['boy', 'baby', 'cab'],
    'CH': ['church', 'match', 'each'],
    'D': ['dog', 'bed', 'add'],
    'DH': ['the', 'that', 'other'],
    'F': ['fish', 'off', 'leaf'],
    'G': ['go', 'big', 'dog'],
    'HH': ['hat', 'ahead', 'who'],
    'JH': ['judge', 'bridge', 'age'],
    'K': ['cat', 'back', 'key'],
    'L': ['love', 'bell', 'feel'],
    'M': ['mom', 'him', 'some'],
    'N': ['no', 'sun', 'on'],
    'NG': ['sing', 'ring', 'long'],
    'P': ['pop', 'top', 'cup'],
    'R': ['red', 'car', 'more'],
    'S': ['see', 'bus', 'ice'],
    'SH': ['she', 'fish', 'wash'],
    'T': ['top', 'cat', 'it'],
    'TH': ['think', 'math', 'with'],
    'V': ['very', 'love', 'give'],
    'W': ['way', 'swim', 'one'],
    'Y': ['yes', 'you', 'use'],
    'Z': ['zoo', 'is', 'his'],
    'ZH': ['measure', 'vision', 'beige'],
}


def is_vowel(phoneme: str) -> bool:
    """Check if phoneme is a vowel."""
    return phoneme.upper() in VOWELS


def is_consonant(phoneme: str) -> bool:
    """Check if phoneme is a consonant."""
    return phoneme.upper() in CONSONANTS
