"""Grapheme-to-phoneme conversion for text-to-phoneme mapping.

Converts English text to ARPABET phonemes, compatible with the existing
phoneme inventory in phonemes.py.
"""

import re
from typing import List, Optional, Dict
from functools import lru_cache


# ARPABET phoneme set (matches phonemes.py)
VALID_PHONEMES = {
    # Vowels
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH',
    # Special
    'SIL', 'UNK'
}


class G2PConverter:
    """Grapheme-to-phoneme converter using g2p_en or CMU dict fallback."""

    def __init__(self, use_g2p_en: bool = True):
        """Initialize G2P converter.

        Args:
            use_g2p_en: If True, try to use g2p_en library. Falls back to CMU dict.
        """
        self._g2p = None
        self._cmu_dict = None
        self._use_g2p_en = use_g2p_en

        if use_g2p_en:
            self._init_g2p_en()

        if self._g2p is None:
            self._init_cmu_dict()

    def _init_g2p_en(self) -> None:
        """Try to initialize g2p_en library."""
        try:
            from g2p_en import G2p
            self._g2p = G2p()
        except ImportError:
            pass  # Fall back to CMU dict

    def _init_cmu_dict(self) -> None:
        """Initialize CMU pronunciation dictionary."""
        try:
            import nltk
            try:
                from nltk.corpus import cmudict
                self._cmu_dict = cmudict.dict()
            except LookupError:
                nltk.download('cmudict', quiet=True)
                from nltk.corpus import cmudict
                self._cmu_dict = cmudict.dict()
        except ImportError:
            raise RuntimeError(
                "Neither g2p_en nor nltk is available. "
                "Install one: pip install g2p_en OR pip install nltk"
            )

    @staticmethod
    def strip_stress(phoneme: str) -> str:
        """Remove stress markers (0, 1, 2) from ARPABET phoneme."""
        return re.sub(r'[012]', '', phoneme)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for G2P conversion."""
        # Convert to uppercase (LibriSpeech uses uppercase)
        text = text.upper()
        # Replace common contractions
        text = text.replace("'S", " S")
        text = text.replace("'T", " T")
        text = text.replace("'RE", " ARE")
        text = text.replace("'VE", " HAVE")
        text = text.replace("'LL", " WILL")
        text = text.replace("'D", " D")
        # Remove remaining punctuation except spaces
        text = re.sub(r"[^A-Z\s']", ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def convert_word(self, word: str) -> List[str]:
        """Convert a single word to phoneme list.

        Args:
            word: Single word to convert

        Returns:
            List of ARPABET phonemes (stress markers removed)
        """
        word = word.upper().strip()
        if not word:
            return []

        # Try g2p_en first
        if self._g2p is not None:
            result = self._g2p(word)
            phonemes = []
            for p in result:
                p = p.strip()
                if p and not p.isspace():
                    stripped = self.strip_stress(p)
                    if stripped in VALID_PHONEMES:
                        phonemes.append(stripped)
            return phonemes

        # Fall back to CMU dict
        word_lower = word.lower()
        # Remove non-alphabetic characters
        word_lower = re.sub(r'[^a-z]', '', word_lower)

        if word_lower in self._cmu_dict:
            # Take first pronunciation
            pron = self._cmu_dict[word_lower][0]
            return [self.strip_stress(p) for p in pron if self.strip_stress(p) in VALID_PHONEMES]

        # Unknown word - return empty (will be handled by caller)
        return []

    def convert(self, text: str) -> List[str]:
        """Convert text to phoneme sequence.

        Args:
            text: Input text (words separated by spaces)

        Returns:
            List of ARPABET phonemes
        """
        text = self.clean_text(text)
        words = text.split()

        all_phonemes = []
        for word in words:
            word_phonemes = self.convert_word(word)
            if word_phonemes:
                all_phonemes.extend(word_phonemes)

        return all_phonemes


# Global converter instance (lazy initialization)
_converter: Optional[G2PConverter] = None


def get_converter() -> G2PConverter:
    """Get or create the global G2P converter instance."""
    global _converter
    if _converter is None:
        _converter = G2PConverter()
    return _converter


def text_to_phonemes(text: str) -> List[str]:
    """Convert text to phoneme sequence.

    This is the main entry point for G2P conversion.

    Args:
        text: Input text (English words)

    Returns:
        List of ARPABET phonemes (e.g., ['HH', 'EH', 'L', 'OW'])

    Example:
        >>> text_to_phonemes("hello world")
        ['HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D']
    """
    return get_converter().convert(text)


@lru_cache(maxsize=10000)
def word_to_phonemes(word: str) -> tuple:
    """Convert a single word to phonemes (cached).

    Args:
        word: Single word

    Returns:
        Tuple of phonemes (for caching hashability)
    """
    return tuple(get_converter().convert_word(word))


def phoneme_count(text: str) -> int:
    """Count number of phonemes in text.

    Useful for filtering by sequence length.
    """
    return len(text_to_phonemes(text))


if __name__ == "__main__":
    # Test the converter
    test_sentences = [
        "HELLO WORLD",
        "HE TOOK A RESOLVE AFTER THIS",
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
    ]

    converter = G2PConverter()
    for sentence in test_sentences:
        phonemes = converter.convert(sentence)
        print(f"{sentence}")
        print(f"  -> {phonemes}")
        print(f"  -> {len(phonemes)} phonemes")
        print()
