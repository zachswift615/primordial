"""Text-to-speech synthesis for speech production.

Supports Piper and Sherpa-ONNX backends for converting phonemes to audio.
"""
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from pathlib import Path

from .config import SpeechConfig
from .phonemes import index_to_phoneme, SILENCE


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    @abstractmethod
    def synthesize_phonemes(
        self,
        phonemes: List[str],
        durations: Optional[List[float]] = None,
        pitches: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Synthesize audio from phoneme sequence.

        Args:
            phonemes: List of ARPABET phoneme strings
            durations: Optional duration for each phoneme (seconds)
            pitches: Optional pitch for each phoneme (Hz)

        Returns:
            Audio waveform as numpy array (sample_rate from config)
        """
        pass

    @abstractmethod
    def synthesize_text(self, text: str) -> np.ndarray:
        """Synthesize audio from text (for generating training targets).

        Args:
            text: Text to synthesize

        Returns:
            Audio waveform as numpy array
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of output audio."""
        pass


class PiperTTS(TTSBackend):
    """Piper TTS backend.

    Piper is a fast, local neural TTS system with direct phoneme synthesis.
    Install: pip install piper-tts

    Models: https://github.com/rhasspy/piper/blob/master/VOICES.md
    """

    # ARPABET to IPA mapping for Piper
    # Based on https://en.wikipedia.org/wiki/ARPABET
    ARPABET_TO_IPA = {
        # Vowels
        'AA': 'ɑ',   # odd, father
        'AE': 'æ',   # at, bat
        'AH': 'ʌ',   # hut, but
        'AO': 'ɔ',   # ought, all
        'AW': 'aʊ',  # cow, how
        'AY': 'aɪ',  # hide, my
        'EH': 'ɛ',   # ed, bet
        'ER': 'ɜɹ',  # hurt, bird (use ɜ + ɹ instead of ɝ which some models lack)
        'EY': 'eɪ',  # ate, say
        'IH': 'ɪ',   # it, bit
        'IY': 'i',   # eat, bee
        'OW': 'oʊ',  # oat, show
        'OY': 'ɔɪ',  # toy, boy
        'UH': 'ʊ',   # hood, could
        'UW': 'u',   # two, blue
        # Consonants
        'B': 'b',
        'CH': 'tʃ',
        'D': 'd',
        'DH': 'ð',   # the, that
        'F': 'f',
        'G': 'ɡ',
        'HH': 'h',
        'JH': 'dʒ',  # judge
        'K': 'k',
        'L': 'l',
        'M': 'm',
        'N': 'n',
        'NG': 'ŋ',   # sing
        'P': 'p',
        'R': 'ɹ',
        'S': 's',
        'SH': 'ʃ',
        'T': 't',
        'TH': 'θ',   # think
        'V': 'v',
        'W': 'w',
        'Y': 'j',
        'Z': 'z',
        'ZH': 'ʒ',   # measure
        # Special
        'SIL': ' ',
        'UNK': '',
    }

    def __init__(self, model_path: str, config: Optional[SpeechConfig] = None):
        """
        Args:
            model_path: Path to Piper .onnx model file
            config: Speech config (optional)
        """
        self.model_path = Path(model_path)
        self.config = config or SpeechConfig()
        self._piper = None
        self._sample_rate = 22050  # Piper default

    def _load_model(self):
        """Lazy load Piper model."""
        if self._piper is None:
            try:
                from piper import PiperVoice
                self._piper = PiperVoice.load(str(self.model_path))
                self._sample_rate = self._piper.config.sample_rate
            except ImportError:
                raise ImportError(
                    "Piper not installed. Install with: pip install piper-tts"
                )

    def synthesize_phonemes(
        self,
        phonemes: List[str],
        durations: Optional[List[float]] = None,
        pitches: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Synthesize from ARPABET phonemes using direct phoneme synthesis.

        Converts ARPABET phonemes to IPA, then uses Piper's phoneme_ids_to_audio
        for high-fidelity phoneme-to-audio synthesis.
        """
        self._load_model()

        # Convert ARPABET to IPA phonemes
        ipa_phonemes = []
        for p in phonemes:
            ipa = self.ARPABET_TO_IPA.get(p.upper(), '')
            if ipa:
                # Add each IPA character separately (some are digraphs)
                ipa_phonemes.extend(list(ipa))

        if not ipa_phonemes:
            return np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)

        try:
            # Use Piper's direct phoneme synthesis
            phoneme_ids = self._piper.phonemes_to_ids(ipa_phonemes)
            audio = self._piper.phoneme_ids_to_audio(phoneme_ids)
            return audio.astype(np.float32)
        except Exception as e:
            # Fallback to text synthesis if direct phoneme fails
            print(f"Warning: Direct phoneme synthesis failed ({e}), using text fallback")
            return self.synthesize_text(self._phonemes_to_approximate_text(phonemes))

    def synthesize_text(self, text: str) -> np.ndarray:
        """Synthesize from text."""
        self._load_model()

        # Piper returns a generator of audio chunks
        audio_chunks = []
        for audio_bytes in self._piper.synthesize_stream_raw(text):
            # Convert bytes to numpy array (16-bit PCM)
            chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_chunks.append(chunk)

        if audio_chunks:
            return np.concatenate(audio_chunks)
        else:
            # Return silence if no audio generated
            return np.zeros(int(self._sample_rate * 0.1), dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _phonemes_to_approximate_text(self, phonemes: List[str]) -> str:
        """Convert phoneme sequence to approximate pronounceable text (fallback)."""
        mapping = {
            'AA': 'ah', 'AE': 'a', 'AH': 'uh', 'AO': 'aw', 'AW': 'ow',
            'AY': 'eye', 'EH': 'eh', 'ER': 'er', 'EY': 'ay', 'IH': 'ih',
            'IY': 'ee', 'OW': 'oh', 'OY': 'oy', 'UH': 'oo', 'UW': 'oo',
            'B': 'b', 'CH': 'ch', 'D': 'd', 'DH': 'th', 'F': 'f',
            'G': 'g', 'HH': 'h', 'JH': 'j', 'K': 'k', 'L': 'l',
            'M': 'm', 'N': 'n', 'NG': 'ng', 'P': 'p', 'R': 'r',
            'S': 's', 'SH': 'sh', 'T': 't', 'TH': 'th', 'V': 'v',
            'W': 'w', 'Y': 'y', 'Z': 'z', 'ZH': 'zh',
            'SIL': ' ', 'UNK': '',
        }
        return ''.join(mapping.get(p.upper(), '') for p in phonemes)


class DummyTTS(TTSBackend):
    """Dummy TTS backend for testing without real TTS.

    Generates simple sine waves for each phoneme.
    """

    def __init__(self, config: Optional[SpeechConfig] = None):
        self.config = config or SpeechConfig()
        self._sample_rate = config.sample_rate if config else 16000

        # Map each ARPABET phoneme to a unique frequency
        # Vowels get lower frequencies (formant-like), consonants higher
        from .phonemes import PHONEME_INVENTORY
        self._phoneme_freqs = {}

        # Vowels: 200-400 Hz range (distinct formant-like)
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
        for i, v in enumerate(vowels):
            self._phoneme_freqs[v] = 200 + i * 15  # 200-410 Hz

        # Consonants: 450-800 Hz range
        consonants = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']
        for i, c in enumerate(consonants):
            self._phoneme_freqs[c] = 450 + i * 15  # 450-795 Hz

        # Special tokens
        self._phoneme_freqs['SIL'] = 0
        self._phoneme_freqs['UNK'] = 100

    def synthesize_phonemes(
        self,
        phonemes: List[str],
        durations: Optional[List[float]] = None,
        pitches: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Generate simple tones for phonemes.

        Each phoneme gets a unique frequency, making classification learnable.
        Adds slight variations to prevent overfitting to exact frequencies.
        """
        if durations is None:
            durations = [0.1] * len(phonemes)  # 100ms per phoneme

        audio_parts = []
        for i, (phoneme, duration) in enumerate(zip(phonemes, durations)):
            # Look up frequency by full phoneme name
            base_freq = self._phoneme_freqs.get(phoneme.upper(), 300)

            # Use pitch override if provided, otherwise add small random variation
            if pitches and i < len(pitches):
                freq = pitches[i]
            else:
                # Add ±5% variation to prevent overfitting to exact frequencies
                freq = base_freq * (1 + np.random.uniform(-0.05, 0.05))

            t = np.linspace(0, duration, int(self._sample_rate * duration))

            if phoneme == SILENCE or base_freq == 0:
                wave = np.zeros_like(t)
            else:
                # Add harmonics for richer sound (more speech-like)
                wave = (0.3 * np.sin(2 * np.pi * freq * t) +
                        0.15 * np.sin(2 * np.pi * freq * 2 * t) +  # 2nd harmonic
                        0.075 * np.sin(2 * np.pi * freq * 3 * t))  # 3rd harmonic

                # Apply simple envelope (attack/decay) for more natural sound
                envelope = np.ones_like(t)
                attack_samples = min(int(0.01 * self._sample_rate), len(t) // 4)
                decay_samples = min(int(0.02 * self._sample_rate), len(t) // 4)
                if attack_samples > 0:
                    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                if decay_samples > 0:
                    envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
                wave *= envelope

            audio_parts.append(wave.astype(np.float32))

        return np.concatenate(audio_parts) if audio_parts else np.zeros(1000, dtype=np.float32)

    def synthesize_text(self, text: str) -> np.ndarray:
        """Generate simple tones for text (one tone per character)."""
        phonemes = list(text.upper().replace(' ', 'SIL'))
        return self.synthesize_phonemes(phonemes)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


def create_tts_backend(config: SpeechConfig) -> TTSBackend:
    """Create TTS backend based on config.

    Args:
        config: Speech config with tts_backend and tts_model_path

    Returns:
        TTSBackend instance
    """
    if config.tts_backend == "piper":
        if not config.tts_model_path:
            print("Warning: No Piper model path specified, using DummyTTS")
            return DummyTTS(config)
        return PiperTTS(config.tts_model_path, config)

    elif config.tts_backend == "sherpa":
        # TODO: Implement Sherpa-ONNX backend
        print("Warning: Sherpa backend not yet implemented, using DummyTTS")
        return DummyTTS(config)

    else:
        return DummyTTS(config)


def phoneme_indices_to_audio(
    indices: torch.Tensor,
    durations: torch.Tensor,
    pitches: torch.Tensor,
    tts: TTSBackend,
    config: SpeechConfig,
) -> np.ndarray:
    """Convert model output to audio.

    Args:
        indices: (batch,) or (batch, seq_len) phoneme indices
        durations: (batch, 1) or (batch, seq_len, 1) normalized durations
        pitches: (batch, 1) or (batch, seq_len, 1) normalized pitches
        tts: TTS backend
        config: Speech config

    Returns:
        Audio waveform (first batch element only for now)
    """
    # Handle batch dimension - just use first element
    if indices.dim() > 1:
        indices = indices[0]
    if durations.dim() > 1:
        durations = durations[0]
    if pitches.dim() > 1:
        pitches = pitches[0]

    # Convert to lists
    phonemes = [index_to_phoneme(i.item()) for i in indices]

    # Denormalize durations and pitches
    duration_secs = (durations.squeeze() * config.max_phoneme_duration).tolist()
    if not isinstance(duration_secs, list):
        duration_secs = [duration_secs]

    min_hz, max_hz = config.pitch_range
    pitch_hz = (min_hz + pitches.squeeze() * (max_hz - min_hz)).tolist()
    if not isinstance(pitch_hz, list):
        pitch_hz = [pitch_hz]

    # Synthesize
    return tts.synthesize_phonemes(phonemes, duration_secs, pitch_hz)
