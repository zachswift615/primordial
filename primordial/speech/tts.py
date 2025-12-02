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

    Piper is a fast, local neural TTS system.
    Install: pip install piper-tts

    Models: https://github.com/rhasspy/piper/blob/master/VOICES.md
    """

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
        """Synthesize from phonemes.

        Note: Piper's phoneme input is limited. This converts phonemes
        to approximate text and synthesizes that. For true phoneme control,
        we'd need to modify Piper's internals or use a different backend.
        """
        self._load_model()

        # Convert ARPABET phonemes to approximate pronunciation
        # This is a simplification - proper phoneme synthesis would need
        # direct access to Piper's phoneme encoder
        text = self._phonemes_to_approximate_text(phonemes)

        return self.synthesize_text(text)

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
        """Convert phoneme sequence to approximate pronounceable text.

        This is a hack for Piper which doesn't have direct phoneme input.
        Maps phonemes to letter combinations that typically produce those sounds.
        """
        # Rough ARPABET to text mapping
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

        text_parts = []
        for p in phonemes:
            text_parts.append(mapping.get(p.upper(), ''))

        return ''.join(text_parts)


class DummyTTS(TTSBackend):
    """Dummy TTS backend for testing without real TTS.

    Generates simple sine waves for each phoneme.
    """

    def __init__(self, config: Optional[SpeechConfig] = None):
        self.config = config or SpeechConfig()
        self._sample_rate = config.sample_rate if config else 16000

        # Map phonemes to frequencies (arbitrary but consistent)
        self._phoneme_freqs = {}
        base_freq = 200
        for i, p in enumerate(['SIL'] + list('AEIOU') + list('BCDFGHJKLMNPQRSTVWXYZ')):
            self._phoneme_freqs[p] = base_freq + i * 20

    def synthesize_phonemes(
        self,
        phonemes: List[str],
        durations: Optional[List[float]] = None,
        pitches: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Generate simple tones for phonemes."""
        if durations is None:
            durations = [0.1] * len(phonemes)  # 100ms per phoneme

        audio_parts = []
        for i, (phoneme, duration) in enumerate(zip(phonemes, durations)):
            freq = pitches[i] if pitches else self._phoneme_freqs.get(phoneme[0], 300)
            t = np.linspace(0, duration, int(self._sample_rate * duration))

            if phoneme == SILENCE:
                wave = np.zeros_like(t)
            else:
                wave = 0.3 * np.sin(2 * np.pi * freq * t)

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
