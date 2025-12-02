"""Configuration for speech learning."""
from dataclasses import dataclass
from typing import Literal

from .phonemes import NUM_PHONEMES


@dataclass
class SpeechConfig:
    """Configuration for speech learning module."""

    # Audio input settings
    sample_rate: int = 16000           # 16kHz standard for speech
    n_mels: int = 80                   # Mel spectrogram bins
    n_fft: int = 400                   # FFT window size (25ms at 16kHz)
    hop_length: int = 160              # Hop between frames (10ms at 16kHz)
    audio_duration: float = 1.0        # Seconds of audio per sample

    # Derived: number of mel frames per sample
    @property
    def n_frames(self) -> int:
        """Number of mel spectrogram frames per audio sample."""
        samples = int(self.sample_rate * self.audio_duration)
        return (samples - self.n_fft) // self.hop_length + 1

    # Model architecture
    hidden_dim: int = 128              # Match LRN hidden dim
    encoder_seq_len: int = 100         # Output sequence length from encoder

    # Speech head output
    num_phonemes: int = NUM_PHONEMES   # 41 (from phonemes.py)
    max_phoneme_duration: float = 0.5  # Max duration for single phoneme (seconds)
    pitch_range: tuple = (50, 400)     # Hz range for pitch output

    # Training settings
    learning_rate: float = 1e-4
    batch_size: int = 32

    # Loss weights
    phoneme_loss_weight: float = 1.0
    duration_loss_weight: float = 0.5
    pitch_loss_weight: float = 0.3
    audio_similarity_weight: float = 1.0

    # TTS settings
    tts_backend: Literal["piper", "sherpa"] = "piper"
    tts_model_path: str = ""           # Path to TTS model (set by user)

    # Training curriculum
    curriculum_phase: Literal["phoneme_classification", "phoneme_production",
                              "sequences", "words"] = "phoneme_classification"

    # Self-listening (agent hears its own output)
    enable_self_listening: bool = True
    self_listening_delay: float = 0.0  # Delay in seconds (0 = immediate)
