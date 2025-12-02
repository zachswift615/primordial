"""Output heads for speech production."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import SpeechConfig
from .phonemes import NUM_PHONEMES


class SpeechHead(nn.Module):
    """Speech production head for LRN.

    Takes pooled features from Fourier mixing layers and outputs:
    - phoneme_logits: (batch, num_phonemes) - probability distribution over phonemes
    - duration: (batch, 1) - how long to hold the phoneme (0-1, scaled to max_duration)
    - pitch: (batch, 1) - pitch/F0 control (0-1, scaled to pitch_range)

    For sequence output (producing multiple phonemes), use SpeechSequenceHead instead.
    """

    def __init__(self, config: SpeechConfig, input_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_phonemes = config.num_phonemes

        # Input dim is typically 3 * hidden_dim from pooling (mean, max, last)
        input_dim = input_dim or (3 * config.hidden_dim)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Phoneme classification head
        self.phoneme_head = nn.Linear(config.hidden_dim, self.num_phonemes)

        # Duration head (outputs 0-1, scaled to max_phoneme_duration)
        self.duration_head = nn.Linear(config.hidden_dim, 1)

        # Pitch head (outputs 0-1, scaled to pitch_range)
        self.pitch_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim) pooled features from Fourier mixing
            return_probs: If True, apply softmax to phoneme logits

        Returns:
            phoneme_logits: (batch, num_phonemes) - raw logits or probabilities
            duration: (batch, 1) - normalized duration (0-1)
            pitch: (batch, 1) - normalized pitch (0-1)
        """
        # Shared features
        features = self.trunk(x)

        # Phoneme prediction
        phoneme_logits = self.phoneme_head(features)
        if return_probs:
            phoneme_logits = F.softmax(phoneme_logits, dim=-1)

        # Duration (sigmoid to 0-1)
        duration = torch.sigmoid(self.duration_head(features))

        # Pitch (sigmoid to 0-1)
        pitch = torch.sigmoid(self.pitch_head(features))

        return phoneme_logits, duration, pitch

    def sample_phoneme(
        self,
        phoneme_logits: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Sample phoneme index from logits.

        Args:
            phoneme_logits: (batch, num_phonemes)
            temperature: Sampling temperature (lower = more greedy)
            greedy: If True, use argmax instead of sampling

        Returns:
            (batch,) phoneme indices
        """
        if greedy:
            return phoneme_logits.argmax(dim=-1)

        # Temperature-scaled sampling
        probs = F.softmax(phoneme_logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def decode_duration(self, duration: torch.Tensor) -> torch.Tensor:
        """Convert normalized duration (0-1) to seconds.

        Args:
            duration: (batch, 1) normalized duration

        Returns:
            (batch, 1) duration in seconds
        """
        return duration * self.config.max_phoneme_duration

    def decode_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        """Convert normalized pitch (0-1) to Hz.

        Args:
            pitch: (batch, 1) normalized pitch

        Returns:
            (batch, 1) pitch in Hz
        """
        min_hz, max_hz = self.config.pitch_range
        return min_hz + pitch * (max_hz - min_hz)


class SpeechSequenceHead(nn.Module):
    """Speech production head that outputs a sequence of phonemes.

    For generating multiple phonemes at once (e.g., a word or phrase).
    Uses the full sequence from Fourier mixing instead of pooled features.

    Input: (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, num_phonemes), (batch, seq_len, 1), (batch, seq_len, 1)
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config
        self.num_phonemes = config.num_phonemes
        self.hidden_dim = config.hidden_dim

        # Per-position prediction (no pooling)
        self.phoneme_head = nn.Linear(self.hidden_dim, self.num_phonemes)
        self.duration_head = nn.Linear(self.hidden_dim, 1)
        self.pitch_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim) sequence from Fourier mixing

        Returns:
            phoneme_logits: (batch, seq_len, num_phonemes)
            duration: (batch, seq_len, 1)
            pitch: (batch, seq_len, 1)
        """
        phoneme_logits = self.phoneme_head(x)
        if return_probs:
            phoneme_logits = F.softmax(phoneme_logits, dim=-1)

        duration = torch.sigmoid(self.duration_head(x))
        pitch = torch.sigmoid(self.pitch_head(x))

        return phoneme_logits, duration, pitch


class ProductionHead(nn.Module):
    """Production head that outputs latent motor commands.

    Takes pooled features and produces:
    - latent: (batch, latent_dim) - latent motor representation (constrained to [-1, 1])
    - duration: (batch, 1) - how long to produce the sound (0-1)
    - pitch: (batch, 1) - pitch/F0 control (0-1)

    This is used for motor learning where the agent learns to control
    a low-dimensional latent space that drives vocalization.
    """

    def __init__(self, config: SpeechConfig, input_dim: int = 384):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

        # Latent projection: input -> 128 -> latent_dim with Tanh
        self.latent_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_dim),
            nn.Tanh(),  # Constrain to [-1, 1]
        )

        # Duration head (outputs 0-1)
        self.duration_head = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

        # Pitch head (outputs 0-1)
        self.pitch_head = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled: (batch, input_dim) pooled features from Fourier mixing

        Returns:
            latent: (batch, latent_dim) - latent motor commands in [-1, 1]
            duration: (batch, 1) - normalized duration (0-1)
            pitch: (batch, 1) - normalized pitch (0-1)
        """
        latent = self.latent_proj(pooled)
        duration = self.duration_head(pooled)
        pitch = self.pitch_head(pooled)

        return latent, duration, pitch


class AudioReconstructionHead(nn.Module):
    """Head for reconstructing audio features (for self-supervised learning).

    Predicts mel spectrogram from Fourier mixing output.
    Used for the sensory prediction task (predict what you'll hear next).
    """

    def __init__(self, config: SpeechConfig, input_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.n_mels = config.n_mels
        self.n_frames = config.encoder_seq_len  # Predict same length

        input_dim = input_dim or (3 * config.hidden_dim)

        # Upsample from pooled features to mel spectrogram
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.n_mels * self.n_frames),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) pooled features

        Returns:
            (batch, n_mels, n_frames) predicted mel spectrogram
        """
        batch_size = x.shape[0]
        x = self.decoder(x)
        return x.view(batch_size, self.n_mels, self.n_frames)
