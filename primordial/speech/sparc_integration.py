"""SPARC integration for articulatory speech synthesis.

SPARC (Speech Articulatory Coding) provides:
- 12D EMA (Electromagnetic Articulography) features for tongue, lips, jaw
- Pitch (F0) and loudness for prosody
- Differentiable decoder for end-to-end training

Reference: "Coding Speech through Vocal Tract Kinematics" (Berkeley, 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .config import SpeechConfig


class SPARCWrapper:
    """Wrapper around SPARC model for encode/decode operations.

    Provides a clean interface for:
    - Encoding audio to articulatory features (EMA + pitch + loudness)
    - Decoding articulatory features back to audio (differentiable)

    Args:
        config: SpeechConfig with SPARC settings
        mock: If True, use mock implementation (for testing without SPARC)
        device: Torch device for computation
    """

    def __init__(
        self,
        config: SpeechConfig,
        mock: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.mock = mock
        self.device = device or torch.device('cpu')
        self.model = None

        if not mock and config.sparc_model_path:
            self._load_model(config.sparc_model_path)

    def _load_model(self, model_path: str) -> None:
        """Load SPARC model from checkpoint."""
        # TODO: Implement actual SPARC loading when available
        # from sparc import SPARC
        # self.model = SPARC.load(model_path)
        raise NotImplementedError(
            "SPARC model loading not yet implemented. "
            "Use mock=True for testing."
        )

    def encode(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode audio to articulatory features.

        Args:
            audio: (batch, samples) waveform at config.sample_rate

        Returns:
            ema: (batch, n_frames, 12) articulator positions
            pitch: (batch, n_frames, 1) F0 in Hz
            loudness: (batch, n_frames, 1) energy
        """
        batch_size = audio.shape[0]
        n_frames = self.config.sparc_n_frames

        if self.mock:
            # Mock implementation for testing
            ema = torch.randn(batch_size, n_frames, 12, device=self.device)
            pitch = torch.abs(torch.randn(batch_size, n_frames, 1, device=self.device)) * 200 + 100
            loudness = torch.sigmoid(torch.randn(batch_size, n_frames, 1, device=self.device))
            return ema, pitch, loudness

        # Real SPARC encoding
        # return self.model.encode(audio)
        raise NotImplementedError("Use mock=True for testing")

    def decode(
        self,
        ema: torch.Tensor,
        pitch: torch.Tensor,
        loudness: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode articulatory features to audio (differentiable).

        Args:
            ema: (batch, n_frames, 12) articulator positions
            pitch: (batch, n_frames, 1) F0 in Hz
            loudness: (batch, n_frames, 1) energy
            spk_emb: (batch, 64) speaker embedding

        Returns:
            audio: (batch, samples) synthesized waveform
        """
        batch_size = ema.shape[0]
        n_samples = int(self.config.sample_rate * self.config.audio_duration)

        if self.mock:
            # Mock implementation: simple differentiable synthesis
            # Combine features and project to audio length
            features = torch.cat([
                ema.mean(dim=1),  # (batch, 12)
                pitch.mean(dim=1),  # (batch, 1)
                loudness.mean(dim=1),  # (batch, 1)
                spk_emb,  # (batch, 64)
            ], dim=-1)  # (batch, 78)

            # Simple projection to audio (differentiable)
            # In real SPARC, this is a neural vocoder
            audio = torch.zeros(batch_size, n_samples, device=ema.device)
            for i in range(n_samples):
                t = i / n_samples
                freq = 100 + features[:, :12].sum(dim=-1) * 10
                audio[:, i] = torch.sin(2 * 3.14159 * freq * t) * loudness.mean(dim=1).squeeze()

            return audio

        # Real SPARC decoding
        # return self.model.decode(ema, pitch, loudness, spk_emb)
        raise NotImplementedError("Use mock=True for testing")


class VoiceIdentity:
    """Manages fixed speaker embedding for consistent voice identity.

    The speaker embedding determines the voice characteristics (timbre,
    formant structure) independent of articulation. Once set, it remains
    fixed during training so the model learns articulation, not voice.

    Args:
        embedding_path: Path to .npy file with 64D speaker embedding
        dim: Embedding dimension (default 64 for SPARC)
    """

    def __init__(
        self,
        embedding_path: Optional[str] = None,
        dim: int = 64,
    ):
        self.dim = dim

        if embedding_path and Path(embedding_path).exists():
            self._embedding = torch.from_numpy(
                np.load(embedding_path)
            ).float()
        else:
            # Random but fixed embedding
            self._embedding = torch.randn(dim)
            self._embedding = self._embedding / self._embedding.norm()

    def get_embedding(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get speaker embedding expanded to batch size.

        Args:
            batch_size: Number of copies to return
            device: Target device

        Returns:
            (batch_size, dim) speaker embedding
        """
        emb = self._embedding.unsqueeze(0).expand(batch_size, -1)
        if device is not None:
            emb = emb.to(device)
        return emb.clone()

    def save(self, path: str) -> None:
        """Save embedding to .npy file."""
        np.save(path, self._embedding.numpy())
