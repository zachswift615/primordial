"""SPARC integration for articulatory speech synthesis.

SPARC (Speech Articulatory Coding) provides:
- 12D EMA (Electromagnetic Articulography) features for tongue, lips, jaw
- Pitch (F0) and loudness for prosody
- Differentiable decoder for end-to-end training

Reference: "Coding Speech through Vocal Tract Kinematics" (Berkeley, 2024)
GitHub: https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

from .config import SpeechConfig


class SPARCWrapper:
    """Wrapper around SPARC model for encode/decode operations.

    Provides a clean interface for:
    - Encoding audio to articulatory features (EMA + pitch + loudness)
    - Decoding articulatory features back to audio

    Args:
        config: SpeechConfig with SPARC settings
        mock: If True, use mock implementation (for testing without SPARC)
        device: Torch device for computation
        model_name: SPARC model variant ('en', 'multi', 'en+')
    """

    def __init__(
        self,
        config: SpeechConfig,
        mock: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        model_name: str = 'en',
    ):
        self.config = config
        self.mock = mock
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = model_name

        if not mock:
            self._load_model()

    def _load_model(self) -> None:
        """Load SPARC model from HuggingFace."""
        from sparc import load_model
        device_str = str(self.device) if isinstance(self.device, torch.device) else self.device
        self.model = load_model(self.model_name, device=device_str)

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
        device = audio.device if isinstance(audio, torch.Tensor) else self.device

        if self.mock:
            # Mock implementation for testing
            ema = torch.randn(batch_size, n_frames, 12, device=device)
            pitch = torch.abs(torch.randn(batch_size, n_frames, 1, device=device)) * 200 + 100
            loudness = torch.sigmoid(torch.randn(batch_size, n_frames, 1, device=device))
            return ema, pitch, loudness

        # Real SPARC encoding
        # Convert to numpy for SPARC (it expects file paths or numpy arrays)
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio

        # SPARC expects list of arrays for batched processing
        if audio_np.ndim == 1:
            audio_list = [audio_np]
        else:
            audio_list = [audio_np[i] for i in range(batch_size)]

        # Encode with SPARC
        results = self.model.encode(audio_list, split_batch=True, reduce=False)

        # Convert results to tensors with consistent shapes
        emas, pitches, loudnesses = [], [], []
        for result in results:
            ema = torch.from_numpy(result['ema']).float()
            pitch = torch.from_numpy(result['pitch']).float()
            loudness = torch.from_numpy(result['loudness']).float()

            # Ensure correct shape: (n_frames, dim)
            if ema.ndim == 1:
                ema = ema.unsqueeze(-1)
            if pitch.ndim == 1:
                pitch = pitch.unsqueeze(-1)
            if loudness.ndim == 1:
                loudness = loudness.unsqueeze(-1)

            # Interpolate to target frame count if needed
            if ema.shape[0] != n_frames:
                ema = F.interpolate(
                    ema.T.unsqueeze(0), size=n_frames, mode='linear', align_corners=False
                ).squeeze(0).T
                pitch = F.interpolate(
                    pitch.T.unsqueeze(0), size=n_frames, mode='linear', align_corners=False
                ).squeeze(0).T
                loudness = F.interpolate(
                    loudness.T.unsqueeze(0), size=n_frames, mode='linear', align_corners=False
                ).squeeze(0).T

            emas.append(ema)
            pitches.append(pitch)
            loudnesses.append(loudness)

        # Stack into batches
        ema_batch = torch.stack(emas, dim=0).to(device)
        pitch_batch = torch.stack(pitches, dim=0).to(device)
        loudness_batch = torch.stack(loudnesses, dim=0).to(device)

        return ema_batch, pitch_batch, loudness_batch

    def decode(
        self,
        ema: torch.Tensor,
        pitch: torch.Tensor,
        loudness: torch.Tensor,
        spk_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode articulatory features to audio.

        Note: SPARC decode is NOT differentiable (uses torch.no_grad internally).
        For end-to-end training, use audio reconstruction loss on mel spectrograms.

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
        # Convert to numpy (SPARC expects numpy arrays)
        ema_np = ema.cpu().numpy() if isinstance(ema, torch.Tensor) else ema
        pitch_np = pitch.cpu().numpy() if isinstance(pitch, torch.Tensor) else pitch
        loudness_np = loudness.cpu().numpy() if isinstance(loudness, torch.Tensor) else loudness
        spk_emb_np = spk_emb.cpu().numpy() if isinstance(spk_emb, torch.Tensor) else spk_emb

        # Handle batch dimension
        if ema_np.ndim == 2:
            # Single sample, add batch dim for consistency
            ema_np = ema_np[np.newaxis, ...]
            pitch_np = pitch_np[np.newaxis, ...]
            loudness_np = loudness_np[np.newaxis, ...]
            spk_emb_np = spk_emb_np[np.newaxis, ...]

        # Decode each sample (SPARC decode doesn't batch well)
        audios = []
        for i in range(batch_size):
            # Squeeze the last dim if it's 1 (SPARC expects (n_frames, dim) not (n_frames, 1))
            p = pitch_np[i].squeeze(-1) if pitch_np[i].shape[-1] == 1 else pitch_np[i]
            l = loudness_np[i].squeeze(-1) if loudness_np[i].shape[-1] == 1 else loudness_np[i]

            audio = self.model.decode(
                ema=ema_np[i],
                pitch=p,
                loudness=l,
                spk_emb=spk_emb_np[i],
            )
            audios.append(audio)

        # Stack and convert to tensor
        # Pad/truncate to expected length
        audio_batch = []
        for audio in audios:
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            elif len(audio) < n_samples:
                audio = np.pad(audio, (0, n_samples - len(audio)))
            audio_batch.append(audio)

        return torch.from_numpy(np.stack(audio_batch)).float().to(ema.device)

    def extract_speaker_embedding(
        self,
        audio: Union[torch.Tensor, np.ndarray, str],
    ) -> np.ndarray:
        """Extract speaker embedding from audio.

        Args:
            audio: Audio waveform (tensor/numpy) or path to audio file

        Returns:
            (64,) speaker embedding
        """
        if self.mock:
            return np.random.randn(64).astype(np.float32)

        # Encode to get speaker embedding
        result = self.model.encode(audio, split_batch=True, reduce=True)
        return result['spk_emb']


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
