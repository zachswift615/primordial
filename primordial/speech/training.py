"""Training loops for speech learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass

from .config import SpeechConfig
from .encoders import MelSpectrogramEncoder, compute_mel_spectrogram
from .heads import SpeechHead, AudioReconstructionHead
from .phonemes import NUM_PHONEMES, phoneme_to_index
from .tts import create_tts_backend, phoneme_indices_to_audio


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    phoneme_loss: float
    phoneme_accuracy: float
    duration_loss: float
    pitch_loss: float
    total_loss: float
    step: int


class SpeechLRN(nn.Module):
    """LRN model configured for speech learning.

    Combines:
    - MelSpectrogramEncoder for audio input
    - Fourier mixing layers (from base LRN)
    - SpeechHead for phoneme/duration/pitch output
    - Optional AudioReconstructionHead for self-supervised learning
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        # Audio encoder
        self.audio_encoder = MelSpectrogramEncoder(config)

        # Import and use LRN's Fourier mixing layers
        from primordial.lrn.lrn_mixing import LRNFourierMixingLayer
        from primordial.lrn.lrn_config import LRNConfig

        # Create compatible LRN config
        lrn_config = LRNConfig(
            hidden_dim=config.hidden_dim,
            num_mixing_layers=6,
        )
        # Override sequence length for speech
        lrn_config.vision_seq_len = config.encoder_seq_len
        lrn_config.audio_seq_len = 0
        lrn_config.proprio_seq_len = 0
        lrn_config.touch_seq_len = 0

        # Fourier mixing layers
        self.mixing_layers = nn.ModuleList([
            LRNFourierMixingLayer(lrn_config)
            for _ in range(6)
        ])

        # Output heads
        pooled_dim = 3 * config.hidden_dim  # mean, max, last pooling
        self.speech_head = SpeechHead(config, input_dim=pooled_dim)
        self.audio_reconstruction_head = AudioReconstructionHead(config, input_dim=pooled_dim)

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spectrogram: (batch, n_mels, n_frames)

        Returns:
            phoneme_logits: (batch, num_phonemes)
            duration: (batch, 1)
            pitch: (batch, 1)
            reconstructed_mel: (batch, n_mels, n_frames)
        """
        # Encode audio
        x = self.audio_encoder(mel_spectrogram)  # (batch, seq_len, hidden_dim)

        # Fourier mixing
        for layer in self.mixing_layers:
            x = layer(x)

        # Pool for output heads
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        last = x[:, -1, :]
        pooled = torch.cat([mean_pool, max_pool, last], dim=1)

        # Speech output
        phoneme_logits, duration, pitch = self.speech_head(pooled)

        # Audio reconstruction (for self-supervised learning)
        reconstructed_mel = self.audio_reconstruction_head(pooled)

        return phoneme_logits, duration, pitch, reconstructed_mel


class PhonemeDataset(Dataset):
    """Dataset of audio samples labeled with phonemes.

    Each sample is a short audio clip containing a single phoneme.
    """

    def __init__(
        self,
        audio_dir: str,
        config: SpeechConfig,
    ):
        """
        Args:
            audio_dir: Directory containing phoneme audio files
                       Expected structure: audio_dir/{phoneme}/*.wav
            config: Speech config
        """
        self.config = config
        self.audio_dir = Path(audio_dir)
        self.samples: List[Tuple[Path, int]] = []  # (audio_path, phoneme_index)

        # Scan directory for samples
        if self.audio_dir.exists():
            for phoneme_dir in self.audio_dir.iterdir():
                if phoneme_dir.is_dir():
                    phoneme = phoneme_dir.name.upper()
                    phoneme_idx = phoneme_to_index(phoneme)

                    for audio_file in phoneme_dir.glob("*.wav"):
                        self.samples.append((audio_file, phoneme_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            mel_spectrogram: (n_mels, n_frames)
            phoneme_index: int
        """
        audio_path, phoneme_idx = self.samples[idx]

        # Load audio (requires torchaudio or similar)
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Compute mel spectrogram
            mel = compute_mel_spectrogram(
                waveform.squeeze(0),
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )

            return mel.squeeze(0), phoneme_idx

        except ImportError:
            # Fallback: return dummy data
            mel = torch.randn(self.config.n_mels, self.config.n_frames)
            return mel, phoneme_idx


class SyntheticPhonemeDataset(Dataset):
    """Synthetic dataset using TTS to generate phoneme audio.

    Useful for bootstrapping when no recorded data is available.
    """

    def __init__(
        self,
        config: SpeechConfig,
        samples_per_phoneme: int = 10,
    ):
        self.config = config
        self.samples_per_phoneme = samples_per_phoneme
        self.tts = create_tts_backend(config)

        # Generate all samples upfront
        self.samples: List[Tuple[torch.Tensor, int]] = []
        self._generate_samples()

    def _generate_samples(self):
        """Generate synthetic audio for each phoneme."""
        from .phonemes import PHONEME_INVENTORY

        for phoneme_idx, phoneme in enumerate(PHONEME_INVENTORY):
            for _ in range(self.samples_per_phoneme):
                # Synthesize single phoneme
                audio = self.tts.synthesize_phonemes([phoneme])

                # Convert to tensor
                waveform = torch.from_numpy(audio).float()

                # Compute mel spectrogram
                mel = compute_mel_spectrogram(
                    waveform,
                    sample_rate=self.tts.sample_rate,
                    n_mels=self.config.n_mels,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                )

                self.samples.append((mel.squeeze(0), phoneme_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


class PhonemeTrainer:
    """Trainer for Phase 1: Phoneme Classification.

    Trains the model to recognize which phoneme it's hearing.
    """

    def __init__(
        self,
        model: SpeechLRN,
        config: SpeechConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        self.step_count = 0

    def train_step(
        self,
        mel: torch.Tensor,
        phoneme_target: torch.Tensor,
    ) -> TrainingMetrics:
        """Single training step.

        Args:
            mel: (batch, n_mels, n_frames)
            phoneme_target: (batch,) phoneme indices

        Returns:
            TrainingMetrics
        """
        self.model.train()
        mel = mel.to(self.device)
        phoneme_target = phoneme_target.to(self.device)

        # Forward pass
        phoneme_logits, duration, pitch, reconstructed = self.model(mel)

        # Phoneme classification loss
        phoneme_loss = F.cross_entropy(phoneme_logits, phoneme_target)

        # Duration/pitch losses (no target for now, just regularize)
        duration_loss = duration.mean() * 0.01  # Encourage shorter durations
        pitch_loss = ((pitch - 0.5) ** 2).mean() * 0.01  # Center pitch

        # Total loss (skip reconstruction for now - variable input sizes)
        total_loss = (
            self.config.phoneme_loss_weight * phoneme_loss +
            self.config.duration_loss_weight * duration_loss +
            self.config.pitch_loss_weight * pitch_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute accuracy
        predicted = phoneme_logits.argmax(dim=-1)
        accuracy = (predicted == phoneme_target).float().mean().item()

        self.step_count += 1

        return TrainingMetrics(
            phoneme_loss=phoneme_loss.item(),
            phoneme_accuracy=accuracy,
            duration_loss=duration_loss.item(),
            pitch_loss=pitch_loss.item(),
            total_loss=total_loss.item(),
            step=self.step_count,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict of average metrics
        """
        total_metrics = {
            'phoneme_loss': 0.0,
            'phoneme_accuracy': 0.0,
            'total_loss': 0.0,
        }
        num_batches = 0

        for mel, phoneme_target in dataloader:
            metrics = self.train_step(mel, phoneme_target)

            total_metrics['phoneme_loss'] += metrics.phoneme_loss
            total_metrics['phoneme_accuracy'] += metrics.phoneme_accuracy
            total_metrics['total_loss'] += metrics.total_loss
            num_batches += 1

            if verbose and self.step_count % 10 == 0:
                print(f"Step {self.step_count}: "
                      f"loss={metrics.total_loss:.4f}, "
                      f"acc={metrics.phoneme_accuracy:.2%}")

        # Average
        for key in total_metrics:
            total_metrics[key] /= max(num_batches, 1)

        return total_metrics

    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
