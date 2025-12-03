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
from .encoders import MelSpectrogramEncoder, CNNMelEncoder, compute_mel_spectrogram
from .heads import SpeechHead, AudioReconstructionHead, ProductionHead
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

        # Audio encoder - choose based on config
        if config.encoder_type == "cnn":
            self.audio_encoder = CNNMelEncoder(config)
        else:
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
        self.production_head = ProductionHead(config, input_dim=pooled_dim)
        self.audio_reconstruction_head = AudioReconstructionHead(config, input_dim=pooled_dim)

    def encode(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to pooled features.

        Args:
            mel_spectrogram: (batch, n_mels, n_frames)

        Returns:
            pooled: (batch, 384) pooled features
        """
        x = self.audio_encoder(mel_spectrogram)
        for layer in self.mixing_layers:
            x = layer(x)
        mean_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        last = x[:, -1, :]
        return torch.cat([mean_pool, max_pool, last], dim=1)

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
        pooled = self.encode(mel_spectrogram)

        # Speech output (perception)
        phoneme_logits, duration, pitch = self.speech_head(pooled)

        # Audio reconstruction (for self-supervised learning)
        reconstructed_mel = self.audio_reconstruction_head(pooled)

        return phoneme_logits, duration, pitch, reconstructed_mel

    def forward_production(
        self,
        mel_spectrogram: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for production head.

        Args:
            mel_spectrogram: (batch, n_mels, n_frames)

        Returns:
            latent: (batch, latent_dim) position in articulatory space
            duration: (batch, 1)
            pitch: (batch, 1)
        """
        pooled = self.encode(mel_spectrogram)
        return self.production_head(pooled)


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
        import torch.nn.functional as F

        # Target frames for consistent batch size
        target_frames = self.config.n_frames

        for phoneme_idx, phoneme in enumerate(PHONEME_INVENTORY):
            for _ in range(self.samples_per_phoneme):
                # Synthesize single phoneme
                audio = self.tts.synthesize_phonemes([phoneme])

                # Convert to tensor
                waveform = torch.from_numpy(audio).float()

                # Resample if TTS sample rate differs from config
                if self.tts.sample_rate != self.config.sample_rate:
                    # Simple resampling via interpolation
                    target_len = int(len(waveform) * self.config.sample_rate / self.tts.sample_rate)
                    waveform = F.interpolate(
                        waveform.view(1, 1, -1),
                        size=target_len,
                        mode='linear',
                        align_corners=False
                    ).squeeze()

                # Compute mel spectrogram at config sample rate
                mel = compute_mel_spectrogram(
                    waveform,
                    sample_rate=self.config.sample_rate,
                    n_mels=self.config.n_mels,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                )

                mel = mel.squeeze(0)  # (n_mels, n_frames)

                # Pad or truncate to target_frames for consistent batching
                n_frames = mel.shape[1]
                if n_frames < target_frames:
                    # Pad with zeros on the right
                    padding = torch.zeros(self.config.n_mels, target_frames - n_frames)
                    mel = torch.cat([mel, padding], dim=1)
                elif n_frames > target_frames:
                    # Truncate
                    mel = mel[:, :target_frames]

                self.samples.append((mel, phoneme_idx))

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


@dataclass
class ProductionMetrics:
    """Metrics from a production training step."""
    latent_loss: float
    match_rate: float
    embed_loss: float
    babbling_ratio: float
    step: int


class ProductionTrainer:
    """Trainer for Phase 2: Speech Production.

    Trains the model to produce phonemes by:
    1. Outputting latent positions in articulatory space
    2. Synthesizing audio via TTS
    3. Listening to its own output
    4. Comparing heard phoneme to intended phoneme

    Uses curriculum of babbling (exploration) → imitation (refinement).
    """

    def __init__(
        self,
        model: SpeechLRN,
        config: SpeechConfig,
        device: str = "cpu",
    ):
        from .latent import get_anchor, snap_to_nearest_anchor, get_all_anchors_tensor
        from .phonemes import PHONEME_INVENTORY, index_to_phoneme

        self.model = model.to(device)
        self.config = config
        self.device = device

        # Import latent space functions
        self.get_anchor = get_anchor
        self.snap_to_nearest = snap_to_nearest_anchor
        self.get_all_anchors = get_all_anchors_tensor
        self.phoneme_inventory = PHONEME_INVENTORY
        self.index_to_phoneme = index_to_phoneme

        # TTS for synthesizing produced sounds
        self.tts = create_tts_backend(config)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        self.step_count = 0
        self.current_babbling_ratio = config.babbling_ratio

    def _get_babbling_ratio(self, epoch: int) -> float:
        """Compute babbling ratio for current epoch."""
        ratio = self.config.babbling_ratio - epoch * self.config.babbling_decay
        return max(ratio, self.config.min_babbling_ratio)

    def _babbling_step(self) -> Tuple[torch.Tensor, str, float]:
        """Generate random latent and see what phoneme it produces.

        Returns:
            (latent, produced_phoneme, distance_to_nearest)
        """
        # Random latent in [-1, 1]^6, with gradient tracking
        latent = torch.randn(self.config.latent_dim, device=self.device, requires_grad=True)
        latent_tanh = latent.tanh()

        # Find nearest phoneme anchor (detach for snapping)
        phoneme, distance = self.snap_to_nearest(latent_tanh.detach())

        return latent_tanh, phoneme, distance

    def _imitation_step(
        self,
        target_phoneme: str,
        context_mel: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Try to produce a target phoneme.

        Args:
            target_phoneme: The phoneme to produce
            context_mel: Optional context audio (e.g., silence)

        Returns:
            Dict with latent, produced_phoneme, perceived_phoneme, losses
        """
        self.model.train()

        # Get target anchor
        target_anchor = self.get_anchor(target_phoneme).to(self.device)

        # If no context, use zeros (silence)
        if context_mel is None:
            context_mel = torch.zeros(
                1, self.config.n_mels, self.config.n_frames,
                device=self.device
            )

        # Production: model outputs latent position
        latent, duration, pitch = self.model.forward_production(context_mel)
        latent = latent.squeeze(0)  # (latent_dim,)

        # Snap to nearest phoneme for TTS
        produced_phoneme, distance = self.snap_to_nearest(latent)

        # Synthesize audio
        audio = self.tts.synthesize_phonemes([produced_phoneme])
        waveform = torch.from_numpy(audio).float()

        # Resample if needed
        if self.tts.sample_rate != self.config.sample_rate:
            target_len = int(len(waveform) * self.config.sample_rate / self.tts.sample_rate)
            waveform = F.interpolate(
                waveform.view(1, 1, -1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze()

        # Compute mel spectrogram of produced audio
        produced_mel = compute_mel_spectrogram(
            waveform,
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        # Pad/truncate to standard size
        produced_mel = produced_mel.squeeze(0)
        n_frames = produced_mel.shape[1]
        if n_frames < self.config.n_frames:
            padding = torch.zeros(self.config.n_mels, self.config.n_frames - n_frames)
            produced_mel = torch.cat([produced_mel, padding], dim=1)
        elif n_frames > self.config.n_frames:
            produced_mel = produced_mel[:, :self.config.n_frames]

        produced_mel = produced_mel.unsqueeze(0).to(self.device)

        # Perception: classify what we hear
        with torch.no_grad():
            perceived_logits, _, _, _ = self.model(produced_mel)
            perceived_idx = perceived_logits.argmax(dim=-1).item()
            perceived_phoneme = self.index_to_phoneme(perceived_idx)

        # Compute losses
        # 1. Latent anchor loss - pull toward target
        latent_loss = F.mse_loss(latent, target_anchor)

        # 2. Match indicator
        match = 1.0 if perceived_phoneme == target_phoneme else 0.0

        # 3. Embedding similarity (use pooled features as embedding)
        target_mel = self._get_target_mel(target_phoneme)
        if target_mel is not None:
            target_mel = target_mel.to(self.device)
            with torch.no_grad():
                target_embed = self.model.encode(target_mel)
            produced_embed = self.model.encode(produced_mel)
            embed_loss = F.mse_loss(produced_embed, target_embed)
        else:
            embed_loss = torch.tensor(0.0, device=self.device)

        return {
            'latent': latent,
            'target_anchor': target_anchor,
            'produced_phoneme': produced_phoneme,
            'perceived_phoneme': perceived_phoneme,
            'latent_loss': latent_loss,
            'match': match,
            'embed_loss': embed_loss,
            'distance': distance,
        }

    def _get_target_mel(self, phoneme: str) -> Optional[torch.Tensor]:
        """Get mel spectrogram for a target phoneme."""
        # Synthesize target phoneme
        audio = self.tts.synthesize_phonemes([phoneme])
        waveform = torch.from_numpy(audio).float()

        # Resample if needed
        if self.tts.sample_rate != self.config.sample_rate:
            target_len = int(len(waveform) * self.config.sample_rate / self.tts.sample_rate)
            waveform = F.interpolate(
                waveform.view(1, 1, -1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze()

        mel = compute_mel_spectrogram(
            waveform,
            sample_rate=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
        )

        mel = mel.squeeze(0)
        n_frames = mel.shape[1]
        if n_frames < self.config.n_frames:
            padding = torch.zeros(self.config.n_mels, self.config.n_frames - n_frames)
            mel = torch.cat([mel, padding], dim=1)
        elif n_frames > self.config.n_frames:
            mel = mel[:, :self.config.n_frames]

        return mel.unsqueeze(0)

    def train_production_step(
        self,
        target_phoneme: str,
        is_babbling: bool = False,
    ) -> ProductionMetrics:
        """Single production training step.

        Args:
            target_phoneme: Target phoneme to produce
            is_babbling: If True, explore randomly; if False, imitate

        Returns:
            ProductionMetrics
        """
        self.model.train()

        if is_babbling:
            # Babbling: random exploration
            latent, produced_phoneme, distance = self._babbling_step()

            # Synthesize and hear
            audio = self.tts.synthesize_phonemes([produced_phoneme])
            waveform = torch.from_numpy(audio).float()

            if self.tts.sample_rate != self.config.sample_rate:
                target_len = int(len(waveform) * self.config.sample_rate / self.tts.sample_rate)
                waveform = F.interpolate(
                    waveform.view(1, 1, -1),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()

            mel = compute_mel_spectrogram(
                waveform,
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            )

            mel = mel.squeeze(0)
            n_frames = mel.shape[1]
            if n_frames < self.config.n_frames:
                padding = torch.zeros(self.config.n_mels, self.config.n_frames - n_frames)
                mel = torch.cat([mel, padding], dim=1)
            elif n_frames > self.config.n_frames:
                mel = mel[:, :self.config.n_frames]

            mel = mel.unsqueeze(0).to(self.device)

            # Perceive what we produced
            with torch.no_grad():
                perceived_logits, _, _, _ = self.model(mel)
                perceived_idx = perceived_logits.argmax(dim=-1).item()
                perceived_phoneme = self.index_to_phoneme(perceived_idx)

            # Learn: "this latent produces this phoneme"
            perceived_anchor = self.get_anchor(perceived_phoneme).to(self.device)
            latent_loss = F.mse_loss(latent, perceived_anchor)

            match = 1.0 if produced_phoneme == perceived_phoneme else 0.0
            embed_loss = torch.tensor(0.0)

        else:
            # Imitation: try to produce target
            result = self._imitation_step(target_phoneme)
            latent_loss = result['latent_loss']
            match = result['match']
            embed_loss = result['embed_loss']

        # Combined loss
        total_loss = latent_loss + 0.5 * embed_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1

        return ProductionMetrics(
            latent_loss=latent_loss.item(),
            match_rate=match,
            embed_loss=embed_loss.item() if isinstance(embed_loss, torch.Tensor) else embed_loss,
            babbling_ratio=self.current_babbling_ratio,
            step=self.step_count,
        )

    def train_production_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """Train production for one epoch.

        Args:
            dataloader: DataLoader with (mel, phoneme_idx) pairs
            epoch: Current epoch number
            verbose: Print step-by-step info

        Returns:
            Dict of average metrics
        """
        self.current_babbling_ratio = self._get_babbling_ratio(epoch)

        total_metrics = {
            'latent_loss': 0.0,
            'match_rate': 0.0,
            'embed_loss': 0.0,
        }
        num_steps = 0

        for mel, phoneme_idx in dataloader:
            # For each sample in batch, do production training
            for i in range(len(phoneme_idx)):
                target_phoneme = self.index_to_phoneme(phoneme_idx[i].item())

                # Decide babbling vs imitation
                is_babbling = np.random.random() < self.current_babbling_ratio

                metrics = self.train_production_step(target_phoneme, is_babbling)

                total_metrics['latent_loss'] += metrics.latent_loss
                total_metrics['match_rate'] += metrics.match_rate
                total_metrics['embed_loss'] += metrics.embed_loss
                num_steps += 1

                if verbose and self.step_count % 20 == 0:
                    print(f"Step {self.step_count}: "
                          f"latent={metrics.latent_loss:.4f}, "
                          f"match={metrics.match_rate:.0%}, "
                          f"babble={self.current_babbling_ratio:.0%}")

        # Average
        for key in total_metrics:
            total_metrics[key] /= max(num_steps, 1)

        total_metrics['babbling_ratio'] = self.current_babbling_ratio

        return total_metrics

    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'current_babbling_ratio': self.current_babbling_ratio,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        if 'current_babbling_ratio' in checkpoint:
            self.current_babbling_ratio = checkpoint['current_babbling_ratio']


@dataclass
class SequenceMetrics:
    """Metrics from a sequence training step."""
    total_loss: float
    discrete_loss: float
    latent_loss: float
    accuracy: float
    step: int


class SequenceTrainer:
    """Trainer for autoregressive phoneme sequence generation.

    Uses teacher forcing with dual losses:
    - Cross-entropy for discrete token prediction
    - MSE for latent anchor prediction (masked to real phonemes)
    """

    def __init__(
        self,
        model,  # SpeechSequenceLRN
        config: SpeechConfig,
        device: str = "cpu",
        lr: float = 1e-4,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.step_count = 0

        # Loss weights
        self.discrete_weight = 1.0
        self.latent_weight = 0.5

    def _get_target_anchors(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Get latent anchors for target token sequence.

        Args:
            target_tokens: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, 6) anchor positions
        """
        from .latent import get_anchor, LATENT_DIM
        from .phonemes import index_to_phoneme

        batch, seq_len = target_tokens.shape
        anchors = torch.zeros(batch, seq_len, LATENT_DIM, device=target_tokens.device)

        for b in range(batch):
            for t in range(seq_len):
                token = target_tokens[b, t].item()
                if token < 41:  # Real phoneme
                    phoneme = index_to_phoneme(token)
                    anchors[b, t] = get_anchor(phoneme)

        return anchors

    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> float:
        """Compute token prediction accuracy."""
        preds = logits.argmax(dim=-1)
        mask = targets != ignore_index
        if mask.sum() == 0:
            return 0.0
        correct = (preds == targets) & mask
        return (correct.sum() / mask.sum()).item()

    def train_step(
        self,
        mel: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict:
        """Single training step with teacher forcing.

        Args:
            mel: (batch, n_mels, n_frames) mel spectrogram
            input_tokens: (batch, seq_len) input with SOS
            target_tokens: (batch, seq_len) target with EOS

        Returns:
            Dict with loss values
        """
        from .latent import EOS_TOKEN

        self.model.train()
        mel = mel.to(self.device)
        input_tokens = input_tokens.to(self.device)
        target_tokens = target_tokens.to(self.device)

        # Forward pass
        discrete_logits, latent_pred = self.model(mel, input_tokens)

        # Discrete loss (cross-entropy)
        # Target uses -100 for padding positions (standard PyTorch ignore_index)
        # EOS tokens in target are NOT ignored - model must learn to predict them!
        discrete_loss = F.cross_entropy(
            discrete_logits.view(-1, discrete_logits.size(-1)),
            target_tokens.view(-1),
            ignore_index=-100,  # Ignore padding only, not EOS
        )

        # Latent loss (MSE, masked to real phonemes only)
        target_anchors = self._get_target_anchors(target_tokens)
        # Real phonemes are 0-40, exclude EOS (42), SOS (41), and padding (-100)
        phoneme_mask = (target_tokens >= 0) & (target_tokens < 41)

        if phoneme_mask.sum() > 0:
            latent_loss = F.mse_loss(
                latent_pred[phoneme_mask],
                target_anchors[phoneme_mask],
            )
        else:
            latent_loss = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = self.discrete_weight * discrete_loss + self.latent_weight * latent_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1

        # Accuracy
        accuracy = self._compute_accuracy(discrete_logits, target_tokens)

        return {
            'total': total_loss.item(),
            'discrete': discrete_loss.item(),
            'latent': latent_loss.item() if isinstance(latent_loss, torch.Tensor) else latent_loss,
            'accuracy': accuracy,
            'step': self.step_count,
        }

    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
