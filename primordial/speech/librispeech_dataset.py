"""LibriSpeech dataset for speaker-diverse training.

Loads audio from LibriSpeech corpus and converts text transcripts to phonemes,
providing (mel_spectrogram, input_tokens, target_tokens) for sequence training.
"""

import torch
import torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import random
import logging

from .config import SpeechConfig
from .encoders import compute_mel_spectrogram
from .phonemes import phoneme_to_index
from .latent import SOS_TOKEN, EOS_TOKEN
from .g2p import text_to_phonemes

logger = logging.getLogger(__name__)


class LibriSpeechDataset(Dataset):
    """Dataset for LibriSpeech audio with phoneme targets.

    Scans LibriSpeech directory structure and provides:
    - mel: Mel spectrogram of utterance
    - input_tokens: [SOS, phoneme1, phoneme2, ...] for teacher forcing
    - target_tokens: [phoneme1, phoneme2, ..., EOS] for loss
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train-clean-100",
        config: Optional[SpeechConfig] = None,
        max_duration: float = 5.0,
        min_duration: float = 0.5,
        max_phonemes: Optional[int] = None,
        min_phonemes: int = 3,
        truncate_phonemes: Optional[int] = None,
        cache_audio: bool = False,
        augment: bool = False,
    ):
        """Initialize LibriSpeech dataset.

        Args:
            root: Path to LibriSpeech directory (containing train-clean-100, etc.)
            split: Which split to load (train-clean-100, dev-clean, etc.)
            config: Speech configuration (uses defaults if None)
            max_duration: Skip utterances longer than this (seconds)
            min_duration: Skip utterances shorter than this (seconds)
            max_phonemes: Filter to utterances with at most this many phonemes
            min_phonemes: Filter to utterances with at least this many phonemes
            truncate_phonemes: Truncate phoneme sequences to first N (keeps all utterances)
            cache_audio: Whether to cache loaded audio in memory
            augment: Whether to apply audio augmentation
        """
        self.root = Path(root)
        self.split = split
        self.config = config or SpeechConfig()
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_phonemes = max_phonemes
        self.min_phonemes = min_phonemes
        self.truncate_phonemes = truncate_phonemes
        self.cache_audio = cache_audio
        self.augment = augment

        self.split_dir = self.root / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        # Scan for all utterances
        self.samples: List[Tuple[Path, str, str]] = []  # (audio_path, text, speaker_id)
        self._scan_directory()

        # Cache for loaded audio
        self._audio_cache: Dict[int, Tuple[torch.Tensor, List[int]]] = {}

        logger.info(
            f"LibriSpeechDataset: {len(self.samples)} utterances from {split}"
        )

    def _scan_directory(self) -> None:
        """Scan LibriSpeech directory structure for utterances."""
        for trans_file in self.split_dir.rglob("*.trans.txt"):
            chapter_dir = trans_file.parent
            speaker_id = chapter_dir.parent.name

            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        audio_path = chapter_dir / f"{utterance_id}.flac"

                        if audio_path.exists():
                            # Convert text to phonemes and filter
                            phonemes = text_to_phonemes(text)

                            if len(phonemes) < self.min_phonemes:
                                continue
                            # Only filter by max_phonemes if not using truncation
                            if self.max_phonemes and not self.truncate_phonemes and len(phonemes) > self.max_phonemes:
                                continue

                            self.samples.append((audio_path, text, speaker_id))

    def _load_audio(self, audio_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess audio file.

        Returns:
            Waveform tensor or None if filtered out by duration
        """
        # Use soundfile for robust FLAC loading
        data, sr = sf.read(audio_path)
        waveform = torch.from_numpy(data).float()

        # Convert to mono if stereo (soundfile returns [samples, channels] for stereo)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=1)

        # Resample if needed (LibriSpeech is 16kHz which matches config)
        if sr != self.config.sample_rate:
            # Simple linear interpolation resampling
            target_len = int(len(waveform) * self.config.sample_rate / sr)
            waveform = F.interpolate(
                waveform.view(1, 1, -1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze()

        # Check duration
        duration = len(waveform) / self.config.sample_rate
        if duration < self.min_duration or duration > self.max_duration:
            return None

        return waveform  # (samples,)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Get a sample.

        Returns:
            mel: (n_mels, n_frames) mel spectrogram
            input_tokens: (seq_len,) input sequence with SOS
            target_tokens: (seq_len,) target sequence with EOS
            speaker_id: str - speaker identifier
        """
        # Check cache
        if self.cache_audio and idx in self._audio_cache:
            mel, phoneme_indices = self._audio_cache[idx]
        else:
            audio_path, text, speaker_id = self.samples[idx]

            # Load audio
            waveform = self._load_audio(audio_path)
            if waveform is None:
                # Duration filter - try next sample (wrap around)
                return self.__getitem__((idx + 1) % len(self.samples))

            # Apply augmentation if enabled
            if self.augment:
                waveform = self._augment_audio(waveform)

            # Compute mel spectrogram
            mel = compute_mel_spectrogram(
                waveform,
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            ).squeeze(0)

            # Convert text to phonemes
            phonemes = text_to_phonemes(text)
            phoneme_indices = [phoneme_to_index(p) for p in phonemes]

            # Cache if enabled
            if self.cache_audio:
                self._audio_cache[idx] = (mel, phoneme_indices)

        # Apply truncation if specified
        if self.truncate_phonemes and len(phoneme_indices) > self.truncate_phonemes:
            phoneme_indices = phoneme_indices[:self.truncate_phonemes]

        # Input: [SOS, phoneme1, phoneme2, ...]
        input_tokens = torch.tensor([SOS_TOKEN] + phoneme_indices, dtype=torch.long)

        # Target: [phoneme1, phoneme2, ..., EOS]
        target_tokens = torch.tensor(phoneme_indices + [EOS_TOKEN], dtype=torch.long)

        return mel, input_tokens, target_tokens, self.samples[idx][2]

    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to audio.

        Args:
            waveform: (samples,) audio tensor

        Returns:
            Augmented waveform
        """
        # Import here to avoid circular dependency
        from .augmentation import augment_waveform
        return augment_waveform(waveform, self.config.sample_rate)

    def get_speaker_ids(self) -> List[str]:
        """Get list of unique speaker IDs in the dataset."""
        return list(set(s[2] for s in self.samples))


class MixedSpeechDataset(Dataset):
    """Combines synthetic (Piper) and real (LibriSpeech) data.

    Implements curriculum learning by adjusting the real_ratio over time.
    """

    def __init__(
        self,
        synthetic_dataset: Dataset,
        real_dataset: Dataset,
        real_ratio: float = 0.5,
    ):
        """Initialize mixed dataset.

        Args:
            synthetic_dataset: WordDataset with Piper TTS data
            real_dataset: LibriSpeechDataset with real audio
            real_ratio: Probability of sampling from real dataset (0.0 to 1.0)
        """
        self.synthetic = synthetic_dataset
        self.real = real_dataset
        self.real_ratio = real_ratio

        # Validate
        if not 0.0 <= real_ratio <= 1.0:
            raise ValueError(f"real_ratio must be 0.0-1.0, got {real_ratio}")

    def set_real_ratio(self, ratio: float) -> None:
        """Update the real/synthetic mixing ratio.

        Call this between epochs for curriculum learning.
        """
        self.real_ratio = max(0.0, min(1.0, ratio))

    def __len__(self) -> int:
        # Return size of larger dataset
        return max(len(self.synthetic), len(self.real))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Get a sample from either synthetic or real dataset.

        Returns:
            mel: (n_mels, n_frames) mel spectrogram
            input_tokens: (seq_len,) input sequence with SOS
            target_tokens: (seq_len,) target sequence with EOS
            source: str - source identifier
        """
        if random.random() < self.real_ratio:
            # Sample from real (LibriSpeech)
            real_idx = idx % len(self.real)
            return self.real[real_idx]
        else:
            # Sample from synthetic (Piper)
            synth_idx = idx % len(self.synthetic)
            return self.synthetic[synth_idx]


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Collate function for variable-length sequences.

    Pads mel spectrograms and token sequences to batch max length.

    Args:
        batch: List of (mel, input_tokens, target_tokens, identifier) tuples

    Returns:
        mels: (batch, n_mels, max_frames) padded mel spectrograms
        input_tokens: (batch, max_seq_len) padded input sequences
        target_tokens: (batch, max_seq_len) padded target sequences
        identifiers: List of source identifiers
    """
    mels, input_seqs, target_seqs, identifiers = zip(*batch)

    # Find max lengths
    max_frames = max(m.shape[1] for m in mels)
    max_seq_len = max(len(s) for s in input_seqs)

    # Pad mels
    padded_mels = []
    for mel in mels:
        if mel.shape[1] < max_frames:
            pad_size = max_frames - mel.shape[1]
            mel = F.pad(mel, (0, pad_size))
        padded_mels.append(mel)
    mels_tensor = torch.stack(padded_mels)

    # Pad sequences (pad with EOS token for targets, 0 for inputs)
    padded_inputs = []
    padded_targets = []
    for inp, tgt in zip(input_seqs, target_seqs):
        if len(inp) < max_seq_len:
            inp = F.pad(inp, (0, max_seq_len - len(inp)), value=0)
            tgt = F.pad(tgt, (0, max_seq_len - len(tgt)), value=EOS_TOKEN)
        padded_inputs.append(inp)
        padded_targets.append(tgt)

    inputs_tensor = torch.stack(padded_inputs)
    targets_tensor = torch.stack(padded_targets)

    return mels_tensor, inputs_tensor, targets_tensor, list(identifiers)


def create_librispeech_dataloaders(
    root: Union[str, Path],
    config: Optional[SpeechConfig] = None,
    train_split: str = "train-clean-100",
    val_split: Optional[str] = None,
    batch_size: int = 32,
    max_phonemes: Optional[int] = None,
    augment: bool = True,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Create train and validation dataloaders for LibriSpeech.

    Args:
        root: Path to LibriSpeech directory
        config: Speech configuration
        train_split: Training split name
        val_split: Validation split name (None to skip)
        batch_size: Batch size for dataloaders
        max_phonemes: Filter by max phoneme sequence length
        augment: Whether to augment training data
        num_workers: Number of dataloader workers

    Returns:
        (train_loader, val_loader) - val_loader is None if val_split is None
    """
    config = config or SpeechConfig()

    # Create training dataset
    train_dataset = LibriSpeechDataset(
        root=root,
        split=train_split,
        config=config,
        max_phonemes=max_phonemes,
        augment=augment,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create validation dataset if requested
    val_loader = None
    if val_split:
        val_dataset = LibriSpeechDataset(
            root=root,
            split=val_split,
            config=config,
            max_phonemes=max_phonemes,
            augment=False,  # No augmentation for validation
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_variable_length,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
