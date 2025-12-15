"""Dataset for pre-processed SPARC features.

SPARC features are pre-computed from LibriSpeech and stored in HDF5 format
for efficient training. This avoids running the SPARC encoder during training.

HDF5 structure:
    mel: (n_samples, n_mels, mel_frames) - input mel spectrograms
    ema: (n_samples, sparc_frames, 12) - target EMA positions
    pitch: (n_samples, sparc_frames, 1) - target F0
    loudness: (n_samples, sparc_frames, 1) - target energy
    speaker_id: (n_samples,) - speaker IDs (for analysis, not training)
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

from .config import SpeechConfig
from .sparc_integration import SPARCWrapper
from .encoders import compute_mel_spectrogram


class SPARCDataset(Dataset):
    """Dataset loading pre-processed SPARC features from HDF5.

    Args:
        hdf5_path: Path to HDF5 file with preprocessed features
        config: SpeechConfig for validation
        transform: Optional transform to apply to mel spectrograms
    """

    def __init__(
        self,
        hdf5_path: str,
        config: SpeechConfig,
        transform: Optional[callable] = None,
    ):
        self.hdf5_path = hdf5_path
        self.config = config
        self.transform = transform

        # Open file to get length (keep closed for multiprocessing)
        with h5py.File(hdf5_path, 'r') as f:
            self._length = len(f['mel'])

        # Will be opened lazily per worker
        self._file = None

    def __len__(self) -> int:
        return self._length

    def _ensure_open(self):
        """Lazily open HDF5 file (for multiprocessing compatibility)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            mel: (n_mels, mel_frames) input mel spectrogram
            ema: (sparc_frames, 12) target EMA
            pitch: (sparc_frames, 1) target pitch
            loudness: (sparc_frames, 1) target loudness
        """
        self._ensure_open()

        mel = torch.from_numpy(self._file['mel'][idx])
        ema = torch.from_numpy(self._file['ema'][idx])
        pitch = torch.from_numpy(self._file['pitch'][idx])
        loudness = torch.from_numpy(self._file['loudness'][idx])

        if self.transform:
            mel = self.transform(mel)

        return mel, ema, pitch, loudness

    def __del__(self):
        if self._file is not None:
            self._file.close()


def preprocess_to_hdf5(
    audio_paths: list,
    output_path: str,
    config: SpeechConfig,
    sparc: SPARCWrapper,
    batch_size: int = 32,
    show_progress: bool = True,
) -> None:
    """Preprocess audio files to HDF5 with SPARC features.

    Args:
        audio_paths: List of (audio_path, speaker_id) tuples
        output_path: Path for output HDF5 file
        config: SpeechConfig
        sparc: SPARCWrapper for encoding
        batch_size: Batch size for SPARC encoding
        show_progress: Show progress bar
    """
    import soundfile as sf

    n_samples = len(audio_paths)
    mel_frames = config.n_frames
    sparc_frames = config.sparc_n_frames

    with h5py.File(output_path, 'w') as f:
        # Create datasets
        mel_ds = f.create_dataset(
            'mel',
            shape=(n_samples, config.n_mels, mel_frames),
            dtype=np.float32,
        )
        ema_ds = f.create_dataset(
            'ema',
            shape=(n_samples, sparc_frames, 12),
            dtype=np.float32,
        )
        pitch_ds = f.create_dataset(
            'pitch',
            shape=(n_samples, sparc_frames, 1),
            dtype=np.float32,
        )
        loudness_ds = f.create_dataset(
            'loudness',
            shape=(n_samples, sparc_frames, 1),
            dtype=np.float32,
        )
        speaker_ds = f.create_dataset(
            'speaker_id',
            shape=(n_samples,),
            dtype=h5py.special_dtype(vlen=str),
        )

        # Process in batches
        iterator = range(0, n_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Preprocessing")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_samples)
            batch_paths = audio_paths[start_idx:end_idx]

            # Load audio batch
            audios = []
            speakers = []
            for audio_path, speaker_id in batch_paths:
                audio_np, sr = sf.read(audio_path)

                # Resample if needed
                if sr != config.sample_rate:
                    import scipy.signal
                    audio_np = scipy.signal.resample(
                        audio_np,
                        int(len(audio_np) * config.sample_rate / sr)
                    )

                # Truncate/pad
                expected_len = int(config.sample_rate * config.audio_duration)
                if len(audio_np) > expected_len:
                    audio_np = audio_np[:expected_len]
                elif len(audio_np) < expected_len:
                    audio_np = np.pad(audio_np, (0, expected_len - len(audio_np)))

                audios.append(audio_np)
                speakers.append(speaker_id)

            # Convert to tensors
            audio_batch = torch.from_numpy(np.stack(audios)).float()

            # Compute mel spectrograms
            mel_batch = compute_mel_spectrogram(audio_batch, config)

            # Encode with SPARC
            ema_batch, pitch_batch, loudness_batch = sparc.encode(audio_batch)

            # Store
            batch_len = end_idx - start_idx
            mel_ds[start_idx:end_idx] = mel_batch.numpy()
            ema_ds[start_idx:end_idx] = ema_batch.numpy()
            pitch_ds[start_idx:end_idx] = pitch_batch.numpy()
            loudness_ds[start_idx:end_idx] = loudness_batch.numpy()
            for i, spk in enumerate(speakers):
                speaker_ds[start_idx + i] = spk
