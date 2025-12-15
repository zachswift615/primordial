"""Tests for SPARC feature dataset."""
import torch
import pytest
import tempfile
import h5py
import numpy as np
from pathlib import Path

from primordial.speech.config import SpeechConfig
from primordial.speech.sparc_dataset import SPARCDataset, preprocess_to_hdf5


class TestSPARCDataset:
    """Tests for pre-processed SPARC feature dataset."""

    @pytest.fixture
    def config(self):
        return SpeechConfig(audio_duration=1.0)

    @pytest.fixture
    def mock_hdf5(self, config):
        """Create a mock HDF5 file with SPARC features."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            path = f.name

        n_samples = 10
        n_frames = config.sparc_n_frames
        n_mels = config.n_mels
        mel_frames = config.n_frames

        with h5py.File(path, 'w') as f:
            f.create_dataset('mel', data=np.random.randn(n_samples, n_mels, mel_frames).astype(np.float32))
            f.create_dataset('ema', data=np.random.randn(n_samples, n_frames, 12).astype(np.float32))
            f.create_dataset('pitch', data=np.abs(np.random.randn(n_samples, n_frames, 1)).astype(np.float32) * 100 + 100)
            f.create_dataset('loudness', data=np.random.rand(n_samples, n_frames, 1).astype(np.float32))
            f.create_dataset('speaker_id', data=np.array([f'spk_{i % 3}'.encode() for i in range(n_samples)]))

        yield path
        Path(path).unlink()

    def test_load_dataset(self, mock_hdf5, config):
        """Should load HDF5 dataset correctly."""
        dataset = SPARCDataset(mock_hdf5, config)

        assert len(dataset) == 10

    def test_getitem_shapes(self, mock_hdf5, config):
        """Dataset items should have correct shapes."""
        dataset = SPARCDataset(mock_hdf5, config)
        mel, ema, pitch, loudness = dataset[0]

        assert mel.shape == (config.n_mels, config.n_frames)
        assert ema.shape == (config.sparc_n_frames, 12)
        assert pitch.shape == (config.sparc_n_frames, 1)
        assert loudness.shape == (config.sparc_n_frames, 1)

    def test_getitem_types(self, mock_hdf5, config):
        """Dataset items should be torch tensors."""
        dataset = SPARCDataset(mock_hdf5, config)
        mel, ema, pitch, loudness = dataset[0]

        assert isinstance(mel, torch.Tensor)
        assert isinstance(ema, torch.Tensor)
        assert mel.dtype == torch.float32
