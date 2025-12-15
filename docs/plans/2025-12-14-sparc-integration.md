# SPARC Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Piper-based phoneme output with SPARC articulatory control, enabling differentiable end-to-end training with interpretable 14D articulator features (12 EMA + pitch + loudness).

**Architecture:** New ArticulatoryHead predicts 14D SPARC features from LRN encoder output. Frozen SPARC decoder synthesizes audio from predictions. Training uses MSE on pre-computed SPARC targets, with optional end-to-end audio loss for refinement.

**Tech Stack:** PyTorch, SPARC (Berkeley), existing SpeechLRN encoder, HDF5 for preprocessed data

**Design Doc:** `docs/sparc_integration_plan.md`

---

## Phase 1: Foundation

### Task 1: Add SPARC Configuration to SpeechConfig

**Files:**
- Modify: `primordial/speech/config.py`
- Test: `tests/speech/test_config.py` (create)

**Step 1: Write the failing test**

Create `tests/speech/test_config.py`:

```python
"""Tests for speech configuration."""
import pytest
from primordial.speech.config import SpeechConfig


def test_sparc_config_defaults():
    """SPARC configuration should have sensible defaults."""
    config = SpeechConfig()

    # SPARC feature dimensions
    assert config.sparc_ema_dim == 12
    assert config.sparc_frame_rate == 50  # Hz

    # Voice embedding
    assert config.sparc_speaker_dim == 64

    # Loss weights
    assert config.ema_loss_weight == 1.0
    assert config.sparc_pitch_loss_weight == 0.5
    assert config.sparc_loudness_loss_weight == 0.3
    assert config.smoothness_loss_weight == 0.1


def test_sparc_output_frames():
    """Should compute correct SPARC frame count for audio duration."""
    config = SpeechConfig(audio_duration=2.0)
    # 2 seconds at 50Hz = 100 frames
    assert config.sparc_n_frames == 100


def test_sparc_mel_to_sparc_ratio():
    """Should compute mel-to-SPARC frame ratio."""
    config = SpeechConfig()
    # Mel at 100Hz (hop=160 @ 16kHz), SPARC at 50Hz
    # Ratio = 100/50 = 2
    assert config.mel_to_sparc_ratio == 2.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_config.py -v`
Expected: FAIL with "AttributeError: 'SpeechConfig' object has no attribute 'sparc_ema_dim'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/config.py` after line 58 (after `min_babbling_ratio`):

```python
    # SPARC integration
    sparc_ema_dim: int = 12                # EMA articulator dimensions
    sparc_frame_rate: int = 50             # SPARC output rate in Hz
    sparc_speaker_dim: int = 64            # Speaker embedding dimension
    sparc_model_path: str = ""             # Path to SPARC checkpoint
    sparc_voice_embedding_path: str = ""   # Path to your_voice_embedding.npy

    # SPARC loss weights
    ema_loss_weight: float = 1.0           # Articulation accuracy (most important)
    sparc_pitch_loss_weight: float = 0.5   # Prosody melody
    sparc_loudness_loss_weight: float = 0.3  # Emphasis patterns
    smoothness_loss_weight: float = 0.1    # Temporal smoothness regularization

    @property
    def sparc_n_frames(self) -> int:
        """Number of SPARC frames per audio sample."""
        return int(self.audio_duration * self.sparc_frame_rate)

    @property
    def mel_to_sparc_ratio(self) -> float:
        """Ratio of mel frame rate to SPARC frame rate."""
        mel_rate = self.sample_rate / self.hop_length  # 16000/160 = 100Hz
        return mel_rate / self.sparc_frame_rate
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/config.py tests/speech/test_config.py
git commit -m "feat(speech): add SPARC configuration parameters"
```

---

### Task 2: Create SPARCWrapper Class

**Files:**
- Create: `primordial/speech/sparc_integration.py`
- Test: `tests/speech/test_sparc_integration.py` (create)

**Step 1: Write the failing test**

Create `tests/speech/test_sparc_integration.py`:

```python
"""Tests for SPARC integration module."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.sparc_integration import SPARCWrapper, VoiceIdentity


class TestSPARCWrapper:
    """Tests for SPARC wrapper (mock mode for CI)."""

    @pytest.fixture
    def config(self):
        return SpeechConfig()

    @pytest.fixture
    def wrapper(self, config):
        # Use mock mode for testing without SPARC installed
        return SPARCWrapper(config, mock=True)

    def test_encode_shape(self, wrapper, config):
        """Encode should return correct feature shapes."""
        batch_size = 4
        audio_len = int(config.sample_rate * config.audio_duration)
        audio = torch.randn(batch_size, audio_len)

        ema, pitch, loudness = wrapper.encode(audio)

        expected_frames = config.sparc_n_frames
        assert ema.shape == (batch_size, expected_frames, 12)
        assert pitch.shape == (batch_size, expected_frames, 1)
        assert loudness.shape == (batch_size, expected_frames, 1)

    def test_decode_shape(self, wrapper, config):
        """Decode should return audio of correct length."""
        batch_size = 2
        n_frames = config.sparc_n_frames

        ema = torch.randn(batch_size, n_frames, 12)
        pitch = torch.randn(batch_size, n_frames, 1)
        loudness = torch.randn(batch_size, n_frames, 1)
        spk_emb = torch.randn(batch_size, 64)

        audio = wrapper.decode(ema, pitch, loudness, spk_emb)

        expected_samples = int(config.sample_rate * config.audio_duration)
        assert audio.shape == (batch_size, expected_samples)

    def test_decode_is_differentiable(self, wrapper, config):
        """Decode should support gradient flow."""
        n_frames = config.sparc_n_frames

        ema = torch.randn(1, n_frames, 12, requires_grad=True)
        pitch = torch.randn(1, n_frames, 1, requires_grad=True)
        loudness = torch.randn(1, n_frames, 1, requires_grad=True)
        spk_emb = torch.randn(1, 64)

        audio = wrapper.decode(ema, pitch, loudness, spk_emb)
        loss = audio.sum()
        loss.backward()

        assert ema.grad is not None
        assert pitch.grad is not None
        assert loudness.grad is not None


class TestVoiceIdentity:
    """Tests for voice identity management."""

    def test_random_embedding(self):
        """Should create random embedding when no file provided."""
        voice = VoiceIdentity(dim=64)
        emb = voice.get_embedding(batch_size=4)

        assert emb.shape == (4, 64)

    def test_embedding_is_fixed(self):
        """Same voice should return same embedding."""
        voice = VoiceIdentity(dim=64)
        emb1 = voice.get_embedding(batch_size=1)
        emb2 = voice.get_embedding(batch_size=1)

        assert torch.allclose(emb1, emb2)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sparc_integration.py -v`
Expected: FAIL with "No module named 'primordial.speech.sparc_integration'"

**Step 3: Write minimal implementation**

Create `primordial/speech/sparc_integration.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sparc_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/sparc_integration.py tests/speech/test_sparc_integration.py
git commit -m "feat(speech): add SPARCWrapper and VoiceIdentity classes"
```

---

### Task 3: Create ArticulatoryHead Module

**Files:**
- Modify: `primordial/speech/heads.py`
- Test: `tests/speech/test_heads.py` (create or modify)

**Step 1: Write the failing test**

Create `tests/speech/test_heads.py`:

```python
"""Tests for speech output heads."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.heads import ArticulatoryHead


class TestArticulatoryHead:
    """Tests for SPARC articulatory output head."""

    @pytest.fixture
    def config(self):
        return SpeechConfig(hidden_dim=128, audio_duration=1.0)

    @pytest.fixture
    def head(self, config):
        return ArticulatoryHead(config, input_dim=384)

    def test_output_shapes(self, head, config):
        """Should output correct SPARC feature shapes."""
        batch_size = 4
        pooled = torch.randn(batch_size, 384)

        output = head(pooled)

        n_frames = config.sparc_n_frames
        assert output['ema'].shape == (batch_size, n_frames, 12)
        assert output['pitch'].shape == (batch_size, n_frames, 1)
        assert output['loudness'].shape == (batch_size, n_frames, 1)

    def test_ema_range(self, head):
        """EMA should be in reasonable range (tanh bounded)."""
        pooled = torch.randn(4, 384)
        output = head(pooled)

        # Tanh output is in [-1, 1]
        assert output['ema'].min() >= -1.0
        assert output['ema'].max() <= 1.0

    def test_pitch_positive(self, head):
        """Pitch should be positive Hz values."""
        pooled = torch.randn(4, 384)
        output = head(pooled)

        # Pitch should be positive (softplus + offset)
        assert output['pitch'].min() > 0

    def test_loudness_range(self, head):
        """Loudness should be in [0, 1]."""
        pooled = torch.randn(4, 384)
        output = head(pooled)

        assert output['loudness'].min() >= 0
        assert output['loudness'].max() <= 1

    def test_differentiable(self, head):
        """Head should support gradient flow."""
        pooled = torch.randn(4, 384, requires_grad=True)
        output = head(pooled)

        loss = output['ema'].sum() + output['pitch'].sum() + output['loudness'].sum()
        loss.backward()

        assert pooled.grad is not None

    def test_from_sequence(self, config):
        """Should work with sequence input (no pooling)."""
        head = ArticulatoryHead(config, input_dim=128, from_sequence=True)

        # Sequence input: (batch, seq_len, hidden_dim)
        seq = torch.randn(4, 100, 128)
        output = head(seq)

        n_frames = config.sparc_n_frames
        assert output['ema'].shape == (4, n_frames, 12)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_heads.py::TestArticulatoryHead -v`
Expected: FAIL with "cannot import name 'ArticulatoryHead'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/heads.py` after `AudioReconstructionHead` class:

```python
class ArticulatoryHead(nn.Module):
    """Articulatory output head for SPARC control.

    Takes pooled or sequence features and outputs:
    - ema: (batch, n_frames, 12) - articulator positions (tongue, lips, jaw)
    - pitch: (batch, n_frames, 1) - F0 in Hz
    - loudness: (batch, n_frames, 1) - energy envelope

    The temporal dimension is produced via a small decoder that upsamples
    from the input to the target SPARC frame rate (50Hz).

    Args:
        config: SpeechConfig with SPARC settings
        input_dim: Dimension of input features (384 for pooled, 128 for sequence)
        from_sequence: If True, input is (batch, seq_len, hidden_dim) not pooled
    """

    def __init__(
        self,
        config: SpeechConfig,
        input_dim: int = 384,
        from_sequence: bool = False,
    ):
        super().__init__()
        self.config = config
        self.from_sequence = from_sequence
        self.n_frames = config.sparc_n_frames

        # Shared feature extraction
        if from_sequence:
            # Process sequence with 1D conv for temporal modeling
            self.temporal = nn.Sequential(
                nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.GELU(),
            )
            feature_dim = 256
        else:
            # Upsample from pooled to sequence
            self.upsample = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.GELU(),
                nn.Linear(512, self.n_frames * 64),
            )
            # Temporal refinement
            self.temporal = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(128, 256, kernel_size=5, padding=2),
                nn.GELU(),
            )
            feature_dim = 256

        # EMA head: 12 articulator positions, bounded by tanh
        self.ema_head = nn.Sequential(
            nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 12, kernel_size=1),
            nn.Tanh(),  # Bound to [-1, 1]
        )

        # Pitch head: positive Hz via softplus
        self.pitch_head = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.pitch_offset = 100.0  # Minimum pitch in Hz
        self.pitch_scale = 300.0   # Range above offset

        # Loudness head: [0, 1] via sigmoid
        self.loudness_head = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, input_dim) pooled features OR
               (batch, seq_len, hidden_dim) sequence features

        Returns:
            dict with:
                ema: (batch, n_frames, 12) articulator positions
                pitch: (batch, n_frames, 1) F0 in Hz
                loudness: (batch, n_frames, 1) energy
        """
        if self.from_sequence:
            # Input: (batch, seq_len, hidden_dim)
            # Conv expects: (batch, channels, length)
            x = x.transpose(1, 2)
            features = self.temporal(x)
            # Interpolate to target frame count
            features = F.interpolate(
                features,
                size=self.n_frames,
                mode='linear',
                align_corners=False
            )
        else:
            # Input: (batch, input_dim) pooled
            batch_size = x.shape[0]
            # Upsample to sequence
            x = self.upsample(x)
            x = x.view(batch_size, 64, self.n_frames)
            features = self.temporal(x)

        # features: (batch, 256, n_frames)

        # EMA: articulator positions
        ema = self.ema_head(features)  # (batch, 12, n_frames)
        ema = ema.transpose(1, 2)  # (batch, n_frames, 12)

        # Pitch: F0 in Hz
        pitch = self.pitch_head(features)  # (batch, 1, n_frames)
        pitch = F.softplus(pitch) * self.pitch_scale + self.pitch_offset
        pitch = pitch.transpose(1, 2)  # (batch, n_frames, 1)

        # Loudness: energy envelope
        loudness = self.loudness_head(features)  # (batch, 1, n_frames)
        loudness = loudness.transpose(1, 2)  # (batch, n_frames, 1)

        return {
            'ema': ema,
            'pitch': pitch,
            'loudness': loudness,
        }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_heads.py::TestArticulatoryHead -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/heads.py tests/speech/test_heads.py
git commit -m "feat(speech): add ArticulatoryHead for SPARC control"
```

---

### Task 4: Export New Classes from Speech Module

**Files:**
- Modify: `primordial/speech/__init__.py`

**Step 1: Write the failing test**

Add to `tests/speech/test_sparc_integration.py`:

```python
def test_public_imports():
    """New classes should be importable from speech module."""
    from primordial.speech import (
        SPARCWrapper,
        VoiceIdentity,
        ArticulatoryHead,
    )

    assert SPARCWrapper is not None
    assert VoiceIdentity is not None
    assert ArticulatoryHead is not None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sparc_integration.py::test_public_imports -v`
Expected: FAIL with "cannot import name 'SPARCWrapper'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/__init__.py`:

```python
from .sparc_integration import SPARCWrapper, VoiceIdentity
from .heads import ArticulatoryHead
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sparc_integration.py::test_public_imports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/__init__.py
git commit -m "feat(speech): export SPARC integration classes"
```

---

## Phase 1.5: Validation (SPARC Roundtrip Quality)

### Task 5: Create SPARC Roundtrip Validation Script

**Files:**
- Create: `primordial/scripts/validate_sparc.py`

**Purpose:** Before training, verify SPARC encode/decode quality on sample audio. This ensures the frozen decoder produces acceptable output from encoded features.

**Step 1: Create validation script**

Create `primordial/scripts/validate_sparc.py`:

```python
#!/usr/bin/env python3
"""Validate SPARC roundtrip quality before training.

Tests:
1. Audio -> SPARC encode -> decode -> compare to original
2. Measures reconstruction quality (mel MSE, PESQ if available)
3. Saves sample audio for manual listening

Usage:
    python -m primordial.scripts.validate_sparc --audio sample.wav
    python -m primordial.scripts.validate_sparc --librispeech /path/to/data --n-samples 10
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import soundfile as sf

from primordial.speech import SpeechConfig, SPARCWrapper, VoiceIdentity
from primordial.speech.encoders import compute_mel_spectrogram


def compute_mel_mse(audio1: torch.Tensor, audio2: torch.Tensor, config: SpeechConfig) -> float:
    """Compute MSE between mel spectrograms of two audio signals."""
    mel1 = compute_mel_spectrogram(audio1.unsqueeze(0), config)
    mel2 = compute_mel_spectrogram(audio2.unsqueeze(0), config)
    return torch.nn.functional.mse_loss(mel1, mel2).item()


def validate_single(
    audio_path: str,
    config: SpeechConfig,
    sparc: SPARCWrapper,
    voice: VoiceIdentity,
    output_dir: Optional[Path] = None,
) -> dict:
    """Validate SPARC roundtrip on a single audio file."""
    # Load audio
    audio_np, sr = sf.read(audio_path)
    if sr != config.sample_rate:
        # Simple resampling
        import scipy.signal
        audio_np = scipy.signal.resample(
            audio_np,
            int(len(audio_np) * config.sample_rate / sr)
        )

    audio = torch.from_numpy(audio_np).float()

    # Truncate/pad to expected duration
    expected_len = int(config.sample_rate * config.audio_duration)
    if len(audio) > expected_len:
        audio = audio[:expected_len]
    elif len(audio) < expected_len:
        audio = torch.nn.functional.pad(audio, (0, expected_len - len(audio)))

    # Encode
    ema, pitch, loudness = sparc.encode(audio.unsqueeze(0))

    # Decode
    spk_emb = voice.get_embedding(batch_size=1, device=audio.device)
    reconstructed = sparc.decode(ema, pitch, loudness, spk_emb)

    # Compute metrics
    mel_mse = compute_mel_mse(audio, reconstructed.squeeze(0), config)

    # Save outputs if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_path).stem

        sf.write(
            output_dir / f"{stem}_original.wav",
            audio.numpy(),
            config.sample_rate,
        )
        sf.write(
            output_dir / f"{stem}_reconstructed.wav",
            reconstructed.squeeze(0).detach().numpy(),
            config.sample_rate,
        )

        # Save features for inspection
        np.savez(
            output_dir / f"{stem}_features.npz",
            ema=ema.detach().numpy(),
            pitch=pitch.detach().numpy(),
            loudness=loudness.detach().numpy(),
        )

    return {
        'mel_mse': mel_mse,
        'ema_mean': ema.abs().mean().item(),
        'ema_std': ema.std().item(),
        'pitch_mean': pitch.mean().item(),
        'pitch_std': pitch.std().item(),
        'loudness_mean': loudness.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SPARC roundtrip quality")
    parser.add_argument("--audio", type=str, help="Single audio file to test")
    parser.add_argument("--librispeech", type=str, help="LibriSpeech root for batch testing")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples for batch")
    parser.add_argument("--output-dir", type=str, default="sparc_validation", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock SPARC (for testing)")
    args = parser.parse_args()

    config = SpeechConfig()
    sparc = SPARCWrapper(config, mock=args.mock)
    voice = VoiceIdentity()
    output_dir = Path(args.output_dir)

    if args.audio:
        results = validate_single(args.audio, config, sparc, voice, output_dir)
        print(f"\nValidation results for {args.audio}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    elif args.librispeech:
        from primordial.speech import LibriSpeechDataset

        # Find audio files
        dataset = LibriSpeechDataset(
            args.librispeech,
            split="dev-clean",
            config=config,
            max_duration=config.audio_duration,
        )

        all_results = []
        for i in range(min(args.n_samples, len(dataset))):
            # Get audio path from dataset
            audio_path = dataset.samples[i]['audio_path']
            results = validate_single(
                audio_path, config, sparc, voice,
                output_dir / f"sample_{i}"
            )
            all_results.append(results)
            print(f"Sample {i}: mel_mse={results['mel_mse']:.4f}")

        # Aggregate
        print("\nAggregate results:")
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            print(f"  {key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Test the script runs**

Run: `python -m primordial.scripts.validate_sparc --mock --audio assets/sample.wav`
(Or create a simple test audio file)

**Step 3: Commit**

```bash
git add primordial/scripts/validate_sparc.py
git commit -m "feat(speech): add SPARC roundtrip validation script"
```

---

## Phase 2: Data Preprocessing

### Task 6: Create SPARC Feature Dataset Class

**Files:**
- Create: `primordial/speech/sparc_dataset.py`
- Test: `tests/speech/test_sparc_dataset.py`

**Step 1: Write the failing test**

Create `tests/speech/test_sparc_dataset.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sparc_dataset.py -v`
Expected: FAIL with "No module named 'primordial.speech.sparc_dataset'"

**Step 3: Write minimal implementation**

Create `primordial/speech/sparc_dataset.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sparc_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/sparc_dataset.py tests/speech/test_sparc_dataset.py
git commit -m "feat(speech): add SPARCDataset for preprocessed features"
```

---

### Task 7: Create LibriSpeech Preprocessing Script

**Files:**
- Create: `primordial/scripts/preprocess_sparc.py`

**Step 1: Create preprocessing script**

Create `primordial/scripts/preprocess_sparc.py`:

```python
#!/usr/bin/env python3
"""Preprocess LibriSpeech with SPARC encoder for training.

Creates HDF5 files with:
- mel: Input mel spectrograms
- ema: Target EMA articulator positions (12D)
- pitch: Target F0 (1D)
- loudness: Target energy (1D)

Usage:
    python -m primordial.scripts.preprocess_sparc \
        --librispeech /path/to/LibriSpeech \
        --output data/sparc_features \
        --splits train-clean-100 dev-clean \
        --duration 2.0
"""
import argparse
from pathlib import Path
import numpy as np

from primordial.speech import SpeechConfig, SPARCWrapper
from primordial.speech.sparc_dataset import preprocess_to_hdf5
from primordial.speech.librispeech_dataset import LibriSpeechDataset


def find_audio_files(librispeech_root: Path, split: str) -> list:
    """Find all audio files in a LibriSpeech split.

    Returns:
        List of (audio_path, speaker_id) tuples
    """
    split_dir = librispeech_root / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    audio_files = []
    for trans_file in split_dir.rglob("*.trans.txt"):
        chapter_dir = trans_file.parent
        speaker_id = chapter_dir.parent.name

        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) >= 1:
                    utterance_id = parts[0]
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if audio_path.exists():
                        audio_files.append((str(audio_path), speaker_id))

    return audio_files


def filter_by_duration(
    audio_files: list,
    min_duration: float,
    max_duration: float,
    sample_rate: int = 16000,
) -> list:
    """Filter audio files by duration."""
    import soundfile as sf

    filtered = []
    for audio_path, speaker_id in audio_files:
        info = sf.info(audio_path)
        duration = info.duration
        if min_duration <= duration <= max_duration:
            filtered.append((audio_path, speaker_id))

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Preprocess LibriSpeech with SPARC")
    parser.add_argument("--librispeech", type=str, required=True, help="LibriSpeech root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train-clean-100"], help="Splits to process")
    parser.add_argument("--duration", type=float, default=2.0, help="Audio duration in seconds")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum duration")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--mock", action="store_true", help="Use mock SPARC (for testing)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    args = parser.parse_args()

    librispeech_root = Path(args.librispeech)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SpeechConfig(audio_duration=args.duration)
    sparc = SPARCWrapper(config, mock=args.mock)

    for split in args.splits:
        print(f"\nProcessing {split}...")

        # Find audio files
        audio_files = find_audio_files(librispeech_root, split)
        print(f"  Found {len(audio_files)} utterances")

        # Filter by duration
        audio_files = filter_by_duration(
            audio_files,
            args.min_duration,
            args.duration,
            config.sample_rate,
        )
        print(f"  {len(audio_files)} after duration filter")

        # Limit samples if requested
        if args.max_samples:
            audio_files = audio_files[:args.max_samples]
            print(f"  Limited to {len(audio_files)} samples")

        # Preprocess
        output_path = output_dir / f"{split}.h5"
        preprocess_to_hdf5(
            audio_files,
            str(output_path),
            config,
            sparc,
            batch_size=args.batch_size,
        )
        print(f"  Saved to {output_path}")

        # Print storage info
        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
```

**Step 2: Test the script runs**

Run: `python -m primordial.scripts.preprocess_sparc --help`
Expected: Shows help message

**Step 3: Commit**

```bash
git add primordial/scripts/preprocess_sparc.py
git commit -m "feat(speech): add LibriSpeech SPARC preprocessing script"
```

---

## Phase 2: Supervised Training

### Task 8: Create SPARC Loss Functions

**Files:**
- Create: `primordial/speech/sparc_losses.py`
- Test: `tests/speech/test_sparc_losses.py`

**Step 1: Write the failing test**

Create `tests/speech/test_sparc_losses.py`:

```python
"""Tests for SPARC loss functions."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.sparc_losses import (
    ema_loss,
    pitch_loss,
    loudness_loss,
    smoothness_loss,
    sparc_combined_loss,
)


class TestSPARCLosses:
    """Tests for SPARC training losses."""

    @pytest.fixture
    def config(self):
        return SpeechConfig()

    def test_ema_loss_perfect(self):
        """EMA loss should be 0 for identical inputs."""
        pred = torch.randn(4, 50, 12)
        loss = ema_loss(pred, pred)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_ema_loss_positive(self):
        """EMA loss should be positive for different inputs."""
        pred = torch.randn(4, 50, 12)
        target = torch.randn(4, 50, 12)
        loss = ema_loss(pred, target)
        assert loss.item() > 0

    def test_pitch_loss_handles_unvoiced(self):
        """Pitch loss should mask unvoiced frames (pitch < threshold)."""
        pred = torch.ones(4, 50, 1) * 150  # All voiced
        target = torch.ones(4, 50, 1) * 150
        target[:, 25:, :] = 0  # Half unvoiced

        loss = pitch_loss(pred, target, unvoiced_threshold=50.0)
        # Should only compute loss on voiced frames
        assert loss.item() >= 0

    def test_smoothness_loss_penalizes_jitter(self):
        """Smoothness loss should penalize rapid changes."""
        # Smooth trajectory
        t = torch.linspace(0, 1, 50).unsqueeze(0).unsqueeze(-1)
        smooth = torch.sin(2 * 3.14159 * t).expand(4, 50, 12)

        # Jittery trajectory
        jittery = torch.randn(4, 50, 12)

        smooth_loss = smoothness_loss(smooth)
        jittery_loss = smoothness_loss(jittery)

        assert jittery_loss > smooth_loss

    def test_combined_loss_structure(self, config):
        """Combined loss should return dict with components."""
        pred = {
            'ema': torch.randn(4, 50, 12),
            'pitch': torch.abs(torch.randn(4, 50, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, 50, 1)),
        }
        target = {
            'ema': torch.randn(4, 50, 12),
            'pitch': torch.abs(torch.randn(4, 50, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, 50, 1)),
        }

        losses = sparc_combined_loss(pred, target, config)

        assert 'total' in losses
        assert 'ema' in losses
        assert 'pitch' in losses
        assert 'loudness' in losses
        assert 'smoothness' in losses
        assert losses['total'] > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sparc_losses.py -v`
Expected: FAIL with "No module named 'primordial.speech.sparc_losses'"

**Step 3: Write minimal implementation**

Create `primordial/speech/sparc_losses.py`:

```python
"""Loss functions for SPARC articulatory training.

Losses:
- EMA MSE: Articulator position accuracy (most important)
- Pitch MSE: Prosody melody matching (with unvoiced masking)
- Loudness MSE: Energy envelope matching
- Smoothness: Temporal regularization to prevent jitter
"""
import torch
import torch.nn.functional as F
from typing import Dict

from .config import SpeechConfig


def ema_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """MSE loss on EMA articulator positions.

    Args:
        pred: (batch, n_frames, 12) predicted EMA
        target: (batch, n_frames, 12) target EMA
        mask: (batch, n_frames) optional frame mask

    Returns:
        Scalar loss
    """
    if mask is not None:
        # Expand mask to match EMA dimensions
        mask = mask.unsqueeze(-1).expand_as(pred)
        diff = (pred - target) ** 2
        return (diff * mask).sum() / mask.sum().clamp(min=1)
    else:
        return F.mse_loss(pred, target)


def pitch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    unvoiced_threshold: float = 50.0,
) -> torch.Tensor:
    """MSE loss on pitch (F0), masking unvoiced frames.

    Args:
        pred: (batch, n_frames, 1) predicted pitch in Hz
        target: (batch, n_frames, 1) target pitch in Hz
        unvoiced_threshold: Frames with target < this are considered unvoiced

    Returns:
        Scalar loss
    """
    # Create mask for voiced frames
    voiced_mask = target > unvoiced_threshold

    if voiced_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Compute MSE only on voiced frames
    diff = (pred - target) ** 2
    return (diff * voiced_mask).sum() / voiced_mask.sum()


def loudness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on loudness envelope.

    Args:
        pred: (batch, n_frames, 1) predicted loudness in [0, 1]
        target: (batch, n_frames, 1) target loudness in [0, 1]

    Returns:
        Scalar loss
    """
    return F.mse_loss(pred, target)


def smoothness_loss(
    trajectory: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Temporal smoothness regularization to prevent jittery output.

    Penalizes rapid frame-to-frame changes in the trajectory.

    Args:
        trajectory: (batch, n_frames, features) any temporal sequence
        order: 1 for velocity penalty, 2 for acceleration penalty

    Returns:
        Scalar loss
    """
    if order == 1:
        # First-order: penalize velocity (frame differences)
        diff = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        return (diff ** 2).mean()
    elif order == 2:
        # Second-order: penalize acceleration (difference of differences)
        diff1 = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
        return (diff2 ** 2).mean()
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


def sparc_combined_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    config: SpeechConfig,
) -> Dict[str, torch.Tensor]:
    """Combined SPARC loss with all components.

    Args:
        pred: Dict with 'ema', 'pitch', 'loudness' predictions
        target: Dict with 'ema', 'pitch', 'loudness' targets
        config: SpeechConfig with loss weights

    Returns:
        Dict with 'total', 'ema', 'pitch', 'loudness', 'smoothness' losses
    """
    # Individual losses
    loss_ema = ema_loss(pred['ema'], target['ema'])
    loss_pitch = pitch_loss(pred['pitch'], target['pitch'])
    loss_loudness = loudness_loss(pred['loudness'], target['loudness'])
    loss_smooth = smoothness_loss(pred['ema'])

    # Weighted combination
    total = (
        config.ema_loss_weight * loss_ema +
        config.sparc_pitch_loss_weight * loss_pitch +
        config.sparc_loudness_loss_weight * loss_loudness +
        config.smoothness_loss_weight * loss_smooth
    )

    return {
        'total': total,
        'ema': loss_ema,
        'pitch': loss_pitch,
        'loudness': loss_loudness,
        'smoothness': loss_smooth,
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sparc_losses.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/sparc_losses.py tests/speech/test_sparc_losses.py
git commit -m "feat(speech): add SPARC loss functions with smoothness regularization"
```

---

### Task 9: Create SPARCTrainer Class

**Files:**
- Modify: `primordial/speech/training.py`
- Test: `tests/speech/test_sparc_trainer.py`

**Step 1: Write the failing test**

Create `tests/speech/test_sparc_trainer.py`:

```python
"""Tests for SPARC trainer."""
import torch
import pytest
from primordial.speech.config import SpeechConfig
from primordial.speech.training import SpeechLRN, SPARCTrainer


class TestSPARCTrainer:
    """Tests for SPARC supervised training."""

    @pytest.fixture
    def config(self):
        return SpeechConfig(encoder_type='cnn', audio_duration=1.0)

    @pytest.fixture
    def model(self, config):
        return SpeechLRN(config, output_head='articulatory')

    @pytest.fixture
    def trainer(self, model, config):
        return SPARCTrainer(model, config, lr=1e-3)

    def test_train_step_returns_losses(self, trainer, config):
        """Train step should return loss dictionary."""
        batch_size = 4
        mel = torch.randn(batch_size, config.n_mels, config.n_frames)
        target = {
            'ema': torch.randn(batch_size, config.sparc_n_frames, 12),
            'pitch': torch.abs(torch.randn(batch_size, config.sparc_n_frames, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(batch_size, config.sparc_n_frames, 1)),
        }

        losses = trainer.train_step(mel, target)

        assert 'total' in losses
        assert 'ema' in losses
        assert losses['total'] > 0

    def test_train_step_updates_weights(self, trainer, config):
        """Train step should update model weights."""
        mel = torch.randn(4, config.n_mels, config.n_frames)
        target = {
            'ema': torch.randn(4, config.sparc_n_frames, 12),
            'pitch': torch.abs(torch.randn(4, config.sparc_n_frames, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, config.sparc_n_frames, 1)),
        }

        # Get initial weights
        initial_weights = trainer.model.articulatory_head.ema_head[0].weight.clone()

        # Train step
        trainer.train_step(mel, target)

        # Check weights changed
        final_weights = trainer.model.articulatory_head.ema_head[0].weight
        assert not torch.allclose(initial_weights, final_weights)

    def test_validation_step(self, trainer, config):
        """Validation step should not update weights."""
        mel = torch.randn(4, config.n_mels, config.n_frames)
        target = {
            'ema': torch.randn(4, config.sparc_n_frames, 12),
            'pitch': torch.abs(torch.randn(4, config.sparc_n_frames, 1)) * 200 + 100,
            'loudness': torch.sigmoid(torch.randn(4, config.sparc_n_frames, 1)),
        }

        initial_weights = trainer.model.articulatory_head.ema_head[0].weight.clone()

        losses = trainer.validation_step(mel, target)

        final_weights = trainer.model.articulatory_head.ema_head[0].weight
        assert torch.allclose(initial_weights, final_weights)
        assert 'total' in losses
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sparc_trainer.py -v`
Expected: FAIL with "cannot import name 'SPARCTrainer'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/training.py` after the existing trainer classes:

```python
from .heads import ArticulatoryHead
from .sparc_losses import sparc_combined_loss


class SPARCTrainer:
    """Trainer for SPARC articulatory control.

    Trains model to predict EMA/pitch/loudness from mel spectrograms.
    Uses pre-computed SPARC targets for supervised learning.

    Args:
        model: SpeechLRN with articulatory head
        config: SpeechConfig
        lr: Learning rate
        device: Torch device
    """

    def __init__(
        self,
        model: 'SpeechLRN',
        config: SpeechConfig,
        lr: float = 1e-4,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.1
        )

    def train_step(
        self,
        mel: torch.Tensor,
        target: dict,
    ) -> dict:
        """Single training step.

        Args:
            mel: (batch, n_mels, mel_frames) input mel spectrogram
            target: Dict with 'ema', 'pitch', 'loudness' targets

        Returns:
            Dict of loss values
        """
        self.model.train()

        mel = mel.to(self.device)
        target = {k: v.to(self.device) for k, v in target.items()}

        # Forward pass
        pred = self.model.forward_articulatory(mel)

        # Compute losses
        losses = sparc_combined_loss(pred, target, self.config)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def validation_step(
        self,
        mel: torch.Tensor,
        target: dict,
    ) -> dict:
        """Validation step (no gradient update).

        Args:
            mel: (batch, n_mels, mel_frames) input mel spectrogram
            target: Dict with 'ema', 'pitch', 'loudness' targets

        Returns:
            Dict of loss values
        """
        self.model.eval()

        mel = mel.to(self.device)
        target = {k: v.to(self.device) for k, v in target.items()}

        with torch.no_grad():
            pred = self.model.forward_articulatory(mel)
            losses = sparc_combined_loss(pred, target, self.config)

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(
        self,
        dataloader: 'DataLoader',
        epoch: int = 0,
    ) -> dict:
        """Train for one epoch.

        Args:
            dataloader: DataLoader yielding (mel, ema, pitch, loudness)
            epoch: Current epoch number

        Returns:
            Dict of average losses
        """
        total_losses = {}
        n_batches = 0

        for mel, ema, pitch, loudness in dataloader:
            target = {
                'ema': ema,
                'pitch': pitch,
                'loudness': loudness,
            }

            losses = self.train_step(mel, target)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v
            n_batches += 1

        # Update scheduler
        self.scheduler.step()

        return {k: v / n_batches for k, v in total_losses.items()}

    def save_checkpoint(self, path: str, epoch: int = 0) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0)
```

Also add to `SpeechLRN` class a method for articulatory forward pass:

```python
def forward_articulatory(self, mel: torch.Tensor) -> dict:
    """Forward pass for articulatory output.

    Args:
        mel: (batch, n_mels, n_frames) mel spectrogram

    Returns:
        Dict with 'ema', 'pitch', 'loudness' predictions
    """
    # Encode and mix
    encoded = self.audio_encoder(mel)
    mixed = self.lrn_layers(encoded)

    # Pool: (mean, max, last) concatenation
    pooled = torch.cat([
        mixed.mean(dim=1),
        mixed.max(dim=1).values,
        mixed[:, -1, :],
    ], dim=-1)

    # Articulatory head
    return self.articulatory_head(pooled)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sparc_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/training.py tests/speech/test_sparc_trainer.py
git commit -m "feat(speech): add SPARCTrainer for supervised articulatory training"
```

---

### Task 10: Create SPARC Training Script

**Files:**
- Create: `primordial/scripts/train_sparc.py`

**Step 1: Create training script**

Create `primordial/scripts/train_sparc.py`:

```python
#!/usr/bin/env python3
"""Train model to predict SPARC articulatory features.

Phase 2 of SPARC integration: supervised training on pre-computed targets.

Usage:
    python -m primordial.scripts.train_sparc \
        --data data/sparc_features/train-clean-100.h5 \
        --val-data data/sparc_features/dev-clean.h5 \
        --epochs 50 \
        --batch-size 32
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from primordial.speech import SpeechConfig, SpeechLRN, SPARCTrainer
from primordial.speech.sparc_dataset import SPARCDataset


def main():
    parser = argparse.ArgumentParser(description="Train SPARC articulatory model")
    parser.add_argument("--data", type=str, required=True, help="Training HDF5 file")
    parser.add_argument("--val-data", type=str, default=None, help="Validation HDF5 file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/sparc", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--encoder", type=str, default="cnn", choices=["linear", "cnn"])
    parser.add_argument("--duration", type=float, default=2.0, help="Audio duration")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Config
    config = SpeechConfig(
        encoder_type=args.encoder,
        audio_duration=args.duration,
    )

    # Data
    train_dataset = SPARCDataset(args.data, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data:
        val_dataset = SPARCDataset(args.val_data, config)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

    # Model
    model = SpeechLRN(config, output_head='articulatory')
    trainer = SPARCTrainer(model, config, lr=args.lr, device=device)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed from epoch {start_epoch}")

    print(f"Training on {len(train_dataset)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)

        log = f"Epoch {epoch+1}/{args.epochs}"
        log += f" | train_loss={train_losses['total']:.4f}"
        log += f" | ema={train_losses['ema']:.4f}"
        log += f" | pitch={train_losses['pitch']:.4f}"

        # Validate
        if val_loader:
            val_losses = {}
            for mel, ema, pitch, loudness in val_loader:
                target = {'ema': ema, 'pitch': pitch, 'loudness': loudness}
                batch_losses = trainer.validation_step(mel, target)
                for k, v in batch_losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v

            n_val = len(val_loader)
            val_losses = {k: v / n_val for k, v in val_losses.items()}
            log += f" | val_loss={val_losses['total']:.4f}"

            # Save best
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                trainer.save_checkpoint(output_dir / "best.pt", epoch)

        print(log)

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(output_dir / f"epoch_{epoch+1}.pt", epoch)

    # Save final
    trainer.save_checkpoint(output_dir / "final.pt", args.epochs - 1)
    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Test the script runs**

Run: `python -m primordial.scripts.train_sparc --help`
Expected: Shows help message

**Step 3: Commit**

```bash
git add primordial/scripts/train_sparc.py
git commit -m "feat(speech): add SPARC supervised training script"
```

---

## Summary

This plan implements **Phase 1** (Foundation) and **Phase 2** (Supervised Training) of the SPARC integration:

| Task | Component | Status |
|------|-----------|--------|
| 1 | SpeechConfig SPARC parameters | New |
| 2 | SPARCWrapper + VoiceIdentity | New |
| 3 | ArticulatoryHead module | New |
| 4 | Module exports | Modified |
| 5 | SPARC validation script | New |
| 6 | SPARCDataset (HDF5) | New |
| 7 | Preprocessing script | New |
| 8 | SPARC loss functions | New |
| 9 | SPARCTrainer | New |
| 10 | Training script | New |

**Key improvements incorporated:**
- Smoothness regularization to prevent jittery output
- Unvoiced masking in pitch loss
- Frame rate alignment handled in ArticulatoryHead
- Validation script for SPARC roundtrip quality check
- Mock mode for testing without SPARC installed

**Next phases (not in this plan):**
- Phase 3: Self-listening with end-to-end audio loss
- Phase 4: RL exploration with phoneme curriculum
- Phase 5: Multimodal integration
