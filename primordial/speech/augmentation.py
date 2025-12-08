"""Audio augmentation utilities for robust speech recognition training.

Applies random transformations to audio to improve speaker invariance:
- Pitch shifting (simulate different vocal ranges)
- Time stretching (simulate different speaking rates)
- Noise injection (robustness to background noise)
- Volume perturbation (handle varying recording levels)
"""

import torch
import torchaudio
import random
from typing import Optional


def augment_waveform(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    pitch_shift_prob: float = 0.3,
    time_stretch_prob: float = 0.3,
    noise_prob: float = 0.2,
    volume_prob: float = 0.3,
    pitch_range: tuple = (-2, 2),
    stretch_range: tuple = (0.9, 1.1),
    noise_level_range: tuple = (0.001, 0.01),
    volume_range: tuple = (0.7, 1.3),
) -> torch.Tensor:
    """Apply random augmentations to audio waveform.

    Args:
        waveform: (samples,) or (batch, samples) audio tensor
        sample_rate: Sample rate in Hz
        pitch_shift_prob: Probability of applying pitch shift
        time_stretch_prob: Probability of applying time stretch
        noise_prob: Probability of adding noise
        volume_prob: Probability of volume perturbation
        pitch_range: (min, max) semitones for pitch shift
        stretch_range: (min, max) rate for time stretch
        noise_level_range: (min, max) noise amplitude
        volume_range: (min, max) volume multiplier

    Returns:
        Augmented waveform (same shape as input)
    """
    # Ensure 1D
    squeeze_output = waveform.dim() == 1
    if squeeze_output:
        waveform = waveform.unsqueeze(0)

    # Pitch shift (requires torchaudio functional)
    if random.random() < pitch_shift_prob:
        waveform = _pitch_shift(waveform, sample_rate, pitch_range)

    # Time stretch
    if random.random() < time_stretch_prob:
        waveform = _time_stretch(waveform, stretch_range, sample_rate)

    # Add noise
    if random.random() < noise_prob:
        waveform = _add_noise(waveform, noise_level_range)

    # Volume perturbation
    if random.random() < volume_prob:
        waveform = _adjust_volume(waveform, volume_range)

    if squeeze_output:
        waveform = waveform.squeeze(0)

    return waveform


def _pitch_shift(
    waveform: torch.Tensor,
    sample_rate: int,
    pitch_range: tuple,
) -> torch.Tensor:
    """Apply pitch shift to waveform.

    Uses resampling-based pitch shifting with proper sinc interpolation.
    """
    shift_semitones = random.uniform(*pitch_range)

    # Convert semitones to rate
    # Positive shift = faster playback = higher pitch
    rate = 2 ** (shift_semitones / 12)

    if rate == 1.0:
        return waveform

    orig_len = waveform.shape[-1]

    # Use torchaudio for high-quality resampling
    # First resample to shift pitch (this changes duration)
    intermediate_rate = int(sample_rate * rate)
    if intermediate_rate <= 0:
        return waveform

    # Resample down to shift pitch up (or vice versa)
    resampler1 = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=intermediate_rate,
        resampling_method='sinc_interp_hann',
    )

    # Resample back to original sample rate (restores duration approximately)
    resampler2 = torchaudio.transforms.Resample(
        orig_freq=intermediate_rate,
        new_freq=sample_rate,
        resampling_method='sinc_interp_hann',
    )

    # Apply pitch shift
    waveform = resampler1(waveform)
    waveform = resampler2(waveform)

    # Pad or truncate to original length
    current_len = waveform.shape[-1]
    if current_len < orig_len:
        waveform = torch.nn.functional.pad(waveform, (0, orig_len - current_len))
    elif current_len > orig_len:
        waveform = waveform[..., :orig_len]

    return waveform


def _time_stretch(
    waveform: torch.Tensor,
    stretch_range: tuple,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Apply time stretch to waveform.

    Note: This changes both tempo AND pitch slightly. For true time-stretch
    without pitch change, a phase vocoder would be needed.
    Uses proper sinc interpolation for better quality.
    """
    rate = random.uniform(*stretch_range)

    if rate == 1.0:
        return waveform

    orig_len = waveform.shape[-1]

    # Use resampling to stretch/compress
    # This is equivalent to playing back at a different speed
    stretched_rate = int(sample_rate / rate)
    if stretched_rate <= 0:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=stretched_rate,
        resampling_method='sinc_interp_hann',
    )
    waveform = resampler(waveform)

    # Pad or truncate to original length
    current_len = waveform.shape[-1]
    if current_len < orig_len:
        waveform = torch.nn.functional.pad(waveform, (0, orig_len - current_len))
    elif current_len > orig_len:
        waveform = waveform[..., :orig_len]

    return waveform


def _add_noise(
    waveform: torch.Tensor,
    noise_level_range: tuple,
) -> torch.Tensor:
    """Add Gaussian noise to waveform."""
    noise_level = random.uniform(*noise_level_range)
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def _adjust_volume(
    waveform: torch.Tensor,
    volume_range: tuple,
) -> torch.Tensor:
    """Adjust volume by random multiplier."""
    volume_factor = random.uniform(*volume_range)
    return waveform * volume_factor


class AudioAugmentor:
    """Configurable audio augmentor with preset augmentation profiles."""

    def __init__(
        self,
        profile: str = "medium",
        sample_rate: int = 16000,
    ):
        """Initialize augmentor.

        Args:
            profile: Augmentation intensity ("light", "medium", "heavy")
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.profile = profile

        # Configure based on profile
        if profile == "light":
            self.pitch_shift_prob = 0.2
            self.time_stretch_prob = 0.2
            self.noise_prob = 0.1
            self.volume_prob = 0.2
            self.pitch_range = (-1, 1)
            self.stretch_range = (0.95, 1.05)
            self.noise_level_range = (0.0005, 0.005)
            self.volume_range = (0.85, 1.15)
        elif profile == "medium":
            self.pitch_shift_prob = 0.3
            self.time_stretch_prob = 0.3
            self.noise_prob = 0.2
            self.volume_prob = 0.3
            self.pitch_range = (-2, 2)
            self.stretch_range = (0.9, 1.1)
            self.noise_level_range = (0.001, 0.01)
            self.volume_range = (0.7, 1.3)
        elif profile == "heavy":
            self.pitch_shift_prob = 0.5
            self.time_stretch_prob = 0.5
            self.noise_prob = 0.4
            self.volume_prob = 0.5
            self.pitch_range = (-4, 4)
            self.stretch_range = (0.8, 1.2)
            self.noise_level_range = (0.005, 0.02)
            self.volume_range = (0.5, 1.5)
        else:
            raise ValueError(f"Unknown profile: {profile}")

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to waveform."""
        return augment_waveform(
            waveform,
            sample_rate=self.sample_rate,
            pitch_shift_prob=self.pitch_shift_prob,
            time_stretch_prob=self.time_stretch_prob,
            noise_prob=self.noise_prob,
            volume_prob=self.volume_prob,
            pitch_range=self.pitch_range,
            stretch_range=self.stretch_range,
            noise_level_range=self.noise_level_range,
            volume_range=self.volume_range,
        )


class SpecAugment:
    """SpecAugment-style augmentation on mel spectrograms.

    Applies frequency and time masking directly to spectrograms.
    Reference: Park et al., "SpecAugment" (2019)
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        """Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram.

        Args:
            mel: (n_mels, n_frames) mel spectrogram

        Returns:
            Augmented mel spectrogram
        """
        mel = mel.clone()
        n_mels, n_frames = mel.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, n_mels - f)
            mel[f0:f0 + f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t = min(t, n_frames)  # Don't exceed spectrogram length
            t0 = random.randint(0, n_frames - t)
            mel[:, t0:t0 + t] = 0

        return mel
