"""Audio encoders for speech learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import SpeechConfig


class MelSpectrogramEncoder(nn.Module):
    """Encodes mel spectrograms to sequence for Fourier mixing.

    Takes mel spectrogram input and produces a sequence representation
    compatible with the LRN's Fourier mixing layers.

    Input: (batch, n_mels, n_frames) - variable n_frames
    Output: (batch, seq_len, hidden_dim) - fixed (batch, 100, 128)
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config
        self.n_mels = config.n_mels
        self.hidden_dim = config.hidden_dim
        self.output_seq_len = config.encoder_seq_len

        # Simple approach: treat each mel frame as a token
        # Project n_mels -> hidden_dim per frame
        self.frame_projection = nn.Linear(config.n_mels, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, n_frames) mel spectrogram

        Returns:
            (batch, seq_len, hidden_dim) sequence for Fourier mixing
        """
        # x: (batch, n_mels, n_frames) -> (batch, n_frames, n_mels)
        x = x.transpose(1, 2)

        # Project each frame: (batch, n_frames, n_mels) -> (batch, n_frames, hidden_dim)
        x = self.frame_projection(x)

        # Interpolate to fixed sequence length
        # (batch, n_frames, hidden_dim) -> (batch, hidden_dim, n_frames)
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=self.output_seq_len, mode='linear', align_corners=False)
        # (batch, hidden_dim, seq_len) -> (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)

        return x


class SimpleAudioEncoder(nn.Module):
    """Simple linear encoder for raw audio or simple features.

    Alternative to MelSpectrogramEncoder for simpler experiments.
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.output_seq_len = config.encoder_seq_len

        # Direct linear projection from mel bins to hidden dim
        self.projection = nn.Linear(config.n_mels, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, n_frames) mel spectrogram

        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Transpose: (batch, n_mels, n_frames) -> (batch, n_frames, n_mels)
        x = x.transpose(1, 2)

        # Project: (batch, n_frames, n_mels) -> (batch, n_frames, hidden_dim)
        x = self.projection(x)

        # Interpolate to target sequence length if needed
        if x.shape[1] != self.output_seq_len:
            x = x.permute(0, 2, 1)  # (batch, hidden_dim, n_frames)
            x = F.interpolate(x, size=self.output_seq_len, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)

        return x


class CNNMelEncoder(nn.Module):
    """CNN encoder for mel spectrograms.

    Uses 2D convolutions to capture local spectral-temporal patterns:
    - Formant structures (vertical patterns in mel)
    - Temporal transitions (horizontal patterns)
    - Phoneme boundaries (rapid changes)

    This should break the 63% ceiling hit by the linear encoder.
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.output_seq_len = config.encoder_seq_len
        self.n_mels = config.n_mels

        # CNN backbone: extract local spectral-temporal features
        # Input: (batch, 1, n_mels, n_frames) - treat mel as 2D image
        self.conv_layers = nn.Sequential(
            # Layer 1: capture fine-grained patterns
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # Layer 2: capture medium-scale patterns
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Layer 3: capture larger patterns, reduce frequency dimension
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Layer 4: final feature extraction
            nn.Conv2d(128, self.hidden_dim, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
        )

        # After 3 stride-2 layers on freq axis: n_mels / 8 = 80 / 8 = 10
        self.freq_out = self.n_mels // 8

        # Collapse frequency dimension and project to sequence
        self.freq_collapse = nn.Linear(self.freq_out * self.hidden_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, n_frames) mel spectrogram

        Returns:
            (batch, seq_len, hidden_dim) sequence for Fourier mixing
        """
        batch_size = x.shape[0]

        # Add channel dimension: (batch, n_mels, n_frames) -> (batch, 1, n_mels, n_frames)
        x = x.unsqueeze(1)

        # CNN feature extraction: (batch, 1, n_mels, n_frames) -> (batch, hidden_dim, freq_out, n_frames)
        x = self.conv_layers(x)

        # Reshape: (batch, hidden_dim, freq_out, n_frames) -> (batch, n_frames, hidden_dim * freq_out)
        x = x.permute(0, 3, 1, 2)  # (batch, n_frames, hidden_dim, freq_out)
        x = x.reshape(batch_size, x.shape[1], -1)  # (batch, n_frames, hidden_dim * freq_out)

        # Collapse frequency: (batch, n_frames, hidden_dim * freq_out) -> (batch, n_frames, hidden_dim)
        x = self.freq_collapse(x)

        # Interpolate to target sequence length
        if x.shape[1] != self.output_seq_len:
            x = x.permute(0, 2, 1)  # (batch, hidden_dim, n_frames)
            x = F.interpolate(x, size=self.output_seq_len, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)

        return x


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    normalize_audio: bool = True,
    apply_cmvn: bool = True,
) -> torch.Tensor:
    """Compute mel spectrogram from waveform.

    Args:
        waveform: (batch, samples) or (samples,) audio waveform
        sample_rate: Sample rate in Hz
        n_mels: Number of mel bins
        n_fft: FFT window size
        hop_length: Hop between frames
        normalize_audio: If True, normalize waveform to unit variance before processing.
            This helps bridge the gap between loud synthetic and quiet real audio.
        apply_cmvn: If True, apply cepstral mean and variance normalization to the
            output mel spectrogram. This normalizes out recording quality differences.

    Returns:
        (batch, n_mels, n_frames) mel spectrogram in log scale
    """
    # Ensure batch dimension
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    batch_size, num_samples = waveform.shape

    # Normalize audio to zero mean and unit variance
    # This bridges the gap between loud Piper audio and quiet LibriSpeech
    if normalize_audio:
        waveform = waveform - waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True)
        waveform = waveform / (std + 1e-8)

    # Compute STFT
    # Window
    window = torch.hann_window(n_fft, device=waveform.device)

    # STFT: (batch, n_fft//2 + 1, n_frames)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )

    # Power spectrogram
    power = stft.abs() ** 2  # (batch, n_fft//2 + 1, n_frames)

    # Mel filterbank
    mel_fb = _create_mel_filterbank(
        n_mels=n_mels,
        n_fft=n_fft,
        sample_rate=sample_rate,
        device=waveform.device,
    )

    # Apply mel filterbank: (n_mels, n_fft//2 + 1) @ (batch, n_fft//2 + 1, n_frames)
    # -> (batch, n_mels, n_frames)
    mel_spec = torch.matmul(mel_fb, power)

    # Log scale
    mel_spec = torch.log(mel_spec + 1e-9)

    # Apply CMVN (Cepstral Mean and Variance Normalization)
    # This normalizes out speaker/recording differences, forcing model to focus on phonetic content
    if apply_cmvn:
        mel_mean = mel_spec.mean(dim=-1, keepdim=True)
        mel_std = mel_spec.std(dim=-1, keepdim=True)
        mel_spec = (mel_spec - mel_mean) / (mel_std + 1e-8)

    return mel_spec


def _create_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    """Create mel filterbank matrix.

    Returns:
        (n_mels, n_fft//2 + 1) filterbank matrix
    """
    n_freqs = n_fft // 2 + 1

    # Frequency range
    f_min = 0.0
    f_max = sample_rate / 2.0

    # Mel scale conversion
    def hz_to_mel(f):
        return 2595 * math.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    # Mel points
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points], device=device)

    # Bin indices
    bin_indices = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    # Create filterbank
    filterbank = torch.zeros(n_mels, n_freqs, device=device)

    for i in range(n_mels):
        left = bin_indices[i]
        center = bin_indices[i + 1]
        right = bin_indices[i + 2]

        # Rising slope
        for j in range(left, center):
            if j < n_freqs:
                filterbank[i, j] = (j - left) / (center - left + 1e-9)

        # Falling slope
        for j in range(center, right):
            if j < n_freqs:
                filterbank[i, j] = (right - j) / (right - center + 1e-9)

    return filterbank
