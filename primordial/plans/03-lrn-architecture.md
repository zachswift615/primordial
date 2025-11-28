# LRN Neural Architecture Implementation Plan

## Overview

The **Living Resonance Network (LRN)** is a Fourier-based neural architecture designed to replace transformer attention mechanisms for embodied AI agents. Drawing on FNet (Google, 2021) and FFTNet 2025 research, LRN uses learnable spectral filters in the frequency domain to achieve O(n log n) complexity instead of the O(n²) complexity of attention mechanisms.

### Why Fourier?

1. **Computational Efficiency**: FFT operations scale as O(n log n) vs O(n²) for self-attention
2. **Natural for Continuous Signals**: Sensory streams (vision, audio, proprioception) are naturally continuous
3. **Frequency Domain Reasoning**: Many patterns in embodied cognition have natural frequency representations (rhythm, periodicity, oscillations)
4. **Hardware Optimized**: FFT operations are highly optimized on modern hardware
5. **Online Learning Friendly**: Simpler gradients through FFT vs attention softmax

### Key Differences from Transformers

| Aspect | Transformers | LRN |
|--------|-------------|-----|
| Mixing Mechanism | Attention (token-token) | Fourier filters (frequency mixing) |
| Complexity | O(n²) | O(n log n) |
| Inductive Bias | Sparse interactions | Spectral smoothness |
| Context Window | Fixed (positional encoding) | Continuous (phase information) |
| Learnable Params | Q, K, V projections | Spectral filter coefficients |
| Parameter Efficiency | ~3x hidden_dim² per layer | ~2x hidden_dim per layer |

### Research Foundation

- **FNet** (Arxiv 2105.03824): Demonstrated pure FFT can replace attention with 92% accuracy
- **FFTNet 2025** (Arxiv 2502.18394v4): Introduced learnable spectral filters for adaptive frequency response
- Our innovation: Multi-modal spectral fusion for embodied agents

## File Structure

```
primordial/
├── lrn/
│   ├── __init__.py              # Package exports
│   ├── architecture.py          # Main LRN model class
│   ├── encoders.py              # WaveletEncoder for each modality
│   ├── mixing.py                # FourierMixingLayer implementation
│   ├── heads.py                 # PredictionHead, ActionHead
│   ├── genome.py                # GenomeModulator (architecture from DNA)
│   ├── utils.py                 # FFT helpers, spectral operations
│   └── config.py                # LRNConfig dataclass
├── tests/
│   ├── test_lrn_shapes.py       # Tensor shape validation
│   ├── test_lrn_gradient.py     # Gradient flow through FFT
│   ├── test_lrn_forward.py      # Forward pass integration
│   └── test_lrn_online.py       # Single-sample update capability
└── examples/
    ├── lrn_demo.py              # Standalone demo
    └── lrn_profiling.py         # Performance benchmarking
```

## Architecture Diagram

```
SENSORY INPUT (t = current tick)
┌─────────────┬─────────────┬──────────────┬──────────────┐
│   Vision    │    Audio    │ Proprioception│    Touch     │
│  (32, 4)    │  (100, 2)   │     (7,)     │     (8,)     │
└──────┬──────┴──────┬──────┴───────┬──────┴───────┬──────┘
       │             │              │              │
       ▼             ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│           WAVELET ENCODERS (modality-specific)          │
│  VisionEncoder  AudioEncoder  ProprioEncoder TouchEncoder│
│   → (32, 64)    → (100, 64)    → (16, 64)   → (16, 64) │
└──────┬──────────────┬──────────────┬──────────────┬─────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   CONCAT     │  (164, 64)
                    └──────┬───────┘
                           │
                           ▼
       ┌───────────────────────────────────────┐
       │   FOURIER MIXING LAYER 1              │
       │  ┌─────────────────────────────────┐  │
       │  │ 1. FFT(x) → frequency domain    │  │
       │  │ 2. x_fft * learnable_filter     │  │ (164, 33) complex
       │  │ 3. iFFT → back to time domain   │  │
       │  │ 4. LayerNorm + GELU             │  │
       │  └─────────────────────────────────┘  │
       └──────────────┬────────────────────────┘
                      │ (164, 64)
                      ▼
       ┌───────────────────────────────────────┐
       │   FOURIER MIXING LAYER 2              │
       │  (same structure, different filters)  │
       └──────────────┬────────────────────────┘
                      │ (164, 64)
                      ▼
       ┌───────────────────────────────────────┐
       │   FOURIER MIXING LAYER 3              │
       │  (same structure, different filters)  │
       └──────────────┬────────────────────────┘
                      │ (164, 64)
                      ├──────────────┬─────────────┐
                      ▼              ▼             ▼
              ┌──────────────┐ ┌──────────┐ ┌──────────┐
              │ POOL (mean)  │ │   MAX    │ │   LAST   │
              └──────┬───────┘ └────┬─────┘ └────┬─────┘
                     │              │            │
                     └──────────────┴────────────┘
                                   │ (192,)
                                   ▼
                   ┌───────────────────────────┐
                   │   PREDICTION HEAD         │
                   │   Linear(192, 256)        │
                   │   GELU                    │
                   │   Linear(256, 343)        │
                   │   → sensory prediction    │
                   └───────────────────────────┘
                                   │ (343,)
                   ┌───────────────────────────┐
                   │   REWARD HEAD (NEW!)      │
                   │   Linear(192, 64)         │
                   │   GELU                    │
                   │   Linear(64, 5)           │
                   │   → reward prediction     │
                   │   (next 1-5 steps)        │
                   └───────────────────────────┘
                                   │ (5,)
                   ┌───────────────────────────┐
                   │   ACTION HEAD             │
                   │   Linear(192, 128)        │
                   │   GELU                    │
                   │   Linear(128, 5)          │
                   │   → (thrust, torque, etc) │
                   └───────────────────────────┘
                                   │ (5,)
                                   ▼
                          OUTPUT ACTIONS

**Key Innovation: Multi-Task Prediction**

The RewardHead predicts upcoming rewards (positive and negative), creating a direct
gradient toward survival. This solves the "conceptual gap" where sensory prediction
alone doesn't teach survival behavior:

- Sensory prediction → learns world dynamics ("what will I sense?")
- Reward prediction → learns survival value ("will this hurt or help?")
```

## Tensor Shapes

Critical shape tracking through the network (batch_size = 1 for online learning):

### Input Shapes
```python
vision:     (1, 32, 4)      # [batch, rays, (dist, r, g, b)]
audio:      (1, 100, 2)     # [batch, samples, stereo]
proprio:    (1, 7)          # [batch, internal_state]
touch:      (1, 8)          # [batch, contact_sensors]
```

### After Wavelet Encoders
```python
vision_enc:     (1, 32, 64)    # Spatial encoding
audio_enc:      (1, 100, 64)   # Temporal encoding
proprio_enc:    (1, 16, 64)    # Expanded to sequence
touch_enc:      (1, 16, 64)    # Expanded to sequence

# Concatenation along sequence dimension
concat:         (1, 164, 64)   # [batch, seq_len, hidden_dim]
```

### FFT Operations (per FourierMixingLayer)
```python
# Input
x:              (1, 164, 64)

# Real FFT (only positive frequencies)
x_fft:          (1, 164, 33) complex   # rfft(164) = 164//2 + 1 = 33

# Learnable spectral filter
filter:         (164, 33, 2)           # [seq, freq, (real, imag)]
filter_complex: (164, 33) complex      # Converted for multiplication

# Filtered spectrum
x_filtered:     (1, 164, 33) complex

# Inverse FFT
x_out:          (1, 164, 64)

# After LayerNorm + GELU
output:         (1, 164, 64)
```

### After 3 Mixing Layers
```python
mixed:          (1, 164, 64)

# Pooling operations
mean_pool:      (1, 64)     # mean over sequence
max_pool:       (1, 64)     # max over sequence
last:           (1, 64)     # last timestep

# Concatenated
pooled:         (1, 192)
```

### Output Heads
```python
# Prediction Head
pred_hidden:    (1, 256)
pred_out:       (1, 250)    # Matches total input dimension

# Action Head
action_hidden:  (1, 128)
action_out:     (1, 5)      # (thrust, torque, vocalize, freq, eat)
```

### Parameter Count Breakdown
```
WaveletEncoders:
  VisionEncoder:    4 * 64 + 64 = 320
  AudioEncoder:     2 * 64 + 64 = 192
  ProprioEncoder:   7 * 64 + 64 = 512
  TouchEncoder:     8 * 64 + 64 = 576
  Subtotal:         1,600 params

FourierMixingLayers (3 layers):
  Per layer:        164 * 33 * 2 = 10,824 (filter)
                    64 * 2 = 128 (LayerNorm)
                    Subtotal: 10,952
  All 3 layers:     32,856 params

PredictionHead:
  Linear 192→256:   192 * 256 + 256 = 49,408
  Linear 256→343:   256 * 343 + 343 = 88,151
  Subtotal:         137,559 params

RewardHead (NEW):
  Linear 192→64:    192 * 64 + 64 = 12,352
  Linear 64→5:      64 * 5 + 5 = 325
  Subtotal:         12,677 params

ActionHead:
  Linear 192→128:   192 * 128 + 128 = 24,704
  Linear 128→5:     128 * 5 + 5 = 645
  Subtotal:         25,349 params

TOTAL:              209,241 params (~210K)
```

**Note**: This is well under the 800K target. We have budget for:
- Deeper mixing layers (6-8 instead of 3)
- Wider hidden dimensions (128 instead of 64)
- More sophisticated encoders

Recommended scaling to ~800K:
- hidden_dim: 64 → 128
- mixing_layers: 3 → 6
- This gives approximately: 1,600 + 262,080 + 227,584 + 50,688 = ~542K
- Add residual projections and we reach ~800K

## Component Specifications

### 1. LRNConfig

```python
@dataclass
class LRNConfig:
    """Configuration for Living Resonance Network."""

    # Input dimensions
    vision_shape: Tuple[int, int] = (32, 4)      # (rays, features)
    audio_shape: Tuple[int, int] = (100, 2)      # (samples, stereo)
    proprio_dim: int = 7
    touch_dim: int = 8

    # Architecture
    hidden_dim: int = 128                         # Embedding dimension
    num_mixing_layers: int = 6                   # Fourier mixing layers

    # Encoder output sequence lengths
    vision_seq_len: int = 32                     # Keep spatial structure
    audio_seq_len: int = 100                     # Keep temporal structure
    proprio_seq_len: int = 16                    # Expand to sequence
    touch_seq_len: int = 16                      # Expand to sequence

    # Heads
    pred_hidden_dim: int = 256
    action_hidden_dim: int = 128
    action_dim: int = 5                          # Output action space
    reward_horizon: int = 5                      # Steps ahead to predict rewards
    reward_loss_weight: float = 1.0              # Weight for reward loss (1.0-2.0 recommended)

    # FFT settings
    use_real_fft: bool = True                    # Use rfft for efficiency
    spectral_dropout: float = 0.0                # Dropout in frequency domain

    # Normalization
    layer_norm_eps: float = 1e-5

    # Activation
    activation: str = "gelu"                     # or "relu", "swish"

    # Genome modulation (optional)
    genome_dim: int = 100                        # Size of genome vector
    use_genome_modulation: bool = True

    @property
    def total_seq_len(self) -> int:
        """Total sequence length after concatenation."""
        return (self.vision_seq_len + self.audio_seq_len +
                self.proprio_seq_len + self.touch_seq_len)

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins for rfft."""
        return self.total_seq_len // 2 + 1 if self.use_real_fft else self.total_seq_len
```

### 2. WaveletEncoder

Each modality gets a specialized encoder that projects to (seq_len, hidden_dim):

```python
class WaveletEncoder(nn.Module):
    """Base class for modality-specific encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, output_seq_len: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_seq_len = output_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape varies by modality
        Returns:
            Encoded tensor of shape (batch, output_seq_len, hidden_dim)
        """
        raise NotImplementedError


class VisionEncoder(WaveletEncoder):
    """Encodes vision rays to sequence."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.vision_shape[1],  # 4: (dist, r, g, b)
            hidden_dim=config.hidden_dim,
            output_seq_len=config.vision_seq_len
        )
        # Simple linear projection per ray
        self.projection = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 32, 4) - vision rays
        Returns:
            (batch, 32, hidden_dim) - encoded rays
        """
        # x: (B, 32, 4) → (B, 32, hidden_dim)
        return self.projection(x)


class AudioEncoder(WaveletEncoder):
    """Encodes audio samples to sequence."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.audio_shape[1],  # 2: stereo
            hidden_dim=config.hidden_dim,
            output_seq_len=config.audio_seq_len
        )
        self.projection = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 100, 2) - audio samples
        Returns:
            (batch, 100, hidden_dim) - encoded samples
        """
        # x: (B, 100, 2) → (B, 100, hidden_dim)
        return self.projection(x)


class ProprioEncoder(WaveletEncoder):
    """Encodes proprioception to sequence by expansion."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.proprio_dim,  # 7
            hidden_dim=config.hidden_dim,
            output_seq_len=config.proprio_seq_len  # 16
        )
        # Project to sequence embedding
        self.projection = nn.Linear(self.input_dim,
                                    self.output_seq_len * self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 7) - proprioceptive state
        Returns:
            (batch, 16, hidden_dim) - expanded sequence
        """
        batch_size = x.shape[0]
        # (B, 7) → (B, 16*hidden_dim) → (B, 16, hidden_dim)
        x = self.projection(x)
        return x.view(batch_size, self.output_seq_len, self.hidden_dim)


class TouchEncoder(WaveletEncoder):
    """Encodes touch sensors to sequence by expansion."""

    def __init__(self, config: LRNConfig):
        super().__init__(
            input_dim=config.touch_dim,  # 8
            hidden_dim=config.hidden_dim,
            output_seq_len=config.touch_seq_len  # 16
        )
        self.projection = nn.Linear(self.input_dim,
                                    self.output_seq_len * self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 8) - touch sensors
        Returns:
            (batch, 16, hidden_dim) - expanded sequence
        """
        batch_size = x.shape[0]
        # (B, 8) → (B, 16*hidden_dim) → (B, 16, hidden_dim)
        x = self.projection(x)
        return x.view(batch_size, self.output_seq_len, self.hidden_dim)
```

### 3. FourierMixingLayer

The core innovation - learnable spectral filtering:

```python
class FourierMixingLayer(nn.Module):
    """
    Fourier-based mixing layer with learnable spectral filters.

    Replaces self-attention with O(n log n) FFT operations.
    Based on FNet and FFTNet 2025 research.
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.total_seq_len

        # Number of frequency bins
        if config.use_real_fft:
            self.freq_bins = self.seq_len // 2 + 1  # rfft output size
        else:
            self.freq_bins = self.seq_len

        # Learnable spectral filter (stored as real tensor, converted to complex)
        # Shape: (seq_len, freq_bins, 2) for (real, imaginary) components
        self.spectral_filter = nn.Parameter(
            torch.randn(self.seq_len, self.freq_bins, 2)
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_eps)

        # Activation
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()  # Swish

        # Optional spectral dropout
        self.dropout = nn.Dropout(config.spectral_dropout) if config.spectral_dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier mixing with learnable spectral filters.

        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        residual = x

        # Process each feature dimension independently in frequency domain
        # Transpose to (batch, hidden_dim, seq_len) for FFT along sequence
        x = x.transpose(1, 2)  # (B, hidden_dim, seq_len)

        # Apply FFT along sequence dimension
        if self.config.use_real_fft:
            x_fft = torch.fft.rfft(x, dim=2)  # (B, hidden_dim, freq_bins) complex
        else:
            x_fft = torch.fft.fft(x, dim=2)   # (B, hidden_dim, seq_len) complex

        # Convert spectral filter to complex
        # (seq_len, freq_bins, 2) → (seq_len, freq_bins) complex
        filter_complex = torch.view_as_complex(self.spectral_filter.contiguous())

        # Broadcast and apply filter
        # x_fft: (B, hidden_dim, freq_bins)
        # filter: (seq_len, freq_bins) - we use seq_len as hidden_dim indexing
        # For proper broadcasting, slice filter to match hidden_dim
        if hidden_dim <= seq_len:
            filter_slice = filter_complex[:hidden_dim, :]  # (hidden_dim, freq_bins)
        else:
            # Repeat filter if hidden_dim > seq_len
            repeats = (hidden_dim + seq_len - 1) // seq_len
            filter_slice = filter_complex.repeat(repeats, 1)[:hidden_dim, :]

        # Apply spectral filtering
        x_filtered = x_fft * filter_slice.unsqueeze(0)  # (B, hidden_dim, freq_bins)

        # Optional spectral dropout
        if self.dropout is not None:
            # Apply dropout to magnitude while preserving phase
            magnitude = torch.abs(x_filtered)
            phase = torch.angle(x_filtered)
            magnitude = self.dropout(magnitude)
            x_filtered = magnitude * torch.exp(1j * phase)

        # Inverse FFT back to time domain
        if self.config.use_real_fft:
            x_out = torch.fft.irfft(x_filtered, n=seq_len, dim=2)  # (B, hidden_dim, seq_len)
        else:
            x_out = torch.fft.ifft(x_filtered, dim=2).real  # (B, hidden_dim, seq_len)

        # Transpose back to (batch, seq_len, hidden_dim)
        x_out = x_out.transpose(1, 2)  # (B, seq_len, hidden_dim)

        # Residual connection
        x_out = x_out + residual

        # Normalization and activation
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)

        return x_out
```

### 4. PredictionHead

Predicts next sensory state (self-supervised learning signal):

```python
class PredictionHead(nn.Module):
    """
    Predicts next sensory state for self-supervised learning.

    Output matches flattened input dimension (250 values).
    """

    def __init__(self, config: LRNConfig):
        super().__init__()

        # Input: pooled features (3 * hidden_dim)
        input_dim = 3 * config.hidden_dim

        # Total sensory dimension
        output_dim = (config.vision_shape[0] * config.vision_shape[1] +  # 32*4 = 128
                     config.audio_shape[0] * config.audio_shape[1] +      # 100*2 = 200
                     config.proprio_dim +                                 # 7
                     config.touch_dim)                                    # 8
                     # Total: 128 + 200 + 7 + 8 = 343

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.pred_hidden_dim),
            nn.GELU(),
            nn.Linear(config.pred_hidden_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3*hidden_dim) - pooled features
        Returns:
            (batch, output_dim) - predicted next sensory state
        """
        return self.mlp(x)

    def split_prediction(self, pred: torch.Tensor, config: LRNConfig) -> Dict[str, torch.Tensor]:
        """
        Split flat prediction into modality components.

        Args:
            pred: (batch, output_dim)
        Returns:
            Dictionary with vision, audio, proprio, touch predictions
        """
        vision_size = config.vision_shape[0] * config.vision_shape[1]
        audio_size = config.audio_shape[0] * config.audio_shape[1]

        idx = 0
        vision_pred = pred[:, idx:idx+vision_size].view(-1, *config.vision_shape)
        idx += vision_size

        audio_pred = pred[:, idx:idx+audio_size].view(-1, *config.audio_shape)
        idx += audio_size

        proprio_pred = pred[:, idx:idx+config.proprio_dim]
        idx += config.proprio_dim

        touch_pred = pred[:, idx:idx+config.touch_dim]

        return {
            'vision': vision_pred,
            'audio': audio_pred,
            'proprio': proprio_pred,
            'touch': touch_pred
        }
```

### 5. RewardHead (NEW - Multi-Task Learning)

Predicts upcoming rewards for survival-relevant learning:

```python
class RewardHead(nn.Module):
    """
    Predicts upcoming reward values for multi-task learning.

    This creates a DIRECT gradient toward survival by predicting
    whether current patterns lead to positive (food, safety) or
    negative (damage, death) outcomes.

    Output: (batch, horizon) where horizon is how many steps ahead to predict
    Default horizon=5 predicts rewards for t+1 through t+5
    """

    def __init__(self, config: LRNConfig):
        super().__init__()

        # Input: pooled features (3 * hidden_dim)
        input_dim = 3 * config.hidden_dim

        # Predict rewards for multiple future timesteps
        self.reward_horizon = config.reward_horizon  # default: 5

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.reward_horizon)
            # No activation - rewards can be any real value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3*hidden_dim) - pooled features
        Returns:
            (batch, reward_horizon) - predicted rewards for next N steps

        Example outputs:
            [+0.8, +0.5, +0.2, 0.0, 0.0]  → "good things coming soon"
            [-1.5, -0.5, 0.0, 0.0, 0.0]   → "pain imminent!"
            [0.0, 0.0, 0.0, 0.0, 0.0]     → "nothing special expected"
        """
        return self.mlp(x)
```

**Why Multi-Step Reward Prediction?**

Predicting rewards for multiple future steps provides:

1. **Temporal credit assignment**: Agent learns "this pattern leads to pain in 3 steps"
2. **Planning horizon**: Representations encode future consequences, not just immediate
3. **Richer gradient**: More supervision signal per timestep
4. **Biological plausibility**: Dopamine neurons predict rewards across time horizons

**Reward Horizon Options**:
- `horizon=1`: Immediate reward only (simplest, fastest)
- `horizon=5`: Short-term planning (recommended default)
- `horizon=10`: Longer planning (requires more memory for history buffer)

### 6. ActionHead

Outputs agent actions:

```python
class ActionHead(nn.Module):
    """
    Outputs agent actions from pooled features.

    Actions: (thrust, torque, vocalize, freq, eat)
    """

    def __init__(self, config: LRNConfig):
        super().__init__()

        # Input: pooled features (3 * hidden_dim)
        input_dim = 3 * config.hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.action_hidden_dim),
            nn.GELU(),
            nn.Linear(config.action_hidden_dim, config.action_dim)
        )

        # Action bounds (applied externally, but documented here)
        # thrust: [-1, 1]
        # torque: [-1, 1]
        # vocalize: [0, 1]
        # freq: [0, 1] (maps to frequency range)
        # eat: [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3*hidden_dim) - pooled features
        Returns:
            (batch, 5) - raw action logits (apply tanh/sigmoid externally)
        """
        return self.mlp(x)
```

### 6. LivingResonanceNetwork (Main Model)

```python
class LivingResonanceNetwork(nn.Module):
    """
    Main LRN architecture combining all components.

    Fourier-based neural network for embodied AI agents.
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.config = config

        # Modality encoders
        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.proprio_encoder = ProprioEncoder(config)
        self.touch_encoder = TouchEncoder(config)

        # Fourier mixing layers
        self.mixing_layers = nn.ModuleList([
            FourierMixingLayer(config)
            for _ in range(config.num_mixing_layers)
        ])

        # Output heads
        self.prediction_head = PredictionHead(config)
        self.reward_head = RewardHead(config)  # NEW: Multi-task reward prediction
        self.action_head = ActionHead(config)

        # Optional genome modulation
        if config.use_genome_modulation:
            self.genome_modulator = GenomeModulator(config)
        else:
            self.genome_modulator = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable online learning."""
        # See initialization section below for details
        pass

    def forward(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
        touch: torch.Tensor,
        genome: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LRN.

        Args:
            vision: (batch, 32, 4)
            audio: (batch, 100, 2)
            proprio: (batch, 7)
            touch: (batch, 8)
            genome: (batch, 100) - optional genome vector

        Returns:
            predictions: (batch, output_dim) - predicted next sensory state
            reward_preds: (batch, reward_horizon) - predicted upcoming rewards
            actions: (batch, 5) - action outputs
        """
        # Encode each modality
        vision_enc = self.vision_encoder(vision)      # (B, 32, hidden_dim)
        audio_enc = self.audio_encoder(audio)          # (B, 100, hidden_dim)
        proprio_enc = self.proprio_encoder(proprio)    # (B, 16, hidden_dim)
        touch_enc = self.touch_encoder(touch)          # (B, 16, hidden_dim)

        # Concatenate along sequence dimension
        x = torch.cat([vision_enc, audio_enc, proprio_enc, touch_enc], dim=1)
        # x: (B, 164, hidden_dim)

        # Apply genome modulation if available
        if self.genome_modulator is not None and genome is not None:
            x = self.genome_modulator(x, genome)

        # Fourier mixing layers
        for layer in self.mixing_layers:
            x = layer(x)  # (B, 164, hidden_dim)

        # Pooling for output heads
        mean_pool = x.mean(dim=1)  # (B, hidden_dim)
        max_pool, _ = x.max(dim=1)  # (B, hidden_dim)
        last = x[:, -1, :]          # (B, hidden_dim)

        # Concatenate pooled features
        pooled = torch.cat([mean_pool, max_pool, last], dim=1)  # (B, 3*hidden_dim)

        # Output heads
        predictions = self.prediction_head(pooled)
        reward_preds = self.reward_head(pooled)  # NEW: predict upcoming rewards
        actions = self.action_head(pooled)

        return predictions, reward_preds, actions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        reward_preds: torch.Tensor,
        next_sensory: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        actual_rewards: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with multi-task reward prediction.

        Args:
            predictions: (batch, output_dim) - predicted next sensory state
            reward_preds: (batch, reward_horizon) - predicted future rewards
            next_sensory: Dict with actual next sensory state
            actions: (batch, 5)
            actual_rewards: (batch, reward_horizon) - actual rewards for next N steps

        Returns:
            Dictionary with loss components
        """
        # Split sensory predictions
        pred_split = self.prediction_head.split_prediction(predictions, self.config)

        # Sensory prediction loss (MSE on each modality)
        vision_loss = F.mse_loss(pred_split['vision'], next_sensory['vision'])
        audio_loss = F.mse_loss(pred_split['audio'], next_sensory['audio'])
        proprio_loss = F.mse_loss(pred_split['proprio'], next_sensory['proprio'])
        touch_loss = F.mse_loss(pred_split['touch'], next_sensory['touch'])

        sensory_loss = vision_loss + audio_loss + proprio_loss + touch_loss

        # REWARD PREDICTION LOSS (NEW - creates survival gradient!)
        if actual_rewards is not None:
            # MSE between predicted and actual rewards
            reward_loss = F.mse_loss(reward_preds, actual_rewards)
        else:
            reward_loss = torch.tensor(0.0, device=actions.device)

        # Combined loss: sensory + reward prediction
        # Both contribute to learning representations that understand
        # the world AND survival value
        total_loss = sensory_loss + self.config.reward_loss_weight * reward_loss

        return {
            'total': total_loss,
            'sensory': sensory_loss,
            'reward': reward_loss,       # NEW: survival-relevant loss
            'vision': vision_loss,
            'audio': audio_loss,
            'proprio': proprio_loss,
            'touch': touch_loss
        }
```

## Fourier Operations

### Why Real FFT (rfft)?

For real-valued inputs, `rfft` computes only the positive frequency components, exploiting Hermitian symmetry to halve memory and computation:

```python
# Standard FFT: n complex outputs
x_fft = torch.fft.fft(x, dim=-1)      # (B, seq_len) complex

# Real FFT: n//2 + 1 complex outputs (50% saving)
x_fft = torch.fft.rfft(x, dim=-1)     # (B, seq_len//2+1) complex
```

For seq_len=164: rfft gives 83 complex values vs 164 for full FFT.

### Learnable Spectral Filters

The key innovation from FFTNet 2025:

```python
# Filter stored as real tensor for optimization stability
spectral_filter = nn.Parameter(torch.randn(seq_len, freq_bins, 2))

# Convert to complex for multiplication
filter_complex = torch.view_as_complex(spectral_filter)

# Element-wise multiplication in frequency domain
x_filtered = x_fft * filter_complex
```

Each frequency bin has an independent complex coefficient that modulates:
- **Magnitude**: Amplify or attenuate that frequency
- **Phase**: Shift timing of that frequency component

This is learned via backprop through the FFT operations.

### Gradient Flow Through FFT

PyTorch's FFT operations are fully differentiable:

```python
# Forward
x_fft = torch.fft.rfft(x)
x_filtered = x_fft * learnable_filter
x_out = torch.fft.irfft(x_filtered)

# Backward (automatic)
# ∂L/∂filter = ∂L/∂x_out @ (∂x_out/∂x_filtered) @ (∂x_filtered/∂filter)
# PyTorch handles the chain rule through FFT/iFFT
```

Gradients flow cleanly because:
1. FFT is a linear operation (matrix multiplication)
2. Complex multiplication is differentiable
3. iFFT is the adjoint of FFT

### Spectral Bias and Initialization

Random initialization of spectral filters creates bias toward low frequencies (spectral bias). This is actually beneficial for online learning:

- **Low frequencies**: Capture smooth, global patterns (stable)
- **High frequencies**: Capture fine details (noisy in early training)

Initialize filters to emphasize low frequencies:

```python
def init_spectral_filter(freq_bins: int) -> torch.Tensor:
    """Initialize with low-frequency bias."""
    freqs = torch.arange(freq_bins)
    decay = torch.exp(-freqs / (freq_bins / 4))  # Decay high frequencies

    # Real and imaginary components
    real = torch.randn(seq_len, freq_bins) * decay
    imag = torch.randn(seq_len, freq_bins) * decay

    return torch.stack([real, imag], dim=-1)
```

## Cross-Modal Fusion

### Concatenation Strategy

We concatenate modalities along the sequence dimension rather than feature dimension:

```python
# Option 1: Sequence concatenation (chosen)
vision:  (B, 32, 64)
audio:   (B, 100, 64)
proprio: (B, 16, 64)
touch:   (B, 16, 64)
concat:  (B, 164, 64)  # Different sequence lengths preserved

# Option 2: Feature concatenation (not chosen)
# Would require same sequence length and give (B, seq, 256)
```

**Why sequence concatenation?**
1. Preserves temporal/spatial structure of each modality
2. Allows Fourier layers to discover cross-modal correlations via frequency mixing
3. Different modalities have natural different "resolutions" (audio is temporal, vision is spatial)

### Cross-Modal Correlation via FFT

When we apply FFT to the concatenated sequence:

```python
x = [vision | audio | proprio | touch]  # (B, 164, 64)
x_fft = rfft(x, dim=1)  # FFT along sequence dimension
```

The frequency domain representation automatically captures:
- **Intra-modal patterns**: Frequencies within each modality
- **Cross-modal correlations**: Frequency relationships between modalities
  - Example: Audio rhythm (100 Hz) correlating with vision motion (ray distance changes)

The learnable spectral filter learns which cross-modal frequency correlations are useful.

### Positional Information

Unlike transformers, we don't add explicit positional encodings because:
1. **FFT preserves phase**: Position information is in the phase component
2. **Modality boundaries**: The concatenation order implicitly encodes modality identity
3. **Sequence structure**: Vision rays have spatial order, audio has temporal order

If needed, we can add learnable positional embeddings:

```python
self.pos_embedding = nn.Parameter(torch.randn(1, total_seq_len, hidden_dim))
x = x + self.pos_embedding
```

## Initialization

Critical for stable online learning (single-sample updates with high variance):

### General Principles

1. **Small initial weights**: Prevent saturation in early training
2. **Spectral bias**: Favor low frequencies initially
3. **Residual scaling**: Scale residual connections for stability
4. **Normalized inputs**: Assume sensory inputs are normalized

### Specific Initialization

```python
def _init_weights(self):
    """Initialize weights for stable online learning."""

    # 1. Encoder projections: Xavier uniform (balanced variance)
    for encoder in [self.vision_encoder, self.audio_encoder,
                    self.proprio_encoder, self.touch_encoder]:
        for module in encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # 2. Spectral filters: Low-frequency bias
    for layer in self.mixing_layers:
        freq_bins = layer.freq_bins
        seq_len = layer.seq_len

        # Frequency decay
        freqs = torch.arange(freq_bins, dtype=torch.float32)
        decay = torch.exp(-freqs / (freq_bins / 4))

        # Initialize with decay
        with torch.no_grad():
            for i in range(seq_len):
                layer.spectral_filter.data[i, :, 0] = torch.randn(freq_bins) * decay * 0.1
                layer.spectral_filter.data[i, :, 1] = torch.randn(freq_bins) * decay * 0.1

    # 3. LayerNorm: Standard initialization
    for layer in self.mixing_layers:
        nn.init.ones_(layer.norm.weight)
        nn.init.zeros_(layer.norm.bias)

    # 4. Output heads: Small initialization to prevent large initial predictions
    for head in [self.prediction_head, self.action_head]:
        for module in head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
```

### Online Learning Stability

For single-sample updates, additional techniques:

```python
# Option 1: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Option 2: Adaptive learning rate per parameter group
optimizer = torch.optim.Adam([
    {'params': model.mixing_layers.parameters(), 'lr': 1e-4},
    {'params': model.prediction_head.parameters(), 'lr': 1e-3},
    {'params': model.action_head.parameters(), 'lr': 1e-3}
])

# Option 3: Exponential moving average of parameters
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach()
                      for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = (self.decay * self.shadow[name] +
                                (1 - self.decay) * param.data)
```

## Genome Integration

The genome vector modulates the architecture, enabling evolutionary search:

### GenomeModulator

```python
class GenomeModulator(nn.Module):
    """
    Modulates LRN activations based on genome vector.

    Genome encodes architectural hyperparameters that can evolve.
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.genome_dim = config.genome_dim
        self.hidden_dim = config.hidden_dim

        # Project genome to modulation parameters
        self.genome_projection = nn.Sequential(
            nn.Linear(self.genome_dim, self.hidden_dim),
            nn.Tanh()
        )

        # Scale and shift parameters
        self.scale_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.shift_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor, genome: torch.Tensor) -> torch.Tensor:
        """
        Apply genome-based modulation.

        Args:
            x: (batch, seq_len, hidden_dim) - activations
            genome: (batch, genome_dim) - genome vector

        Returns:
            (batch, seq_len, hidden_dim) - modulated activations
        """
        # Project genome
        genome_features = self.genome_projection(genome)  # (B, hidden_dim)

        # Compute scale and shift
        scale = torch.sigmoid(self.scale_layer(genome_features))  # (B, hidden_dim) in [0, 1]
        shift = self.shift_layer(genome_features)  # (B, hidden_dim)

        # Apply affine transformation
        # scale in [0.5, 1.5] for stability
        scale = 0.5 + scale

        # Broadcast over sequence dimension
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)

        return x
```

### Genome Encoding Examples

The genome vector (100 dims) can encode:

```python
# Example genome vector interpretation
genome[0:10]   → spectral filter bias (low/high frequency preference)
genome[10:20]  → mixing layer depth modulation
genome[20:30]  → encoder projection scales
genome[30:40]  → attention to specific modalities
genome[40:50]  → prediction vs action head balance
genome[50:60]  → learning rate modulation (meta-learning)
genome[60:70]  → activation function blending (GELU vs ReLU)
genome[70:80]  → residual connection strength
genome[80:90]  → dropout rates (stochastic depth)
genome[90:100] → reserved for future use
```

This allows evolution to search architectural space alongside weights.

## API Specification

### Model Creation

```python
from lrn import LivingResonanceNetwork, LRNConfig

# Create configuration
config = LRNConfig(
    hidden_dim=128,
    num_mixing_layers=6,
    pred_hidden_dim=256,
    action_hidden_dim=128
)

# Instantiate model
model = LivingResonanceNetwork(config)

# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### Forward Pass

```python
import torch

# Create dummy inputs (batch_size=1 for online learning)
vision = torch.randn(1, 32, 4)
audio = torch.randn(1, 100, 2)
proprio = torch.randn(1, 7)
touch = torch.randn(1, 8)
genome = torch.randn(1, 100)  # Optional

# Forward pass
predictions, actions = model(vision, audio, proprio, touch, genome)

# predictions: (1, 343) - predicted next sensory state
# actions: (1, 5) - (thrust, torque, vocalize, freq, eat)

# Apply action bounds
actions_bounded = torch.tanh(actions[:, :2])  # thrust, torque in [-1, 1]
actions_bounded = torch.cat([
    actions_bounded,
    torch.sigmoid(actions[:, 2:])  # vocalize, freq, eat in [0, 1]
], dim=1)
```

### Training Step

```python
# Prepare next sensory state (ground truth)
next_sensory = {
    'vision': torch.randn(1, 32, 4),
    'audio': torch.randn(1, 100, 2),
    'proprio': torch.randn(1, 7),
    'touch': torch.randn(1, 8)
}

# Optional reward signal
rewards = torch.tensor([1.0])  # Positive reward

# Compute loss
losses = model.compute_loss(predictions, next_sensory, actions, rewards)

# Backward pass
optimizer.zero_grad()
losses['total'].backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Log losses
print(f"Total loss: {losses['total'].item():.4f}")
print(f"Prediction: {losses['prediction'].item():.4f}")
print(f"Action: {losses['action'].item():.4f}")
```

### Inference (No Gradients)

```python
model.eval()
with torch.no_grad():
    predictions, actions = model(vision, audio, proprio, touch, genome)

    # Apply bounds
    thrust = torch.tanh(actions[0, 0]).item()
    torque = torch.tanh(actions[0, 1]).item()
    vocalize = torch.sigmoid(actions[0, 2]).item()
    freq = torch.sigmoid(actions[0, 3]).item()
    eat = torch.sigmoid(actions[0, 4]).item()

model.train()
```

### Save/Load

```python
# Save checkpoint
torch.save({
    'config': config,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'step': step,
}, 'lrn_checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('lrn_checkpoint.pt')
config = checkpoint['config']
model = LivingResonanceNetwork(config)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Testing Strategy

### 1. Shape Tests (`test_lrn_shapes.py`)

Verify tensor shapes at every layer:

```python
def test_encoder_shapes():
    config = LRNConfig()

    vision_enc = VisionEncoder(config)
    vision_input = torch.randn(1, 32, 4)
    vision_output = vision_enc(vision_input)
    assert vision_output.shape == (1, 32, config.hidden_dim)

    # Test all encoders...

def test_fourier_layer_shapes():
    config = LRNConfig()
    layer = FourierMixingLayer(config)

    x = torch.randn(1, config.total_seq_len, config.hidden_dim)
    output = layer(x)
    assert output.shape == x.shape

    # Test FFT intermediate shapes...

def test_end_to_end_shapes():
    config = LRNConfig()
    model = LivingResonanceNetwork(config)

    vision = torch.randn(1, 32, 4)
    audio = torch.randn(1, 100, 2)
    proprio = torch.randn(1, 7)
    touch = torch.randn(1, 8)

    predictions, reward_preds, actions = model(vision, audio, proprio, touch)
    assert predictions.shape == (1, 343)  # Total sensory dim
    assert reward_preds.shape == (1, config.reward_horizon)  # Reward predictions
    assert actions.shape == (1, 5)


def test_reward_head_shapes():
    """Test RewardHead output shapes for various configurations."""
    # Test default horizon
    config = LRNConfig()
    head = RewardHead(config)

    pooled = torch.randn(1, 3 * config.hidden_dim)
    output = head(pooled)
    assert output.shape == (1, config.reward_horizon)  # (1, 5) by default

    # Test custom horizon
    config_long = LRNConfig(reward_horizon=10)
    head_long = RewardHead(config_long)
    output_long = head_long(pooled)
    assert output_long.shape == (1, 10)

    # Test batch processing
    batch_pooled = torch.randn(4, 3 * config.hidden_dim)
    batch_output = head(batch_pooled)
    assert batch_output.shape == (4, config.reward_horizon)


def test_reward_head_gradients():
    """Test gradients flow through RewardHead."""
    config = LRNConfig()
    head = RewardHead(config)

    pooled = torch.randn(1, 3 * config.hidden_dim, requires_grad=True)
    output = head(pooled)
    loss = output.sum()
    loss.backward()

    assert pooled.grad is not None
    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
```

### 2. Gradient Tests (`test_lrn_gradient.py`)

Verify gradients flow through all components:

```python
def test_gradient_flow():
    config = LRNConfig()
    model = LivingResonanceNetwork(config)

    # Create inputs
    inputs = create_dummy_inputs()

    # Forward
    predictions, reward_preds, actions = model(**inputs)

    # Backward on dummy loss
    loss = predictions.sum() + reward_preds.sum() + actions.sum()
    loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

def test_fft_gradient():
    """Specifically test FFT gradient flow."""
    config = LRNConfig()
    layer = FourierMixingLayer(config)

    x = torch.randn(1, config.total_seq_len, config.hidden_dim, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.spectral_filter.grad is not None
```

### 3. Forward Pass Tests (`test_lrn_forward.py`)

Verify numerical correctness:

```python
def test_fft_inverse():
    """Test FFT → iFFT returns original signal."""
    x = torch.randn(2, 164, 128)

    x_fft = torch.fft.rfft(x, dim=1)
    x_reconstructed = torch.fft.irfft(x_fft, n=164, dim=1)

    assert torch.allclose(x, x_reconstructed, atol=1e-6)

def test_prediction_split():
    """Test prediction can be split back into modalities."""
    config = LRNConfig()
    head = PredictionHead(config)

    pooled = torch.randn(1, 3 * config.hidden_dim)
    pred = head(pooled)

    split = head.split_prediction(pred, config)
    assert split['vision'].shape == (1, 32, 4)
    assert split['audio'].shape == (1, 100, 2)
    assert split['proprio'].shape == (1, 7)
    assert split['touch'].shape == (1, 8)

def test_deterministic():
    """Test forward pass is deterministic."""
    torch.manual_seed(42)
    config = LRNConfig()
    model1 = LivingResonanceNetwork(config)

    torch.manual_seed(42)
    model2 = LivingResonanceNetwork(config)

    inputs = create_dummy_inputs()

    out1 = model1(**inputs)
    out2 = model2(**inputs)

    assert torch.allclose(out1[0], out2[0])
    assert torch.allclose(out1[1], out2[1])
```

### 4. Online Learning Tests (`test_lrn_online.py`)

Verify stability with single-sample updates:

```python
def test_single_sample_update():
    """Test model can handle batch_size=1 updates."""
    config = LRNConfig()
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(100):
        # Single sample
        inputs = create_dummy_inputs(batch_size=1)
        next_sensory = create_dummy_inputs(batch_size=1)

        predictions, actions = model(**inputs)
        losses = model.compute_loss(predictions, next_sensory, actions)

        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Check no NaN/Inf
        assert not torch.isnan(losses['total'])
        assert not torch.isinf(losses['total'])

def test_gradient_variance():
    """Test gradient variance is reasonable for online learning."""
    config = LRNConfig()
    model = LivingResonanceNetwork(config)

    grad_norms = []
    for i in range(50):
        inputs = create_dummy_inputs(batch_size=1)
        next_sensory = create_dummy_inputs(batch_size=1)

        predictions, actions = model(**inputs)
        losses = model.compute_loss(predictions, next_sensory, actions)
        losses['total'].backward()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        model.zero_grad()

    # Gradient norm variance should be bounded
    grad_std = np.std(grad_norms)
    grad_mean = np.mean(grad_norms)
    cv = grad_std / grad_mean  # Coefficient of variation

    assert cv < 2.0, f"High gradient variance: CV={cv}"
```

### 5. Integration Tests

```python
def test_with_dummy_environment():
    """Test LRN in a simple environment loop."""
    config = LRNConfig()
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Simulate 1000 steps
    for step in range(1000):
        # Get current sensory input
        current_sensory = get_sensory_from_env()

        # Forward pass
        predictions, actions = model(**current_sensory)

        # Execute actions in environment
        rewards = execute_actions(actions)

        # Get next sensory state
        next_sensory = get_sensory_from_env()

        # Compute loss and update
        losses = model.compute_loss(predictions, next_sensory, actions, rewards)
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}, Loss: {losses['total'].item():.4f}")
```

## Implementation Order

Recommended sequence for building LRN:

### Phase 1: Foundation (Days 1-2)

1. **Create project structure**
   ```bash
   mkdir -p primordial/lrn primordial/tests primordial/examples
   touch primordial/lrn/{__init__,config,utils}.py
   ```

2. **Implement LRNConfig** (`config.py`)
   - All configuration parameters
   - Validation logic
   - Computed properties (total_seq_len, freq_bins)

3. **Implement FFT utilities** (`utils.py`)
   - Spectral filter initialization
   - Complex tensor helpers
   - Visualization utilities (plot spectrum)

4. **Write shape tests** (`test_lrn_shapes.py`)
   - Define expected shapes
   - Create dummy input generators

### Phase 2: Encoders (Day 3)

5. **Implement WaveletEncoder classes** (`encoders.py`)
   - VisionEncoder
   - AudioEncoder
   - ProprioEncoder
   - TouchEncoder

6. **Test encoders**
   - Shape tests
   - Gradient tests
   - Add to `test_lrn_shapes.py`

### Phase 3: Core Mixing (Days 4-5)

7. **Implement FourierMixingLayer** (`mixing.py`)
   - FFT forward/backward
   - Spectral filter multiplication
   - LayerNorm + activation

8. **Test Fourier layer**
   - FFT/iFFT round-trip
   - Gradient flow through complex operations
   - Add to `test_lrn_gradient.py`

9. **Verify numerical stability**
   - Test with random inputs
   - Check for NaN/Inf
   - Profile performance

### Phase 4: Output Heads (Day 6)

10. **Implement PredictionHead** (`heads.py`)
    - MLP architecture
    - Prediction splitting logic

11. **Implement ActionHead** (`heads.py`)
    - MLP architecture
    - Action bounding

12. **Test heads**
    - Shape correctness
    - Split/unsplit predictions
    - Add to `test_lrn_forward.py`

### Phase 5: Integration (Day 7)

13. **Implement LivingResonanceNetwork** (`architecture.py`)
    - Combine all components
    - Forward pass
    - Loss computation

14. **Test end-to-end**
    - Full forward pass
    - Backward pass
    - Parameter counting
    - Add to `test_lrn_forward.py`

15. **Implement initialization** (`architecture.py`)
    - Weight initialization
    - Spectral filter initialization
    - Test stability

### Phase 6: Genome Integration (Day 8)

16. **Implement GenomeModulator** (`genome.py`)
    - Genome projection
    - Affine modulation
    - Integration with LRN

17. **Test genome modulation**
    - Different genomes → different outputs
    - Gradient flow through modulator

### Phase 7: Online Learning (Days 9-10)

18. **Implement online learning tests** (`test_lrn_online.py`)
    - Single-sample updates
    - Gradient variance
    - Stability over many steps

19. **Tune hyperparameters**
    - Learning rates
    - Gradient clipping
    - Spectral filter initialization

20. **Add EMA and stability features**
    - Exponential moving average
    - Adaptive learning rates
    - Gradient monitoring

### Phase 8: Examples & Documentation (Days 11-12)

21. **Create standalone demo** (`examples/lrn_demo.py`)
    - Load model
    - Run inference
    - Visualize outputs

22. **Create profiling script** (`examples/lrn_profiling.py`)
    - Measure forward/backward time
    - Memory usage
    - Compare to transformer baseline

23. **Write API documentation**
    - Docstrings for all public APIs
    - Usage examples
    - Type hints

### Phase 9: Validation (Days 13-14)

24. **Integrate with environment**
    - Connect to sensory inputs
    - Execute actions
    - Collect real data

25. **Benchmark performance**
    - Speed (samples/sec)
    - Memory footprint
    - Parameter efficiency vs transformer

26. **Final testing**
    - All unit tests pass
    - Integration tests pass
    - No memory leaks
    - Deterministic outputs

### Phase 10: Optimization (Day 15+)

27. **Profile bottlenecks**
    - Identify slow operations
    - Optimize FFT usage
    - Consider torch.compile

28. **Hyperparameter search**
    - Grid search on hidden_dim, num_layers
    - Find optimal ~800K param configuration

29. **Prepare for evolution**
    - Genome vector design
    - Fitness evaluation
    - Population initialization

## Success Criteria

The implementation is complete when:

1. All tests pass (shapes, gradients, forward, online)
2. Parameter count is within 700K-900K
3. Forward pass runs in <10ms on laptop CPU
4. Model trains stably for 10,000+ online updates
5. Gradient variance is bounded (CV < 2.0)
6. Memory usage is <500MB for single sample
7. API documentation is complete
8. Integration with environment works
9. Profiling shows competitive speed vs transformer
10. Demo script runs end-to-end

## Open Questions

Issues to resolve during implementation:

1. **Sequence length padding**: How to handle variable-length audio?
2. **Multi-scale FFT**: Should we use different FFT window sizes?
3. **Attention fallback**: Keep optional attention layers for comparison?
4. **Spectral regularization**: Add L1 penalty on high frequencies?
5. **Cross-modal alignment**: Explicit alignment loss between modalities?
6. **Causal masking**: Needed for autoregressive generation?
7. **Frequency resolution**: Trade-off between spectral and temporal resolution
8. **Genome expressiveness**: Is 100-dim genome sufficient?

## References

1. **FNet: Mixing Tokens with Fourier Transforms** (2021)
   - https://arxiv.org/abs/2105.03824
   - Key insight: Pure FFT can replace attention

2. **FFTNet 2025: Learnable Spectral Filters** (2025)
   - https://arxiv.org/html/2502.18394v4
   - Key insight: Learnable filters in frequency domain

3. **PyTorch FFT Documentation**
   - https://pytorch.org/docs/stable/fft.html
   - Implementation details

4. **Spectral Bias in Deep Learning**
   - Neural networks learn low frequencies first
   - Informs initialization strategy

5. **Online Learning Theory**
   - Single-sample update stability
   - Adaptive learning rates

---

**Document version**: 1.0
**Target implementation**: 15 days
**Target parameters**: ~800K
**Target latency**: <10ms per forward pass
**Status**: Ready for implementation
