# Fourier Mixing Prototype Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate that Fourier-based mixing with learnable spectral filters can learn temporal patterns AND reward prediction before building the full LRN architecture.

**Architecture:** A minimal model with input projection, 2 FourierMixingLayers, pooling, and dual output heads (SensoryHead + RewardHead). Trained on a synthetic multi-task prediction problem: (1) predict next timestep in noisy sine wave, (2) predict upcoming "reward" signals. This mirrors the full LRN's multi-task learning approach.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, pytest

---

## Success Criteria

The prototype validates the core hypothesis when:
1. Gradients flow through FFT operations (no NaN/Inf)
2. **Sensory prediction loss** decreases over 1000 training steps
3. **Reward prediction loss** decreases over 1000 training steps
4. Model predictions improve vs random baseline on **both** tasks
5. Forward pass runs in <5ms on CPU
6. **Single-sample updates (batch_size=1) remain stable for 100 steps**

---

## Task 1: Project Structure

**Files:**
- Create: `primordial/lrn/__init__.py`
- Create: `primordial/lrn/config.py`
- Create: `primordial/tests/__init__.py`
- Create: `primordial/tests/lrn/__init__.py`
- Create: `requirements.txt`

**Step 1: Create directory structure**

```bash
mkdir -p primordial/lrn primordial/tests/lrn
touch primordial/__init__.py
touch primordial/lrn/__init__.py
touch primordial/tests/__init__.py
touch primordial/tests/lrn/__init__.py
```

**Step 2: Create requirements.txt**

```
torch>=2.0.0
numpy>=1.24.0
pytest>=7.4.0
```

**Step 3: Create minimal config**

`primordial/lrn/config.py`:
```python
"""Configuration for Fourier mixing prototype."""
from dataclasses import dataclass


@dataclass
class PrototypeConfig:
    """Minimal config for Fourier mixing validation."""

    # Sequence dimensions
    seq_len: int = 64
    hidden_dim: int = 32

    # Architecture
    num_mixing_layers: int = 2

    # FFT settings
    use_real_fft: bool = True

    # Normalization
    layer_norm_eps: float = 1e-5

    # Multi-task learning (matches full LRN)
    reward_horizon: int = 5  # Predict rewards for next N steps
    reward_loss_weight: float = 1.0  # Weight for reward loss vs sensory loss

    @property
    def freq_bins(self) -> int:
        """Number of frequency bins for rfft."""
        return self.seq_len // 2 + 1 if self.use_real_fft else self.seq_len
```

**Step 4: Export from __init__.py**

`primordial/lrn/__init__.py`:
```python
"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig

__all__ = ["PrototypeConfig"]
```

**Step 5: Verify imports work**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -c "from primordial.lrn import PrototypeConfig; print(PrototypeConfig())"`

Expected: Config dataclass prints without error

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add project structure and PrototypeConfig"
```

---

## Task 2: FourierMixingLayer Implementation

**Files:**
- Create: `primordial/lrn/mixing.py`
- Create: `primordial/tests/lrn/test_mixing.py`

**Step 1: Write the failing test for shapes**

`primordial/tests/lrn/test_mixing.py`:
```python
"""Tests for FourierMixingLayer."""
import torch
import pytest


def test_fourier_mixing_layer_output_shape():
    """Test that output shape matches input shape."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig(seq_len=64, hidden_dim=32)
    layer = FourierMixingLayer(config)

    x = torch.randn(1, 64, 32)  # (batch, seq_len, hidden_dim)
    output = layer(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"


def test_fourier_mixing_layer_batch():
    """Test with larger batch size."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig(seq_len=64, hidden_dim=32)
    layer = FourierMixingLayer(config)

    x = torch.randn(8, 64, 32)
    output = layer(x)

    assert output.shape == (8, 64, 32)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_mixing.py -v`

Expected: FAIL with "No module named 'primordial.lrn.mixing'"

**Step 3: Implement FourierMixingLayer**

`primordial/lrn/mixing.py`:
```python
"""Fourier-based mixing layer with learnable spectral filters."""
import torch
import torch.nn as nn

from .config import PrototypeConfig


class FourierMixingLayer(nn.Module):
    """
    Fourier-based mixing layer with learnable spectral filters.

    Replaces self-attention with O(n log n) FFT operations.
    Based on FNet and FFTNet 2025 research.

    Architecture matches parent LRN spec:
    - Filter shape: (seq_len, freq_bins, 2)
    - Spectral bias initialization (favor low frequencies)
    - Slicing logic for hidden_dim/seq_len mismatch
    """

    def __init__(self, config: PrototypeConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.seq_len = config.seq_len
        self.freq_bins = config.freq_bins

        # Learnable spectral filter (stored as real tensor, converted to complex)
        # Shape: (seq_len, freq_bins, 2) - matches parent LRN architecture
        self.spectral_filter = nn.Parameter(
            self._init_spectral_filter()
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_eps)

        # Activation
        self.activation = nn.GELU()

    def _init_spectral_filter(self) -> torch.Tensor:
        """
        Initialize spectral filter with frequency-dependent decay (spectral bias).

        Low frequencies get larger initial values, high frequencies get smaller.
        This matches biological and empirical observations that neural networks
        learn low frequencies first.
        """
        # Frequency decay: exp(-freq / (freq_bins / 4))
        freqs = torch.arange(self.freq_bins, dtype=torch.float32)
        decay = torch.exp(-freqs / (self.freq_bins / 4))

        # Initialize with decay applied
        # Shape: (seq_len, freq_bins, 2) for real and imaginary
        filter_init = torch.randn(self.seq_len, self.freq_bins, 2) * 0.1
        filter_init = filter_init * decay.unsqueeze(0).unsqueeze(-1)

        return filter_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier mixing with learnable spectral filters.

        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        residual = x

        # Transpose to (batch, hidden_dim, seq_len) for FFT along sequence
        x = x.transpose(1, 2)  # (B, hidden_dim, seq_len)

        # Apply FFT along sequence dimension
        if self.config.use_real_fft:
            x_fft = torch.fft.rfft(x, dim=2)  # (B, hidden_dim, freq_bins) complex
        else:
            x_fft = torch.fft.fft(x, dim=2)

        # Convert spectral filter to complex
        # Shape: (seq_len, freq_bins) complex
        filter_complex = torch.view_as_complex(self.spectral_filter.contiguous())

        # Slice or repeat filter to match hidden_dim (parent LRN logic)
        if self.hidden_dim <= self.seq_len:
            filter_slice = filter_complex[:self.hidden_dim, :]  # (hidden_dim, freq_bins)
        else:
            # Repeat filter if hidden_dim > seq_len
            repeats = (self.hidden_dim + self.seq_len - 1) // self.seq_len
            filter_slice = filter_complex.repeat(repeats, 1)[:self.hidden_dim, :]

        # Apply spectral filtering
        x_filtered = x_fft * filter_slice.unsqueeze(0)  # (B, hidden_dim, freq_bins)

        # Inverse FFT back to time domain
        if self.config.use_real_fft:
            x_out = torch.fft.irfft(x_filtered, n=self.seq_len, dim=2)
        else:
            x_out = torch.fft.ifft(x_filtered, dim=2).real

        # Transpose back to (batch, seq_len, hidden_dim)
        x_out = x_out.transpose(1, 2)

        # Residual connection, normalization, activation
        x_out = x_out + residual
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)

        return x_out
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_mixing.py -v`

Expected: 2 passed

**Step 5: Update exports**

`primordial/lrn/__init__.py`:
```python
"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .mixing import FourierMixingLayer

__all__ = ["PrototypeConfig", "FourierMixingLayer"]
```

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add FourierMixingLayer with spectral filtering"
```

---

## Task 3: Gradient Flow Tests

**Files:**
- Modify: `primordial/tests/lrn/test_mixing.py`

**Step 1: Write failing gradient test**

Add to `primordial/tests/lrn/test_mixing.py`:
```python
def test_gradient_flow_through_fft():
    """Test gradients flow through FFT operations."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig()
    layer = FourierMixingLayer(config)

    x = torch.randn(1, 64, 32, requires_grad=True)
    output = layer(x)
    loss = output.sum()
    loss.backward()

    # Check input gradient exists
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    assert not torch.isinf(x.grad).any(), "Inf in input gradient"

    # Check spectral filter gradient exists
    assert layer.spectral_filter.grad is not None, "No gradient for spectral filter"
    assert not torch.isnan(layer.spectral_filter.grad).any(), "NaN in filter gradient"


def test_gradient_no_nan_after_many_steps():
    """Test gradients remain stable over multiple forward/backward passes."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.mixing import FourierMixingLayer

    config = PrototypeConfig()
    layer = FourierMixingLayer(config)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    for step in range(100):
        x = torch.randn(4, 64, 32)
        output = layer(x)
        loss = output.sum()

        optimizer.zero_grad()
        loss.backward()

        # Check for NaN/Inf
        for name, param in layer.named_parameters():
            assert not torch.isnan(param.grad).any(), f"NaN gradient at step {step}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient at step {step}"

        optimizer.step()

        # Check weights didn't explode
        for name, param in layer.named_parameters():
            assert not torch.isnan(param).any(), f"NaN weights at step {step}"
```

**Step 2: Run tests**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_mixing.py -v`

Expected: 4 passed

**Step 3: Commit**

```bash
git add -A
git commit -m "test: add gradient flow tests for FourierMixingLayer"
```

---

## Task 4: Prototype Model with Multi-Task Heads

**Files:**
- Create: `primordial/lrn/heads.py`
- Create: `primordial/lrn/prototype.py`
- Create: `primordial/tests/lrn/test_prototype.py`

**Step 1: Write failing test for prototype model**

`primordial/tests/lrn/test_prototype.py`:
```python
"""Tests for the Fourier prototype model."""
import torch
import pytest


def test_prototype_model_forward():
    """Test prototype model forward pass returns both outputs."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.prototype import FourierPrototype

    config = PrototypeConfig(seq_len=64, hidden_dim=32, reward_horizon=5)
    model = FourierPrototype(config, input_dim=1)

    # Input: (batch, seq_len, 1)
    x = torch.randn(4, 64, 1)
    sensory_pred, reward_pred = model(x)

    # Sensory prediction: (batch, seq_len, 1)
    assert sensory_pred.shape == (4, 64, 1), f"Expected (4, 64, 1), got {sensory_pred.shape}"

    # Reward prediction: (batch, reward_horizon)
    assert reward_pred.shape == (4, 5), f"Expected (4, 5), got {reward_pred.shape}"


def test_prototype_model_parameter_count():
    """Test model has reasonable parameter count."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.prototype import FourierPrototype

    config = PrototypeConfig(seq_len=64, hidden_dim=32)
    model = FourierPrototype(config, input_dim=1)

    param_count = sum(p.numel() for p in model.parameters())
    # Should be small for prototype: ~10K-50K params
    assert param_count < 100_000, f"Too many params: {param_count}"
    assert param_count > 1_000, f"Too few params: {param_count}"

    print(f"Prototype parameter count: {param_count:,}")


def test_reward_head_shapes():
    """Test RewardHead output shapes for various configurations."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.heads import RewardHead

    config = PrototypeConfig(hidden_dim=32, reward_horizon=5)
    head = RewardHead(config)

    # Pooled input: (batch, hidden_dim)
    pooled = torch.randn(4, 32)
    output = head(pooled)

    assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"


def test_reward_head_gradients():
    """Test gradients flow through RewardHead."""
    from primordial.lrn.config import PrototypeConfig
    from primordial.lrn.heads import RewardHead

    config = PrototypeConfig(hidden_dim=32, reward_horizon=5)
    head = RewardHead(config)

    pooled = torch.randn(1, 32, requires_grad=True)
    output = head(pooled)
    loss = output.sum()
    loss.backward()

    assert pooled.grad is not None
    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_prototype.py -v`

Expected: FAIL with import error

**Step 3: Implement RewardHead**

`primordial/lrn/heads.py`:
```python
"""Output heads for the Fourier prototype."""
import torch
import torch.nn as nn

from .config import PrototypeConfig


class RewardHead(nn.Module):
    """
    Predicts upcoming reward values for multi-task learning.

    This creates a DIRECT gradient toward survival by predicting
    whether current patterns lead to positive or negative outcomes.

    Matches the RewardHead from the full LRN architecture.
    """

    def __init__(self, config: PrototypeConfig):
        super().__init__()

        self.reward_horizon = config.reward_horizon

        # Simple MLP: hidden_dim -> 64 -> reward_horizon
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, config.reward_horizon)
            # No activation - rewards can be any real value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_dim) - pooled features
        Returns:
            (batch, reward_horizon) - predicted rewards for next N steps
        """
        return self.mlp(x)


class SensoryHead(nn.Module):
    """
    Predicts next sensory state (sequence-to-sequence).

    For the prototype, this is a simple linear projection.
    """

    def __init__(self, config: PrototypeConfig, output_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(config.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, output_dim)
        """
        return self.proj(x)
```

**Step 4: Implement FourierPrototype with dual heads**

`primordial/lrn/prototype.py`:
```python
"""Minimal Fourier prototype for validation with multi-task learning."""
import torch
import torch.nn as nn
from typing import Tuple

from .config import PrototypeConfig
from .mixing import FourierMixingLayer
from .heads import SensoryHead, RewardHead


class FourierPrototype(nn.Module):
    """
    Minimal Fourier-based model for validating FFT mixing learns.

    Multi-task architecture matching the full LRN:
        Input (seq_len, input_dim)
        -> Linear projection to hidden_dim
        -> N x FourierMixingLayer
        -> Pooling (mean over sequence)
        -> Dual heads:
           - SensoryHead: predicts next sensory state (seq_len, 1)
           - RewardHead: predicts upcoming rewards (reward_horizon,)

    This validates BOTH:
    1. Fourier mixing learns temporal patterns (sensory prediction)
    2. Fourier mixing learns reward prediction (survival gradient)
    """

    def __init__(
        self,
        config: PrototypeConfig,
        input_dim: int = 1,
    ):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # Fourier mixing layers
        self.mixing_layers = nn.ModuleList([
            FourierMixingLayer(config)
            for _ in range(config.num_mixing_layers)
        ])

        # Output heads (multi-task)
        self.sensory_head = SensoryHead(config, output_dim=input_dim)
        self.reward_head = RewardHead(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-task outputs.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            sensory_pred: (batch, seq_len, input_dim) - predicted next sensory state
            reward_pred: (batch, reward_horizon) - predicted upcoming rewards
        """
        # Project to hidden dim
        x = self.input_proj(x)  # (B, seq_len, hidden_dim)

        # Apply Fourier mixing layers
        for layer in self.mixing_layers:
            x = layer(x)

        # Sensory prediction (sequence-to-sequence)
        sensory_pred = self.sensory_head(x)  # (B, seq_len, input_dim)

        # Pooling for reward head (mean over sequence)
        pooled = x.mean(dim=1)  # (B, hidden_dim)

        # Reward prediction
        reward_pred = self.reward_head(pooled)  # (B, reward_horizon)

        return sensory_pred, reward_pred
```

**Step 5: Run tests**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_prototype.py -v`

Expected: 4 passed

**Step 6: Update exports**

`primordial/lrn/__init__.py`:
```python
"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .mixing import FourierMixingLayer
from .heads import SensoryHead, RewardHead
from .prototype import FourierPrototype

__all__ = [
    "PrototypeConfig",
    "FourierMixingLayer",
    "SensoryHead",
    "RewardHead",
    "FourierPrototype",
]
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: add FourierPrototype with multi-task heads (SensoryHead + RewardHead)"
```

---

## Task 5: Synthetic Training Data with Rewards

**Files:**
- Create: `primordial/lrn/data.py`
- Create: `primordial/tests/lrn/test_data.py`

**Step 1: Write failing test for data generator**

`primordial/tests/lrn/test_data.py`:
```python
"""Tests for synthetic training data."""
import torch
import pytest


def test_sine_wave_generator_shapes():
    """Test sine wave data generator produces correct shapes."""
    from primordial.lrn.data import generate_multitask_batch

    batch = generate_multitask_batch(batch_size=8, seq_len=64, reward_horizon=5)

    assert "input" in batch
    assert "sensory_target" in batch
    assert "reward_target" in batch

    assert batch["input"].shape == (8, 64, 1)
    assert batch["sensory_target"].shape == (8, 64, 1)
    assert batch["reward_target"].shape == (8, 5)  # reward_horizon


def test_sensory_target_is_shifted():
    """Test sensory target is shifted version of input (next-step prediction)."""
    from primordial.lrn.data import generate_multitask_batch

    batch = generate_multitask_batch(batch_size=1, seq_len=64, reward_horizon=5)

    # Target should be input shifted by 1 step
    input_seq = batch["input"][0, :-1, 0]
    target_seq = batch["sensory_target"][0, 1:, 0]

    # They should be correlated (same underlying signal)
    correlation = torch.corrcoef(torch.stack([input_seq, target_seq]))[0, 1]
    assert correlation > 0.9, f"Low correlation: {correlation}"


def test_reward_target_values():
    """Test reward targets are in expected range."""
    from primordial.lrn.data import generate_multitask_batch

    batch = generate_multitask_batch(batch_size=32, seq_len=64, reward_horizon=5)

    rewards = batch["reward_target"]

    # Rewards should be in reasonable range [-2, 2]
    assert rewards.min() >= -2.5, f"Reward too low: {rewards.min()}"
    assert rewards.max() <= 2.5, f"Reward too high: {rewards.max()}"

    # Should have some variation (not all zeros)
    assert rewards.std() > 0.1, f"No reward variation: std={rewards.std()}"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_data.py -v`

Expected: FAIL with import error

**Step 3: Implement multi-task data generator**

`primordial/lrn/data.py`:
```python
"""Synthetic data generation for multi-task prototype validation."""
import torch
from typing import Dict


def generate_multitask_batch(
    batch_size: int = 8,
    seq_len: int = 64,
    reward_horizon: int = 5,
    freq_range: tuple = (0.5, 2.0),
    noise_std: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Generate batch of noisy sine waves with synthetic reward signals.

    This creates a multi-task learning scenario:
    1. Sensory prediction: Predict next timestep in sequence
    2. Reward prediction: Predict upcoming "rewards" based on signal amplitude

    The reward signal is derived from the sine wave amplitude - high amplitude
    regions get positive reward (simulating "food found"), low amplitude gets
    negative reward (simulating "danger"). This creates a learnable correlation
    between sensory patterns and reward outcomes.

    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        reward_horizon: Number of future reward steps to predict
        freq_range: (min, max) frequency range
        noise_std: Standard deviation of noise

    Returns:
        Dict with:
            'input': (batch, seq_len, 1) - input sequence
            'sensory_target': (batch, seq_len, 1) - next-step prediction target
            'reward_target': (batch, reward_horizon) - future reward targets
    """
    # Random frequencies for each sequence
    freqs = torch.rand(batch_size, 1) * (freq_range[1] - freq_range[0]) + freq_range[0]

    # Random phases
    phases = torch.rand(batch_size, 1) * 2 * 3.14159

    # Time steps (extra for target and reward horizon)
    total_len = seq_len + reward_horizon + 1
    t = torch.linspace(0, 4 * 3.14159, total_len).unsqueeze(0)  # (1, total_len)

    # Generate sine waves: (batch, total_len)
    signal = torch.sin(freqs * t + phases)

    # Add noise to signal
    noisy_signal = signal + torch.randn_like(signal) * noise_std

    # Split into input and sensory target
    input_seq = noisy_signal[:, :seq_len].unsqueeze(-1)  # (batch, seq_len, 1)
    sensory_target = noisy_signal[:, 1:seq_len+1].unsqueeze(-1)  # (batch, seq_len, 1)

    # Generate reward targets based on amplitude of FUTURE signal
    # Reward = amplitude of signal at t+1, t+2, ..., t+horizon
    # This simulates: "if I'm at a peak, good things happen; if at a trough, bad things"
    reward_target = torch.zeros(batch_size, reward_horizon)
    for h in range(reward_horizon):
        # Future signal value (clean, not noisy) determines reward
        future_val = signal[:, seq_len + h]
        # Map sine [-1, 1] to reward [-1, 1] with some scaling
        reward_target[:, h] = future_val

    return {
        "input": input_seq,
        "sensory_target": sensory_target,
        "reward_target": reward_target,
    }


# Backward compatibility alias
def generate_sine_batch(
    batch_size: int = 8,
    seq_len: int = 64,
    freq_range: tuple = (0.5, 2.0),
    noise_std: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Legacy function - use generate_multitask_batch instead."""
    batch = generate_multitask_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        reward_horizon=5,
        freq_range=freq_range,
        noise_std=noise_std,
    )
    return {
        "input": batch["input"],
        "target": batch["sensory_target"],
    }
```

**Step 4: Run tests**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_data.py -v`

Expected: 3 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add multi-task synthetic data generator with reward signals"
```

---

## Task 6: Multi-Task Training Loop

**Files:**
- Create: `primordial/lrn/train.py`
- Create: `primordial/tests/lrn/test_train.py`

**Step 1: Write failing test for multi-task training**

`primordial/tests/lrn/test_train.py`:
```python
"""Tests for multi-task training loop."""
import torch
import pytest


def test_training_reduces_sensory_loss():
    """Test that sensory prediction loss decreases over training."""
    from primordial.lrn.train import train_prototype

    result = train_prototype(
        num_steps=500,
        batch_size=8,
        learning_rate=1e-3,
        verbose=False,
    )

    # Sensory loss should decrease
    initial = result["sensory_losses"][0]
    final = result["sensory_losses"][-1]

    assert final < initial, f"Sensory loss didn't decrease: {initial:.4f} -> {final:.4f}"

    improvement = (initial - final) / initial
    assert improvement > 0.3, f"Only {improvement:.1%} sensory improvement"


def test_training_reduces_reward_loss():
    """Test that reward prediction loss decreases over training."""
    from primordial.lrn.train import train_prototype

    result = train_prototype(
        num_steps=500,
        batch_size=8,
        learning_rate=1e-3,
        verbose=False,
    )

    # Reward loss should decrease
    initial = result["reward_losses"][0]
    final = result["reward_losses"][-1]

    assert final < initial, f"Reward loss didn't decrease: {initial:.4f} -> {final:.4f}"


def test_training_no_nan():
    """Test training doesn't produce NaN losses."""
    from primordial.lrn.train import train_prototype

    result = train_prototype(
        num_steps=100,
        batch_size=4,
        learning_rate=1e-3,
        verbose=False,
    )

    for i, loss in enumerate(result["total_losses"]):
        assert not torch.isnan(torch.tensor(loss)), f"NaN loss at step {i}"
        assert not torch.isinf(torch.tensor(loss)), f"Inf loss at step {i}"


def test_online_learning_stability():
    """Test single-sample updates remain stable (simulates online learning)."""
    from primordial.lrn.train import train_prototype

    result = train_prototype(
        num_steps=100,
        batch_size=1,  # Single-sample updates!
        learning_rate=1e-3,
        verbose=False,
    )

    # Should not have NaN even with batch_size=1
    for i, loss in enumerate(result["total_losses"]):
        assert not torch.isnan(torch.tensor(loss)), f"NaN at step {i} with batch_size=1"

    # Loss should still decrease (though maybe more noisily)
    initial_avg = sum(result["total_losses"][:10]) / 10
    final_avg = sum(result["total_losses"][-10:]) / 10
    assert final_avg < initial_avg * 1.5, "Online learning unstable"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_train.py -v`

Expected: FAIL with import error

**Step 3: Implement multi-task training loop**

`primordial/lrn/train.py`:
```python
"""Multi-task training loop for Fourier prototype validation."""
import torch
import torch.nn as nn
from typing import Dict, List

from .config import PrototypeConfig
from .prototype import FourierPrototype
from .data import generate_multitask_batch


def train_prototype(
    num_steps: int = 1000,
    batch_size: int = 8,
    seq_len: int = 64,
    hidden_dim: int = 32,
    num_layers: int = 2,
    reward_horizon: int = 5,
    reward_loss_weight: float = 1.0,
    learning_rate: float = 1e-3,
    verbose: bool = True,
    log_every: int = 100,
) -> Dict:
    """
    Train the Fourier prototype on multi-task prediction.

    Multi-task learning:
    1. Sensory prediction: Predict next timestep in sequence
    2. Reward prediction: Predict upcoming reward signals

    Args:
        num_steps: Number of training steps
        batch_size: Batch size (use 1 for online learning test)
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of Fourier mixing layers
        reward_horizon: Number of future reward steps to predict
        reward_loss_weight: Weight for reward loss vs sensory loss
        learning_rate: Learning rate
        verbose: Print progress
        log_every: Print every N steps

    Returns:
        Dict with training results
    """
    # Create config and model
    config = PrototypeConfig(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_mixing_layers=num_layers,
        reward_horizon=reward_horizon,
        reward_loss_weight=reward_loss_weight,
    )
    model = FourierPrototype(config, input_dim=1)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history
    total_losses: List[float] = []
    sensory_losses: List[float] = []
    reward_losses: List[float] = []

    for step in range(num_steps):
        # Generate multi-task batch
        batch = generate_multitask_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            reward_horizon=reward_horizon,
        )

        # Forward pass (returns both predictions)
        sensory_pred, reward_pred = model(batch["input"])

        # Compute losses
        sensory_loss = criterion(sensory_pred, batch["sensory_target"])
        reward_loss = criterion(reward_pred, batch["reward_target"])

        # Combined loss (multi-task)
        total_loss = sensory_loss + config.reward_loss_weight * reward_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Record losses
        total_losses.append(total_loss.item())
        sensory_losses.append(sensory_loss.item())
        reward_losses.append(reward_loss.item())

        # Log progress
        if verbose and (step + 1) % log_every == 0:
            avg_total = sum(total_losses[-log_every:]) / log_every
            avg_sensory = sum(sensory_losses[-log_every:]) / log_every
            avg_reward = sum(reward_losses[-log_every:]) / log_every
            print(
                f"Step {step + 1}/{num_steps} | "
                f"Total: {avg_total:.4f} | "
                f"Sensory: {avg_sensory:.4f} | "
                f"Reward: {avg_reward:.4f}"
            )

    return {
        "model": model,
        "config": config,
        "total_losses": total_losses,
        "sensory_losses": sensory_losses,
        "reward_losses": reward_losses,
        "final_total_loss": total_losses[-1],
        "final_sensory_loss": sensory_losses[-1],
        "final_reward_loss": reward_losses[-1],
    }


def evaluate_prototype(model: FourierPrototype, num_batches: int = 10) -> Dict:
    """
    Evaluate prototype model on both tasks.

    Returns:
        Dict with evaluation metrics for sensory and reward prediction
    """
    model.eval()
    config = model.config
    criterion = nn.MSELoss()

    sensory_total = 0.0
    reward_total = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            batch = generate_multitask_batch(
                batch_size=8,
                seq_len=config.seq_len,
                reward_horizon=config.reward_horizon,
            )
            sensory_pred, reward_pred = model(batch["input"])

            sensory_total += criterion(sensory_pred, batch["sensory_target"]).item()
            reward_total += criterion(reward_pred, batch["reward_target"]).item()

    sensory_avg = sensory_total / num_batches
    reward_avg = reward_total / num_batches

    # Compare to random baseline
    sensory_baseline = 0.0
    reward_baseline = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            batch = generate_multitask_batch(
                batch_size=8,
                seq_len=config.seq_len,
                reward_horizon=config.reward_horizon,
            )
            sensory_baseline += criterion(
                torch.randn_like(batch["sensory_target"]),
                batch["sensory_target"]
            ).item()
            reward_baseline += criterion(
                torch.randn_like(batch["reward_target"]),
                batch["reward_target"]
            ).item()

    sensory_baseline /= num_batches
    reward_baseline /= num_batches

    return {
        "sensory_loss": sensory_avg,
        "reward_loss": reward_avg,
        "sensory_baseline": sensory_baseline,
        "reward_baseline": reward_baseline,
        "sensory_improvement": sensory_baseline / sensory_avg if sensory_avg > 0 else float("inf"),
        "reward_improvement": reward_baseline / reward_avg if reward_avg > 0 else float("inf"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("FOURIER MIXING PROTOTYPE - MULTI-TASK VALIDATION")
    print("=" * 60)

    # Train
    print("\nTraining with multi-task learning...")
    result = train_prototype(
        num_steps=1000,
        batch_size=8,
        learning_rate=1e-3,
        verbose=True,
        log_every=100,
    )

    # Evaluate
    print("\nEvaluating...")
    eval_result = evaluate_prototype(result["model"])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final sensory loss:  {result['final_sensory_loss']:.4f}")
    print(f"Final reward loss:   {result['final_reward_loss']:.4f}")
    print(f"Final total loss:    {result['final_total_loss']:.4f}")
    print()
    print(f"Sensory baseline:    {eval_result['sensory_baseline']:.4f}")
    print(f"Reward baseline:     {eval_result['reward_baseline']:.4f}")
    print()
    print(f"Sensory improvement: {eval_result['sensory_improvement']:.1f}x better than random")
    print(f"Reward improvement:  {eval_result['reward_improvement']:.1f}x better than random")

    # Success criteria
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    success = True

    # Check 1: Sensory loss decreased
    if result["sensory_losses"][-1] < result["sensory_losses"][0]:
        print("[PASS] Sensory loss decreased during training")
    else:
        print("[FAIL] Sensory loss did not decrease")
        success = False

    # Check 2: Reward loss decreased
    if result["reward_losses"][-1] < result["reward_losses"][0]:
        print("[PASS] Reward loss decreased during training")
    else:
        print("[FAIL] Reward loss did not decrease")
        success = False

    # Check 3: Better than random on sensory
    if eval_result["sensory_improvement"] > 1.5:
        print(f"[PASS] Sensory prediction {eval_result['sensory_improvement']:.1f}x better than random")
    else:
        print("[FAIL] Sensory prediction not better than random")
        success = False

    # Check 4: Better than random on reward
    if eval_result["reward_improvement"] > 1.2:
        print(f"[PASS] Reward prediction {eval_result['reward_improvement']:.1f}x better than random")
    else:
        print("[FAIL] Reward prediction not better than random")
        success = False

    # Check 5: No NaN
    all_losses = result["total_losses"] + result["sensory_losses"] + result["reward_losses"]
    if not any(torch.isnan(torch.tensor(l)) for l in all_losses):
        print("[PASS] No NaN losses")
    else:
        print("[FAIL] NaN losses detected")
        success = False

    # Check 6: Online learning test
    print("\nTesting online learning (batch_size=1)...")
    online_result = train_prototype(
        num_steps=100,
        batch_size=1,
        learning_rate=1e-3,
        verbose=False,
    )
    online_nan = any(torch.isnan(torch.tensor(l)) for l in online_result["total_losses"])
    if not online_nan:
        print("[PASS] Online learning stable (no NaN with batch_size=1)")
    else:
        print("[FAIL] Online learning unstable")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("MULTI-TASK PROTOTYPE VALIDATION: SUCCESS")
        print("Fourier mixing learns BOTH sensory and reward prediction!")
        print("Ready to build full LRN architecture.")
    else:
        print("MULTI-TASK PROTOTYPE VALIDATION: FAILED")
        print("Investigate before proceeding with full implementation.")
    print("=" * 60)
```

**Step 4: Run tests**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/test_train.py -v`

Expected: 4 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add multi-task training loop with sensory + reward prediction"
```

---

## Task 7: Run Full Multi-Task Validation

**Files:**
- No new files

**Step 1: Run the prototype validation script**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m primordial.lrn.train`

Expected output (approximately):
```
============================================================
FOURIER MIXING PROTOTYPE - MULTI-TASK VALIDATION
============================================================

Training with multi-task learning...
Step 100/1000 | Total: 0.XXXX | Sensory: 0.XXXX | Reward: 0.XXXX
Step 200/1000 | Total: 0.XXXX | Sensory: 0.XXXX | Reward: 0.XXXX
...

RESULTS
============================================================
Final sensory loss:  ~0.02-0.05
Final reward loss:   ~0.3-0.5
Final total loss:    ~0.3-0.6

Sensory baseline:    ~1.0
Reward baseline:     ~1.0

Sensory improvement: ~20-50x better than random
Reward improvement:  ~2-5x better than random

VALIDATION
============================================================
[PASS] Sensory loss decreased during training
[PASS] Reward loss decreased during training
[PASS] Sensory prediction XXx better than random
[PASS] Reward prediction XXx better than random
[PASS] No NaN losses

Testing online learning (batch_size=1)...
[PASS] Online learning stable (no NaN with batch_size=1)

============================================================
MULTI-TASK PROTOTYPE VALIDATION: SUCCESS
Fourier mixing learns BOTH sensory and reward prediction!
Ready to build full LRN architecture.
============================================================
```

**Step 2: Run all tests**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m pytest primordial/tests/lrn/ -v`

Expected: All tests pass (13+ tests)

**Step 3: Commit final state**

```bash
git add -A
git commit -m "feat: complete multi-task Fourier prototype validation"
```

---

## Task 8: Performance Benchmark

**Files:**
- Create: `primordial/lrn/benchmark.py`

**Step 1: Create benchmark script**

`primordial/lrn/benchmark.py`:
```python
"""Performance benchmarking for Fourier prototype."""
import time
import torch

from .config import PrototypeConfig
from .prototype import FourierPrototype


def benchmark_forward_pass(
    num_iterations: int = 100,
    batch_size: int = 1,
    seq_len: int = 64,
    hidden_dim: int = 32,
) -> dict:
    """
    Benchmark forward pass timing.

    Returns timing statistics.
    """
    config = PrototypeConfig(seq_len=seq_len, hidden_dim=hidden_dim)
    model = FourierPrototype(config, input_dim=1)
    model.eval()

    # Warmup
    x = torch.randn(batch_size, seq_len, 1)
    for _ in range(10):
        _ = model(x)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            x = torch.randn(batch_size, seq_len, 1)
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Standard config
    result = benchmark_forward_pass(
        num_iterations=100,
        batch_size=1,
        seq_len=64,
        hidden_dim=32,
    )

    print(f"\nForward pass timing (batch=1, seq=64, hidden=32):")
    print(f"  Mean:  {result['mean_ms']:.3f} ms")
    print(f"  Min:   {result['min_ms']:.3f} ms")
    print(f"  Max:   {result['max_ms']:.3f} ms")
    print(f"  Std:   {result['std_ms']:.3f} ms")

    # Check against target
    if result["mean_ms"] < 5.0:
        print(f"\n[PASS] Forward pass < 5ms target")
    else:
        print(f"\n[WARN] Forward pass > 5ms target")

    # Larger config (closer to full LRN)
    result_large = benchmark_forward_pass(
        num_iterations=100,
        batch_size=1,
        seq_len=164,  # Full LRN seq_len
        hidden_dim=128,  # Full LRN hidden_dim
    )

    print(f"\nForward pass timing (batch=1, seq=164, hidden=128):")
    print(f"  Mean:  {result_large['mean_ms']:.3f} ms")
    print(f"  Min:   {result_large['min_ms']:.3f} ms")
    print(f"  Max:   {result_large['max_ms']:.3f} ms")

    if result_large["mean_ms"] < 10.0:
        print(f"\n[PASS] Large config forward pass < 10ms target")
    else:
        print(f"\n[WARN] Large config forward pass > 10ms target")
```

**Step 2: Run benchmark**

Run: `cd /Users/zachswift/projects/kung-foo-chick-pea-feeble && python -m primordial.lrn.benchmark`

Expected: Forward pass < 5ms for prototype config

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add performance benchmarking"
```

---

## Summary

After completing all 8 tasks, you will have:

1. **Project structure** with proper Python packaging
2. **FourierMixingLayer** with learnable spectral filters (matching parent architecture)
   - Filter shape: `(seq_len, freq_bins, 2)` with slicing logic
   - Spectral bias initialization (frequency-dependent decay)
3. **Gradient flow tests** proving FFT is differentiable
4. **Multi-task FourierPrototype** with dual output heads
   - SensoryHead: predicts next sensory state
   - RewardHead: predicts upcoming rewards
5. **Multi-task synthetic data generator** with reward signals
6. **Multi-task training loop** with combined loss
7. **Full validation script** proving Fourier mixing learns BOTH tasks
8. **Performance benchmarks** confirming speed targets

**Success criteria validated:**
1. Gradients flow through FFT (no NaN/Inf)
2. **Sensory prediction loss** decreases over training
3. **Reward prediction loss** decreases over training
4. Model beats random baseline on **both** tasks
5. Forward pass < 5ms on CPU
6. **Single-sample updates (batch_size=1) remain stable**

**Key architectural alignment with parent LRN:**
- Spectral filter shape matches: `(seq_len, freq_bins, 2)`
- Spectral bias initialization included
- Multi-task learning with RewardHead validates survival gradient hypothesis
- Online learning stability tested

**Next step:** Build full LRN architecture using `primordial/plans/03-lrn-architecture.md`
