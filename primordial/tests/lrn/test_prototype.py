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
