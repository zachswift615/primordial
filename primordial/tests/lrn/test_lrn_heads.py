"""Tests for LRN output heads."""
import torch
import pytest
from primordial.lrn.lrn_config import LRNConfig
from primordial.lrn.lrn_heads import PredictionHead, LRNRewardHead, ActionHead


# ============================================================================
# PredictionHead Tests
# ============================================================================


def test_prediction_head_output_shape():
    """Test PredictionHead produces correct output shape (B, 343)."""
    config = LRNConfig()
    head = PredictionHead(config)

    # Input: (batch, 3*hidden_dim) = (batch, 384)
    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    output = head(x)

    # Expected: (batch, 343)
    expected_shape = (batch_size, config.total_sensory_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_prediction_head_split_prediction():
    """Test split_prediction returns correct shapes for each modality."""
    config = LRNConfig()
    head = PredictionHead(config)

    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    pred = head(x)

    # Split prediction into modalities
    split = head.split_prediction(pred, config)

    # Check all keys present
    assert 'vision' in split
    assert 'audio' in split
    assert 'proprio' in split
    assert 'touch' in split

    # Check vision shape: (batch, 32, 4)
    assert split['vision'].shape == (batch_size, *config.vision_shape), \
        f"Vision shape mismatch: {split['vision'].shape}"

    # Check audio shape: (batch, 100, 2)
    assert split['audio'].shape == (batch_size, *config.audio_shape), \
        f"Audio shape mismatch: {split['audio'].shape}"

    # Check proprio shape: (batch, 7)
    assert split['proprio'].shape == (batch_size, config.proprio_dim), \
        f"Proprio shape mismatch: {split['proprio'].shape}"

    # Check touch shape: (batch, 8)
    assert split['touch'].shape == (batch_size, config.touch_dim), \
        f"Touch shape mismatch: {split['touch'].shape}"


def test_prediction_head_split_prediction_values():
    """Test split_prediction correctly slices the prediction tensor."""
    config = LRNConfig()
    head = PredictionHead(config)

    batch_size = 2
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    pred = head(x)

    split = head.split_prediction(pred, config)

    # Reconstruct from split and verify it matches original
    vision_flat = split['vision'].reshape(batch_size, -1)
    audio_flat = split['audio'].reshape(batch_size, -1)
    proprio_flat = split['proprio']
    touch_flat = split['touch']

    reconstructed = torch.cat([vision_flat, audio_flat, proprio_flat, touch_flat], dim=1)

    assert reconstructed.shape == pred.shape
    assert torch.allclose(reconstructed, pred), "Split prediction doesn't match original"


def test_prediction_head_batch_processing():
    """Test PredictionHead handles different batch sizes."""
    config = LRNConfig()
    head = PredictionHead(config)

    input_dim = 3 * config.hidden_dim

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, input_dim)
        output = head(x)
        assert output.shape == (batch_size, config.total_sensory_dim)


def test_prediction_head_gradient_flow():
    """Test gradients flow through PredictionHead."""
    config = LRNConfig()
    head = PredictionHead(config)

    batch_size = 2
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    output = head(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()


def test_prediction_head_output_dim():
    """Test PredictionHead output_dim attribute is correct."""
    config = LRNConfig()
    head = PredictionHead(config)

    assert head.output_dim == config.total_sensory_dim
    assert head.output_dim == 343


# ============================================================================
# LRNRewardHead Tests
# ============================================================================


def test_lrn_reward_head_output_shape():
    """Test LRNRewardHead produces correct output shape (B, 5)."""
    config = LRNConfig()
    head = LRNRewardHead(config)

    # Input: (batch, 3*hidden_dim) = (batch, 384)
    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    output = head(x)

    # Expected: (batch, reward_horizon) = (batch, 5)
    expected_shape = (batch_size, config.reward_horizon)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_lrn_reward_head_batch_processing():
    """Test LRNRewardHead handles different batch sizes."""
    config = LRNConfig()
    head = LRNRewardHead(config)

    input_dim = 3 * config.hidden_dim

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, input_dim)
        output = head(x)
        assert output.shape == (batch_size, config.reward_horizon)


def test_lrn_reward_head_gradient_flow():
    """Test gradients flow through LRNRewardHead."""
    config = LRNConfig()
    head = LRNRewardHead(config)

    batch_size = 2
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    output = head(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()


def test_lrn_reward_head_no_activation():
    """Test LRNRewardHead outputs raw values (no activation)."""
    config = LRNConfig()
    head = LRNRewardHead(config)

    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    output = head(x)

    # Output should be able to have any real value (positive or negative)
    # Just check it's not all zeros or all the same
    assert not torch.allclose(output, torch.zeros_like(output))


def test_lrn_reward_head_horizon_attribute():
    """Test LRNRewardHead stores reward_horizon correctly."""
    config = LRNConfig()
    head = LRNRewardHead(config)

    assert head.reward_horizon == config.reward_horizon
    assert head.reward_horizon == 5


# ============================================================================
# ActionHead Tests
# ============================================================================


def test_action_head_output_shape():
    """Test ActionHead produces correct output shape (B, 5)."""
    config = LRNConfig()
    head = ActionHead(config)

    # Input: (batch, 3*hidden_dim) = (batch, 384)
    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    output = head(x)

    # Expected: (batch, action_dim) = (batch, 5)
    expected_shape = (batch_size, config.action_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


def test_action_head_batch_processing():
    """Test ActionHead handles different batch sizes."""
    config = LRNConfig()
    head = ActionHead(config)

    input_dim = 3 * config.hidden_dim

    for batch_size in [1, 8, 16]:
        x = torch.randn(batch_size, input_dim)
        output = head(x)
        assert output.shape == (batch_size, config.action_dim)


def test_action_head_gradient_flow():
    """Test gradients flow through ActionHead."""
    config = LRNConfig()
    head = ActionHead(config)

    batch_size = 2
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    output = head(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()


def test_action_head_action_dim():
    """Test ActionHead outputs correct action dimension."""
    config = LRNConfig()
    head = ActionHead(config)

    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)
    output = head(x)

    # Should have 5 actions: thrust, torque, vocalize, freq, eat
    assert output.shape[1] == 5


# ============================================================================
# Integration Tests
# ============================================================================


def test_all_heads_with_same_input():
    """Test all heads can process the same pooled features."""
    config = LRNConfig()

    pred_head = PredictionHead(config)
    reward_head = LRNRewardHead(config)
    action_head = ActionHead(config)

    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)

    # All heads should work with same input
    pred_output = pred_head(x)
    reward_output = reward_head(x)
    action_output = action_head(x)

    assert pred_output.shape == (batch_size, 343)
    assert reward_output.shape == (batch_size, 5)
    assert action_output.shape == (batch_size, 5)


def test_heads_are_independent():
    """Test that heads produce independent outputs."""
    config = LRNConfig()

    pred_head = PredictionHead(config)
    reward_head = LRNRewardHead(config)
    action_head = ActionHead(config)

    batch_size = 4
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim)

    pred_output = pred_head(x)
    reward_output = reward_head(x)
    action_output = action_head(x)

    # Outputs should be different (different heads)
    # Check that they're not accidentally sharing parameters
    assert pred_head.mlp[0].weight.data_ptr() != reward_head.mlp[0].weight.data_ptr()
    assert reward_head.mlp[0].weight.data_ptr() != action_head.mlp[0].weight.data_ptr()


def test_heads_gradient_isolation():
    """Test that gradients to each head are isolated."""
    config = LRNConfig()

    pred_head = PredictionHead(config)
    reward_head = LRNRewardHead(config)
    action_head = ActionHead(config)

    batch_size = 2
    input_dim = 3 * config.hidden_dim
    x = torch.randn(batch_size, input_dim, requires_grad=True)

    # Forward pass through all heads
    pred_output = pred_head(x)
    reward_output = reward_head(x)
    action_output = action_head(x)

    # Backward through only prediction head
    pred_loss = pred_output.sum()
    pred_loss.backward()

    # Check that gradients exist for input
    assert x.grad is not None

    # Check that other heads' parameters don't have gradients
    for param in reward_head.parameters():
        assert param.grad is None

    for param in action_head.parameters():
        assert param.grad is None


def test_custom_config_values():
    """Test heads work with custom config values."""
    config = LRNConfig(
        hidden_dim=64,  # Smaller hidden dim
        pred_hidden_dim=128,
        action_hidden_dim=64,
        reward_horizon=3,
        action_dim=5
    )

    pred_head = PredictionHead(config)
    reward_head = LRNRewardHead(config)
    action_head = ActionHead(config)

    batch_size = 2
    input_dim = 3 * config.hidden_dim  # 192
    x = torch.randn(batch_size, input_dim)

    pred_output = pred_head(x)
    reward_output = reward_head(x)
    action_output = action_head(x)

    assert pred_output.shape == (batch_size, config.total_sensory_dim)
    assert reward_output.shape == (batch_size, 3)  # custom reward_horizon
    assert action_output.shape == (batch_size, 5)
