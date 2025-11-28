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

    # Use low noise for this test to ensure clear correlation
    batch = generate_multitask_batch(batch_size=1, seq_len=64, reward_horizon=5, noise_std=0.01)

    # Target[t] should equal Input[t+1] (shifted by 1 step)
    # So Input[1:] should correlate highly with Target[:-1]
    input_shifted = batch["input"][0, 1:, 0]  # input positions 1 to seq_len-1
    target_early = batch["sensory_target"][0, :-1, 0]  # target positions 0 to seq_len-2

    # These should be the same position in the original noisy signal
    correlation = torch.corrcoef(torch.stack([input_shifted, target_early]))[0, 1]
    assert correlation > 0.99, f"Low correlation: {correlation}"


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
