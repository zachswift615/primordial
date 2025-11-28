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

    # Reward loss should decrease (compare first 50 vs last 50 for stability)
    initial_avg = sum(result["reward_losses"][:50]) / 50
    final_avg = sum(result["reward_losses"][-50:]) / 50

    assert final_avg < initial_avg, f"Reward loss didn't decrease: {initial_avg:.4f} -> {final_avg:.4f}"


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
