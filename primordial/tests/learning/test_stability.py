"""
Tests for stability measures in online learning.

Tests gradient clipping, EMA, gradient accumulation, and monitoring.
"""

import pytest
import torch
import torch.nn as nn

from primordial.learning.stability import (
    GradientClipper,
    GradientAccumulator,
    ExponentialMovingAverage,
    GradientMonitor,
)


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def compute_total_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def test_gradient_clipping_norm():
    """Test gradient clipping by norm prevents explosion."""
    model = SimpleModel()
    clipper = GradientClipper(clip_type='norm', max_norm=1.0)

    # Create large gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100

    norm_before = compute_total_norm(model)
    assert norm_before > 1.0

    grad_norm = clipper.clip(model)
    norm_after = compute_total_norm(model)
    assert norm_after <= 1.01  # Allow small tolerance


def test_gradient_clipping_value():
    """Test gradient clipping by value."""
    model = SimpleModel()
    clipper = GradientClipper(clip_type='value', max_value=5.0)

    # Create large gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100

    clipper.clip(model)

    # Check all gradient values are within bounds
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(torch.abs(p.grad) <= 5.0)


def test_gradient_accumulator():
    """Test gradient accumulator counts correctly."""
    accumulator = GradientAccumulator(accumulation_steps=4)

    # First 3 steps should not update
    assert not accumulator.should_update()
    assert not accumulator.should_update()
    assert not accumulator.should_update()

    # 4th step should update
    assert accumulator.should_update()

    # Then cycle repeats
    assert not accumulator.should_update()


def test_gradient_accumulator_scale_loss():
    """Test gradient accumulator scales loss correctly."""
    accumulator = GradientAccumulator(accumulation_steps=4)

    loss = torch.tensor(10.0)
    scaled = accumulator.scale_loss(loss)

    assert torch.isclose(scaled, torch.tensor(2.5))


def test_ema_stability():
    """Test EMA provides smoother weights than raw updates."""
    model = SimpleModel()
    ema = ExponentialMovingAverage(model, decay=0.99)

    # Get initial parameter
    initial_param = list(model.parameters())[0].data.clone()

    # Make several large updates
    for _ in range(10):
        for p in model.parameters():
            p.data += torch.randn_like(p) * 0.5
        ema.update()

    # EMA weights should be closer to initial than current
    ema.apply_shadow()
    shadow_param = list(model.parameters())[0].data.clone()
    ema.restore()
    current_param = list(model.parameters())[0].data.clone()

    shadow_distance = torch.norm(shadow_param - initial_param)
    current_distance = torch.norm(current_param - initial_param)

    assert shadow_distance < current_distance


def test_ema_apply_restore():
    """Test EMA apply and restore works correctly."""
    model = SimpleModel()
    ema = ExponentialMovingAverage(model, decay=0.99)

    # Get original weights
    original_weights = {
        name: p.data.clone() for name, p in model.named_parameters()
    }

    # Update model and EMA
    for p in model.parameters():
        p.data += torch.randn_like(p) * 0.1
    ema.update()

    # Apply shadow
    ema.apply_shadow()
    shadow_weights = {
        name: p.data.clone() for name, p in model.named_parameters()
    }

    # Restore
    ema.restore()
    restored_weights = {
        name: p.data.clone() for name, p in model.named_parameters()
    }

    # Shadow weights should be different from restored
    for name in shadow_weights:
        assert not torch.equal(shadow_weights[name], restored_weights[name])

    # Restored should NOT equal original (model was updated)
    for name in original_weights:
        assert not torch.equal(original_weights[name], restored_weights[name])


def test_gradient_monitor_stats():
    """Test gradient monitor computes statistics correctly."""
    model = SimpleModel()
    monitor = GradientMonitor(window_size=10)

    # Record some gradients
    for i in range(5):
        for p in model.parameters():
            p.grad = torch.randn_like(p) * (1 + i * 0.1)
        monitor.record(model)

    stats = monitor.get_statistics()

    assert 'grad_norm_mean' in stats
    assert 'grad_norm_std' in stats
    assert 'grad_var_mean' in stats
    assert 'is_stable' in stats

    assert stats['grad_norm_mean'] > 0
    assert stats['is_stable'] is True


def test_gradient_monitor_empty():
    """Test gradient monitor handles empty state."""
    monitor = GradientMonitor()

    stats = monitor.get_statistics()

    assert stats['grad_norm_mean'] == 0.0
    assert stats['is_stable'] is True


def test_gradient_monitor_window():
    """Test gradient monitor respects window size."""
    monitor = GradientMonitor(window_size=5)
    model = SimpleModel()

    # Record 10 gradients
    for i in range(10):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        monitor.record(model)

    # Should only keep last 5
    assert len(monitor.grad_norms) == 5
    assert len(monitor.grad_vars) == 5
