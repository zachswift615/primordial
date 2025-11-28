"""Tests for learning metrics and visualization."""

import pytest
import torch

from primordial.learning.metrics import LearningMetrics, LearningVisualizer


def test_learning_metrics_record():
    """Test recording metrics."""
    metrics = LearningMetrics(log_interval=10)

    # Record a step
    metrics.record_step(
        prediction_loss=0.5,
        survival_reward=1.0,
        teaching_reward=0.5,
        total_reward=1.5,
        gradient_norm=0.1,
        modulation_factor=1.2,
        learning_rate=1e-4
    )

    assert metrics.step_count == 1
    assert len(metrics.prediction_losses) == 1
    assert metrics.prediction_losses[0] == 0.5
    assert metrics.rewards_survival[0] == 1.0
    assert metrics.rewards_teaching[0] == 0.5


def test_learning_metrics_summary():
    """Test getting summary statistics."""
    metrics = LearningMetrics(log_interval=10)

    # Record multiple steps
    for i in range(5):
        metrics.record_step(
            prediction_loss=float(i),
            survival_reward=1.0,
            teaching_reward=0.5,
            total_reward=1.5,
            gradient_norm=0.1,
            modulation_factor=1.2,
            learning_rate=1e-4
        )

    summary = metrics.get_summary()

    assert summary['step'] == 5
    assert summary['loss/prediction_mean'] == 2.0  # mean of [0,1,2,3,4]
    assert summary['reward/survival_mean'] == 1.0
    assert summary['reward/teaching_mean'] == 0.5
    assert summary['reward/total_mean'] == 1.5
    assert summary['training/gradient_norm_mean'] == 0.1
    assert summary['training/modulation_factor_mean'] == 1.2
    assert summary['training/learning_rate'] == 1e-4


def test_learning_metrics_clear():
    """Test clearing metric buffers."""
    metrics = LearningMetrics(log_interval=10)

    # Record steps
    for i in range(5):
        metrics.record_step(
            prediction_loss=0.5,
            survival_reward=1.0,
            teaching_reward=0.5,
            total_reward=1.5,
            gradient_norm=0.1,
            modulation_factor=1.2,
            learning_rate=1e-4
        )

    assert len(metrics.prediction_losses) == 5

    # Clear buffers
    metrics.clear_buffers()

    assert len(metrics.prediction_losses) == 0
    assert len(metrics.rewards_survival) == 0
    assert len(metrics.rewards_teaching) == 0
    assert len(metrics.rewards_total) == 0
    assert len(metrics.gradient_norms) == 0
    assert len(metrics.modulation_factors) == 0
    assert len(metrics.learning_rates) == 0

    # Step count should be preserved
    assert metrics.step_count == 5


def test_learning_metrics_should_log():
    """Test should_log interval."""
    metrics = LearningMetrics(log_interval=5)

    # First 4 steps should not log
    for i in range(4):
        metrics.record_step(
            prediction_loss=0.5,
            survival_reward=1.0,
            teaching_reward=0.5,
            total_reward=1.5,
            gradient_norm=0.1,
            modulation_factor=1.2,
            learning_rate=1e-4
        )
        assert not metrics.should_log()

    # 5th step should log
    metrics.record_step(
        prediction_loss=0.5,
        survival_reward=1.0,
        teaching_reward=0.5,
        total_reward=1.5,
        gradient_norm=0.1,
        modulation_factor=1.2,
        learning_rate=1e-4
    )
    assert metrics.should_log()


def test_learning_metrics_empty_summary():
    """Test getting summary with no data."""
    metrics = LearningMetrics(log_interval=10)

    summary = metrics.get_summary()

    assert summary == {}


def test_learning_visualizer_init_no_backends():
    """Test initializing visualizer without backends."""
    viz = LearningVisualizer(use_tensorboard=False, use_wandb=False)

    assert not viz.use_tensorboard
    assert not viz.use_wandb
    assert viz.tb_writer is None
    assert viz.wandb is None


def test_learning_visualizer_log_metrics_no_backends():
    """Test logging metrics with no backends (should not crash)."""
    viz = LearningVisualizer(use_tensorboard=False, use_wandb=False)

    metrics = {
        'loss/prediction_mean': 0.5,
        'reward/total_mean': 1.5
    }

    # Should not crash even with no backends
    viz.log_metrics(metrics, step=1)


def test_learning_visualizer_log_predictions_no_backends():
    """Test logging predictions with no backends (should not crash)."""
    viz = LearningVisualizer(use_tensorboard=False, use_wandb=False)

    predicted = torch.randn(1, 10)
    actual = torch.randn(1, 10)

    # Should not crash even with no backends
    viz.log_model_predictions(predicted, actual, step=1)


def test_learning_visualizer_close_no_backends():
    """Test closing visualizer with no backends (should not crash)."""
    viz = LearningVisualizer(use_tensorboard=False, use_wandb=False)

    # Should not crash
    viz.close()


def test_learning_metrics_std_calculation():
    """Test standard deviation calculation in summary."""
    metrics = LearningMetrics(log_interval=10)

    # Record steps with varying losses
    losses = [1.0, 2.0, 3.0, 4.0, 5.0]
    for loss in losses:
        metrics.record_step(
            prediction_loss=loss,
            survival_reward=1.0,
            teaching_reward=0.5,
            total_reward=1.5,
            gradient_norm=0.1,
            modulation_factor=1.2,
            learning_rate=1e-4
        )

    summary = metrics.get_summary()

    # Mean should be 3.0
    assert summary['loss/prediction_mean'] == 3.0

    # Std should be sqrt(2.5) ≈ 1.58
    import numpy as np
    expected_std = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(summary['loss/prediction_std'] - expected_std) < 1e-6


def test_learning_metrics_multiple_record_clear_cycles():
    """Test multiple record-clear cycles."""
    metrics = LearningMetrics(log_interval=10)

    # Cycle 1
    for i in range(5):
        metrics.record_step(
            prediction_loss=1.0,
            survival_reward=1.0,
            teaching_reward=0.5,
            total_reward=1.5,
            gradient_norm=0.1,
            modulation_factor=1.2,
            learning_rate=1e-4
        )

    assert metrics.step_count == 5
    summary1 = metrics.get_summary()
    assert summary1['loss/prediction_mean'] == 1.0

    metrics.clear_buffers()

    # Cycle 2
    for i in range(5):
        metrics.record_step(
            prediction_loss=2.0,
            survival_reward=2.0,
            teaching_reward=1.0,
            total_reward=3.0,
            gradient_norm=0.2,
            modulation_factor=1.5,
            learning_rate=5e-5
        )

    assert metrics.step_count == 10
    summary2 = metrics.get_summary()
    assert summary2['loss/prediction_mean'] == 2.0
    assert summary2['reward/survival_mean'] == 2.0


def test_learning_metrics_learning_rate_tracking():
    """Test that learning rate is tracked correctly."""
    metrics = LearningMetrics(log_interval=10)

    # Record steps with changing learning rate
    learning_rates = [1e-4, 9e-5, 8e-5, 7e-5, 6e-5]
    for lr in learning_rates:
        metrics.record_step(
            prediction_loss=0.5,
            survival_reward=1.0,
            teaching_reward=0.5,
            total_reward=1.5,
            gradient_norm=0.1,
            modulation_factor=1.2,
            learning_rate=lr
        )

    summary = metrics.get_summary()

    # Should report the most recent learning rate
    assert summary['training/learning_rate'] == 6e-5


def test_learning_metrics_zero_values():
    """Test recording zero values doesn't cause issues."""
    metrics = LearningMetrics(log_interval=10)

    # Record steps with all zeros
    for i in range(5):
        metrics.record_step(
            prediction_loss=0.0,
            survival_reward=0.0,
            teaching_reward=0.0,
            total_reward=0.0,
            gradient_norm=0.0,
            modulation_factor=0.0,
            learning_rate=0.0
        )

    summary = metrics.get_summary()

    assert summary['loss/prediction_mean'] == 0.0
    assert summary['loss/prediction_std'] == 0.0
    assert summary['reward/survival_mean'] == 0.0
    assert summary['training/gradient_norm_mean'] == 0.0
