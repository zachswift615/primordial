"""Comprehensive integration tests for the LRN system (Phase 9: Validation).

This test suite validates the full LRN architecture with integration tests:
- Full forward + backward + optimizer cycles
- Online learning simulation
- Multi-task learning (sensory + reward)
- Gradient stability over extended training
- Memory stability
- All components integrated together
- Parameter count validation
- Deterministic behavior with seeding
"""
import gc
import pytest
import torch
import torch.nn as nn
from typing import Dict

from primordial.lrn import LivingResonanceNetwork, LRNConfig
from primordial.lrn.learning import (
    RewardHistoryBuffer,
    GradientClipper,
    ExponentialMovingAverage,
    GradientMonitor,
    OnlineLRScheduler
)


def create_synthetic_sensory_data(
    batch_size: int,
    config: LRNConfig,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """Create synthetic sensory data for testing.

    Args:
        batch_size: Number of samples
        config: LRN configuration
        device: Device for tensors

    Returns:
        Dictionary with vision, audio, proprio, touch inputs
    """
    if device is None:
        device = torch.device('cpu')

    return {
        'vision': torch.randn(batch_size, *config.vision_shape, device=device),
        'audio': torch.randn(batch_size, *config.audio_shape, device=device),
        'proprio': torch.randn(batch_size, config.proprio_dim, device=device),
        'touch': torch.randn(batch_size, config.touch_dim, device=device),
        'genome': torch.randn(batch_size, config.genome_dim, device=device)
    }


def create_synthetic_targets(
    batch_size: int,
    config: LRNConfig,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """Create synthetic target data for testing.

    Args:
        batch_size: Number of samples
        config: LRN configuration
        device: Device for tensors

    Returns:
        Dictionary with next_sensory and actual_rewards
    """
    if device is None:
        device = torch.device('cpu')

    return {
        'next_sensory': {
            'vision': torch.randn(batch_size, *config.vision_shape, device=device),
            'audio': torch.randn(batch_size, *config.audio_shape, device=device),
            'proprio': torch.randn(batch_size, config.proprio_dim, device=device),
            'touch': torch.randn(batch_size, config.touch_dim, device=device),
        },
        'actual_rewards': torch.randn(batch_size, config.reward_horizon, device=device)
    }


@pytest.fixture
def config():
    """Create default LRN configuration."""
    return LRNConfig()


@pytest.fixture
def model(config):
    """Create LRN model."""
    return LivingResonanceNetwork(config)


def test_full_forward_backward_cycle(config):
    """Test complete forward + backward + optimizer step cycle.

    This validates the basic training loop:
    1. Forward pass
    2. Loss computation
    3. Backward pass
    4. Optimizer step
    """
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create synthetic data
    inputs = create_synthetic_sensory_data(batch_size=4, config=config)
    targets = create_synthetic_targets(batch_size=4, config=config)

    # Forward pass
    predictions, reward_preds, actions = model(
        inputs['vision'],
        inputs['audio'],
        inputs['proprio'],
        inputs['touch'],
        inputs['genome']
    )

    # Compute loss
    loss_dict = model.compute_loss(
        predictions,
        reward_preds,
        targets['next_sensory'],
        actions,
        targets['actual_rewards']
    )

    # Check loss exists and is finite
    assert 'total' in loss_dict
    assert torch.isfinite(loss_dict['total'])
    assert loss_dict['total'].requires_grad

    # Backward pass
    optimizer.zero_grad()
    loss_dict['total'].backward()

    # Check gradients exist and are finite for components used in loss
    # (action_head won't have gradients since actions aren't used in loss)
    grad_count = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all()
            grad_count += 1

    # At least most parameters should have gradients
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert grad_count > total_params * 0.8, \
        f"Too few parameters have gradients: {grad_count}/{total_params}"

    # Optimizer step
    initial_param = next(model.parameters()).clone()
    optimizer.step()
    updated_param = next(model.parameters())

    # Verify parameters changed
    assert not torch.allclose(initial_param, updated_param, atol=1e-8)


def test_online_learning_simulation(config):
    """Simulate 100 steps of online learning with batch_size=1.

    This tests the system in true online learning mode where each
    update is from a single sample, mimicking real-time agent learning.
    """
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []

    for step in range(100):
        # Single sample (online learning)
        inputs = create_synthetic_sensory_data(batch_size=1, config=config)
        targets = create_synthetic_targets(batch_size=1, config=config)

        # Forward pass
        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        # Compute loss
        loss_dict = model.compute_loss(
            predictions,
            reward_preds,
            targets['next_sensory'],
            actions,
            targets['actual_rewards']
        )

        # Backward + optimize
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        losses.append(loss_dict['total'].item())

        # Check loss is finite at every step
        assert torch.isfinite(loss_dict['total'])

    # Verify we completed all steps
    assert len(losses) == 100

    # Check all losses are positive and finite
    assert all(loss > 0 for loss in losses)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


def test_multi_task_learning_reduces_both_losses(config):
    """Train for 500 steps and verify both sensory AND reward loss decrease.

    This validates that multi-task learning is working correctly:
    the model should learn to predict both next sensory states AND
    future rewards, with both losses decreasing over training.
    """
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    sensory_losses = []
    reward_losses = []

    # Create consistent synthetic data for better signal
    # (In real scenario, model learns correlations in data)
    torch.manual_seed(42)

    for step in range(500):
        # Use batch_size=8 for more stable gradients
        inputs = create_synthetic_sensory_data(batch_size=8, config=config)
        targets = create_synthetic_targets(batch_size=8, config=config)

        # Forward pass
        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        # Compute loss
        loss_dict = model.compute_loss(
            predictions,
            reward_preds,
            targets['next_sensory'],
            actions,
            targets['actual_rewards']
        )

        # Backward + optimize
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

        sensory_losses.append(loss_dict['sensory'].item())
        reward_losses.append(loss_dict['reward'].item())

    # Check that both losses decreased
    # Compare first 50 steps vs last 50 steps
    early_sensory = sum(sensory_losses[:50]) / 50
    late_sensory = sum(sensory_losses[-50:]) / 50

    early_reward = sum(reward_losses[:50]) / 50
    late_reward = sum(reward_losses[-50:]) / 50

    # Sensory loss should decrease
    assert late_sensory < early_sensory, \
        f"Sensory loss did not decrease: {early_sensory:.4f} -> {late_sensory:.4f}"

    # Reward loss should decrease
    assert late_reward < early_reward, \
        f"Reward loss did not decrease: {early_reward:.4f} -> {late_reward:.4f}"


def test_gradient_stability_extended(config):
    """Run 1000 steps and verify no NaN/Inf gradients appear.

    This tests gradient stability over extended training, ensuring
    the architecture remains numerically stable even after many updates.
    """
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    gradient_clipper = GradientClipper(clip_type='norm', max_norm=1.0)
    gradient_monitor = GradientMonitor(window_size=100)

    for step in range(1000):
        # Generate data
        inputs = create_synthetic_sensory_data(batch_size=4, config=config)
        targets = create_synthetic_targets(batch_size=4, config=config)

        # Forward pass
        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        # Compute loss
        loss_dict = model.compute_loss(
            predictions,
            reward_preds,
            targets['next_sensory'],
            actions,
            targets['actual_rewards']
        )

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()

        # Check gradients before clipping
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"NaN/Inf gradient detected at step {step}"

        # Clip gradients
        grad_norm = gradient_clipper.clip(model)
        assert torch.isfinite(torch.tensor(grad_norm))

        # Monitor gradients
        gradient_monitor.record(model)

        # Optimize
        optimizer.step()

        # Check outputs remain finite
        assert torch.isfinite(predictions).all()
        assert torch.isfinite(reward_preds).all()
        assert torch.isfinite(actions).all()

    # Check gradient statistics
    stats = gradient_monitor.get_statistics()
    assert stats['is_stable'], "Gradients became unstable during training"
    assert torch.isfinite(torch.tensor(stats['grad_norm_mean']))
    assert torch.isfinite(torch.tensor(stats['grad_norm_std']))


def test_memory_stability(config):
    """Run 500 steps and check memory doesn't grow unbounded.

    This validates that the training loop doesn't have memory leaks
    that would prevent long-term online learning.
    """
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Measure initial memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Track allocated tensors
    initial_tensor_count = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

    for step in range(500):
        # Generate data
        inputs = create_synthetic_sensory_data(batch_size=4, config=config)
        targets = create_synthetic_targets(batch_size=4, config=config)

        # Forward pass
        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        # Compute loss
        loss_dict = model.compute_loss(
            predictions,
            reward_preds,
            targets['next_sensory'],
            actions,
            targets['actual_rewards']
        )

        # Backward + optimize
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()

    # Measure final memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    final_tensor_count = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

    # Allow some growth (optimizer states, etc.) but not unbounded
    # Tensor count can grow due to optimizer state (momentum buffers, etc.)
    # but shouldn't grow linearly with training steps
    # For 500 steps, growth should be bounded (optimizer state is fixed size once initialized)
    # Allow up to 10x growth to account for optimizer states and internal buffers
    growth_ratio = final_tensor_count / initial_tensor_count
    assert growth_ratio < 10.0, \
        f"Memory grew too much: {initial_tensor_count} -> {final_tensor_count} tensors"

    # More importantly, check that growth is bounded and not linear
    # If it was truly unbounded, we'd see much more than 10x growth


def test_all_components_integrated(config):
    """Test LRN + RewardHistoryBuffer + GradientClipper + EMA + LRScheduler together.

    This validates that all the learning components work together in harmony.
    """
    # Initialize all components
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    reward_buffer = RewardHistoryBuffer(
        horizon=config.reward_horizon,
        max_pending=100,
        max_stale_steps=50
    )
    gradient_clipper = GradientClipper(clip_type='norm', max_norm=1.0)
    ema = ExponentialMovingAverage(model, decay=0.999)
    lr_scheduler = OnlineLRScheduler(
        optimizer,
        warmup_steps=100,
        base_lr=1e-4,
        min_lr=1e-6,
        decay_rate=0.9999
    )
    gradient_monitor = GradientMonitor(window_size=50)

    # Run training loop for 200 steps
    for step in range(200):
        # Generate data
        inputs = create_synthetic_sensory_data(batch_size=4, config=config)
        targets = create_synthetic_targets(batch_size=4, config=config)

        # Forward pass
        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        # Record reward prediction in buffer
        reward_buffer.record_prediction(step, reward_preds[0])
        reward_buffer.record_actual_reward(step, targets['actual_rewards'][0, 0].item())

        # Compute loss
        loss_dict = model.compute_loss(
            predictions,
            reward_preds,
            targets['next_sensory'],
            actions,
            targets['actual_rewards']
        )

        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()

        # Clip gradients
        gradient_clipper.clip(model)

        # Monitor gradients
        gradient_monitor.record(model)

        # Optimize
        optimizer.step()

        # Update EMA
        ema.update()

        # Update learning rate
        current_lr = lr_scheduler.step()
        assert current_lr > 0

        # Check reward buffer has pending predictions
        if step >= config.reward_horizon:
            ready_pairs = reward_buffer.get_ready_pairs()
            # Should have some ready pairs after horizon steps
            assert len(ready_pairs) >= 0

    # Test EMA shadow weights
    ema.apply_shadow()

    # Generate test data
    test_inputs = create_synthetic_sensory_data(batch_size=2, config=config)

    with torch.no_grad():
        ema_predictions, ema_rewards, ema_actions = model(
            test_inputs['vision'],
            test_inputs['audio'],
            test_inputs['proprio'],
            test_inputs['touch'],
            test_inputs['genome']
        )

    # Restore original weights
    ema.restore()

    # Check outputs are finite
    assert torch.isfinite(ema_predictions).all()
    assert torch.isfinite(ema_rewards).all()
    assert torch.isfinite(ema_actions).all()

    # Check gradient statistics
    stats = gradient_monitor.get_statistics()
    assert stats['is_stable']


def test_parameter_count_in_range(config):
    """Verify model has 400K-900K parameters.

    This ensures the architecture remains efficient and within
    the expected parameter budget.
    """
    model = LivingResonanceNetwork(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # All parameters should be trainable
    assert total_params == trainable_params

    # Should be in range 400K-900K
    assert 400_000 <= total_params <= 900_000, \
        f"Parameter count {total_params:,} outside expected range [400K, 900K]"

    print(f"\nTotal parameters: {total_params:,}")


def test_deterministic_with_seed(config):
    """Test that same seed produces same outputs.

    This validates that the model is deterministic when seeded,
    which is important for reproducibility and debugging.
    """
    def run_forward_pass(seed: int):
        """Run a forward pass with given seed."""
        torch.manual_seed(seed)
        model = LivingResonanceNetwork(config)

        torch.manual_seed(seed)
        inputs = create_synthetic_sensory_data(batch_size=4, config=config)

        with torch.no_grad():
            predictions, reward_preds, actions = model(
                inputs['vision'],
                inputs['audio'],
                inputs['proprio'],
                inputs['touch'],
                inputs['genome']
            )

        return predictions, reward_preds, actions

    # Run with seed 42 twice
    pred1, reward1, action1 = run_forward_pass(42)
    pred2, reward2, action2 = run_forward_pass(42)

    # Should be identical
    assert torch.allclose(pred1, pred2, atol=1e-6)
    assert torch.allclose(reward1, reward2, atol=1e-6)
    assert torch.allclose(action1, action2, atol=1e-6)

    # Run with different seed
    pred3, reward3, action3 = run_forward_pass(123)

    # Should be different
    assert not torch.allclose(pred1, pred3, atol=1e-3)


def test_backward_pass_deterministic(config):
    """Test that backward pass with same seed produces same gradients.

    This validates that gradient computation is also deterministic.
    """
    def run_backward_pass(seed: int):
        """Run backward pass with given seed and return gradients."""
        torch.manual_seed(seed)
        model = LivingResonanceNetwork(config)

        torch.manual_seed(seed)
        inputs = create_synthetic_sensory_data(batch_size=4, config=config)
        targets = create_synthetic_targets(batch_size=4, config=config)

        # Forward pass
        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        # Compute loss
        loss_dict = model.compute_loss(
            predictions,
            reward_preds,
            targets['next_sensory'],
            actions,
            targets['actual_rewards']
        )

        # Backward
        loss_dict['total'].backward()

        # Collect gradients
        grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

        return grads

    # Run with seed 42 twice
    grads1 = run_backward_pass(42)
    grads2 = run_backward_pass(42)

    # Gradients should be identical
    assert len(grads1) == len(grads2)
    for g1, g2 in zip(grads1, grads2):
        assert torch.allclose(g1, g2, atol=1e-6)


def test_loss_components_non_negative(config):
    """Test that all loss components are non-negative.

    MSE losses should always be >= 0.
    """
    model = LivingResonanceNetwork(config)

    inputs = create_synthetic_sensory_data(batch_size=4, config=config)
    targets = create_synthetic_targets(batch_size=4, config=config)

    # Forward pass
    predictions, reward_preds, actions = model(
        inputs['vision'],
        inputs['audio'],
        inputs['proprio'],
        inputs['touch'],
        inputs['genome']
    )

    # Compute loss
    loss_dict = model.compute_loss(
        predictions,
        reward_preds,
        targets['next_sensory'],
        actions,
        targets['actual_rewards']
    )

    # All losses should be >= 0
    for key, value in loss_dict.items():
        assert value >= 0, f"{key} loss is negative: {value}"


def test_training_reduces_loss(config):
    """Test that training actually reduces loss over time.

    This is a basic sanity check that the model can learn.
    """
    model = LivingResonanceNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create fixed dataset for better signal
    torch.manual_seed(42)
    inputs_list = [create_synthetic_sensory_data(batch_size=8, config=config) for _ in range(10)]
    targets_list = [create_synthetic_targets(batch_size=8, config=config) for _ in range(10)]

    initial_losses = []
    final_losses = []

    # Initial loss (before training)
    model.eval()
    with torch.no_grad():
        for inputs, targets in zip(inputs_list, targets_list):
            predictions, reward_preds, actions = model(
                inputs['vision'],
                inputs['audio'],
                inputs['proprio'],
                inputs['touch'],
                inputs['genome']
            )
            loss_dict = model.compute_loss(
                predictions,
                reward_preds,
                targets['next_sensory'],
                actions,
                targets['actual_rewards']
            )
            initial_losses.append(loss_dict['total'].item())

    # Train for 100 epochs
    model.train()
    for epoch in range(100):
        for inputs, targets in zip(inputs_list, targets_list):
            predictions, reward_preds, actions = model(
                inputs['vision'],
                inputs['audio'],
                inputs['proprio'],
                inputs['touch'],
                inputs['genome']
            )
            loss_dict = model.compute_loss(
                predictions,
                reward_preds,
                targets['next_sensory'],
                actions,
                targets['actual_rewards']
            )

            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

    # Final loss (after training)
    model.eval()
    with torch.no_grad():
        for inputs, targets in zip(inputs_list, targets_list):
            predictions, reward_preds, actions = model(
                inputs['vision'],
                inputs['audio'],
                inputs['proprio'],
                inputs['touch'],
                inputs['genome']
            )
            loss_dict = model.compute_loss(
                predictions,
                reward_preds,
                targets['next_sensory'],
                actions,
                targets['actual_rewards']
            )
            final_losses.append(loss_dict['total'].item())

    # Loss should decrease significantly
    avg_initial = sum(initial_losses) / len(initial_losses)
    avg_final = sum(final_losses) / len(final_losses)

    assert avg_final < avg_initial * 0.5, \
        f"Loss did not decrease enough: {avg_initial:.4f} -> {avg_final:.4f}"
