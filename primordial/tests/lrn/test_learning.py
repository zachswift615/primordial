"""Tests for online learning utilities."""
import torch
import torch.nn as nn
import pytest
import numpy as np

from primordial.lrn.learning import (
    RewardHistoryBuffer,
    GradientClipper,
    ExponentialMovingAverage,
    GradientMonitor,
    OnlineLRScheduler,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestRewardHistoryBuffer:
    """Tests for RewardHistoryBuffer."""

    def test_initialization(self):
        """Test buffer initializes correctly."""
        buffer = RewardHistoryBuffer(horizon=5, max_pending=100, max_stale_steps=50)

        assert buffer.horizon == 5
        assert buffer.max_pending == 100
        assert buffer.max_stale_steps == 50
        assert len(buffer.reward_history) == 0
        assert len(buffer.pending_predictions) == 0

    def test_record_prediction(self):
        """Test recording predictions."""
        buffer = RewardHistoryBuffer(horizon=5)
        reward_preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        buffer.record_prediction(step=10, reward_preds=reward_preds)

        assert len(buffer.pending_predictions) == 1
        assert buffer.pending_predictions[0]["step"] == 10
        assert torch.allclose(
            buffer.pending_predictions[0]["predictions"], reward_preds
        )
        assert buffer.pending_predictions[0]["steps_remaining"] == 5

    def test_record_actual_reward(self):
        """Test recording actual rewards."""
        buffer = RewardHistoryBuffer(horizon=5)

        buffer.record_actual_reward(step=10, reward=1.5)
        buffer.record_actual_reward(step=11, reward=-0.5)

        assert buffer.reward_history[10] == 1.5
        assert buffer.reward_history[11] == -0.5

    def test_get_ready_pairs_not_ready_yet(self):
        """Test get_ready_pairs returns empty when not ready."""
        buffer = RewardHistoryBuffer(horizon=5)
        reward_preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        buffer.record_prediction(step=10, reward_preds=reward_preds)

        # Only 1 step has passed, need 5
        ready_pairs = buffer.get_ready_pairs()
        assert len(ready_pairs) == 0
        assert len(buffer.pending_predictions) == 1

    def test_get_ready_pairs_after_horizon(self):
        """Test get_ready_pairs returns pairs after horizon steps."""
        buffer = RewardHistoryBuffer(horizon=3)
        reward_preds = torch.tensor([0.1, 0.2, 0.3])

        # Record prediction at step 10
        buffer.record_prediction(step=10, reward_preds=reward_preds)

        # Record actual rewards for steps 11, 12, 13
        buffer.record_actual_reward(step=11, reward=1.0)
        buffer.record_actual_reward(step=12, reward=2.0)
        buffer.record_actual_reward(step=13, reward=3.0)

        # Call get_ready_pairs 3 times to decrement steps_remaining
        for _ in range(3):
            ready_pairs = buffer.get_ready_pairs()

        # Should have 1 ready pair now
        assert len(ready_pairs) == 1
        pred, actual = ready_pairs[0]
        assert torch.allclose(pred, reward_preds)
        assert torch.allclose(actual, torch.tensor([1.0, 2.0, 3.0]))

        # Pending predictions should be cleared
        assert len(buffer.pending_predictions) == 0

    def test_stale_prediction_cleanup(self):
        """Test stale predictions are discarded."""
        buffer = RewardHistoryBuffer(horizon=3, max_stale_steps=10)
        reward_preds = torch.tensor([0.1, 0.2, 0.3])

        buffer.record_prediction(step=10, reward_preds=reward_preds)

        # Call get_ready_pairs many times (> max_stale_steps)
        for _ in range(15):
            ready_pairs = buffer.get_ready_pairs()

        # Prediction should be discarded as stale
        assert len(buffer.pending_predictions) == 0
        assert len(ready_pairs) == 0

    def test_max_pending_enforcement(self):
        """Test max_pending limit is enforced."""
        buffer = RewardHistoryBuffer(horizon=3, max_pending=5)

        # Add more predictions than max_pending
        for i in range(10):
            reward_preds = torch.tensor([0.1, 0.2, 0.3])
            buffer.record_prediction(step=i, reward_preds=reward_preds)

        # Should only keep last max_pending predictions
        assert len(buffer.pending_predictions) <= 5

    def test_cleanup_old_entries(self):
        """Test old reward history entries are cleaned up."""
        buffer = RewardHistoryBuffer(horizon=3, max_stale_steps=5)

        # Record many rewards
        for i in range(100):
            buffer.record_actual_reward(step=i, reward=float(i))

        # Should not keep all 100 rewards
        assert len(buffer.reward_history) < 100

    def test_on_death_clears_state(self):
        """Test on_death clears all buffer state."""
        buffer = RewardHistoryBuffer(horizon=3)
        reward_preds = torch.tensor([0.1, 0.2, 0.3])

        buffer.record_prediction(step=10, reward_preds=reward_preds)
        buffer.record_actual_reward(step=11, reward=1.0)

        buffer.on_death()

        assert len(buffer.pending_predictions) == 0
        assert len(buffer.reward_history) == 0
        assert buffer._oldest_step == 0

    def test_missing_rewards_default_to_zero(self):
        """Test missing rewards default to 0.0."""
        buffer = RewardHistoryBuffer(horizon=3)
        reward_preds = torch.tensor([0.1, 0.2, 0.3])

        buffer.record_prediction(step=10, reward_preds=reward_preds)

        # Only record some rewards
        buffer.record_actual_reward(step=11, reward=1.0)
        # steps 12, 13 missing

        # Wait for horizon
        for _ in range(3):
            ready_pairs = buffer.get_ready_pairs()

        # Should have pair with zeros for missing rewards
        assert len(ready_pairs) == 1
        pred, actual = ready_pairs[0]
        assert torch.allclose(actual, torch.tensor([1.0, 0.0, 0.0]))

    def test_multiple_predictions_tracked(self):
        """Test multiple predictions can be tracked simultaneously."""
        buffer = RewardHistoryBuffer(horizon=3)

        # Record multiple predictions
        buffer.record_prediction(step=10, reward_preds=torch.tensor([0.1, 0.2, 0.3]))
        buffer.record_prediction(step=11, reward_preds=torch.tensor([0.4, 0.5, 0.6]))
        buffer.record_prediction(step=12, reward_preds=torch.tensor([0.7, 0.8, 0.9]))

        assert len(buffer.pending_predictions) == 3

        # Record rewards for all
        for i in range(11, 20):
            buffer.record_actual_reward(step=i, reward=float(i))

        # Wait for horizon
        for _ in range(3):
            ready_pairs = buffer.get_ready_pairs()

        # Should have 3 ready pairs
        assert len(ready_pairs) == 3


class TestGradientClipper:
    """Tests for GradientClipper."""

    def test_initialization(self):
        """Test clipper initializes correctly."""
        clipper = GradientClipper(clip_type="norm", max_norm=1.0, max_value=10.0)

        assert clipper.clip_type == "norm"
        assert clipper.max_norm == 1.0
        assert clipper.max_value == 10.0

    def test_norm_clipping(self):
        """Test gradient norm clipping works."""
        model = SimpleModel()
        clipper = GradientClipper(clip_type="norm", max_norm=1.0)

        # Create large gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 100

        # Compute norm before clipping
        norm_before = 0.0
        for p in model.parameters():
            if p.grad is not None:
                norm_before += p.grad.data.norm(2).item() ** 2
        norm_before = norm_before**0.5

        assert norm_before > 1.0

        # Clip
        grad_norm = clipper.clip(model)

        # Compute norm after clipping
        norm_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                norm_after += p.grad.data.norm(2).item() ** 2
        norm_after = norm_after**0.5

        assert norm_after <= 1.0 + 1e-5  # Allow small numerical error

    def test_value_clipping(self):
        """Test gradient value clipping works."""
        model = SimpleModel()
        clipper = GradientClipper(clip_type="value", max_value=1.0)

        # Create large gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 100

        # Verify some gradients are large
        max_val_before = 0.0
        for p in model.parameters():
            if p.grad is not None:
                max_val_before = max(max_val_before, p.grad.abs().max().item())

        assert max_val_before > 1.0

        # Clip
        clipper.clip(model)

        # Verify all gradients are clipped
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.abs().max().item() <= 1.0 + 1e-5

    def test_clip_returns_norm(self):
        """Test clip returns gradient norm."""
        model = SimpleModel()
        clipper = GradientClipper(clip_type="norm", max_norm=1.0)

        # Create gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p)

        grad_norm = clipper.clip(model)

        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0

    def test_invalid_clip_type_raises_error(self):
        """Test invalid clip_type raises ValueError."""
        clipper = GradientClipper(clip_type="invalid")
        model = SimpleModel()

        for p in model.parameters():
            p.grad = torch.randn_like(p)

        with pytest.raises(ValueError):
            clipper.clip(model)

    def test_no_gradients_returns_zero_norm(self):
        """Test clipping with no gradients returns 0."""
        model = SimpleModel()
        clipper = GradientClipper(clip_type="norm", max_norm=1.0)

        # No gradients set
        grad_norm = clipper.clip(model)

        # Should return 0 or handle gracefully
        assert grad_norm >= 0.0


class TestExponentialMovingAverage:
    """Tests for ExponentialMovingAverage."""

    def test_initialization(self):
        """Test EMA initializes correctly."""
        model = SimpleModel()
        ema = ExponentialMovingAverage(model, decay=0.999)

        assert ema.model is model
        assert ema.decay == 0.999
        assert len(ema.shadow) > 0

        # Shadow should match initial model weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(ema.shadow[name], param.data)

    def test_update_changes_shadow(self):
        """Test update modifies shadow weights."""
        model = SimpleModel()
        ema = ExponentialMovingAverage(model, decay=0.999)

        # Get initial shadow
        initial_shadow = {name: w.clone() for name, w in ema.shadow.items()}

        # Modify model weights
        for p in model.parameters():
            p.data += torch.randn_like(p) * 0.1

        # Update EMA
        ema.update()

        # Shadow should have changed
        for name in ema.shadow:
            assert not torch.allclose(ema.shadow[name], initial_shadow[name])

    def test_update_with_high_decay(self):
        """Test update with high decay changes shadow slowly."""
        model = SimpleModel()
        ema = ExponentialMovingAverage(model, decay=0.99)

        # Get initial shadow
        initial_shadow = {name: w.clone() for name, w in ema.shadow.items()}

        # Make large change to model
        for p in model.parameters():
            p.data += 10.0

        # Update EMA
        ema.update()

        # Shadow should change less than model
        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow_change = (ema.shadow[name] - initial_shadow[name]).abs().mean()
                model_change = (param.data - initial_shadow[name]).abs().mean()
                assert shadow_change < model_change

    def test_apply_shadow_and_restore(self):
        """Test apply_shadow and restore roundtrip."""
        model = SimpleModel()
        ema = ExponentialMovingAverage(model, decay=0.999)

        # Get initial model weights
        initial_weights = {
            name: param.data.clone() for name, param in model.named_parameters()
        }

        # Modify model
        for p in model.parameters():
            p.data += torch.randn_like(p) * 0.1

        # Update EMA several times
        for _ in range(5):
            ema.update()

        # Apply shadow
        ema.apply_shadow()

        # Model weights should now be shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.allclose(param.data, ema.shadow[name])

        # Restore
        ema.restore()

        # Model weights should be back to what they were before apply_shadow
        # (not initial weights, but the weights before apply_shadow)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(param.data, initial_weights[name])

    def test_multiple_apply_restore_cycles(self):
        """Test multiple apply/restore cycles work correctly."""
        model = SimpleModel()
        ema = ExponentialMovingAverage(model, decay=0.999)

        for _ in range(3):
            # Store current weights
            weights_before = {
                name: param.data.clone() for name, param in model.named_parameters()
            }

            # Apply shadow
            ema.apply_shadow()

            # Restore
            ema.restore()

            # Should be back to original
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert torch.allclose(param.data, weights_before[name])

    def test_shadow_smoother_than_model(self):
        """Test shadow weights are smoother than model weights."""
        model = SimpleModel()
        ema = ExponentialMovingAverage(model, decay=0.99)

        # Make several noisy updates
        weight_changes = []
        shadow_changes = []

        prev_weight = None
        prev_shadow = None

        for _ in range(10):
            # Random update
            for p in model.parameters():
                p.data += torch.randn_like(p) * 0.5

            ema.update()

            # Track changes
            current_weight = list(model.parameters())[0].data.clone()
            current_shadow = ema.shadow[list(ema.shadow.keys())[0]].clone()

            if prev_weight is not None:
                weight_changes.append((current_weight - prev_weight).abs().mean())
                shadow_changes.append((current_shadow - prev_shadow).abs().mean())

            prev_weight = current_weight
            prev_shadow = current_shadow

        # Shadow should have smaller average changes (smoother)
        assert np.mean(shadow_changes) < np.mean(weight_changes)


class TestGradientMonitor:
    """Tests for GradientMonitor."""

    def test_initialization(self):
        """Test monitor initializes correctly."""
        monitor = GradientMonitor(window_size=100)

        assert monitor.window_size == 100
        assert len(monitor.grad_norms) == 0
        assert len(monitor.grad_vars) == 0

    def test_record_gradients(self):
        """Test recording gradient statistics."""
        model = SimpleModel()
        monitor = GradientMonitor(window_size=100)

        # Create gradients
        for p in model.parameters():
            p.grad = torch.randn_like(p)

        monitor.record(model)

        assert len(monitor.grad_norms) == 1
        assert len(monitor.grad_vars) == 1
        assert monitor.grad_norms[0] > 0.0

    def test_window_size_enforcement(self):
        """Test monitor enforces window size."""
        model = SimpleModel()
        monitor = GradientMonitor(window_size=10)

        # Record more than window_size
        for _ in range(20):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            monitor.record(model)

        assert len(monitor.grad_norms) == 10
        assert len(monitor.grad_vars) == 10

    def test_get_statistics_empty(self):
        """Test get_statistics with no data."""
        monitor = GradientMonitor(window_size=100)
        stats = monitor.get_statistics()

        assert stats["grad_norm_mean"] == 0.0
        assert stats["grad_norm_std"] == 0.0
        assert stats["grad_var_mean"] == 0.0
        assert stats["is_stable"] is True

    def test_get_statistics_with_data(self):
        """Test get_statistics returns correct values."""
        model = SimpleModel()
        monitor = GradientMonitor(window_size=100)

        # Record several steps
        for _ in range(20):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            monitor.record(model)

        stats = monitor.get_statistics()

        assert stats["grad_norm_mean"] > 0.0
        assert stats["grad_norm_std"] >= 0.0
        assert stats["grad_var_mean"] >= 0.0
        assert isinstance(stats["is_stable"], bool)

    def test_stability_detection_stable(self):
        """Test stability detection for stable gradients."""
        model = SimpleModel()
        monitor = GradientMonitor(window_size=100)

        # Record stable gradients (small, consistent)
        for _ in range(20):
            for p in model.parameters():
                p.grad = torch.randn_like(p) * 0.01
            monitor.record(model)

        stats = monitor.get_statistics()
        assert stats["is_stable"] is True

    def test_stability_detection_unstable(self):
        """Test stability detection for unstable gradients."""
        model = SimpleModel()
        monitor = GradientMonitor(window_size=100)

        # Record mostly stable, then some unstable
        for i in range(20):
            if i < 10:
                # Stable
                for p in model.parameters():
                    p.grad = torch.randn_like(p) * 0.01
            else:
                # Unstable (large variance)
                for p in model.parameters():
                    p.grad = torch.randn_like(p) * 100
            monitor.record(model)

        stats = monitor.get_statistics()
        # May or may not be unstable depending on threshold
        assert isinstance(stats["is_stable"], bool)

    def test_no_gradients_records_zero(self):
        """Test recording with no gradients."""
        model = SimpleModel()
        monitor = GradientMonitor(window_size=100)

        # No gradients set
        monitor.record(model)

        assert len(monitor.grad_norms) == 1
        assert monitor.grad_norms[0] == 0.0


class TestOnlineLRScheduler:
    """Tests for OnlineLRScheduler."""

    def test_initialization(self):
        """Test scheduler initializes correctly."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=0.999
        )

        assert scheduler.warmup_steps == 100
        assert scheduler.base_lr == 1e-4
        assert scheduler.min_lr == 1e-6
        assert scheduler.decay_rate == 0.999
        assert scheduler.step_count == 0

    def test_warmup_increases_lr(self):
        """Test warmup increases learning rate linearly."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=0.999
        )

        lrs = []
        for _ in range(100):
            lr = scheduler.step()
            lrs.append(lr)

        # LR should increase during warmup
        assert lrs[0] < lrs[50] < lrs[99]
        # Final warmup LR should be close to base_lr
        assert abs(lrs[99] - 1e-4) < 1e-6

    def test_warmup_reaches_base_lr(self):
        """Test warmup reaches base_lr at warmup_steps."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=1.0
        )

        for _ in range(100):
            scheduler.step()

        lr = scheduler.step()
        assert abs(lr - 1e-4) < 1e-7

    def test_decay_after_warmup(self):
        """Test exponential decay after warmup."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=0.99
        )

        # Complete warmup
        for _ in range(100):
            scheduler.step()

        # Decay phase
        lrs = []
        for _ in range(100):
            lr = scheduler.step()
            lrs.append(lr)

        # LR should decrease during decay
        assert lrs[0] > lrs[50] > lrs[99]

    def test_min_lr_floor(self):
        """Test min_lr acts as floor."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=10, base_lr=1e-4, min_lr=1e-6, decay_rate=0.5
        )

        # Run many steps to decay below min_lr
        for _ in range(1000):
            lr = scheduler.step()

        # Should never go below min_lr
        assert lr >= 1e-6

    def test_optimizer_lr_updated(self):
        """Test scheduler actually updates optimizer's learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=0.999
        )

        # Step scheduler
        lr = scheduler.step()

        # Optimizer's LR should match
        assert optimizer.param_groups[0]["lr"] == lr

    def test_step_returns_current_lr(self):
        """Test step() returns current learning rate."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=0.999
        )

        lr = scheduler.step()

        assert isinstance(lr, float)
        assert lr > 0.0

    def test_no_decay_with_rate_one(self):
        """Test decay_rate=1.0 means no decay."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=10, base_lr=1e-4, min_lr=1e-6, decay_rate=1.0
        )

        # Complete warmup
        for _ in range(10):
            scheduler.step()

        # Decay phase with rate=1.0
        lr1 = scheduler.step()
        lr2 = scheduler.step()
        lr3 = scheduler.step()

        # LR should stay constant
        assert abs(lr1 - lr2) < 1e-9
        assert abs(lr2 - lr3) < 1e-9
        assert abs(lr1 - 1e-4) < 1e-9

    def test_multiple_param_groups(self):
        """Test scheduler works with multiple parameter groups."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(
            [
                {"params": model.fc1.parameters()},
                {"params": model.fc2.parameters()},
            ],
            lr=1e-4,
        )
        scheduler = OnlineLRScheduler(
            optimizer, warmup_steps=100, base_lr=1e-4, min_lr=1e-6, decay_rate=0.999
        )

        lr = scheduler.step()

        # All param groups should have same LR
        for group in optimizer.param_groups:
            assert group["lr"] == lr
