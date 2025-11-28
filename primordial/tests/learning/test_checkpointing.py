"""Tests for death handling and checkpointing."""

import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from primordial.learning.checkpointing import DeathHandler, DeathReplay
from primordial.learning.losses import PredictionLoss


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    yield str(checkpoint_dir)
    # Cleanup
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def test_death_handler_checkpoint_save(temp_checkpoint_dir):
    """Test that DeathHandler saves checkpoints on death."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    handler = DeathHandler(checkpoint_dir=temp_checkpoint_dir)

    # Trigger death
    result = handler.on_death(model, optimizer)

    # Check results
    assert result['death_count'] == 1
    assert 'checkpoint_path' in result

    # Check checkpoint file exists
    checkpoint_path = Path(result['checkpoint_path'])
    assert checkpoint_path.exists()

    # Check checkpoint content
    checkpoint = torch.load(checkpoint_path)
    assert 'death_count' in checkpoint
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert checkpoint['death_count'] == 1


def test_death_handler_checkpoint_load(temp_checkpoint_dir):
    """Test that DeathHandler can load checkpoints."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    handler = DeathHandler(checkpoint_dir=temp_checkpoint_dir)

    # Get initial weights
    initial_weights = model.fc.weight.data.clone()

    # Modify weights
    with torch.no_grad():
        model.fc.weight.data += 1.0

    # Save death checkpoint
    handler.on_death(model, optimizer)

    # Reset to initial weights
    with torch.no_grad():
        model.fc.weight.data = initial_weights.clone()

    # Load checkpoint
    checkpoint = handler.load_latest_checkpoint(model, optimizer)

    # Check weights were restored (should be initial + 1.0)
    assert checkpoint is not None
    assert not torch.equal(model.fc.weight.data, initial_weights)
    assert torch.allclose(model.fc.weight.data, initial_weights + 1.0)


def test_death_handler_lr_reduction(temp_checkpoint_dir):
    """Test that DeathHandler reduces learning rate on death."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    handler = DeathHandler(
        checkpoint_dir=temp_checkpoint_dir,
        lr_reduction_factor=0.5
    )

    initial_lr = optimizer.param_groups[0]['lr']

    # Trigger death
    handler.on_death(model, optimizer, lr_scheduler=True)

    # Check LR was reduced
    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr == initial_lr * 0.5


def test_death_handler_optimizer_reset(temp_checkpoint_dir):
    """Test that DeathHandler resets optimizer state on death."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    handler = DeathHandler(
        checkpoint_dir=temp_checkpoint_dir,
        reset_optimizer=True
    )

    # Perform a few optimization steps to build up momentum
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()

    # Check that optimizer has momentum
    param = next(model.parameters())
    state = optimizer.state[param]
    assert 'exp_avg' in state
    assert 'exp_avg_sq' in state

    # Store momentum values before reset
    exp_avg_before = state['exp_avg'].clone()
    exp_avg_sq_before = state['exp_avg_sq'].clone()

    # Both should be non-zero
    assert exp_avg_before.abs().sum() > 0
    assert exp_avg_sq_before.abs().sum() > 0

    # Trigger death
    handler.on_death(model, optimizer)

    # Check momentum was reset
    exp_avg_after = state['exp_avg']
    exp_avg_sq_after = state['exp_avg_sq']

    assert exp_avg_after.abs().sum() == 0
    assert exp_avg_sq_after.abs().sum() == 0


def test_death_handler_multiple_deaths(temp_checkpoint_dir):
    """Test multiple death checkpoints."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    handler = DeathHandler(checkpoint_dir=temp_checkpoint_dir)

    # Trigger multiple deaths
    result1 = handler.on_death(model, optimizer)
    result2 = handler.on_death(model, optimizer)
    result3 = handler.on_death(model, optimizer)

    assert result1['death_count'] == 1
    assert result2['death_count'] == 2
    assert result3['death_count'] == 3

    # Check all checkpoint files exist
    checkpoint_dir = Path(temp_checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('death_*.pt'))
    assert len(checkpoints) == 3


def test_death_replay_add_experience():
    """Test adding experiences to replay buffer."""
    replay = DeathReplay(replay_buffer_size=10)

    # Add experiences
    for i in range(15):
        senses = torch.randn(1, 10)
        action = torch.randn(1, 5)
        prediction = torch.randn(1, 10)
        next_senses = torch.randn(1, 10)
        reward = float(i)

        replay.add_experience(senses, action, prediction, next_senses, reward)

    # Buffer should be at max size
    assert len(replay.buffer) == 10

    # Most recent reward should be 14 (last added)
    assert replay.buffer[-1]['reward'] == 14.0


def test_death_replay_on_death():
    """Test replaying experiences on death."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = PredictionLoss()

    replay = DeathReplay(replay_buffer_size=20, replay_iterations=5)

    # Add some experiences
    for i in range(20):
        senses = torch.randn(1, 10)
        action = torch.randn(1, 5)
        with torch.no_grad():
            prediction = model(senses)
        next_senses = torch.randn(1, 10)
        reward = 1.0

        replay.add_experience(senses, action, prediction, next_senses, reward)

    # Get initial weights
    initial_weights = model.fc.weight.data.clone()

    # Replay on death
    replay.replay_on_death(model, optimizer, loss_fn)

    # Weights should have changed
    assert not torch.equal(model.fc.weight.data, initial_weights)


def test_death_replay_clear():
    """Test clearing replay buffer."""
    replay = DeathReplay(replay_buffer_size=10)

    # Add experiences
    for i in range(5):
        senses = torch.randn(1, 10)
        action = torch.randn(1, 5)
        prediction = torch.randn(1, 10)
        next_senses = torch.randn(1, 10)
        reward = float(i)

        replay.add_experience(senses, action, prediction, next_senses, reward)

    assert len(replay.buffer) == 5

    # Clear buffer
    replay.clear()

    assert len(replay.buffer) == 0


def test_death_handler_min_lr(temp_checkpoint_dir):
    """Test that learning rate doesn't go below minimum."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    handler = DeathHandler(
        checkpoint_dir=temp_checkpoint_dir,
        lr_reduction_factor=0.5,
        min_lr=1e-6
    )

    # Set LR to just above minimum
    optimizer.param_groups[0]['lr'] = 1.5e-6

    # Trigger death - should reduce but not below min
    handler.on_death(model, optimizer, lr_scheduler=True)

    # Check LR hit minimum
    new_lr = optimizer.param_groups[0]['lr']
    assert new_lr == 1e-6


def test_death_handler_no_optimizer_reset(temp_checkpoint_dir):
    """Test that optimizer state is preserved when reset_optimizer=False."""
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    handler = DeathHandler(
        checkpoint_dir=temp_checkpoint_dir,
        reset_optimizer=False
    )

    # Perform optimization steps to build up momentum
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()

    # Get momentum before death
    param = next(model.parameters())
    state = optimizer.state[param]
    exp_avg_before = state['exp_avg'].clone()
    exp_avg_sq_before = state['exp_avg_sq'].clone()

    # Trigger death
    handler.on_death(model, optimizer)

    # Check momentum was NOT reset
    exp_avg_after = state['exp_avg']
    exp_avg_sq_after = state['exp_avg_sq']

    assert torch.equal(exp_avg_after, exp_avg_before)
    assert torch.equal(exp_avg_sq_after, exp_avg_sq_before)
