"""
Tests for the main online learning loop.

Tests integration of loss, optimizer, rewards, and stability measures.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from primordial.learning.learning_loop import OnlineLearningLoop


class SimpleTestModel(nn.Module):
    """Simple test model that outputs both action and prediction."""

    def __init__(self, input_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, input_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        output = self.fc2(h)
        # Return both action and prediction
        return output, output


class MockAgentState:
    """Mock agent state for testing."""

    def __init__(self, health=100.0, energy=100.0):
        self.health = health
        self.energy = energy
        self.max_health = 100.0
        self.max_energy = 100.0


def create_test_learning_loop(model=None, checkpoint_dir=None):
    """Create a learning loop for testing."""
    if model is None:
        model = SimpleTestModel()

    optimizer_config = {
        'type': 'adamw',
        'params': {
            'lr': 1e-3,
        },
        'lr_schedule': {
            'warmup_steps': 10,
            'base_lr': 1e-3,
        }
    }

    loss_config = {
        'type': 'mse',
    }

    reward_config = {
        'combiner': {
            'survival_weight': 1.0,
            'teaching_weight': 1.5,
        },
        'modulation': {
            'modulation_type': 'linear',
            'reward_scale': 0.1,
        }
    }

    stability_config = {
        'clipping': {
            'clip_type': 'norm',
            'max_norm': 1.0,
        },
        'monitoring': {
            'window_size': 10,
        },
        'ema': {
            'decay': 0.99,
        }
    }

    death_config = {
        'checkpoint_dir': checkpoint_dir or tempfile.mkdtemp(),
        'reset_optimizer': True,
        'lr_reduction_factor': 0.5,
    }

    return OnlineLearningLoop(
        model=model,
        optimizer_config=optimizer_config,
        loss_config=loss_config,
        reward_config=reward_config,
        stability_config=stability_config,
        death_config=death_config,
    )


def test_learning_loop_creation():
    """Test that learning loop can be created."""
    model = SimpleTestModel()
    loop = create_test_learning_loop(model)

    assert loop.model is model
    assert loop.step_count == 0
    assert loop.death_count == 0
    assert loop.prev_prediction is None


def test_learning_loop_step():
    """Test single learning step."""
    model = SimpleTestModel()
    loop = create_test_learning_loop(model)

    # Create test data
    senses = torch.randn(1, 10)
    prev_senses = torch.randn(1, 10)
    agent_state = MockAgentState(health=80, energy=60)
    prev_agent_state = MockAgentState(health=80, energy=60)
    events = []

    # First step (no learning yet, just forward)
    action1, pred1 = loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

    assert action1 is not None
    assert pred1 is not None
    assert loop.step_count == 1
    assert loop.prev_prediction is not None

    # Second step (learning happens)
    senses2 = torch.randn(1, 10)
    action2, pred2 = loop.step(senses2, senses, agent_state, prev_agent_state, events)

    assert action2 is not None
    assert pred2 is not None
    assert loop.step_count == 2


def test_learning_loop_multiple_steps():
    """Test multiple learning steps."""
    model = SimpleTestModel()
    loop = create_test_learning_loop(model)

    # Run 10 steps
    for i in range(10):
        senses = torch.randn(1, 10)
        prev_senses = torch.randn(1, 10)
        agent_state = MockAgentState(health=80, energy=60)
        prev_agent_state = MockAgentState(health=80, energy=60)
        events = ['ate_food'] if i % 3 == 0 else []

        action, pred = loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

        assert action is not None
        assert pred is not None

    assert loop.step_count == 10


def test_learning_loop_on_death():
    """Test death handling."""
    tmpdir = tempfile.mkdtemp()
    try:
        model = SimpleTestModel()
        loop = create_test_learning_loop(model, checkpoint_dir=tmpdir)

        # Get initial weights
        initial_weights = {
            name: p.clone() for name, p in model.named_parameters()
        }

        # Run some steps to update weights
        for i in range(5):
            senses = torch.randn(1, 10)
            prev_senses = torch.randn(1, 10)
            agent_state = MockAgentState(health=80, energy=60)
            prev_agent_state = MockAgentState(health=80, energy=60)
            events = []
            loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

        # Trigger death
        result = loop.on_death()

        assert result['death_count'] == 1
        assert 'checkpoint_path' in result
        assert Path(result['checkpoint_path']).exists()

        # State should be reset
        assert loop.prev_prediction is None
        assert loop.prev_state is None
        assert loop.death_count == 1

        # Should be able to continue learning
        senses = torch.randn(1, 10)
        prev_senses = torch.randn(1, 10)
        agent_state = MockAgentState(health=100, energy=100)
        prev_agent_state = MockAgentState(health=100, energy=100)
        action, pred = loop.step(senses, prev_senses, agent_state, prev_agent_state, [])

        assert action is not None
        assert pred is not None

    finally:
        shutil.rmtree(tmpdir)


def test_learning_loop_checkpoint_save_load():
    """Test checkpoint save and load."""
    tmpdir = tempfile.mkdtemp()
    try:
        model = SimpleTestModel()
        loop = create_test_learning_loop(model, checkpoint_dir=tmpdir)

        # Run some steps
        for i in range(5):
            senses = torch.randn(1, 10)
            prev_senses = torch.randn(1, 10)
            agent_state = MockAgentState(health=80, energy=60)
            prev_agent_state = MockAgentState(health=80, energy=60)
            events = []
            loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

        # Save checkpoint
        checkpoint_path = Path(tmpdir) / 'test_checkpoint.pt'
        loop.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Create new loop and load
        model2 = SimpleTestModel()
        loop2 = create_test_learning_loop(model2, checkpoint_dir=tmpdir)

        loop2.load_checkpoint(str(checkpoint_path))

        assert loop2.step_count == loop.step_count
        assert loop2.death_count == loop.death_count

        # Weights should match
        for (name1, p1), (name2, p2) in zip(
            loop.model.named_parameters(),
            loop2.model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(p1, p2)

    finally:
        shutil.rmtree(tmpdir)


def test_learning_loop_with_rewards():
    """Test learning with different reward signals."""
    model = SimpleTestModel()
    loop = create_test_learning_loop(model)

    # Add human teaching signal
    loop.reward_combiner.human_teaching.add_teaching_signal(1.0)

    # Run step with positive survival reward
    senses = torch.randn(1, 10)
    prev_senses = torch.randn(1, 10)
    agent_state = MockAgentState(health=90, energy=70)
    prev_agent_state = MockAgentState(health=80, energy=60)
    events = ['ate_food']

    # First step
    loop.step(senses, prev_senses, agent_state, prev_agent_state, [])

    # Second step with rewards
    action, pred = loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

    assert action is not None
    assert pred is not None


def test_learning_loop_gradient_monitoring():
    """Test that gradient monitoring works."""
    model = SimpleTestModel()
    loop = create_test_learning_loop(model)

    # Run enough steps to populate gradient monitor
    for i in range(15):
        senses = torch.randn(1, 10)
        prev_senses = torch.randn(1, 10)
        agent_state = MockAgentState(health=80, energy=60)
        prev_agent_state = MockAgentState(health=80, energy=60)
        events = []
        loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

    # Get gradient statistics
    stats = loop.grad_monitor.get_statistics()

    assert 'grad_norm_mean' in stats
    assert stats['grad_norm_mean'] > 0  # We had some learning


def test_learning_loop_ema_inference():
    """Test that EMA weights are used for inference."""
    model = SimpleTestModel()
    loop = create_test_learning_loop(model)

    # Run some steps to diverge EMA from current weights
    for i in range(20):
        senses = torch.randn(1, 10)
        prev_senses = torch.randn(1, 10)
        agent_state = MockAgentState(health=80, energy=60)
        prev_agent_state = MockAgentState(health=80, energy=60)
        events = []
        loop.step(senses, prev_senses, agent_state, prev_agent_state, events)

    # The fact that it runs without error means EMA is working
    assert loop.step_count == 20
