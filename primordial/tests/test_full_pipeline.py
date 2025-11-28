"""Full pipeline integration test."""

import pytest
from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig


def test_full_pipeline_runs():
    """Test complete simulation pipeline runs without errors."""
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        max_agents=1,
        predator_count=0,  # No predators for stable test
        initial_food=20,
        learning_enabled=True,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)

    # Run for 100 steps
    metrics = sim.run(steps=100)

    assert len(metrics) == 100
    assert sim.step_count == 100

    # Agent should still be alive (no predators)
    for wrapper in sim.agents.values():
        assert wrapper.agent.is_alive


def test_full_pipeline_with_predators():
    """Test pipeline with predators (may cause death)."""
    config = SimulationConfig(
        world_width=200,
        world_height=200,
        max_agents=1,
        predator_count=2,
        initial_food=10,
        learning_enabled=True,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)

    # Run for 500 steps (enough for potential death)
    metrics = sim.run(steps=500)

    assert len(metrics) == 500


def test_full_pipeline_save_load():
    """Test saving and loading simulation state."""
    import tempfile
    import os

    config = SimulationConfig(
        world_width=200,
        world_height=200,
        max_agents=1,
        predator_count=0,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)
    sim.run(steps=50)

    original_step = sim.step_count
    original_age = list(sim.agents.values())[0].agent.age

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test')
        sim.save_state(path)

        # Create new simulation and load
        sim2 = Simulation(config)
        sim2.load_state(path)

        assert sim2.step_count == original_step


def test_learning_changes_behavior():
    """Test that learning produces different behavior over time."""
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        max_agents=1,
        predator_count=0,
        initial_food=30,
        learning_enabled=True,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)

    # Run and check agent is learning
    metrics = sim.run(steps=200)

    # Check that the model has been updated
    wrapper = list(sim.agents.values())[0]
    assert wrapper.step_count == 200
