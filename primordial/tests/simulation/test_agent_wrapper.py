import pytest
import torch
from primordial.simulation.agent_wrapper import AgentWrapper
from primordial.simulation.config import SimulationConfig
from primordial.world.world import World


@pytest.fixture
def world():
    return World(width=500, height=500)


@pytest.fixture
def config():
    return SimulationConfig(learning_enabled=True)


def test_agent_wrapper_creation(world, config):
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    assert wrapper.agent is not None
    assert wrapper.model is not None
    assert wrapper.learning_loop is not None


def test_agent_wrapper_step(world, config):
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    world.add_entity(wrapper.agent)

    # Perform one step
    action, metrics = wrapper.step(world)

    assert action is not None
    assert 'loss' in metrics or metrics == {}  # No loss on first step


def test_agent_wrapper_learning_disabled(world):
    config = SimulationConfig(learning_enabled=False)
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    world.add_entity(wrapper.agent)

    # Should still produce action without learning
    action, metrics = wrapper.step(world)
    assert action is not None


def test_agent_wrapper_death_handling(world, config):
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    world.add_entity(wrapper.agent)

    # Simulate death
    wrapper.agent.die("test")
    result = wrapper.on_death()

    assert 'death_count' in result
