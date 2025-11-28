import pytest
from primordial.simulation.config import SimulationConfig


def test_simulation_config_defaults():
    config = SimulationConfig()
    assert config.world_width == 1000.0
    assert config.world_height == 1000.0
    assert config.tick_rate == 60
    assert config.learning_enabled is True


def test_simulation_config_custom():
    config = SimulationConfig(
        world_width=500.0,
        learning_enabled=False
    )
    assert config.world_width == 500.0
    assert config.learning_enabled is False


def test_simulation_config_to_dict():
    config = SimulationConfig()
    d = config.to_dict()
    assert 'world_width' in d
    assert 'learning_enabled' in d
