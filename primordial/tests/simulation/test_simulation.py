import pytest
from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig


@pytest.fixture
def config():
    return SimulationConfig(
        world_width=500,
        world_height=500,
        predator_count=1,
        initial_food=10,
        render_enabled=False
    )


def test_simulation_creation(config):
    sim = Simulation(config)
    assert sim.world is not None
    assert len(sim.agents) == config.max_agents


def test_simulation_tick(config):
    sim = Simulation(config)

    metrics = sim.tick()

    assert 'step' in metrics
    assert metrics['step'] == 1


def test_simulation_run_steps(config):
    sim = Simulation(config)

    all_metrics = sim.run(steps=10)

    assert len(all_metrics) == 10


def test_simulation_agent_survival_time(config):
    sim = Simulation(config)

    # Run for a bit
    sim.run(steps=100)

    # Check we can get survival time
    survival = sim.get_agent_survival_time("agent_0")
    assert survival >= 0


def test_simulation_reset(config):
    sim = Simulation(config)
    sim.run(steps=50)

    sim.reset()

    assert sim.step_count == 0
    assert all(a.agent.is_alive for a in sim.agents.values())
