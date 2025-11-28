import pytest
from primordial.experiments.teaching_impact import TeachingImpactExperiment
from primordial.simulation.config import SimulationConfig


def test_teaching_experiment_creation():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        render_enabled=False
    )
    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=100,
        teaching_interval=10
    )
    assert exp.teaching_interval == 10


def test_teaching_experiment_run():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=50,
        teaching_interval=10,
        num_trials=1
    )

    results = exp.run()

    assert 'with_teaching_mean_survival' in results
    assert 'without_teaching_mean_survival' in results
    assert 'teaching_acceleration' in results
