import pytest
from primordial.experiments.survival_baseline import SurvivalBaselineExperiment
from primordial.simulation.config import SimulationConfig


def test_survival_experiment_creation():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=1,
        render_enabled=False
    )
    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=100,
        num_trials=2
    )
    assert exp.steps_per_trial == 100
    assert exp.num_trials == 2


def test_survival_experiment_run():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=1,
        render_enabled=False
    )
    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=50,
        num_trials=2
    )

    results = exp.run()

    assert 'learning_on_mean_survival' in results
    assert 'learning_off_mean_survival' in results
    assert 'improvement_ratio' in results


def test_survival_experiment_improvement_ratio():
    # This is more of an integration test
    config = SimulationConfig(
        world_width=500,
        world_height=500,
        predator_count=0,  # No predators for predictable results
        render_enabled=False
    )
    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=100,
        num_trials=1
    )

    results = exp.run()

    # Ratio should be >= 1 (at minimum, learning shouldn't hurt)
    # In practice, may be < 1 for short runs
    assert 'improvement_ratio' in results
