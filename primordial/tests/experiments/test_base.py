import pytest
import tempfile
import os
from primordial.experiments.base import BaseExperiment
from primordial.simulation.config import SimulationConfig


class DummyExperiment(BaseExperiment):
    """Test experiment implementation."""

    def run(self):
        # Run 100 steps
        for _ in range(100):
            self.simulation.tick()
            self.collector.record({
                'step': self.simulation.step_count,
                'survival_time': self.get_mean_survival_time()
            })
        return self.collector.summary()


def test_base_experiment_creation():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = DummyExperiment(config)
    assert exp.simulation is not None
    assert exp.collector is not None


def test_base_experiment_run():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = DummyExperiment(config)

    results = exp.run()

    assert 'total_records' in results


def test_base_experiment_export():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = DummyExperiment(config)
    exp.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        exp.export_results(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, 'metrics.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'summary.json'))
