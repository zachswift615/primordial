"""Base class for experiments."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import json

from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig
from primordial.experiments.metrics_collector import MetricsCollector


class BaseExperiment(ABC):
    """Base class for Primordial experiments.

    Provides common infrastructure:
    - Simulation setup
    - Metrics collection
    - Results export

    Subclasses implement the `run()` method with experiment logic.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize experiment.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.simulation = Simulation(config)
        self.collector = MetricsCollector()

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment.

        Returns:
            Summary results.
        """
        pass

    def get_mean_survival_time(self) -> float:
        """Get mean survival time across all agents."""
        times = []
        for agent_id, wrapper in self.simulation.agents.items():
            times.append(wrapper.agent.age)
        return sum(times) / len(times) if times else 0.0

    def get_alive_count(self) -> int:
        """Get count of alive agents."""
        return sum(
            1 for w in self.simulation.agents.values()
            if w.agent.is_alive
        )

    def reset_simulation(self) -> None:
        """Reset simulation for another trial."""
        self.simulation.reset()

    def export_results(self, output_dir: str) -> None:
        """Export experiment results.

        Args:
            output_dir: Output directory path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export metrics
        self.collector.export_csv(str(output_path / 'metrics.csv'))

        # Export summary
        summary = self.collector.summary()
        summary['config'] = self.config.to_dict()

        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def print_summary(self) -> None:
        """Print experiment summary to console."""
        summary = self.collector.summary()

        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS")
        print("=" * 50)

        for key, value in sorted(summary.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("=" * 50 + "\n")
