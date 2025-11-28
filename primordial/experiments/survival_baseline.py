"""Survival baseline experiment: Learning ON vs OFF.

Measures whether online learning improves agent survival time
compared to a random/untrained baseline.

Success criterion: Agent survives >5x longer with learning ON vs OFF.
"""

from typing import Dict, Any, List
from copy import deepcopy

from primordial.simulation.config import SimulationConfig
from primordial.experiments.base import BaseExperiment


class SurvivalBaselineExperiment(BaseExperiment):
    """Compares survival with learning enabled vs disabled.

    Runs multiple trials in each condition and computes
    mean survival time.
    """

    def __init__(
        self,
        config: SimulationConfig,
        steps_per_trial: int = 10000,
        num_trials: int = 5,
    ):
        """Initialize experiment.

        Args:
            config: Base simulation configuration.
            steps_per_trial: Steps to run each trial.
            num_trials: Number of trials per condition.
        """
        super().__init__(config)
        self.steps_per_trial = steps_per_trial
        self.num_trials = num_trials

        self.learning_on_survivals: List[float] = []
        self.learning_off_survivals: List[float] = []

    def run(self) -> Dict[str, Any]:
        """Run the experiment.

        Returns:
            Results with survival times and improvement ratio.
        """
        print("Starting Survival Baseline Experiment")
        print(f"  Steps per trial: {self.steps_per_trial}")
        print(f"  Trials per condition: {self.num_trials}")
        print()

        # Condition 1: Learning ON
        print("Running: Learning ON")
        self.learning_on_survivals = self._run_condition(learning_enabled=True)

        # Condition 2: Learning OFF
        print("Running: Learning OFF")
        self.learning_off_survivals = self._run_condition(learning_enabled=False)

        # Compute results
        results = self._compute_results()

        # Record to collector
        self.collector.record(results)

        return results

    def _run_condition(self, learning_enabled: bool) -> List[float]:
        """Run trials for a condition.

        Args:
            learning_enabled: Whether learning is enabled.

        Returns:
            List of survival times.
        """
        survival_times = []

        for trial in range(self.num_trials):
            # Create fresh config with learning setting
            trial_config = SimulationConfig(
                **{**self.config.to_dict(), 'learning_enabled': learning_enabled}
            )

            # Create fresh simulation
            from primordial.simulation.simulation import Simulation
            sim = Simulation(trial_config)

            # Run trial
            for step in range(self.steps_per_trial):
                sim.tick()

                # Early termination if all agents dead
                if all(not w.agent.is_alive for w in sim.agents.values()):
                    break

            # Collect survival times
            trial_survival = 0.0
            for agent_id, wrapper in sim.agents.items():
                trial_survival += wrapper.agent.age
            trial_survival /= len(sim.agents)

            survival_times.append(trial_survival)
            print(f"  Trial {trial + 1}/{self.num_trials}: {trial_survival:.2f}s")

        return survival_times

    def _compute_results(self) -> Dict[str, Any]:
        """Compute experiment results."""
        on_mean = sum(self.learning_on_survivals) / len(self.learning_on_survivals)
        off_mean = sum(self.learning_off_survivals) / len(self.learning_off_survivals)

        # Avoid division by zero
        ratio = on_mean / max(off_mean, 0.001)

        return {
            'learning_on_mean_survival': on_mean,
            'learning_off_mean_survival': off_mean,
            'learning_on_trials': self.learning_on_survivals,
            'learning_off_trials': self.learning_off_survivals,
            'improvement_ratio': ratio,
            'target_ratio': 5.0,
            'target_met': ratio >= 5.0,
        }


def main():
    """Run survival baseline experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run survival baseline experiment'
    )
    parser.add_argument('--steps', type=int, default=10000,
                       help='Steps per trial')
    parser.add_argument('--trials', type=int, default=5,
                       help='Trials per condition')
    parser.add_argument('--output', type=str, default='./results/survival',
                       help='Output directory')

    args = parser.parse_args()

    config = SimulationConfig(
        world_width=1000,
        world_height=1000,
        predator_count=3,
        initial_food=50,
        render_enabled=False,
    )

    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials
    )

    results = exp.run()
    exp.print_summary()
    exp.export_results(args.output)


if __name__ == '__main__':
    main()
