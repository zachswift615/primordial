"""Teaching impact experiment: measures effect of human teaching.

Compares learning speed with and without simulated teaching signals.

Success criterion: Human teaching accelerates learning >2x.
"""

from typing import Dict, Any, List
import random

from primordial.simulation.config import SimulationConfig
from primordial.simulation.simulation import Simulation
from primordial.experiments.base import BaseExperiment


class TeachingImpactExperiment(BaseExperiment):
    """Measures the impact of teaching signals on learning.

    Simulates teaching by providing reward signals when agent
    performs good behaviors (eating, avoiding predators).
    """

    def __init__(
        self,
        config: SimulationConfig,
        steps_per_trial: int = 10000,
        num_trials: int = 3,
        teaching_interval: int = 10,
    ):
        """Initialize experiment.

        Args:
            config: Simulation configuration.
            steps_per_trial: Steps per trial.
            num_trials: Trials per condition.
            teaching_interval: Steps between teaching signals.
        """
        super().__init__(config)
        self.steps_per_trial = steps_per_trial
        self.num_trials = num_trials
        self.teaching_interval = teaching_interval

        self.with_teaching_survivals: List[float] = []
        self.without_teaching_survivals: List[float] = []

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        print("Starting Teaching Impact Experiment")
        print(f"  Steps per trial: {self.steps_per_trial}")
        print(f"  Teaching interval: {self.teaching_interval}")
        print()

        # Condition 1: With teaching
        print("Running: With Teaching")
        self.with_teaching_survivals = self._run_condition(teaching=True)

        # Condition 2: Without teaching
        print("Running: Without Teaching")
        self.without_teaching_survivals = self._run_condition(teaching=False)

        results = self._compute_results()
        self.collector.record(results)

        return results

    def _run_condition(self, teaching: bool) -> List[float]:
        """Run trials for a condition."""
        survival_times = []

        for trial in range(self.num_trials):
            sim = Simulation(self.config)

            for step in range(self.steps_per_trial):
                metrics = sim.tick()

                # Inject teaching signals
                if teaching and step % self.teaching_interval == 0:
                    self._inject_teaching(sim)

                # Early termination
                if all(not w.agent.is_alive for w in sim.agents.values()):
                    break

            # Collect survival
            trial_survival = sum(
                w.agent.age for w in sim.agents.values()
            ) / len(sim.agents)

            survival_times.append(trial_survival)
            print(f"  Trial {trial + 1}/{self.num_trials}: {trial_survival:.2f}s")

        return survival_times

    def _inject_teaching(self, sim: Simulation) -> None:
        """Inject simulated teaching signals.

        Rewards eating, punishes taking damage.
        """
        for agent_id, wrapper in sim.agents.items():
            if not wrapper.agent.is_alive:
                continue

            # Reward eating
            if wrapper.agent.is_eating:
                # Add 'reward' event to trigger positive modulation
                if wrapper.learning_loop is not None:
                    # The events list is processed in step()
                    # We add to the events that will be processed
                    wrapper.events.append('human_reward')

            # Punish low health
            if wrapper.agent.health < wrapper.agent.genome.max_health * 0.3:
                if wrapper.learning_loop is not None:
                    wrapper.events.append('human_punish')

    def _compute_results(self) -> Dict[str, Any]:
        """Compute results."""
        with_mean = sum(self.with_teaching_survivals) / len(self.with_teaching_survivals)
        without_mean = sum(self.without_teaching_survivals) / len(self.without_teaching_survivals)

        acceleration = with_mean / max(without_mean, 0.001)

        return {
            'with_teaching_mean_survival': with_mean,
            'without_teaching_mean_survival': without_mean,
            'with_teaching_trials': self.with_teaching_survivals,
            'without_teaching_trials': self.without_teaching_survivals,
            'teaching_acceleration': acceleration,
            'target_acceleration': 2.0,
            'target_met': acceleration >= 2.0,
        }


def main():
    """Run teaching impact experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run teaching impact experiment'
    )
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--output', type=str, default='./results/teaching')

    args = parser.parse_args()

    config = SimulationConfig(
        world_width=1000,
        world_height=1000,
        predator_count=3,
        render_enabled=False,
    )

    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials
    )

    results = exp.run()
    exp.print_summary()
    exp.export_results(args.output)


if __name__ == '__main__':
    main()
