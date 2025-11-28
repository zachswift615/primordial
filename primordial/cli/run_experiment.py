"""CLI for running experiments."""

import argparse
from pathlib import Path

from primordial.simulation.config import SimulationConfig


EXPERIMENTS = {
    'survival': 'SurvivalBaselineExperiment',
    'teaching': 'TeachingImpactExperiment',
}


def list_experiments():
    """List available experiments."""
    print("\nAvailable Experiments:")
    print("=" * 40)
    print("  survival  - Learning ON vs OFF comparison")
    print("             Target: >5x survival improvement")
    print()
    print("  teaching  - Human teaching impact")
    print("             Target: >2x learning acceleration")
    print()
    print("Usage:")
    print("  python -m primordial experiment survival --steps 10000")
    print("  python -m primordial experiment teaching --output ./results")
    print()


def run_survival(args):
    """Run survival baseline experiment."""
    from primordial.experiments.survival_baseline import SurvivalBaselineExperiment

    config = SimulationConfig(
        world_width=args.world_size,
        world_height=args.world_size,
        predator_count=args.predators,
        initial_food=args.food,
        seed=args.seed,
        render_enabled=False,
    )

    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials
    )

    results = exp.run()
    exp.print_summary()

    if args.output:
        exp.export_results(args.output)
        print(f"\nResults exported to: {args.output}")

    return results


def run_teaching(args):
    """Run teaching impact experiment."""
    from primordial.experiments.teaching_impact import TeachingImpactExperiment

    config = SimulationConfig(
        world_width=args.world_size,
        world_height=args.world_size,
        predator_count=args.predators,
        initial_food=args.food,
        seed=args.seed,
        render_enabled=False,
    )

    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials,
        teaching_interval=args.teaching_interval
    )

    results = exp.run()
    exp.print_summary()

    if args.output:
        exp.export_results(args.output)
        print(f"\nResults exported to: {args.output}")

    return results


def add_experiment_parser(subparsers):
    """Add experiment subparser."""
    exp_parser = subparsers.add_parser(
        'experiment',
        help='Run experiments'
    )

    exp_parser.add_argument(
        'name',
        nargs='?',
        choices=['survival', 'teaching'],
        help='Experiment name'
    )

    exp_parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )

    # Common args
    exp_parser.add_argument('--steps', type=int, default=10000,
                           help='Steps per trial (default: 10000)')
    exp_parser.add_argument('--trials', type=int, default=5,
                           help='Trials per condition (default: 5)')
    exp_parser.add_argument('--output', type=str, default=None,
                           help='Output directory for results')
    exp_parser.add_argument('--world-size', type=int, default=1000,
                           help='World size (default: 1000)')
    exp_parser.add_argument('--predators', type=int, default=3,
                           help='Number of predators (default: 3)')
    exp_parser.add_argument('--food', type=int, default=50,
                           help='Initial food count (default: 50)')
    exp_parser.add_argument('--seed', type=int, default=None,
                           help='Random seed for reproducibility')

    # Teaching-specific
    exp_parser.add_argument('--teaching-interval', type=int, default=10,
                           help='Teaching signal interval (default: 10)')

    return exp_parser


def run_experiment_command(args):
    """Handle experiment command."""
    if args.list or args.name is None:
        list_experiments()
        return

    if args.name == 'survival':
        run_survival(args)
    elif args.name == 'teaching':
        run_teaching(args)
