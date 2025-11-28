"""CLI entry point for Primordial."""

import argparse
import sys


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Primordial - Human Teaching Interface for AI Agents"
    )

    subparsers = parser.add_subparsers(dest='command')

    # Interface command (default)
    interface_parser = subparsers.add_parser(
        'interface',
        help='Run the teaching interface'
    )
    interface_parser.add_argument('--fps', type=int, default=60)
    interface_parser.add_argument('--width', type=int, default=960)
    interface_parser.add_argument('--height', type=int, default=720)
    interface_parser.add_argument('--no-audio', action='store_true')

    # Experiment command
    from primordial.cli.run_experiment import add_experiment_parser, run_experiment_command
    add_experiment_parser(subparsers)

    # Simulate command (headless)
    sim_parser = subparsers.add_parser(
        'simulate',
        help='Run headless simulation'
    )
    sim_parser.add_argument('--steps', type=int, default=1000)
    sim_parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'experiment':
        run_experiment_command(args)
    elif args.command == 'simulate':
        run_simulate(args)
    elif args.command == 'interface' or args.command is None:
        run_interface(args if args.command else parser.parse_args(['interface']))


def run_interface(args):
    """Run the teaching interface."""
    from primordial.interface.config import UIConfig
    from primordial.interface.app import TeachingApp

    config = UIConfig()
    config.fps = args.fps
    config.window_width = args.width
    config.window_height = args.height

    app = TeachingApp(config)

    if args.no_audio:
        app.audio_capture.stop()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        app.stop()
        sys.exit(0)


def run_simulate(args):
    """Run headless simulation."""
    from primordial.simulation.simulation import Simulation
    from primordial.simulation.config import SimulationConfig

    config = SimulationConfig(render_enabled=False)
    sim = Simulation(config)

    print(f"Running simulation for {args.steps} steps...")
    metrics = sim.run(args.steps)

    print(f"\nCompleted {len(metrics)} steps")
    print(f"Final agent survival times:")
    for agent_id, wrapper in sim.agents.items():
        print(f"  {agent_id}: {wrapper.agent.age:.2f}s")

    if args.output:
        sim.save_state(args.output)
        print(f"\nState saved to: {args.output}")


if __name__ == "__main__":
    main()
