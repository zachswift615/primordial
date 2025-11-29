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

    # Agents command (database management)
    agents_parser = subparsers.add_parser(
        'agents',
        help='Manage saved agents database'
    )
    agents_subparsers = agents_parser.add_subparsers(dest='agents_command')

    # agents list
    list_parser = agents_subparsers.add_parser('list', help='List saved agents')
    list_parser.add_argument('--sort', choices=['longest_life', 'total_food_eaten', 'offspring_count', 'generation', 'saved_at'],
                             default='longest_life', help='Sort by column')
    list_parser.add_argument('--limit', type=int, default=20, help='Max results')
    list_parser.add_argument('--asc', action='store_true', help='Sort ascending')

    # agents show
    show_parser = agents_subparsers.add_parser('show', help='Show agent details')
    show_parser.add_argument('id', type=int, help='Agent database ID')

    # agents delete
    del_parser = agents_subparsers.add_parser('delete', help='Delete an agent')
    del_parser.add_argument('id', type=int, help='Agent database ID')

    # agents stats
    agents_subparsers.add_parser('stats', help='Show database statistics')

    args = parser.parse_args()

    if args.command == 'experiment':
        run_experiment_command(args)
    elif args.command == 'simulate':
        run_simulate(args)
    elif args.command == 'agents':
        run_agents(args)
    elif args.command == 'interface' or args.command is None:
        run_interface(args if args.command else parser.parse_args(['interface']))


def run_interface(args):
    """Run the teaching interface with live simulation."""
    from primordial.interface.config import UIConfig
    from primordial.interface.integrated_app import IntegratedTeachingApp
    from primordial.simulation.config import SimulationConfig

    ui_config = UIConfig()
    ui_config.fps = args.fps
    ui_config.window_width = args.width
    ui_config.window_height = args.height

    sim_config = SimulationConfig(
        world_width=640,
        world_height=480,
        max_agents=10,
        predator_count=2,
        initial_food=30,
        learning_enabled=True,
    )

    app = IntegratedTeachingApp(ui_config, sim_config)

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


def run_agents(args):
    """Manage the agents database."""
    from primordial.simulation.agent_database import AgentDatabase

    db = AgentDatabase()

    if args.agents_command == 'list':
        agents = db.list_agents(
            order_by=args.sort,
            descending=not args.asc,
            limit=args.limit
        )
        print(f"\n=== Agents (sorted by {args.sort}) ===")
        print(db.format_agent_list(agents))
        print()

    elif args.agents_command == 'show':
        agent = db.get_agent(args.id)
        if agent:
            print(f"\n=== Agent #{agent.id}: {agent.name} ===")
            print(f"  Generation: {agent.generation}")
            print(f"  Total food eaten: {agent.total_food_eaten}")
            print(f"  Times bred: {agent.times_bred}")
            print(f"  Offspring count: {agent.offspring_count}")
            print(f"  Deaths: {agent.deaths}")
            print(f"  Total time alive: {agent.total_time_alive:.1f}s")
            print(f"  Longest life: {agent.longest_life:.1f}s")
            print(f"  Damage taken: {agent.damage_taken:.1f}")
            print(f"  Notes: {agent.notes or '(none)'}")
            print(f"  Model path: {agent.model_path}")
            print()
        else:
            print(f"Agent {args.id} not found")

    elif args.agents_command == 'delete':
        if db.delete_agent(args.id):
            print(f"Deleted agent {args.id}")
        else:
            print(f"Agent {args.id} not found")

    elif args.agents_command == 'stats':
        stats = db.get_stats()
        print(f"\n=== Agent Database Statistics ===")
        print(f"  Total agents: {stats['total_agents']}")
        print(f"  Best longest life: {stats['max_longest_life']:.1f}s")
        print(f"  Most food eaten: {stats['max_food_eaten']}")
        print(f"  Most offspring: {stats['max_offspring']}")
        print(f"  Highest generation: {stats['max_generation']}")
        print()

    else:
        print("Use: python -m primordial agents [list|show|delete|stats]")
        print("  list  - List saved agents (--sort, --limit, --asc)")
        print("  show  - Show agent details (id)")
        print("  delete - Delete an agent (id)")
        print("  stats - Show database statistics")


if __name__ == "__main__":
    main()
