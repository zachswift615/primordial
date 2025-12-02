#!/usr/bin/env python3
"""Run LRN agent in Minecraft via MineRL.

Usage:
    python -m primordial.scripts.run_minecraft [options]

Examples:
    # Run with rendering (watch the agent play)
    python -m primordial.scripts.run_minecraft --render

    # Run headless for faster training
    python -m primordial.scripts.run_minecraft --no-render --episodes 100

    # Resume from checkpoint
    python -m primordial.scripts.run_minecraft --checkpoint ./checkpoints/minecraft/agent.pt

    # Use 128x128 resolution (more detail, slower)
    python -m primordial.scripts.run_minecraft --rgb-size 128
"""

import argparse
import sys
from pathlib import Path


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []

    try:
        import minerl
    except ImportError:
        missing.append("minerl")

    try:
        import gym
    except ImportError:
        missing.append("gym")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        print("\nNote: MineRL also requires Java 8. See https://minerl.readthedocs.io/")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LRN agent in Minecraft",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="MineRLNavigateDense-v0",
        help="MineRL environment name (default: MineRLNavigateDense-v0)"
    )
    parser.add_argument(
        "--rgb-size",
        type=int,
        default=64,
        choices=[64, 128],
        help="RGB frame size (default: 64)"
    )

    # Rendering
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Show game window (default: True)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless (faster training)"
    )

    # Training
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Max steps per episode (default: 6000, ~5 min)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/minecraft",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N episodes (default: 5)"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run on (default: auto)"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed logs"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )

    return parser.parse_args()


def get_device(requested: str) -> str:
    """Determine best available device."""
    import torch

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    args = parse_args()

    # Check dependencies
    check_dependencies()

    # Handle render flag
    render = args.render and not args.no_render

    # Get device
    device = get_device(args.device)

    print("=" * 60)
    print("Primordial LRN - Minecraft Training")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"RGB Size: {args.rgb_size}x{args.rgb_size}")
    print(f"Device: {device}")
    print(f"Render: {render}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Import after dependency check
    from primordial.minecraft.config import MinecraftConfig
    from primordial.minecraft.wrapper import MinecraftAgentWrapper

    # Create config
    config = MinecraftConfig(
        env_name=args.env,
        rgb_size=args.rgb_size,
        render=render,
        max_episode_steps=args.max_steps,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_episodes=args.save_every,
        verbose=not args.quiet,
    )

    # Create agent
    print("\nInitializing agent...")
    agent = MinecraftAgentWrapper(config=config, device=device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load_checkpoint(args.checkpoint)

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    try:
        for episode in range(args.episodes):
            stats = agent.run_episode(render=render, max_steps=args.max_steps)

            print(f"\nEpisode {stats['episode']}: "
                  f"steps={stats['steps']}, "
                  f"reward={stats['reward']:.2f}, "
                  f"fps={stats['fps']:.1f}")

            # Save checkpoint periodically
            if (episode + 1) % args.save_every == 0:
                agent.save_checkpoint()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    finally:
        # Save final checkpoint
        print("\nSaving final checkpoint...")
        agent.save_checkpoint(
            Path(args.checkpoint_dir) / "minecraft_agent_final.pt"
        )
        agent.close()

    print("\nTraining complete!")
    print(f"Total steps: {agent.step_count}")
    print(f"Total episodes: {agent.episode_count}")


if __name__ == "__main__":
    main()
