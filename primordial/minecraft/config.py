"""Configuration for Minecraft environment."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MinecraftConfig:
    """Configuration for Minecraft training environment."""

    # Environment settings
    env_name: str = "MineRLNavigateDense-v0"  # MineRL environment name
    rgb_size: int = 64  # Resize frames to this size (64x64 or 128x128)

    # Rendering
    render: bool = True  # Show game window
    render_fps: int = 20  # Target FPS for rendering

    # Training
    max_episode_steps: int = 6000  # ~5 minutes at 20fps
    learning_rate: float = 1e-4
    reward_scale: float = 1.0

    # Reward shaping weights
    health_reward_scale: float = 2.0
    food_reward_scale: float = 0.5
    navigation_reward_scale: float = 1.0
    movement_bonus: float = 0.01
    idle_penalty: float = -0.01
    death_penalty: float = -10.0

    # Checkpointing
    checkpoint_dir: str = "./checkpoints/minecraft"
    save_every_n_episodes: int = 10

    # Logging
    log_every_n_steps: int = 100
    verbose: bool = True
