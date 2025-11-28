"""Configuration for simulation."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class SimulationConfig:
    """Configuration for running a simulation.

    Attributes:
        world_width: World width in units.
        world_height: World height in units.
        tick_rate: Simulation ticks per second.
        learning_enabled: Whether online learning is active.
        max_agents: Maximum number of agents.
        initial_food: Initial food items to spawn.
        max_food: Maximum food in world.
        predator_count: Number of predators.
        checkpoint_interval: Steps between checkpoints.
        metrics_interval: Steps between metrics logging.
        render_enabled: Whether to render visualization.
        seed: Random seed for reproducibility.
    """
    # World settings
    world_width: float = 1000.0
    world_height: float = 1000.0
    tick_rate: int = 60

    # Agent settings
    max_agents: int = 1
    initial_food: int = 50
    max_food: int = 100
    predator_count: int = 3

    # Learning settings
    learning_enabled: bool = True
    checkpoint_interval: int = 1000
    metrics_interval: int = 100

    # LRN settings (passed to LRNConfig)
    lrn_hidden_dim: int = 128
    lrn_num_mixing_layers: int = 6

    # Optimizer settings
    learning_rate: float = 1e-4
    reward_modulation_scale: float = 1.0

    # Rendering
    render_enabled: bool = False

    # Reproducibility
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
