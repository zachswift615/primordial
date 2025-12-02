"""Training level configurations for single-player mode.

Each training level is designed to develop specific agent capabilities
through environmental pressure. Levels progress from basic survival
to advanced tactical challenges.

Philosophy: "Stats are autobiography, not biography" - agents develop
through what they actually experience, not through assigned values.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from primordial.simulation.config import SimulationConfig


@dataclass
class TrainingLevel:
    """Definition of a training level.

    Attributes:
        name: Display name of the level.
        description: Brief description of the level's challenge.
        trains: Stats this level primarily develops.
        config_overrides: SimulationConfig values to override.
        star_thresholds: Survival times (seconds) for 1/2/3 stars.
        unlocked_by: Level name that must be completed to unlock this.
    """
    name: str
    description: str
    trains: str  # e.g., "Vision Range, Basic Movement"
    config_overrides: Dict[str, Any]
    star_thresholds: tuple  # (1-star seconds, 2-star, 3-star)
    unlocked_by: Optional[str] = None  # None means always unlocked


# Level Definitions
TRAINING_LEVELS: Dict[str, TrainingLevel] = {
    "boot_camp": TrainingLevel(
        name="Boot Camp",
        description="Open space with abundant food. No predators. Learn to find and eat food efficiently.",
        trains="Vision Range, Basic Movement",
        config_overrides={
            "world_width": 600,
            "world_height": 600,
            "max_agents": 3,
            "initial_food": 80,  # Abundant food
            "max_food": 120,
            "predator_count": 0,  # No predators
            "vegetation_clusters": 2,  # Minimal obstacles
            "water_bodies": 1,
        },
        star_thresholds=(60, 120, 180),  # 1/2/3 minutes
        unlocked_by=None,  # Always unlocked
    ),

    "gauntlet": TrainingLevel(
        name="The Gauntlet",
        description="Narrow corridor with one slow predator. Learn evasion timing and when to run.",
        trains="Reaction Time, Speed",
        config_overrides={
            "world_width": 400,  # Narrow
            "world_height": 800,  # Long corridor
            "max_agents": 2,
            "initial_food": 30,
            "max_food": 50,
            "predator_count": 1,  # One slow predator
            "vegetation_clusters": 4,  # Some cover
            "water_bodies": 1,
        },
        star_thresholds=(45, 90, 150),
        unlocked_by="boot_camp",
    ),

    "maze": TrainingLevel(
        name="The Maze",
        description="Complex space with scattered obstacles. No predators. Learn pathfinding and exploration.",
        trains="Vision Range, Navigation",
        config_overrides={
            "world_width": 700,
            "world_height": 700,
            "max_agents": 2,
            "initial_food": 20,  # Sparse, scattered
            "max_food": 40,
            "predator_count": 0,
            "vegetation_clusters": 15,  # Many obstacles
            "water_bodies": 3,
        },
        star_thresholds=(60, 120, 200),
        unlocked_by="gauntlet",
    ),

    "arena": TrainingLevel(
        name="The Arena",
        description="Open arena with 2-3 varied predators and scattered cover. Handle multiple threats.",
        trains="Agility, Reaction Time",
        config_overrides={
            "world_width": 800,
            "world_height": 800,
            "max_agents": 3,
            "initial_food": 40,
            "max_food": 60,
            "predator_count": 3,  # Multiple predators
            "vegetation_clusters": 6,  # Some cover
            "water_bodies": 2,
        },
        star_thresholds=(30, 75, 120),
        unlocked_by="maze",
    ),

    "drought": TrainingLevel(
        name="The Drought",
        description="Sparse food, one water source, patrolling predator. Master energy management.",
        trains="Energy Efficiency, Stamina",
        config_overrides={
            "world_width": 600,
            "world_height": 600,
            "max_agents": 2,
            "initial_food": 10,  # Very sparse
            "max_food": 20,
            "predator_count": 1,
            "vegetation_clusters": 3,
            "water_bodies": 1,  # Single water source
        },
        star_thresholds=(60, 120, 240),
        unlocked_by="arena",
    ),

    "fortress": TrainingLevel(
        name="The Fortress",
        description="Many obstacles, aggressive predators. Learn to use environment defensively.",
        trains="Tactics",
        config_overrides={
            "world_width": 700,
            "world_height": 700,
            "max_agents": 2,
            "initial_food": 30,
            "max_food": 50,
            "predator_count": 2,
            "vegetation_clusters": 12,  # Many hiding spots
            "water_bodies": 2,
        },
        star_thresholds=(45, 90, 150),
        unlocked_by="drought",
    ),

    "pack_hunt": TrainingLevel(
        name="The Pack Hunt",
        description="Multiple coordinated predators. Survive against hunting packs.",
        trains="All Evasion Stats",
        config_overrides={
            "world_width": 900,
            "world_height": 900,
            "max_agents": 3,
            "initial_food": 40,
            "max_food": 60,
            "predator_count": 4,  # Many predators
            "vegetation_clusters": 8,
            "water_bodies": 2,
        },
        star_thresholds=(30, 60, 120),
        unlocked_by="fortress",
    ),

    "ultimate": TrainingLevel(
        name="Ultimate Challenge",
        description="All hazards combined. The true test of mastery. Can your agent survive?",
        trains="Everything",
        config_overrides={
            "world_width": 1000,
            "world_height": 1000,
            "max_agents": 4,
            "initial_food": 25,  # Sparse
            "max_food": 40,
            "predator_count": 5,  # Many predators
            "vegetation_clusters": 10,
            "water_bodies": 2,
        },
        star_thresholds=(30, 60, 120),
        unlocked_by="pack_hunt",
    ),
}


def get_level_config(level_key: str, base_config: Optional[SimulationConfig] = None) -> SimulationConfig:
    """Get a SimulationConfig for a training level.

    Args:
        level_key: Key of the level in TRAINING_LEVELS.
        base_config: Base config to modify. If None, uses defaults.

    Returns:
        SimulationConfig with level overrides applied.

    Raises:
        KeyError: If level_key is not found.
    """
    if level_key not in TRAINING_LEVELS:
        raise KeyError(f"Unknown training level: {level_key}")

    level = TRAINING_LEVELS[level_key]

    # Start with base config or defaults
    config_dict = (base_config.to_dict() if base_config else SimulationConfig().to_dict())

    # Apply level overrides
    config_dict.update(level.config_overrides)

    return SimulationConfig.from_dict(config_dict)


def get_level_list() -> list:
    """Get list of all training levels with their info.

    Returns:
        List of dicts with level info.
    """
    levels = []
    for key, level in TRAINING_LEVELS.items():
        levels.append({
            'key': key,
            'name': level.name,
            'description': level.description,
            'trains': level.trains,
            'unlocked_by': level.unlocked_by,
            'star_thresholds': level.star_thresholds,
        })
    return levels


def check_level_unlocked(level_key: str, completed_levels: set) -> bool:
    """Check if a level is unlocked based on completed levels.

    Args:
        level_key: Level to check.
        completed_levels: Set of completed level keys.

    Returns:
        True if level is unlocked.
    """
    if level_key not in TRAINING_LEVELS:
        return False

    level = TRAINING_LEVELS[level_key]

    # No prerequisite means always unlocked
    if level.unlocked_by is None:
        return True

    # Check if prerequisite is completed
    return level.unlocked_by in completed_levels


def get_star_rating(level_key: str, survival_time: float) -> int:
    """Get star rating for a survival time on a level.

    Args:
        level_key: Level played.
        survival_time: Time survived in seconds.

    Returns:
        Star rating (0-3).
    """
    if level_key not in TRAINING_LEVELS:
        return 0

    level = TRAINING_LEVELS[level_key]
    thresholds = level.star_thresholds

    stars = 0
    for threshold in thresholds:
        if survival_time >= threshold:
            stars += 1
        else:
            break

    return stars
