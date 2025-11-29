"""Agent genome for heritable hyperparameters.

The genome system defines agent capabilities and is designed for future
breeding and evolution. Each trait can be inherited from parents and
mutated during reproduction.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List


# Stats that can be trained through behavior
TRAINABLE_STATS: List[str] = [
    'max_speed',
    'vision_range',
    'audio_range',  # hearing
    'max_angular_speed',  # agility/reaction time
    'eating_efficiency',  # energy efficiency
    'max_health',
    'damage_resistance',  # stamina
]

# Configuration for stat training
TRAINING_CONFIG = {
    'permanent_ratio': 0.20,  # 20% of gains become permanent
    'decay_rate': 0.001,  # 1% per minute = 0.01/60 per second
    'base_gain': 0.1,  # Base stat gain per qualifying event
    'diminishing_factor': 0.1,  # Higher = faster diminishing returns
}


@dataclass
class AgentGenome:
    """Heritable hyperparameters that define agent capabilities.

    Attributes:
        Physical traits:
            max_speed: Maximum velocity in units/sec.
            max_angular_speed: Maximum rotation rate in radians/sec.
            thrust_force: Forward/backward force magnitude.
            torque_force: Rotational force magnitude.
            radius: Collision radius in units.
            mass: Physics mass affecting acceleration.

        Sensory capabilities:
            vision_range: Maximum vision ray distance.
            vision_fov: Field of view in degrees.
            vision_rays: Number of vision rays cast.
            audio_range: Maximum hearing distance.
            touch_range: Touch sensor reach beyond radius.

        Metabolic parameters:
            base_energy_cost: Energy/sec while idle.
            movement_energy_mult: Energy multiplier for thrust/torque.
            vocalize_energy_mult: Energy multiplier for sound production.
            eating_efficiency: Energy gained per food unit.

        Health parameters:
            max_health: Maximum health value.
            max_energy: Maximum energy value.
            damage_resistance: Multiplier on incoming damage (higher = less damage).
            healing_rate: Health/sec when energy > 50%.

        Neural architecture hints:
            hidden_dim: Hidden dimension for LRN.
            num_layers: Number of layers for LRN.
            learning_rate: Learning rate for online training.

        Mutation parameters:
            mutation_rate: Probability per gene of mutating.
            mutation_scale: Std dev of gaussian mutation as fraction of value.
    """

    # Physical traits
    max_speed: float = 150.0
    max_angular_speed: float = 3.0
    thrust_force: float = 500.0
    torque_force: float = 1000.0
    radius: float = 8.0
    mass: float = 1.0

    # Sensory capabilities
    vision_range: float = 200.0
    vision_fov: float = 120.0  # degrees
    vision_rays: int = 32
    audio_range: float = 300.0
    touch_range: float = 15.0

    # Metabolic parameters (reduced costs to give agents more time to learn)
    base_energy_cost: float = 0.1  # 50% slower drain (was 0.2)
    movement_energy_mult: float = 0.5  # 50% slower drain (was 1.0)
    vocalize_energy_mult: float = 1.0  # Reduced from 1.5
    eating_efficiency: float = 0.9  # Increased from 0.8 - get more from food

    # Health parameters
    max_health: float = 100.0
    max_energy: float = 100.0
    damage_resistance: float = 1.0
    healing_rate: float = 0.1

    # Neural architecture hints
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 0.001

    # Mutation parameters
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1

    # Training gains (earned through behavior, not mutation)
    # These are stored separately and added to base stats
    permanent_gains: Dict[str, float] = field(default_factory=dict)
    active_gains: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize training gains dicts if empty."""
        if not self.permanent_gains:
            self.permanent_gains = {stat: 0.0 for stat in TRAINABLE_STATS}
        if not self.active_gains:
            self.active_gains = {stat: 0.0 for stat in TRAINABLE_STATS}

    def get_effective_stat(self, stat_name: str) -> float:
        """Get the effective value of a stat including training gains.

        Args:
            stat_name: Name of the stat to retrieve.

        Returns:
            Base stat + permanent gains + active gains.
        """
        base_value = getattr(self, stat_name)
        permanent = self.permanent_gains.get(stat_name, 0.0)
        active = self.active_gains.get(stat_name, 0.0)
        return base_value + permanent + active

    def apply_training_gain(self, stat_name: str, amount: float) -> float:
        """Apply a training gain to a stat with diminishing returns.

        Splits gain into permanent (20%) and active (80%) portions.

        Args:
            stat_name: Stat to train.
            amount: Raw gain amount (before diminishing returns).

        Returns:
            Actual gain applied after diminishing returns.
        """
        if stat_name not in TRAINABLE_STATS:
            return 0.0

        # Calculate diminishing returns based on current total gains
        current_gains = self.permanent_gains.get(stat_name, 0.0) + self.active_gains.get(stat_name, 0.0)
        base_value = getattr(self, stat_name)
        gain_ratio = current_gains / base_value if base_value > 0 else 0

        # Diminishing returns: gain decreases as total gains increase
        diminishing = 1.0 / (1.0 + gain_ratio * TRAINING_CONFIG['diminishing_factor'] * 10)
        actual_gain = amount * diminishing

        # Split into permanent and active
        permanent_portion = actual_gain * TRAINING_CONFIG['permanent_ratio']
        active_portion = actual_gain * (1.0 - TRAINING_CONFIG['permanent_ratio'])

        self.permanent_gains[stat_name] = self.permanent_gains.get(stat_name, 0.0) + permanent_portion
        self.active_gains[stat_name] = self.active_gains.get(stat_name, 0.0) + active_portion

        return actual_gain

    def decay_active_gains(self, dt: float) -> None:
        """Apply decay to active gains over time.

        Args:
            dt: Time step in seconds.
        """
        decay_factor = 1.0 - (TRAINING_CONFIG['decay_rate'] * dt)
        for stat_name in TRAINABLE_STATS:
            if stat_name in self.active_gains:
                self.active_gains[stat_name] *= decay_factor

    def get_training_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all training gains.

        Returns:
            Dict with stat names as keys, containing base/permanent/active/total.
        """
        summary = {}
        for stat_name in TRAINABLE_STATS:
            base = getattr(self, stat_name)
            permanent = self.permanent_gains.get(stat_name, 0.0)
            active = self.active_gains.get(stat_name, 0.0)
            summary[stat_name] = {
                'base': base,
                'permanent': permanent,
                'active': active,
                'total': base + permanent + active,
            }
        return summary

    def mutate(self) -> AgentGenome:
        """Create mutated copy for offspring.

        Applies gaussian mutation to each numeric trait with probability
        equal to mutation_rate. The mutation magnitude is scaled by
        mutation_scale * current_value.

        Returns:
            New AgentGenome with mutated values.
        """
        child = copy.deepcopy(self)

        # Physical traits that can mutate
        physical_traits = [
            "max_speed",
            "max_angular_speed",
            "thrust_force",
            "torque_force",
            "radius",
            "mass",
        ]
        for trait in physical_traits:
            if random.random() < self.mutation_rate:
                current = getattr(child, trait)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, trait, max(0.1, current + delta))

        # Sensory traits
        sensory_traits = ["vision_range", "vision_fov", "audio_range", "touch_range"]
        for trait in sensory_traits:
            if random.random() < self.mutation_rate:
                current = getattr(child, trait)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, trait, max(10.0, current + delta))

        # Metabolic traits
        metabolic_traits = [
            "base_energy_cost",
            "movement_energy_mult",
            "vocalize_energy_mult",
            "eating_efficiency",
        ]
        for trait in metabolic_traits:
            if random.random() < self.mutation_rate:
                current = getattr(child, trait)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, trait, max(0.01, current + delta))

        # Health traits
        health_traits = ["max_health", "max_energy", "damage_resistance", "healing_rate"]
        for trait in health_traits:
            if random.random() < self.mutation_rate:
                current = getattr(child, trait)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, trait, max(0.01, current + delta))

        return child

    def to_dict(self) -> Dict[str, Any]:
        """Serialize genome for saving/loading.

        Returns:
            Dictionary with all genome values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentGenome:
        """Deserialize genome from saved data.

        Args:
            data: Dictionary with genome values.

        Returns:
            New AgentGenome instance.
        """
        # Filter to only known fields
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


def create_default_genome() -> AgentGenome:
    """Create a baseline genome for initial population.

    Returns:
        AgentGenome with default values tuned for ~60 second survival.
    """
    return AgentGenome()


def breed(parent1: AgentGenome, parent2: AgentGenome, mutation_chance: float = 0.1) -> AgentGenome:
    """Create offspring genome from two parents.

    Each trait is randomly inherited from either parent. There's a chance
    that mutation is applied to the result.

    Args:
        parent1: First parent genome.
        parent2: Second parent genome.
        mutation_chance: Probability of applying mutation (default 10%).

    Returns:
        New child genome (possibly mutated).
    """
    # Start with copy of parent1
    child = copy.deepcopy(parent1)

    # Randomly inherit each trait from either parent
    for f in fields(child):
        if random.random() < 0.5:
            setattr(child, f.name, getattr(parent2, f.name))

    # 10% chance of mutation
    if random.random() < mutation_chance:
        return child.mutate()
    return child
