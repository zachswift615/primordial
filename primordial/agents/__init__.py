"""Agent module for Primordial simulation.

This module provides the physical embodiment of agents including:
- AgentGenome: Heritable hyperparameters for agent capabilities
- AgentState: Complete physical and internal state
- AgentAction: Continuous action outputs from neural network
- AgentBody: Physical agent entity with sensors and actuators
"""

from primordial.agents.genome import AgentGenome, create_default_genome, breed
from primordial.agents.actions import AgentAction
from primordial.agents.body import AgentBody

__all__ = [
    "AgentGenome",
    "AgentAction",
    "AgentBody",
    "create_default_genome",
    "breed",
]
