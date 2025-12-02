"""Minecraft integration for Primordial LRN.

This module provides integration with MineDojo to train LRN agents
in Minecraft environments.
"""

from .config import MinecraftConfig
from .wrapper import MinecraftAgentWrapper

__all__ = ['MinecraftConfig', 'MinecraftAgentWrapper']
