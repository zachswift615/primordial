"""
Online Learning System for Primordial agents.

This module implements continual learning through real-time experience,
without batch training. Agents learn by minimizing prediction error while
survival and human teaching signals modulate the learning process.
"""

from primordial.learning.losses import PredictionLoss
from primordial.learning.rewards import (
    RewardHistoryBuffer,
    SurvivalRewards,
    HumanTeaching,
    RewardCombiner,
)
from primordial.learning.optimizer import (
    RewardModulatedOptimizer,
    OnlineLRScheduler,
)
from primordial.learning.stability import (
    GradientClipper,
    GradientAccumulator,
    ExponentialMovingAverage,
    GradientMonitor,
)
from primordial.learning.learning_loop import OnlineLearningLoop
from primordial.learning.checkpointing import DeathHandler, DeathReplay
from primordial.learning.metrics import LearningMetrics, LearningVisualizer

__all__ = [
    "PredictionLoss",
    "RewardHistoryBuffer",
    "SurvivalRewards",
    "HumanTeaching",
    "RewardCombiner",
    "RewardModulatedOptimizer",
    "OnlineLRScheduler",
    "GradientClipper",
    "GradientAccumulator",
    "ExponentialMovingAverage",
    "GradientMonitor",
    "OnlineLearningLoop",
    "DeathHandler",
    "DeathReplay",
    "LearningMetrics",
    "LearningVisualizer",
]
