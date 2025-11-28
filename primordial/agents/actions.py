"""Agent action definitions.

Actions represent the continuous outputs from the neural network
that control agent behavior in the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AgentAction:
    """Continuous action outputs from neural network.

    All action values are continuous and bounded:
    - thrust: Forward/backward force (-1.0 to 1.0)
    - torque: Rotation force (-1.0 to 1.0)
    - vocalize: Sound production [frequency, amplitude] (0.0 to 1.0 each)
    - eat: Eating effort (0.0 to 1.0, only works when touching food)

    Attributes:
        thrust: Forward (positive) or backward (negative) thrust.
        torque: Counter-clockwise (positive) or clockwise (negative) rotation.
        vocalize: Array of [frequency, amplitude] for sound production.
        eat: Effort applied to consuming nearby food.
    """

    thrust: float
    torque: float
    vocalize: np.ndarray  # (2,) [frequency, amplitude]
    eat: float

    def __post_init__(self):
        """Ensure vocalize is proper array."""
        if not isinstance(self.vocalize, np.ndarray):
            self.vocalize = np.array(self.vocalize, dtype=np.float32)

    @classmethod
    def from_tensor(cls, action_tensor: torch.Tensor) -> AgentAction:
        """Create action from neural network output tensor.

        Expected tensor shape: (5,) with layout:
        [thrust, torque, vocalize_freq, vocalize_amp, eat]

        Network outputs should use:
        - tanh for thrust and torque (range: -1 to 1)
        - sigmoid for vocalize_freq, vocalize_amp, eat (range: 0 to 1)

        Args:
            action_tensor: Output tensor from neural network.

        Returns:
            AgentAction with clamped values.

        Raises:
            AssertionError: If tensor shape is not (5,).
        """
        assert action_tensor.shape == (5,), f"Expected shape (5,), got {action_tensor.shape}"

        arr = action_tensor.detach().cpu().numpy()
        return cls(
            thrust=float(np.clip(arr[0], -1.0, 1.0)),
            torque=float(np.clip(arr[1], -1.0, 1.0)),
            vocalize=np.clip(arr[2:4], 0.0, 1.0).astype(np.float32),
            eat=float(np.clip(arr[4], 0.0, 1.0)),
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert action to tensor for logging/analysis.

        Returns:
            Tensor of shape (5,) with action values.
        """
        return torch.tensor(
            [
                self.thrust,
                self.torque,
                self.vocalize[0],
                self.vocalize[1],
                self.eat,
            ],
            dtype=torch.float32,
        )

    @classmethod
    def zero(cls) -> AgentAction:
        """Create a zero/idle action.

        Returns:
            AgentAction with all values at zero/neutral.
        """
        return cls(
            thrust=0.0,
            torque=0.0,
            vocalize=np.zeros(2, dtype=np.float32),
            eat=0.0,
        )

    @classmethod
    def random(cls) -> AgentAction:
        """Create a random action (useful for testing).

        Returns:
            AgentAction with random values in valid ranges.
        """
        return cls(
            thrust=np.random.uniform(-1.0, 1.0),
            torque=np.random.uniform(-1.0, 1.0),
            vocalize=np.random.uniform(0.0, 1.0, size=2).astype(np.float32),
            eat=np.random.uniform(0.0, 1.0),
        )
