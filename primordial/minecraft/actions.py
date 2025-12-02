"""Action conversion for Minecraft environments."""
import torch
import numpy as np
from typing import Dict, Any


class ActionConverter:
    """Converts LRN action outputs to Minecraft action dictionaries."""

    # Camera sensitivity (degrees per unit output)
    CAMERA_SENSITIVITY = 15.0

    def convert(self, action_tensor: torch.Tensor) -> Dict[str, Any]:
        """Convert LRN action output to MineRL/MineDojo action dict.

        Args:
            action_tensor: (1, 8) tensor from LRN action head
                [0]: forward
                [1]: back
                [2]: left
                [3]: right
                [4]: jump
                [5]: camera_pitch (vertical look)
                [6]: camera_yaw (horizontal look)
                [7]: attack

        Returns:
            Action dict compatible with MineRL/MineDojo
        """
        a = action_tensor.squeeze(0).detach().cpu().numpy()

        # Binary actions: threshold at 0.5 after sigmoid
        def to_binary(x):
            return int(torch.sigmoid(torch.tensor(x)).item() > 0.5)

        # Continuous actions: tanh then scale
        def to_camera(x):
            return float(np.tanh(x) * self.CAMERA_SENSITIVITY)

        action = {
            # Movement (binary)
            'forward': to_binary(a[0]),
            'back': to_binary(a[1]),
            'left': to_binary(a[2]),
            'right': to_binary(a[3]),
            'jump': to_binary(a[4]),

            # Camera (continuous, in degrees)
            'camera': np.array([to_camera(a[5]), to_camera(a[6])]),

            # Combat (binary)
            'attack': to_binary(a[7]),

            # Unused actions (set to defaults)
            'sneak': 0,
            'sprint': 0,
            'use': 0,
            'drop': 0,
            'inventory': 0,
            'hotbar.1': 0,
            'hotbar.2': 0,
            'hotbar.3': 0,
            'hotbar.4': 0,
            'hotbar.5': 0,
            'hotbar.6': 0,
            'hotbar.7': 0,
            'hotbar.8': 0,
            'hotbar.9': 0,
        }

        return action

    def get_no_op(self) -> Dict[str, Any]:
        """Return a no-operation action (agent does nothing)."""
        return {
            'forward': 0,
            'back': 0,
            'left': 0,
            'right': 0,
            'jump': 0,
            'camera': np.array([0.0, 0.0]),
            'attack': 0,
            'sneak': 0,
            'sprint': 0,
            'use': 0,
            'drop': 0,
            'inventory': 0,
            'hotbar.1': 0,
            'hotbar.2': 0,
            'hotbar.3': 0,
            'hotbar.4': 0,
            'hotbar.5': 0,
            'hotbar.6': 0,
            'hotbar.7': 0,
            'hotbar.8': 0,
            'hotbar.9': 0,
        }

    def add_exploration_noise(self, action: Dict[str, Any], scale: float = 0.3) -> Dict[str, Any]:
        """Add exploration noise to action.

        Args:
            action: Action dict
            scale: Noise scale (probability of flipping binary, std for camera)

        Returns:
            Modified action dict with noise
        """
        # Randomly flip binary actions
        for key in ['forward', 'back', 'left', 'right', 'jump', 'attack']:
            if np.random.random() < scale * 0.5:  # Lower flip rate
                action[key] = 1 - action[key]

        # Add noise to camera
        action['camera'] = action['camera'] + np.random.normal(0, scale * 5, size=2)

        return action
