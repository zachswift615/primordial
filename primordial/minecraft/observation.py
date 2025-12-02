"""Observation processing for Minecraft environments."""
import torch
import numpy as np
from typing import Dict, Any, Tuple
import cv2


class ObservationProcessor:
    """Converts MineDojo/MineRL observations to LRN input format."""

    def __init__(self, rgb_size: int = 64):
        """
        Args:
            rgb_size: Target size for RGB frames (64 or 128)
        """
        self.rgb_size = rgb_size

    def process(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process raw observation into LRN input tensors.

        Args:
            obs: Raw observation dict from MineDojo/MineRL

        Returns:
            vision: (1, 3, rgb_size, rgb_size) - RGB frame
            audio: (1, 100, 2) - placeholder audio
            proprio: (1, 10) - life stats
            touch: (1, 8) - damage source
        """
        vision = self._process_vision(obs)
        audio = self._process_audio(obs)
        proprio = self._process_proprio(obs)
        touch = self._process_touch(obs)

        return vision, audio, proprio, touch

    def _process_vision(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Convert RGB observation to tensor.

        Handles both MineRL (obs['pov']) and MineDojo (obs['rgb']) formats.
        """
        # Get RGB frame
        if 'pov' in obs:
            # MineRL format: (H, W, 3) uint8
            rgb = obs['pov']
        elif 'rgb' in obs:
            # MineDojo format: (H, W, 3) uint8
            rgb = obs['rgb']
        else:
            # Fallback: black frame
            rgb = np.zeros((self.rgb_size, self.rgb_size, 3), dtype=np.uint8)

        # Resize to target size
        if rgb.shape[0] != self.rgb_size or rgb.shape[1] != self.rgb_size:
            rgb = cv2.resize(rgb, (self.rgb_size, self.rgb_size), interpolation=cv2.INTER_AREA)

        # Convert to tensor: (H, W, 3) -> (1, 3, H, W), normalized to [0, 1]
        rgb_tensor = torch.from_numpy(rgb).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        return rgb_tensor

    def _process_audio(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Create placeholder audio tensor.

        MineDojo doesn't provide raw audio, so we use a placeholder.
        Could be extended to use sound event data in the future.
        """
        # Placeholder: silent audio (100 samples, stereo)
        audio = torch.zeros(1, 100, 2)
        return audio

    def _process_proprio(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Extract proprioceptive state from life_stats and location_stats.

        Output: 10 values normalized to [0, 1] or [-1, 1]:
            0: health (0-20 -> 0-1)
            1: food (0-20 -> 0-1)
            2: oxygen (0-300 -> 0-1)
            3: armor (0-20 -> 0-1)
            4: saturation (0-20 -> 0-1)
            5: xp (0-1395 -> 0-1)
            6: yaw (-180 to 180 -> -1 to 1)
            7: pitch (-90 to 90 -> -1 to 1)
            8: is_sleeping (0 or 1)
            9: compass_angle (for navigation tasks, -1 to 1)
        """
        proprio = np.zeros(10, dtype=np.float32)

        # Life stats (MineDojo format)
        if 'life_stats' in obs:
            stats = obs['life_stats']
            proprio[0] = stats.get('life', 20) / 20.0
            proprio[1] = stats.get('food', 20) / 20.0
            proprio[2] = stats.get('oxygen', 300) / 300.0
            proprio[3] = stats.get('armor', 0) / 20.0
            proprio[4] = stats.get('saturation', 5) / 20.0
            proprio[5] = stats.get('xp', 0) / 1395.0
            proprio[8] = 1.0 if stats.get('is_sleeping', False) else 0.0

        # Location stats (MineDojo format)
        if 'location_stats' in obs:
            loc = obs['location_stats']
            proprio[6] = loc.get('yaw', 0) / 180.0  # Normalize to [-1, 1]
            proprio[7] = loc.get('pitch', 0) / 90.0

        # Compass angle (MineRL NavigateDense provides this)
        if 'compassAngle' in obs:
            # Normalize compass angle to [-1, 1]
            proprio[9] = obs['compassAngle'] / 180.0

        return torch.from_numpy(proprio).unsqueeze(0)  # (1, 10)

    def _process_touch(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Extract damage source information.

        Output: 8 values:
            0: damage_amount (0-40 -> 0-1)
            1: damage_yaw (-180 to 180 -> -1 to 1)
            2: damage_pitch (-90 to 90 -> -1 to 1)
            3: damage_distance (0-100 -> 0-1)
            4: is_explosion (0 or 1)
            5: is_fire_damage (0 or 1)
            6: is_magic_damage (0 or 1)
            7: is_projectile (0 or 1)
        """
        touch = np.zeros(8, dtype=np.float32)

        if 'damage_source' in obs:
            dmg = obs['damage_source']
            touch[0] = min(dmg.get('damage_amount', 0) / 40.0, 1.0)
            touch[1] = dmg.get('damage_yaw', 0) / 180.0
            touch[2] = dmg.get('damage_pitch', 0) / 90.0
            touch[3] = min(dmg.get('damage_distance', 0) / 100.0, 1.0)
            touch[4] = 1.0 if dmg.get('is_explosion', False) else 0.0
            touch[5] = 1.0 if dmg.get('is_fire_damage', False) else 0.0
            touch[6] = 1.0 if dmg.get('is_magic_damage', False) else 0.0
            touch[7] = 1.0 if dmg.get('is_projectile', False) else 0.0

        return torch.from_numpy(touch).unsqueeze(0)  # (1, 8)

    def get_health(self, obs: Dict[str, Any]) -> float:
        """Extract current health from observation."""
        if 'life_stats' in obs:
            return obs['life_stats'].get('life', 20) / 20.0
        return 1.0

    def get_food(self, obs: Dict[str, Any]) -> float:
        """Extract current food level from observation."""
        if 'life_stats' in obs:
            return obs['life_stats'].get('food', 20) / 20.0
        return 1.0
