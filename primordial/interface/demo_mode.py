"""Demonstration mode where human directly controls the agent."""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import math


@dataclass
class DemoAction:
    """A single demonstration action."""
    action_type: str  # "move", "interact", "wait"
    move_direction: Tuple[float, float] = (0.0, 0.0)
    intensity: float = 1.0

    @classmethod
    def from_key_input(cls, up: bool, down: bool, left: bool,
                      right: bool) -> "DemoAction":
        """Create action from keyboard directional input."""
        dx = 0.0
        dy = 0.0

        if up:
            dy -= 1.0
        if down:
            dy += 1.0
        if left:
            dx -= 1.0
        if right:
            dx += 1.0

        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            magnitude = math.sqrt(dx * dx + dy * dy)
            dx /= magnitude
            dy /= magnitude

        action_type = "move" if (dx != 0 or dy != 0) else "wait"

        return cls(
            action_type=action_type,
            move_direction=(dx, dy)
        )

    @classmethod
    def from_joystick(cls, axis_x: float, axis_y: float,
                     deadzone: float = 0.1) -> "DemoAction":
        """Create action from joystick analog input."""
        # Apply deadzone
        if abs(axis_x) < deadzone:
            axis_x = 0.0
        if abs(axis_y) < deadzone:
            axis_y = 0.0

        magnitude = math.sqrt(axis_x * axis_x + axis_y * axis_y)
        action_type = "move" if magnitude > deadzone else "wait"

        return cls(
            action_type=action_type,
            move_direction=(axis_x, axis_y),
            intensity=min(magnitude, 1.0)
        )


class DemonstrationController:
    """Controls agent during human demonstration."""

    def __init__(self):
        self._active = False
        self._current_action: Optional[DemoAction] = None
        self._recorded_sequence: List[Tuple[float, DemoAction]] = []

    def activate(self) -> None:
        """Enter demonstration mode."""
        self._active = True
        self._recorded_sequence.clear()

    def deactivate(self) -> None:
        """Exit demonstration mode."""
        self._active = False
        self._current_action = None

    def is_active(self) -> bool:
        """Check if demonstration mode is active."""
        return self._active

    def set_current_action(self, action: DemoAction) -> None:
        """Set the current demonstration action."""
        self._current_action = action

    def get_current_action(self) -> Optional[DemoAction]:
        """Get the current demonstration action."""
        return self._current_action

    def record_action(self, action: DemoAction, timestamp: float) -> None:
        """Record an action in the demonstration sequence."""
        self._recorded_sequence.append((timestamp, action))

    def get_recorded_sequence(self) -> List[Tuple[float, DemoAction]]:
        """Get the full recorded demonstration sequence."""
        return self._recorded_sequence.copy()

    def clear_recording(self) -> None:
        """Clear the recorded sequence."""
        self._recorded_sequence.clear()

    def apply_to_agent(self, agent: Any, dt: float) -> Optional[Dict[str, Any]]:
        """
        Apply current demonstration action to agent.

        Args:
            agent: The agent to control
            dt: Delta time in seconds

        Returns:
            Dictionary describing the applied action, or None
        """
        if not self._active or self._current_action is None:
            return None

        action = self._current_action

        if action.action_type == "move":
            dx, dy = action.move_direction
            speed = 100.0 * action.intensity  # pixels per second

            # Apply movement (assuming agent has position attribute)
            if hasattr(agent, 'position'):
                agent.position[0] += dx * speed * dt
                agent.position[1] += dy * speed * dt

            return {
                "action": "move",
                "direction": (dx, dy),
                "speed": speed
            }

        return None
