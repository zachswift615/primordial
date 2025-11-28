"""Unified input handling for keyboard, mouse, and controller."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pygame
from primordial.interface.config import UIConfig
from primordial.interface.teaching_signals import TeachingSignal


@dataclass
class InputState:
    """Current state of input devices."""
    mouse_pos: Tuple[int, int] = (0, 0)
    control_mode: bool = False
    zoom_level: float = 1.0
    movement_vector: Tuple[float, float] = (0.0, 0.0)
    controller: Optional[pygame.joystick.Joystick] = None


class InputHandler:
    """Processes input from keyboard, mouse, and game controller."""

    def __init__(self, config: UIConfig):
        self.config = config
        self.state = InputState()

        # Initialize controller if available
        self._init_controller()

    def _init_controller(self) -> None:
        """Initialize game controller if connected."""
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            self.state.controller = pygame.joystick.Joystick(0)
            self.state.controller.init()
            print(f"Controller connected: {self.state.controller.get_name()}")

    def process_event(self, event: pygame.event.Event,
                     timestamp: float) -> List[TeachingSignal]:
        """
        Process a single input event and generate teaching signals.

        Args:
            event: Pygame event
            timestamp: Current timestamp in seconds

        Returns:
            List of teaching signals generated from this event
        """
        signals = []

        if event.type == pygame.KEYDOWN:
            signals.extend(self._handle_keydown(event, timestamp))
        elif event.type == pygame.KEYUP:
            signals.extend(self._handle_keyup(event, timestamp))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            signals.extend(self._handle_mouse_button(event, timestamp))
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event)
        elif event.type == pygame.MOUSEWHEEL:
            self._handle_mouse_wheel(event)
        elif event.type == pygame.JOYBUTTONDOWN:
            signals.extend(self._handle_joy_button(event, timestamp))
        elif event.type == pygame.JOYAXISMOTION:
            self._handle_joy_axis(event)

        return signals

    def _handle_keydown(self, event: pygame.event.Event,
                       timestamp: float) -> List[TeachingSignal]:
        """Handle keyboard key press."""
        signals = []

        if event.key == self.config.keys.REWARD:
            signals.append(TeachingSignal.reward(timestamp))
        elif event.key == self.config.keys.PUNISH:
            signals.append(TeachingSignal.punish(timestamp))
        elif event.key == self.config.keys.CONTROL:
            self.state.control_mode = not self.state.control_mode

        # Handle movement keys in control mode
        if self.state.control_mode:
            if event.key == self.config.keys.MOVE_UP:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_up"))
            elif event.key == self.config.keys.MOVE_DOWN:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_down"))
            elif event.key == self.config.keys.MOVE_LEFT:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_left"))
            elif event.key == self.config.keys.MOVE_RIGHT:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_right"))

        return signals

    def _handle_keyup(self, event: pygame.event.Event,
                     timestamp: float) -> List[TeachingSignal]:
        """Handle keyboard key release."""
        # Currently no signals on key release
        return []

    def _handle_mouse_button(self, event: pygame.event.Event,
                            timestamp: float) -> List[TeachingSignal]:
        """Handle mouse button click."""
        signals = []

        if event.button == 1:  # Left click
            x, y = event.pos
            signals.append(TeachingSignal.point(timestamp, x, y))

        return signals

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        """Handle mouse movement."""
        self.state.mouse_pos = event.pos

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        """Handle mouse wheel for zooming."""
        if event.y > 0:  # Scroll up
            self.state.zoom_level = min(
                self.config.zoom_max,
                self.state.zoom_level + self.config.zoom_step
            )
        elif event.y < 0:  # Scroll down
            self.state.zoom_level = max(
                self.config.zoom_min,
                self.state.zoom_level - self.config.zoom_step
            )

    def _handle_joy_button(self, event: pygame.event.Event,
                          timestamp: float) -> List[TeachingSignal]:
        """Handle game controller button press."""
        signals = []

        if event.button == self.config.controller.REWARD:
            signals.append(TeachingSignal.reward(timestamp))
        elif event.button == self.config.controller.PUNISH:
            signals.append(TeachingSignal.punish(timestamp))
        elif event.button == self.config.controller.CONTROL:
            self.state.control_mode = not self.state.control_mode

        return signals

    def _handle_joy_axis(self, event: pygame.event.Event) -> None:
        """Handle game controller joystick movement."""
        # Left stick for movement (axis 0=horizontal, 1=vertical)
        if event.axis == 0:  # Horizontal
            x, y = self.state.movement_vector
            self.state.movement_vector = (event.value, y)
        elif event.axis == 1:  # Vertical
            x, y = self.state.movement_vector
            self.state.movement_vector = (x, event.value)

    def get_continuous_movement(self) -> Tuple[float, float]:
        """Get current movement vector from controller or keyboard state."""
        return self.state.movement_vector
