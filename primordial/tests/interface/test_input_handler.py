import pytest
import pygame
from unittest.mock import Mock
from primordial.interface.input_handler import InputHandler, InputState
from primordial.interface.config import UIConfig
from primordial.interface.teaching_signals import TeachingSignalType


def test_input_state_initialization():
    state = InputState()
    assert state.mouse_pos == (0, 0)
    assert not state.control_mode
    assert state.zoom_level == 1.0


def test_input_handler_processes_keyboard():
    config = UIConfig()
    handler = InputHandler(config)

    # Mock keyboard event
    event = Mock()
    event.type = pygame.KEYDOWN
    event.key = pygame.K_SPACE

    signals = handler.process_event(event, timestamp=1.0)
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.REWARD


def test_input_handler_punish_key():
    config = UIConfig()
    handler = InputHandler(config)

    event = Mock()
    event.type = pygame.KEYDOWN
    event.key = pygame.K_x

    signals = handler.process_event(event, timestamp=2.0)
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.PUNISH


def test_input_handler_control_mode_toggle():
    config = UIConfig()
    handler = InputHandler(config)

    event = Mock()
    event.type = pygame.KEYDOWN
    event.key = pygame.K_c

    handler.process_event(event, timestamp=1.0)
    assert handler.state.control_mode

    handler.process_event(event, timestamp=2.0)
    assert not handler.state.control_mode


def test_input_handler_mouse_click_pointing():
    config = UIConfig()
    handler = InputHandler(config)

    event = Mock()
    event.type = pygame.MOUSEBUTTONDOWN
    event.button = 1  # Left click
    event.pos = (100, 200)

    signals = handler.process_event(event, timestamp=3.0)
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.POINT
    assert signals[0].data["x"] == 100
    assert signals[0].data["y"] == 200


def test_input_handler_mouse_scroll_zoom():
    config = UIConfig()
    handler = InputHandler(config)

    # Scroll up
    event = Mock()
    event.type = pygame.MOUSEWHEEL
    event.y = 1

    handler.process_event(event, timestamp=1.0)
    assert handler.state.zoom_level > 1.0

    # Scroll down
    event.y = -1
    handler.process_event(event, timestamp=2.0)
    assert handler.state.zoom_level == 1.0  # Back to default
