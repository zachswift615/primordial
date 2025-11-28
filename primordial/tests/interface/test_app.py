import pytest
import pygame
from unittest.mock import Mock, patch
from primordial.interface.app import TeachingApp
from primordial.interface.config import UIConfig


def test_app_initialization():
    config = UIConfig()
    app = TeachingApp(config)

    assert app.running is False
    assert app.renderer is not None
    assert app.input_handler is not None


def test_app_start_stop():
    config = UIConfig()
    app = TeachingApp(config)

    # App starts not running
    assert not app.running

    # Can manually stop
    app.stop()
    assert not app.running


@patch('pygame.event.get')
def test_app_processes_quit_event(mock_event_get):
    config = UIConfig()
    app = TeachingApp(config)

    # Mock quit event
    quit_event = Mock()
    quit_event.type = pygame.QUIT
    mock_event_get.return_value = [quit_event]

    app.running = True
    app._process_events(0.0)

    assert not app.running
