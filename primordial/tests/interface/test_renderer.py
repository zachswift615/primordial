import pytest
import pygame
from unittest.mock import Mock
from primordial.interface.renderer import Renderer
from primordial.interface.config import UIConfig


def test_renderer_initialization():
    config = UIConfig()
    renderer = Renderer(config)
    assert renderer.screen is not None
    assert renderer.clock is not None


def test_renderer_frame_timing():
    config = UIConfig()
    renderer = Renderer(config)

    # First frame
    dt1 = renderer.tick()
    assert dt1 >= 0

    # Second frame should have elapsed time
    dt2 = renderer.tick()
    assert dt2 > 0


def test_renderer_get_fps():
    config = UIConfig()
    renderer = Renderer(config)

    renderer.tick()
    fps = renderer.get_fps()
    assert fps >= 0
