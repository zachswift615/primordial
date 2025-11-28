import pytest
from primordial.interface.config import UIConfig, Colors, Layout, KeyBindings


def test_ui_config_creates_valid_dimensions():
    config = UIConfig()
    assert config.window_width == 960
    assert config.window_height == 720
    assert config.fps == 60


def test_layout_viewport_dimensions():
    layout = Layout()
    # World view
    assert layout.world_view_rect == (0, 40, 640, 480)
    # Agent POV
    assert layout.agent_pov_rect == (640, 40, 320, 240)
    # Status panel
    assert layout.status_rect == (640, 280, 320, 120)


def test_keybindings_are_defined():
    kb = KeyBindings()
    assert kb.REWARD is not None
    assert kb.PUNISH is not None
    assert kb.CONTROL is not None
    assert kb.POINT is not None
