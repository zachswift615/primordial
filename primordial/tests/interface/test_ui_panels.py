import pytest
import pygame
import numpy as np
from unittest.mock import Mock
from primordial.interface.ui_panels import (
    HeaderPanel, WorldViewPanel, AgentPOVPanel,
    StatusPanel, MetricsPanel, WaveformPanel, ControlsPanel
)
from primordial.interface.config import UIConfig


@pytest.fixture
def pygame_surface():
    """Create a test pygame surface."""
    pygame.init()
    return pygame.Surface((960, 720))


def test_header_panel_renders(pygame_surface):
    config = UIConfig()
    panel = HeaderPanel(config)

    # Should not raise exception
    panel.render(pygame_surface, fps=60, recording=True)


def test_status_panel_displays_agent_state(pygame_surface):
    config = UIConfig()
    panel = StatusPanel(config)

    agent_state = {
        "energy": 0.8,
        "health": 1.0,
        "age": 42.5,
        "survival_time": 83.0
    }

    # Should not raise exception
    panel.render(pygame_surface, agent_state=agent_state, mode="OBSERVE")


def test_waveform_panel_renders_audio(pygame_surface):
    config = UIConfig()
    panel = WaveformPanel(config)

    # Create test waveform
    waveform = np.sin(np.linspace(0, 10 * np.pi, 1000))

    # Should not raise exception
    panel.render(pygame_surface, waveform=waveform)


def test_metrics_panel_displays_learning_stats(pygame_surface):
    config = UIConfig()
    panel = MetricsPanel(config)

    metrics = {
        "loss": 0.0234,
        "rewards": 12,
        "punishments": 3,
        "demonstrations": 5,
        "voice_samples": 47
    }

    # Should not raise exception
    panel.render(pygame_surface, metrics=metrics)
