"""Configuration constants for the teaching interface."""

from dataclasses import dataclass
from typing import Tuple
import pygame


@dataclass
class Colors:
    """Color palette for UI."""
    BACKGROUND = (20, 20, 30)
    PANEL_BG = (30, 30, 40)
    TEXT = (200, 200, 200)
    TEXT_BRIGHT = (255, 255, 255)
    AGENT = (100, 200, 255)
    ENTITY = (255, 200, 100)
    REWARD = (100, 255, 100)
    PUNISH = (255, 100, 100)
    POINT_CURSOR = (255, 255, 0)
    WAVEFORM = (0, 255, 150)
    GRID = (50, 50, 60)


@dataclass
class Layout:
    """UI layout dimensions (x, y, width, height)."""
    # Header bar
    header_rect: Tuple[int, int, int, int] = (0, 0, 960, 40)

    # Main panels
    world_view_rect: Tuple[int, int, int, int] = (0, 40, 640, 480)
    agent_pov_rect: Tuple[int, int, int, int] = (640, 40, 320, 240)
    status_rect: Tuple[int, int, int, int] = (640, 280, 320, 120)
    metrics_rect: Tuple[int, int, int, int] = (640, 400, 320, 120)

    # Bottom panels
    waveform_rect: Tuple[int, int, int, int] = (0, 520, 640, 80)
    controls_rect: Tuple[int, int, int, int] = (0, 600, 640, 120)


@dataclass
class KeyBindings:
    """Keyboard input mappings."""
    REWARD = pygame.K_SPACE
    PUNISH = pygame.K_x
    CONTROL = pygame.K_c
    POINT = 1  # Mouse button 1 (left click)
    MOVE_UP = pygame.K_UP
    MOVE_DOWN = pygame.K_DOWN
    MOVE_LEFT = pygame.K_LEFT
    MOVE_RIGHT = pygame.K_RIGHT
    SAVE = pygame.K_s
    LOAD = pygame.K_l
    QUIT = pygame.K_ESCAPE


@dataclass
class ControllerBindings:
    """Game controller button mappings."""
    REWARD = 0      # A button
    PUNISH = 1      # B button
    CONTROL = 2     # X button
    MENU = 3        # Y button


@dataclass
class UIConfig:
    """Main UI configuration."""
    window_width: int = 960
    window_height: int = 720
    fps: int = 60
    font_size: int = 16
    font_size_small: int = 12
    font_size_large: int = 20

    # Zoom settings
    zoom_min: float = 0.5
    zoom_max: float = 4.0
    zoom_step: float = 0.1

    # Audio settings
    audio_sample_rate: int = 44100
    audio_channels: int = 1
    audio_buffer_size: int = 1024
    waveform_history: int = 2048  # Samples to display

    colors: Colors = None
    layout: Layout = None
    keys: KeyBindings = None
    controller: ControllerBindings = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = Colors()
        if self.layout is None:
            self.layout = Layout()
        if self.keys is None:
            self.keys = KeyBindings()
        if self.controller is None:
            self.controller = ControllerBindings()
