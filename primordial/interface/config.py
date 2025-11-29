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
    AGENT = (100, 200, 255)       # Blue - the learning agent
    FOOD = (100, 255, 100)        # Green - food to eat
    PREDATOR = (255, 80, 80)      # Red - dangerous predators
    VEGETATION = (30, 80, 30)     # Dark green - hiding spots
    WATER = (50, 100, 180)        # Blue - water bodies
    ENTITY = (255, 200, 100)      # Orange - generic entity
    REWARD = (100, 255, 100)
    PUNISH = (255, 100, 100)
    POINT_CURSOR = (255, 255, 0)
    WAVEFORM = (0, 255, 150)
    GRID = (50, 50, 60)


@dataclass
class Layout:
    """UI layout dimensions (x, y, width, height)."""
    # Header bar (wider window: 1120px total)
    header_rect: Tuple[int, int, int, int] = (0, 0, 1120, 40)

    # Main panels - right panel is now 480px wide
    world_view_rect: Tuple[int, int, int, int] = (0, 40, 640, 480)
    agent_table_rect: Tuple[int, int, int, int] = (640, 40, 480, 240)
    status_rect: Tuple[int, int, int, int] = (640, 280, 480, 120)
    metrics_rect: Tuple[int, int, int, int] = (640, 400, 480, 120)

    # Bottom panels
    waveform_rect: Tuple[int, int, int, int] = (0, 520, 640, 80)
    controls_rect: Tuple[int, int, int, int] = (0, 600, 640, 120)

    # Legacy (keep for compatibility)
    agent_pov_rect: Tuple[int, int, int, int] = (640, 40, 480, 240)


@dataclass
class KeyBindings:
    """Keyboard input mappings."""
    REWARD = pygame.K_r          # R for reward
    PUNISH = pygame.K_x          # X for punish
    PUSH_TO_TALK = pygame.K_SPACE  # SPACE for microphone
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
    window_width: int = 1120  # Wider to fit agent table
    window_height: int = 720
    fps: int = 60
    font_size: int = 24        # Increased from 16
    font_size_small: int = 18  # Increased from 12
    font_size_large: int = 28  # Increased from 20

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
