"""Individual UI panel renderers."""

from typing import Dict, Any, Optional
import pygame
import numpy as np
from primordial.interface.config import UIConfig


class BasePanel:
    """Base class for UI panels."""

    def __init__(self, config: UIConfig):
        self.config = config
        self.font = pygame.font.Font(None, config.font_size)
        self.font_small = pygame.font.Font(None, config.font_size_small)
        self.font_large = pygame.font.Font(None, config.font_size_large)

    def draw_panel_background(self, surface: pygame.Surface, rect: tuple) -> None:
        """Draw panel background."""
        pygame.draw.rect(surface, self.config.colors.PANEL_BG, rect)
        pygame.draw.rect(surface, self.config.colors.GRID, rect, 1)


class HeaderPanel(BasePanel):
    """Top header bar with FPS and recording status."""

    def render(self, surface: pygame.Surface, fps: int, recording: bool) -> None:
        """Render header panel."""
        rect = self.config.layout.header_rect
        self.draw_panel_background(surface, rect)

        # Title
        title_text = self.font_large.render(
            "Primordial Teaching Interface",
            True,
            self.config.colors.TEXT_BRIGHT
        )
        surface.blit(title_text, (10, 10))

        # FPS counter
        fps_text = self.font.render(f"FPS: {fps}", True, self.config.colors.TEXT)
        surface.blit(fps_text, (rect[2] - 150, 10))

        # Recording indicator
        if recording:
            rec_text = self.font.render("[●REC]", True, self.config.colors.REWARD)
            surface.blit(rec_text, (rect[2] - 70, 10))


class WorldViewPanel(BasePanel):
    """Top-down view of the world."""

    def render(self, surface: pygame.Surface, world_state: Dict[str, Any],
               zoom: float = 1.0, offset: tuple = (0, 0)) -> None:
        """Render world view."""
        rect = self.config.layout.world_view_rect
        self.draw_panel_background(surface, rect)

        # Create subsurface for clipping
        viewport = surface.subsurface(rect)

        # Draw grid
        self._draw_grid(viewport, zoom, offset)

        # Draw entities if provided
        if "entities" in world_state:
            self._draw_entities(viewport, world_state["entities"], zoom, offset)

    def _draw_grid(self, surface: pygame.Surface, zoom: float, offset: tuple) -> None:
        """Draw background grid."""
        grid_size = int(50 * zoom)
        w, h = surface.get_size()

        for x in range(0, w, grid_size):
            pygame.draw.line(surface, self.config.colors.GRID, (x, 0), (x, h))
        for y in range(0, h, grid_size):
            pygame.draw.line(surface, self.config.colors.GRID, (0, y), (w, y))

    def _draw_entities(self, surface: pygame.Surface, entities: list,
                      zoom: float, offset: tuple) -> None:
        """Draw world entities."""
        import math

        # Draw in order: food first, then predators, then agents on top
        draw_order = {"food": 0, "predator": 1, "agent": 2}
        sorted_entities = sorted(entities, key=lambda e: draw_order.get(e.get("type"), 0))

        for entity in sorted_entities:
            pos = entity.get("position", (0, 0))
            entity_type = entity.get("type", "unknown")
            entity_radius = entity.get("radius", 10)

            # Transform position with zoom and offset
            screen_x = int((pos[0] - offset[0]) * zoom)
            screen_y = int((pos[1] - offset[1]) * zoom)

            # Choose color and size based on type
            if entity_type == "agent":
                color = self.config.colors.AGENT
                radius = max(8, int(entity_radius * zoom))
                # Draw agent with direction indicator
                pygame.draw.circle(surface, color, (screen_x, screen_y), radius)
                # Draw outline
                pygame.draw.circle(surface, (255, 255, 255), (screen_x, screen_y), radius, 2)
                # Draw direction indicator if angle provided
                angle = entity.get("angle", 0)
                dir_len = radius + 8
                end_x = screen_x + int(math.cos(angle) * dir_len)
                end_y = screen_y + int(math.sin(angle) * dir_len)
                pygame.draw.line(surface, (255, 255, 255), (screen_x, screen_y), (end_x, end_y), 3)

            elif entity_type == "food":
                color = self.config.colors.FOOD
                radius = max(4, int(entity_radius * zoom * 0.8))
                pygame.draw.circle(surface, color, (screen_x, screen_y), radius)

            elif entity_type == "predator":
                color = self.config.colors.PREDATOR
                radius = max(6, int(entity_radius * zoom))
                # Draw predator as a triangle pointing in direction
                angle = entity.get("angle", 0)
                points = []
                for i in range(3):
                    a = angle + i * (2 * math.pi / 3)
                    px = screen_x + int(math.cos(a) * radius)
                    py = screen_y + int(math.sin(a) * radius)
                    points.append((px, py))
                pygame.draw.polygon(surface, color, points)
                pygame.draw.polygon(surface, (255, 100, 100), points, 2)

            else:
                color = self.config.colors.ENTITY
                radius = max(4, int(entity_radius * zoom))
                pygame.draw.circle(surface, color, (screen_x, screen_y), radius)


class AgentPOVPanel(BasePanel):
    """First-person view from agent's perspective."""

    def render(self, surface: pygame.Surface, agent_view: Optional[np.ndarray]) -> None:
        """Render agent POV."""
        rect = self.config.layout.agent_pov_rect
        self.draw_panel_background(surface, rect)

        if agent_view is not None:
            # Convert numpy array to pygame surface
            # Assuming agent_view is RGB image array
            view_surface = pygame.surfarray.make_surface(agent_view)
            view_surface = pygame.transform.scale(view_surface, (rect[2], rect[3]))
            surface.blit(view_surface, (rect[0], rect[1]))
        else:
            # Placeholder text
            text = self.font.render("No agent view", True, self.config.colors.TEXT)
            text_rect = text.get_rect(center=(rect[0] + rect[2]//2, rect[1] + rect[3]//2))
            surface.blit(text, text_rect)


class StatusPanel(BasePanel):
    """Agent status information."""

    def render(self, surface: pygame.Surface, agent_state: Dict[str, Any],
               mode: str) -> None:
        """Render status panel."""
        rect = self.config.layout.status_rect
        self.draw_panel_background(surface, rect)

        x, y = rect[0] + 10, rect[1] + 10
        line_height = 25

        # Energy bar
        energy = agent_state.get("energy", 0.0)
        self._draw_progress_bar(surface, "Energy", energy, x, y,
                               self.config.colors.REWARD)

        # Health bar
        health = agent_state.get("health", 0.0)
        self._draw_progress_bar(surface, "Health", health, x, y + line_height,
                               self.config.colors.PUNISH)

        # Age and survival time
        age = agent_state.get("age", 0.0)
        survival = agent_state.get("survival_time", 0.0)

        text = self.font_small.render(
            f"Age: {age:.1f}s  Survival: {survival:.1f}s",
            True,
            self.config.colors.TEXT
        )
        surface.blit(text, (x, y + 2 * line_height))

        # Mode and signal
        text = self.font_small.render(
            f"Mode: {mode}  Signal: NONE",
            True,
            self.config.colors.TEXT
        )
        surface.blit(text, (x, y + 3 * line_height))

    def _draw_progress_bar(self, surface: pygame.Surface, label: str,
                          value: float, x: int, y: int, color: tuple) -> None:
        """Draw a labeled progress bar."""
        bar_width = 200
        bar_height = 15

        # Label
        text = self.font_small.render(f"{label}:", True, self.config.colors.TEXT)
        surface.blit(text, (x, y))

        # Background
        bar_x = x + 70
        pygame.draw.rect(surface, self.config.colors.GRID,
                        (bar_x, y, bar_width, bar_height))

        # Fill
        fill_width = int(bar_width * value)
        pygame.draw.rect(surface, color, (bar_x, y, fill_width, bar_height))

        # Percentage
        pct_text = self.font_small.render(f"{int(value * 100)}%", True,
                                          self.config.colors.TEXT_BRIGHT)
        surface.blit(pct_text, (bar_x + bar_width + 10, y))


class MetricsPanel(BasePanel):
    """Learning metrics display."""

    def render(self, surface: pygame.Surface, metrics: Dict[str, Any]) -> None:
        """Render metrics panel."""
        rect = self.config.layout.metrics_rect
        self.draw_panel_background(surface, rect)

        x, y = rect[0] + 10, rect[1] + 10
        line_height = 22

        # Loss
        loss = metrics.get("loss", 0.0)
        text = self.font.render(f"Loss: {loss:.4f} ↓", True,
                               self.config.colors.TEXT_BRIGHT)
        surface.blit(text, (x, y))

        # Counts
        rewards = metrics.get("rewards", 0)
        punishments = metrics.get("punishments", 0)
        text = self.font_small.render(
            f"Rewards: {rewards}  Punishments: {punishments}",
            True,
            self.config.colors.TEXT
        )
        surface.blit(text, (x, y + line_height))

        demos = metrics.get("demonstrations", 0)
        text = self.font_small.render(f"Demonstrations: {demos}", True,
                                      self.config.colors.TEXT)
        surface.blit(text, (x, y + 2 * line_height))

        voice = metrics.get("voice_samples", 0)
        text = self.font_small.render(f"Voice Samples: {voice}", True,
                                      self.config.colors.TEXT)
        surface.blit(text, (x, y + 3 * line_height))


class WaveformPanel(BasePanel):
    """Audio waveform visualization."""

    def render(self, surface: pygame.Surface, waveform: np.ndarray) -> None:
        """Render waveform panel."""
        rect = self.config.layout.waveform_rect
        self.draw_panel_background(surface, rect)

        if len(waveform) == 0:
            return

        # Downsample waveform to fit panel width
        panel_width = rect[2] - 20
        if len(waveform) > panel_width:
            step = len(waveform) // panel_width
            waveform = waveform[::step][:panel_width]

        # Draw waveform
        center_y = rect[1] + rect[3] // 2
        scale = (rect[3] - 20) / 2

        points = []
        for i, sample in enumerate(waveform):
            x = rect[0] + 10 + i
            y = center_y - int(sample * scale)
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(surface, self.config.colors.WAVEFORM, False, points, 2)

        # Center line
        pygame.draw.line(surface, self.config.colors.GRID,
                        (rect[0], center_y), (rect[0] + rect[2], center_y))


class ControlsPanel(BasePanel):
    """Control hints and help."""

    def render(self, surface: pygame.Surface) -> None:
        """Render controls panel."""
        rect = self.config.layout.controls_rect
        self.draw_panel_background(surface, rect)

        x, y = rect[0] + 10, rect[1] + 10
        line_height = 18

        # Legend and controls
        controls = [
            ("Legend & Controls", self.config.colors.TEXT_BRIGHT, self.font),
            ("BLUE circle = Agent (learning AI)    GREEN = Food    RED triangle = Predator", self.config.colors.TEXT, self.font_small),
            ("", self.config.colors.TEXT, self.font_small),
            ("TEACHING: SPACE=Reward (good!)  X=Punish (bad!)  - Agent learns from your feedback", self.config.colors.REWARD, self.font_small),
            ("CONTROL:  C=Toggle control mode, then ARROWS to move agent toward food", self.config.colors.AGENT, self.font_small),
            ("Agent eats GREEN food when close + hungry. Avoid RED predators! ESC=Quit", self.config.colors.TEXT, self.font_small),
        ]

        for i, (line, color, font) in enumerate(controls):
            text = font.render(line, True, color)
            surface.blit(text, (x, y + i * line_height))
