"""Main rendering engine for the teaching interface."""

from typing import Dict, Any, Optional, List
import pygame
import numpy as np
from primordial.interface.config import UIConfig
from primordial.interface.ui_panels import (
    HeaderPanel, WorldViewPanel, AgentPOVPanel, AgentTablePanel,
    StatusPanel, MetricsPanel, WaveformPanel, ControlsPanel
)


class Renderer:
    """Main renderer managing all UI panels."""

    def __init__(self, config: UIConfig):
        """
        Initialize renderer.

        Args:
            config: UI configuration
        """
        self.config = config

        # Initialize Pygame
        pygame.init()

        # Create window
        self.screen = pygame.display.set_mode(
            (config.window_width, config.window_height)
        )
        pygame.display.set_caption("Primordial Teaching Interface")

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Initialize panels
        self.header = HeaderPanel(config)
        self.world_view = WorldViewPanel(config)
        self.agent_pov = AgentPOVPanel(config)
        self.agent_table = AgentTablePanel(config)
        self.status = StatusPanel(config)
        self.metrics = MetricsPanel(config)
        self.waveform = WaveformPanel(config)
        self.controls = ControlsPanel(config)

        # Recording state
        self.recording = False

    def tick(self) -> float:
        """
        Advance one frame and control FPS.

        Returns:
            Delta time in seconds since last frame
        """
        dt = self.clock.tick(self.config.fps) / 1000.0  # Convert ms to seconds
        return dt

    def get_fps(self) -> int:
        """Get current FPS."""
        return int(self.clock.get_fps())

    def render_frame(self,
                    world_state: Dict[str, Any],
                    agent_state: Dict[str, Any],
                    agent_view: Optional[np.ndarray],
                    waveform: np.ndarray,
                    metrics: Dict[str, Any],
                    mode: str,
                    zoom: float = 1.0,
                    offset: tuple = (0, 0),
                    agents_table_data: Optional[List[Dict[str, Any]]] = None,
                    selected_agent_id: Optional[str] = None) -> None:
        """
        Render complete frame with all panels.

        Args:
            world_state: World state data
            agent_state: Agent status data
            agent_view: Agent's first-person view (RGB image)
            waveform: Audio waveform samples
            metrics: Learning metrics
            mode: Current interaction mode
            zoom: World view zoom level
            offset: World view camera offset
            agents_table_data: List of all agents' data for table display
            selected_agent_id: Currently selected agent ID
        """
        # Clear screen
        self.screen.fill(self.config.colors.BACKGROUND)

        # Render all panels
        self.header.render(self.screen, self.get_fps(), self.recording)
        self.world_view.render(self.screen, world_state, zoom, offset)

        # Use agent table if data provided, otherwise fall back to POV
        if agents_table_data is not None:
            self.agent_table.render(self.screen, agents_table_data, selected_agent_id)
        else:
            self.agent_pov.render(self.screen, agent_view)

        self.status.render(self.screen, agent_state, mode)
        self.metrics.render(self.screen, metrics)
        self.waveform.render(self.screen, waveform)
        self.controls.render(self.screen)

        # Update display
        pygame.display.flip()

    def set_recording(self, recording: bool) -> None:
        """Set recording indicator state."""
        self.recording = recording

    def cleanup(self) -> None:
        """Cleanup resources."""
        pygame.quit()
