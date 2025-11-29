"""Cockpit-style teaching interface with comprehensive controls."""

import pygame
import pygame_gui
from pathlib import Path
from typing import Optional, Dict, Any, List

from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig
from primordial.simulation.agent_database import AgentDatabase


class CockpitApp:
    """Cockpit-style interface with collapsible panels and live controls."""

    # Color constants (matching theme.json)
    BG_DARKEST = (10, 10, 18)
    BG_DARK = (18, 18, 26)
    BG_PANEL = (26, 26, 36)
    CYAN = (0, 255, 255)
    CYAN_DIM = (0, 170, 170)
    GREEN = (0, 255, 136)
    RED = (255, 68, 102)
    TEXT_BRIGHT = (255, 255, 255)
    TEXT_NORMAL = (204, 204, 221)
    TEXT_DIM = (136, 136, 153)

    # Layout constants
    TOPBAR_HEIGHT = 44
    BOTTOMBAR_HEIGHT = 50
    PANEL_WIDTH = 280

    def __init__(
        self,
        window_width: int = 1280,
        window_height: int = 800,
        sim_config: Optional[SimulationConfig] = None,
    ):
        """Initialize cockpit app."""
        self.window_width = window_width
        self.window_height = window_height

        # Simulation config
        self.sim_config = sim_config or SimulationConfig(
            world_width=800,
            world_height=600,
            max_agents=10,
            predator_count=2,
            initial_food=30,
            learning_enabled=True,
            render_enabled=False,
        )

        # State
        self.running = False
        self.left_panel_visible = True
        self.right_panel_visible = True
        self.selected_agent_id: Optional[str] = None
        self.time_scale = 1.0

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Primordial - Cockpit Interface")
        self.clock = pygame.time.Clock()

        # Initialize pygame-gui with error handling for theme loading
        theme_path = Path(__file__).parent / "theme.json"
        try:
            if theme_path.exists():
                self.ui_manager = pygame_gui.UIManager(
                    (window_width, window_height),
                    theme_path=str(theme_path)
                )
            else:
                print(f"Warning: theme.json not found at {theme_path}, using defaults")
                self.ui_manager = pygame_gui.UIManager((window_width, window_height))
        except Exception as e:
            print(f"Error loading theme: {e}, using defaults")
            self.ui_manager = pygame_gui.UIManager((window_width, window_height))

        # Create simulation
        self.simulation = Simulation(self.sim_config)

        # Agent database
        self.agent_db = AgentDatabase()

        # Teaching stats
        self.stats = {
            "rewards": 0,
            "punishments": 0,
            "demonstrations": 0,
        }

        # Build UI
        self._build_ui()

    def _build_ui(self) -> None:
        """Build all UI elements."""
        # Will be implemented in subsequent tasks
        pass

    def _handle_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_TAB:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        self.right_panel_visible = not self.right_panel_visible
                    else:
                        self.left_panel_visible = not self.left_panel_visible
                    self._rebuild_layout()

            # Pass to pygame-gui
            self.ui_manager.process_events(event)

    def _update(self, dt: float) -> None:
        """Update simulation and UI."""
        # Update simulation
        scaled_dt = dt * self.time_scale
        self.simulation.tick(scaled_dt)

        # Update pygame-gui
        self.ui_manager.update(dt)

    def _render_world(self) -> None:
        """Render world view as background."""
        # Calculate world view area
        left = self.PANEL_WIDTH if self.left_panel_visible else 0
        right = self.window_width - (self.PANEL_WIDTH if self.right_panel_visible else 0)
        top = self.TOPBAR_HEIGHT
        bottom = self.window_height - self.BOTTOMBAR_HEIGHT

        world_rect = pygame.Rect(left, top, right - left, bottom - top)

        # Draw grid background
        pygame.draw.rect(self.screen, self.BG_DARKEST, world_rect)

        # Draw grid lines
        grid_size = 50
        for x in range(world_rect.left, world_rect.right, grid_size):
            pygame.draw.line(self.screen, (20, 30, 30), (x, world_rect.top), (x, world_rect.bottom))
        for y in range(world_rect.top, world_rect.bottom, grid_size):
            pygame.draw.line(self.screen, (20, 30, 30), (world_rect.left, y), (world_rect.right, y))

        # TODO: Render entities (agents, food, predators, etc.)

    def _render_topbar(self) -> None:
        """Render HUD top bar."""
        rect = pygame.Rect(0, 0, self.window_width, self.TOPBAR_HEIGHT)
        pygame.draw.rect(self.screen, self.BG_DARK, rect)
        pygame.draw.line(self.screen, self.CYAN_DIM, (0, self.TOPBAR_HEIGHT - 1),
                        (self.window_width, self.TOPBAR_HEIGHT - 1))

        # TODO: Render FPS, speed controls, etc.

    def _render_bottombar(self) -> None:
        """Render HUD bottom bar."""
        rect = pygame.Rect(0, self.window_height - self.BOTTOMBAR_HEIGHT,
                          self.window_width, self.BOTTOMBAR_HEIGHT)
        pygame.draw.rect(self.screen, self.BG_DARK, rect)
        pygame.draw.line(self.screen, self.CYAN_DIM, (0, rect.top), (self.window_width, rect.top))

        # TODO: Render teaching buttons, audio visualizer, etc.

    def _render(self) -> None:
        """Render full frame."""
        self.screen.fill(self.BG_DARKEST)

        # Render world (background)
        self._render_world()

        # Render HUD bars
        self._render_topbar()
        self._render_bottombar()

        # Render pygame-gui
        self.ui_manager.draw_ui(self.screen)

        pygame.display.flip()

    def _rebuild_layout(self) -> None:
        """Rebuild UI layout after panel toggle or resize."""
        # Will be implemented to reposition panels
        pass

    def run(self) -> None:
        """Main loop."""
        self.running = True

        while self.running:
            dt = self.clock.tick(60) / 1000.0

            self._handle_events()
            self._update(dt)
            self._render()

        self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        pygame.quit()


def main():
    """Run cockpit interface."""
    app = CockpitApp()
    app.run()


if __name__ == "__main__":
    main()
