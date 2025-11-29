"""Cockpit-style teaching interface with comprehensive controls."""

import math
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

        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 32)

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
        # Get transform (uses helper from Task 3.4, or calculate inline for now)
        if hasattr(self, '_get_world_transform'):
            world_rect, scale, offset_x, offset_y = self._get_world_transform()
        else:
            # Inline calculation (will be replaced by helper in Task 3.4)
            left = self.PANEL_WIDTH if self.left_panel_visible else 0
            right = self.window_width - (self.PANEL_WIDTH if self.right_panel_visible else 0)
            top = self.TOPBAR_HEIGHT
            bottom = self.window_height - self.BOTTOMBAR_HEIGHT
            world_rect = pygame.Rect(left, top, right - left, bottom - top)
            scale_x = world_rect.width / self.simulation.world.width
            scale_y = world_rect.height / self.simulation.world.height
            scale = min(scale_x, scale_y)
            offset_x = world_rect.left + (world_rect.width - self.simulation.world.width * scale) / 2
            offset_y = world_rect.top + (world_rect.height - self.simulation.world.height * scale) / 2

        # Draw background
        pygame.draw.rect(self.screen, self.BG_DARKEST, world_rect)

        # Draw grid lines
        grid_size = 50
        grid_color = (15, 25, 25)
        for x in range(world_rect.left, world_rect.right, grid_size):
            pygame.draw.line(self.screen, grid_color, (x, world_rect.top), (x, world_rect.bottom))
        for y in range(world_rect.top, world_rect.bottom, grid_size):
            pygame.draw.line(self.screen, grid_color, (world_rect.left, y), (world_rect.right, y))

        def world_to_screen(pos):
            return (int(offset_x + pos.x * scale), int(offset_y + pos.y * scale))

        # Render water bodies
        from primordial.world.entities import Water
        for entity in self.simulation.world.static_entities:
            if isinstance(entity, Water):
                pos = world_to_screen(entity.position)
                radius = int(entity.radius * scale)
                pygame.draw.circle(self.screen, (50, 100, 180), pos, radius)
                pygame.draw.circle(self.screen, (70, 130, 200), pos, radius, 2)

        # Render vegetation
        for veg in self.simulation.world.vegetation:
            pos = world_to_screen(veg.position)
            radius = int(veg.radius * scale)
            pygame.draw.circle(self.screen, (30, 80, 30), pos, radius)

        # Render food
        for food in self.simulation.world.food_items:
            if food.is_active:
                pos = world_to_screen(food.position)
                radius = max(4, int(food.radius * scale))
                pygame.draw.circle(self.screen, self.GREEN, pos, radius)
                # Glow effect
                pygame.draw.circle(self.screen, (0, 200, 100), pos, radius + 2, 1)

        # Render predators
        for predator in self.simulation.world.predators:
            if predator.is_active:
                pos = world_to_screen(predator.position)
                # Get angle from velocity
                angle = 0
                if hasattr(predator, 'velocity') and predator.velocity.magnitude() > 0.1:
                    angle = math.atan2(predator.velocity.y, predator.velocity.x)

                # Draw triangle pointing in direction
                size = max(8, int(predator.radius * scale))
                points = [
                    (pos[0] + math.cos(angle) * size, pos[1] + math.sin(angle) * size),
                    (pos[0] + math.cos(angle + 2.4) * size * 0.7, pos[1] + math.sin(angle + 2.4) * size * 0.7),
                    (pos[0] + math.cos(angle - 2.4) * size * 0.7, pos[1] + math.sin(angle - 2.4) * size * 0.7),
                ]
                pygame.draw.polygon(self.screen, self.RED, points)

        # Render agents
        for agent_id, wrapper in self.simulation.agents.items():
            if wrapper.agent.is_alive:
                agent = wrapper.agent
                pos = world_to_screen(agent.position)
                radius = max(6, int(agent.radius * scale))

                # Agent color (highlight if selected)
                color = (100, 200, 255)
                if agent_id == self.selected_agent_id:
                    color = self.CYAN
                    # Selection ring
                    pygame.draw.circle(self.screen, self.CYAN, pos, radius + 4, 2)

                # Body
                pygame.draw.circle(self.screen, color, pos, radius)
                pygame.draw.circle(self.screen, self.TEXT_BRIGHT, pos, radius, 1)

                # Direction indicator
                end_x = pos[0] + math.cos(agent.angle) * radius * 1.5
                end_y = pos[1] + math.sin(agent.angle) * radius * 1.5
                pygame.draw.line(self.screen, self.TEXT_BRIGHT, pos, (end_x, end_y), 2)

    def _render_topbar(self) -> None:
        """Render HUD top bar."""
        rect = pygame.Rect(0, 0, self.window_width, self.TOPBAR_HEIGHT)
        pygame.draw.rect(self.screen, self.BG_DARK, rect)
        pygame.draw.line(self.screen, self.CYAN_DIM, (0, self.TOPBAR_HEIGHT - 1),
                        (self.window_width, self.TOPBAR_HEIGHT - 1))

        x = 16

        # Logo
        logo = self.font_large.render("PRIMORDIAL", True, self.CYAN)
        self.screen.blit(logo, (x, 8))
        x += logo.get_width() + 20

        # Divider
        pygame.draw.line(self.screen, (40, 40, 50), (x, 8), (x, self.TOPBAR_HEIGHT - 8))
        x += 20

        # FPS
        fps = int(self.clock.get_fps())
        fps_text = self.font_small.render(f"FPS: {fps}", True, self.TEXT_DIM)
        self.screen.blit(fps_text, (x, 14))
        x += fps_text.get_width() + 20

        # Divider
        pygame.draw.line(self.screen, (40, 40, 50), (x, 8), (x, self.TOPBAR_HEIGHT - 8))
        x += 20

        # Speed control
        speed_label = self.font_small.render(f"{self.time_scale:.1f}x", True, self.CYAN)
        # Slow button
        slow_rect = pygame.Rect(x, 10, 24, 24)
        pygame.draw.rect(self.screen, (37, 37, 48), slow_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.CYAN_DIM, slow_rect, 1, border_radius=4)
        slow_text = self.font_small.render("<", True, self.CYAN)
        self.screen.blit(slow_text, (x + 8, 12))
        x += 28

        # Speed value
        self.screen.blit(speed_label, (x, 14))
        x += speed_label.get_width() + 4

        # Fast button
        fast_rect = pygame.Rect(x, 10, 24, 24)
        pygame.draw.rect(self.screen, (37, 37, 48), fast_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.CYAN_DIM, fast_rect, 1, border_radius=4)
        fast_text = self.font_small.render(">", True, self.CYAN)
        self.screen.blit(fast_text, (x + 8, 12))
        x += 44

        # Divider
        pygame.draw.line(self.screen, (40, 40, 50), (x, 8), (x, self.TOPBAR_HEIGHT - 8))
        x += 20

        # Day/Night
        env = self.simulation.world.environment
        if hasattr(env, 'brightness'):
            is_night = env.brightness < 0.3
            icon = "N" if is_night else "D"
            text = "Night" if is_night else "Day"
            color = (170, 102, 255) if is_night else (255, 170, 0)
            day_text = self.font_small.render(f"{icon} {text}", True, color)
            self.screen.blit(day_text, (x, 14))
            x += day_text.get_width() + 20

        # Divider
        pygame.draw.line(self.screen, (40, 40, 50), (x, 8), (x, self.TOPBAR_HEIGHT - 8))
        x += 20

        # Generation (highest)
        max_gen = max((w.generation for w in self.simulation.agents.values()), default=0)
        gen_text = self.font_small.render(f"Gen: {max_gen}", True, self.TEXT_NORMAL)
        self.screen.blit(gen_text, (x, 14))
        x += gen_text.get_width() + 20

        # Divider
        pygame.draw.line(self.screen, (40, 40, 50), (x, 8), (x, self.TOPBAR_HEIGHT - 8))
        x += 20

        # Population
        alive = sum(1 for w in self.simulation.agents.values() if w.agent.is_alive)
        total = len(self.simulation.agents)
        pop_text = self.font_small.render(f"Pop: {alive}/{total}", True, self.TEXT_NORMAL)
        self.screen.blit(pop_text, (x, 14))

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
