# Cockpit UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the existing pygame interface into a sci-fi/neon "cockpit" layout - **Phase 1 MVP** with core controls, expandable in later phases.

**Architecture:** World view becomes full-screen background with manually-rendered overlay panels. pygame-gui is used for modals only (Phase 6+). Raw pygame drawing provides better performance and control for the main UI panels.

**Tech Stack:** pygame 2.x, pygame-gui 0.6+ (for modals only), existing simulation/world/lrn modules

**Reference:** `docs/mockups/primordial-ui-mockup.html` and `docs/plans/2025-01-28-cockpit-ui-design.md`

---

## Scope

### Phase 1-5 MVP Includes:
- Core layout (world view, top/bottom bars, collapsible side panels)
- World tab controls (population, environment basics)
- Agent table with selection
- Teaching buttons (reward/punish)
- All keyboard shortcuts from IntegratedTeachingApp
- Save/load agent functionality

### Phase 6-7 Future (Not in MVP):
- Database browser modal
- Genome editor modal
- Help overlay modal
- Agents tab (default genome sliders)
- Learn tab (learning rate, hidden dim, etc.)
- Rewards tab (all 12 reward values)
- Predators tab (patrol, detection, damage, etc.)
- Presets tab (save/load configurations)

---

## Phase 1: Setup and Core Layout

### Task 1.1: Install pygame-gui and Create Theme

**Files:**
- Modify: `requirements.txt` (add pygame-gui)
- Create: `primordial/interface/theme.json`

**Step 1: Add pygame-gui to requirements**

Add to `requirements.txt`:
```
pygame-gui>=0.6.0
```

**Step 2: Run pip install**

Run: `pip install pygame-gui`
Expected: Successfully installed pygame-gui

**Step 3: Create theme.json with sci-fi colors**

Create `primordial/interface/theme.json`:
```json
{
    "#defaults": {
        "colours": {
            "dark_bg": "#0a0a12",
            "normal_bg": "#1a1a24",
            "hovered_bg": "#2a2a38",
            "disabled_bg": "#151520",
            "normal_text": "#ccccdd",
            "hovered_text": "#ffffff",
            "disabled_text": "#555566",
            "normal_border": "#00aaaa",
            "hovered_border": "#00ffff"
        },
        "font": {
            "name": "fira_code",
            "size": "13",
            "regular_path": "font/FiraCode-Regular.ttf"
        }
    },
    "UIPanel": {
        "colours": {
            "dark_bg": "#1a1a24",
            "normal_bg": "#1a1a24"
        }
    },
    "UIButton": {
        "colours": {
            "normal_bg": "#252530",
            "hovered_bg": "#2a2a38",
            "active_bg": "#00aaaa",
            "normal_border": "#00aaaa",
            "hovered_border": "#00ffff"
        },
        "misc": {
            "border_width": "1",
            "shape_corner_radius": "4"
        }
    },
    "UIHorizontalSlider": {
        "colours": {
            "normal_bg": "#252530",
            "hovered_bg": "#2a2a38",
            "normal_border": "#2a2a38"
        }
    },
    "UITextEntryLine": {
        "colours": {
            "dark_bg": "#252530",
            "normal_bg": "#252530",
            "normal_border": "#2a2a38",
            "focused_border": "#00ffff"
        }
    },
    "UISelectionList": {
        "colours": {
            "dark_bg": "#1a1a24",
            "normal_bg": "#1a1a24"
        }
    },
    "UIWindow": {
        "colours": {
            "dark_bg": "#1a1a24",
            "normal_bg": "#1a1a24",
            "normal_border": "#00aaaa"
        }
    },
    "UIDropDownMenu": {
        "colours": {
            "normal_bg": "#252530",
            "hovered_bg": "#2a2a38",
            "normal_border": "#2a2a38"
        }
    },
    "#reward_button": {
        "colours": {
            "normal_bg": "#00aa66",
            "hovered_bg": "#00ff88",
            "normal_border": "#00ff88"
        }
    },
    "#punish_button": {
        "colours": {
            "normal_bg": "#aa3344",
            "hovered_bg": "#ff4466",
            "normal_border": "#ff4466"
        }
    },
    "#tab_button": {
        "colours": {
            "normal_bg": "#252530",
            "selected_bg": "#2a2a38",
            "normal_border": "#252530",
            "selected_border": "#00ffff"
        }
    },
    "#status_bar_energy": {
        "colours": {
            "filled_bar": "#ffaa00",
            "unfilled_bar": "#252530"
        }
    },
    "#status_bar_health": {
        "colours": {
            "filled_bar": "#00ff88",
            "unfilled_bar": "#252530"
        }
    },
    "#status_bar_breed": {
        "colours": {
            "filled_bar": "#ff66aa",
            "unfilled_bar": "#252530"
        }
    },
    "#status_bar_social": {
        "colours": {
            "filled_bar": "#aa66ff",
            "unfilled_bar": "#252530"
        }
    }
}
```

**Step 4: Commit**

```bash
git add requirements.txt primordial/interface/theme.json
git commit -m "feat: add pygame-gui and sci-fi theme"
```

---

### Task 1.2: Create New CockpitApp Class Shell

**Files:**
- Create: `primordial/interface/cockpit_app.py`

**Step 1: Create basic CockpitApp with pygame-gui manager**

Create `primordial/interface/cockpit_app.py`:
```python
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
```

**Step 2: Test that the shell runs**

Run: `python -m primordial.interface.cockpit_app`
Expected: Window opens with dark background, grid lines visible, closes on ESC

**Step 3: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: create CockpitApp shell with pygame-gui"
```

---

### Task 1.3: Implement World Renderer

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add entity rendering to _render_world**

Add this method to CockpitApp (replace the existing `_render_world`).

**NOTE:** This uses `_get_world_transform()` which will be added in Task 3.4. For now, include the transform calculation inline - it will be refactored in Task 3.4:

```python
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
    import math
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
```

**Step 2: Test entity rendering**

Run: `python -m primordial.interface.cockpit_app`
Expected: Agents (blue circles), food (green), predators (red triangles), vegetation, water all visible

**Step 3: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement world entity rendering in cockpit"
```

---

## Phase 2: HUD Bars

### Task 2.1: Implement Top Bar with Stats

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add font initialization to __init__**

Add after pygame.init():
```python
# Fonts
self.font = pygame.font.Font(None, 24)
self.font_small = pygame.font.Font(None, 18)
self.font_large = pygame.font.Font(None, 32)
```

**Step 2: Implement _render_topbar**

Replace `_render_topbar`:
```python
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
```

**Step 3: Test top bar**

Run: `python -m primordial.interface.cockpit_app`
Expected: Top bar shows logo, FPS, speed controls, day/night, generation, population

**Step 4: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement top bar with stats display"
```

---

### Task 2.2: Implement Bottom Bar with Teaching Buttons

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add reward/punish button rects to __init__**

Add to __init__:
```python
# Button rects (will be set in _render_bottombar)
self.reward_btn_rect = pygame.Rect(0, 0, 0, 0)
self.punish_btn_rect = pygame.Rect(0, 0, 0, 0)
self.pause_btn_rect = pygame.Rect(0, 0, 0, 0)

# Pause state
self.paused = False
```

**Step 2: Implement _render_bottombar**

Replace `_render_bottombar`:
```python
def _render_bottombar(self) -> None:
    """Render HUD bottom bar."""
    bar_top = self.window_height - self.BOTTOMBAR_HEIGHT
    rect = pygame.Rect(0, bar_top, self.window_width, self.BOTTOMBAR_HEIGHT)
    pygame.draw.rect(self.screen, self.BG_DARK, rect)
    pygame.draw.line(self.screen, self.CYAN_DIM, (0, bar_top), (self.window_width, bar_top))

    x = 16

    # Reward button
    self.reward_btn_rect = pygame.Rect(x, bar_top + 8, 100, 34)
    btn_color = (0, 170, 102) if not pygame.mouse.get_pressed()[0] else (0, 255, 136)
    pygame.draw.rect(self.screen, btn_color, self.reward_btn_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.GREEN, self.reward_btn_rect, 2, border_radius=4)
    reward_text = self.font_small.render("REWARD (R)", True, self.BG_DARKEST)
    text_x = self.reward_btn_rect.centerx - reward_text.get_width() // 2
    text_y = self.reward_btn_rect.centery - reward_text.get_height() // 2
    self.screen.blit(reward_text, (text_x, text_y))
    x += 108

    # Punish button
    self.punish_btn_rect = pygame.Rect(x, bar_top + 8, 100, 34)
    btn_color = (170, 51, 68) if not pygame.mouse.get_pressed()[0] else (255, 68, 102)
    pygame.draw.rect(self.screen, btn_color, self.punish_btn_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.RED, self.punish_btn_rect, 2, border_radius=4)
    punish_text = self.font_small.render("PUNISH (X)", True, self.BG_DARKEST)
    text_x = self.punish_btn_rect.centerx - punish_text.get_width() // 2
    text_y = self.punish_btn_rect.centery - punish_text.get_height() // 2
    self.screen.blit(punish_text, (text_x, text_y))
    x += 120

    # Divider
    pygame.draw.line(self.screen, (40, 40, 50), (x, bar_top + 8), (x, bar_top + 42))
    x += 20

    # Audio visualizer placeholder
    audio_rect = pygame.Rect(x, bar_top + 12, 100, 26)
    pygame.draw.rect(self.screen, (37, 37, 48), audio_rect, border_radius=4)
    mic_text = self.font_small.render("MIC", True, self.TEXT_DIM)
    self.screen.blit(mic_text, (x + 8, bar_top + 16))
    # Fake bars
    bar_heights = [4, 8, 14, 20, 12, 6, 10, 4]
    bar_x = x + 40
    for h in bar_heights:
        bar_rect = pygame.Rect(bar_x, bar_top + 12 + (26 - h) // 2, 4, h)
        pygame.draw.rect(self.screen, self.CYAN, bar_rect, border_radius=2)
        bar_x += 6

    # Right side buttons
    x = self.window_width - 260

    # Pause button
    self.pause_btn_rect = pygame.Rect(x, bar_top + 8, 70, 34)
    pygame.draw.rect(self.screen, (37, 37, 48), self.pause_btn_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.TEXT_DIM, self.pause_btn_rect, 1, border_radius=4)
    pause_label = "Play" if self.paused else "Pause"
    pause_text = self.font_small.render(pause_label, True, self.TEXT_NORMAL)
    text_x = self.pause_btn_rect.centerx - pause_text.get_width() // 2
    self.screen.blit(pause_text, (text_x, bar_top + 16))
    x += 78

    # Record button (placeholder)
    rec_rect = pygame.Rect(x, bar_top + 8, 60, 34)
    pygame.draw.rect(self.screen, (37, 37, 48), rec_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.TEXT_DIM, rec_rect, 1, border_radius=4)
    rec_text = self.font_small.render("Rec", True, self.TEXT_NORMAL)
    self.screen.blit(rec_text, (x + 18, bar_top + 16))
    x += 68

    # Help button
    help_rect = pygame.Rect(x, bar_top + 8, 50, 34)
    pygame.draw.rect(self.screen, (37, 37, 48), help_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.TEXT_DIM, help_rect, 1, border_radius=4)
    help_text = self.font_small.render("?", True, self.TEXT_NORMAL)
    self.screen.blit(help_text, (x + 20, bar_top + 14))
```

**Step 3: Add button click handling to _handle_events**

Add to the event loop in `_handle_events`, inside the `for event` loop:
```python
# Mouse clicks
if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
    mouse_pos = pygame.mouse.get_pos()

    # Teaching buttons
    if self.reward_btn_rect.collidepoint(mouse_pos):
        self._send_reward()
    elif self.punish_btn_rect.collidepoint(mouse_pos):
        self._send_punish()
    elif self.pause_btn_rect.collidepoint(mouse_pos):
        self.paused = not self.paused
```

**Step 4: Add keyboard handling for R/X**

Add to the `if event.type == pygame.KEYDOWN` block:
```python
elif event.key == pygame.K_r:
    self._send_reward()
elif event.key == pygame.K_x:
    self._send_punish()
elif event.key == pygame.K_p:
    self.paused = not self.paused
```

**Step 5: Add teaching signal methods**

Add these methods to CockpitApp:
```python
def _send_reward(self) -> None:
    """Send reward signal to selected agent."""
    self.stats["rewards"] += 1
    wrapper = self._get_target_agent_wrapper()
    if wrapper:
        wrapper.events.append('human_reward')
        print(f"Reward sent to {wrapper.agent_id}")

def _send_punish(self) -> None:
    """Send punish signal to selected agent."""
    self.stats["punishments"] += 1
    wrapper = self._get_target_agent_wrapper()
    if wrapper:
        wrapper.events.append('human_punish')
        print(f"Punish sent to {wrapper.agent_id}")

def _get_target_agent_wrapper(self):
    """Get wrapper for selected agent or first living agent."""
    if self.selected_agent_id and self.selected_agent_id in self.simulation.agents:
        wrapper = self.simulation.agents[self.selected_agent_id]
        if wrapper.agent.is_alive:
            return wrapper

    for wrapper in self.simulation.agents.values():
        if wrapper.agent.is_alive:
            return wrapper
    return None
```

**Step 6: Update _update to respect pause**

Modify `_update`:
```python
def _update(self, dt: float) -> None:
    """Update simulation and UI."""
    if not self.paused:
        scaled_dt = dt * self.time_scale
        self.simulation.tick(scaled_dt)

    self.ui_manager.update(dt)
```

**Step 7: Test bottom bar**

Run: `python -m primordial.interface.cockpit_app`
Expected: Bottom bar shows reward/punish buttons, clicking or pressing R/X prints messages, P pauses

**Step 8: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement bottom bar with teaching buttons"
```

---

## Phase 3: Side Panels

### Task 3.1: Create Left Panel Framework with Tabs

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add panel state to __init__**

Add to __init__:
```python
# Left panel tab state
self.left_panel_tab = "world"  # world, agents, learn, rewards, predators, presets
```

**Step 2: Create _render_left_panel method**

Add this method:
```python
def _render_left_panel(self) -> None:
    """Render left control panel."""
    if not self.left_panel_visible:
        # Draw collapse button only
        btn_rect = pygame.Rect(8, self.TOPBAR_HEIGHT + 8, 24, 24)
        pygame.draw.rect(self.screen, self.BG_PANEL, btn_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.CYAN_DIM, btn_rect, 1, border_radius=4)
        arrow = self.font_small.render(">", True, self.CYAN)
        self.screen.blit(arrow, (btn_rect.x + 8, btn_rect.y + 4))
        return

    panel_height = self.window_height - self.TOPBAR_HEIGHT - self.BOTTOMBAR_HEIGHT
    panel_rect = pygame.Rect(0, self.TOPBAR_HEIGHT, self.PANEL_WIDTH, panel_height)

    # Panel background
    pygame.draw.rect(self.screen, self.BG_PANEL, panel_rect)
    pygame.draw.line(self.screen, self.CYAN_DIM,
                    (self.PANEL_WIDTH - 1, self.TOPBAR_HEIGHT),
                    (self.PANEL_WIDTH - 1, self.window_height - self.BOTTOMBAR_HEIGHT))

    # Header
    header_rect = pygame.Rect(0, self.TOPBAR_HEIGHT, self.PANEL_WIDTH, 36)
    pygame.draw.rect(self.screen, self.BG_DARK, header_rect)
    pygame.draw.line(self.screen, self.CYAN_DIM, (0, header_rect.bottom), (self.PANEL_WIDTH, header_rect.bottom))

    title = self.font_small.render("CONTROLS", True, self.CYAN)
    self.screen.blit(title, (12, self.TOPBAR_HEIGHT + 10))

    # Collapse button
    btn_rect = pygame.Rect(self.PANEL_WIDTH - 32, self.TOPBAR_HEIGHT + 6, 24, 24)
    pygame.draw.rect(self.screen, (37, 37, 48), btn_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.TEXT_DIM, btn_rect, 1, border_radius=4)
    arrow = self.font_small.render("<", True, self.TEXT_DIM)
    self.screen.blit(arrow, (btn_rect.x + 8, btn_rect.y + 4))

    # Tabs
    tabs = ["World", "Agents", "Learn", "Rewards", "Pred", "Presets"]
    tab_keys = ["world", "agents", "learn", "rewards", "predators", "presets"]
    tab_y = self.TOPBAR_HEIGHT + 44
    tab_x = 8

    for i, (label, key) in enumerate(zip(tabs, tab_keys)):
        tab_width = 42 if label in ["Pred"] else 46
        tab_rect = pygame.Rect(tab_x, tab_y, tab_width, 24)

        is_active = self.left_panel_tab == key
        bg_color = (42, 42, 56) if is_active else (37, 37, 48)
        border_color = self.CYAN if is_active else (37, 37, 48)
        text_color = self.CYAN if is_active else self.TEXT_DIM

        pygame.draw.rect(self.screen, bg_color, tab_rect, border_radius=4)
        pygame.draw.rect(self.screen, border_color, tab_rect, 1, border_radius=4)

        tab_text = self.font_small.render(label, True, text_color)
        self.screen.blit(tab_text, (tab_x + 4, tab_y + 5))

        tab_x += tab_width + 4
        if tab_x > self.PANEL_WIDTH - 50:
            tab_x = 8
            tab_y += 28

    # Content area
    content_top = tab_y + 32
    self._render_left_panel_content(content_top)

def _render_left_panel_content(self, top: int) -> None:
    """Render content for current tab."""
    # Placeholder - will be implemented per tab
    content_text = self.font_small.render(f"[{self.left_panel_tab.upper()} controls]", True, self.TEXT_DIM)
    self.screen.blit(content_text, (16, top + 16))
```

**Step 3: Add tab click handling**

Add to mouse click handling in `_handle_events`:
```python
# Left panel tab clicks
if self.left_panel_visible:
    tabs = ["world", "agents", "learn", "rewards", "predators", "presets"]
    tab_y = self.TOPBAR_HEIGHT + 44
    tab_x = 8
    for key in tabs:
        tab_width = 42 if key == "predators" else 46
        tab_rect = pygame.Rect(tab_x, tab_y, tab_width, 24)
        if tab_rect.collidepoint(mouse_pos):
            self.left_panel_tab = key
            break
        tab_x += tab_width + 4
        if tab_x > self.PANEL_WIDTH - 50:
            tab_x = 8
            tab_y += 28
```

**Step 4: Call _render_left_panel in _render**

Add to `_render`, after `_render_bottombar()`:
```python
# Render side panels
self._render_left_panel()
```

**Step 5: Test left panel**

Run: `python -m primordial.interface.cockpit_app`
Expected: Left panel visible with tabs, clicking tabs changes active tab, TAB toggles panel

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement left panel framework with tabs"
```

---

### Task 3.2: Implement World Tab Controls

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add slider state tracking**

Add to __init__:
```python
# Control values (live-edit the sim_config)
self.control_values = {
    "max_agents": self.sim_config.max_agents,
    "initial_food": self.sim_config.initial_food,
    "max_food": self.sim_config.max_food,
    "predator_count": self.sim_config.predator_count,
    "tick_rate": self.sim_config.tick_rate,
}

# Active slider being dragged
self.active_slider = None
```

**Step 2: Create slider rendering helper**

Add this method:
```python
def _render_slider(self, x: int, y: int, width: int, label: str, value: float,
                   min_val: float, max_val: float, key: str, default: float) -> int:
    """Render a slider control. Returns height used."""
    # Label row
    label_text = self.font_small.render(label, True, self.TEXT_NORMAL)
    self.screen.blit(label_text, (x, y))

    default_text = self.font_small.render(f"Default: {default}", True, self.TEXT_DIM)
    self.screen.blit(default_text, (x + width - default_text.get_width(), y))

    y += 18

    # Slider track
    track_rect = pygame.Rect(x, y + 4, width - 70, 6)
    pygame.draw.rect(self.screen, (37, 37, 48), track_rect, border_radius=3)

    # Slider fill
    pct = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    fill_width = int(track_rect.width * pct)
    fill_rect = pygame.Rect(x, y + 4, fill_width, 6)
    pygame.draw.rect(self.screen, self.CYAN_DIM, fill_rect, border_radius=3)

    # Slider thumb
    thumb_x = x + fill_width
    thumb_rect = pygame.Rect(thumb_x - 6, y, 12, 14)
    pygame.draw.rect(self.screen, self.CYAN, thumb_rect, border_radius=6)

    # Store rect for interaction
    setattr(self, f"slider_{key}_rect", pygame.Rect(x, y, width - 70, 14))
    setattr(self, f"slider_{key}_range", (min_val, max_val))

    # Value input
    input_rect = pygame.Rect(x + width - 55, y - 2, 50, 18)
    pygame.draw.rect(self.screen, (37, 37, 48), input_rect, border_radius=4)
    pygame.draw.rect(self.screen, (42, 42, 56), input_rect, 1, border_radius=4)

    val_str = str(int(value)) if value == int(value) else f"{value:.1f}"
    val_text = self.font_small.render(val_str, True, self.TEXT_BRIGHT)
    self.screen.blit(val_text, (input_rect.right - val_text.get_width() - 4, y))

    return 38  # Height used
```

**Step 3: Implement world tab content**

Replace `_render_left_panel_content`:
```python
def _render_left_panel_content(self, top: int) -> None:
    """Render content for current tab."""
    x = 12
    y = top
    width = self.PANEL_WIDTH - 24

    if self.left_panel_tab == "world":
        # Population section
        section = self.font_small.render("POPULATION", True, self.TEXT_DIM)
        self.screen.blit(section, (x, y))
        pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
        y += 24

        y += self._render_slider(x, y, width, "Max Agents",
                                 self.control_values["max_agents"], 1, 20, "max_agents", 5)
        y += self._render_slider(x, y, width, "Initial Food",
                                 self.control_values["initial_food"], 10, 200, "initial_food", 50)
        y += self._render_slider(x, y, width, "Max Food",
                                 self.control_values["max_food"], 20, 500, "max_food", 100)
        y += self._render_slider(x, y, width, "Predator Count",
                                 self.control_values["predator_count"], 0, 10, "predator_count", 2)

        y += 8

        # Environment section
        section = self.font_small.render("ENVIRONMENT", True, self.TEXT_DIM)
        self.screen.blit(section, (x, y))
        pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
        y += 24

        y += self._render_slider(x, y, width, "Tick Rate",
                                 self.control_values["tick_rate"], 15, 120, "tick_rate", 60)

    else:
        # Placeholder for other tabs
        content_text = self.font_small.render(f"[{self.left_panel_tab.upper()} controls]", True, self.TEXT_DIM)
        self.screen.blit(content_text, (x, y + 16))
```

**Step 4: Add slider interaction handling**

Add to `_handle_events`, in the MOUSEBUTTONDOWN section:
```python
# Slider interaction
if self.left_panel_visible and self.left_panel_tab == "world":
    for key in ["max_agents", "initial_food", "max_food", "predator_count", "tick_rate"]:
        rect = getattr(self, f"slider_{key}_rect", None)
        if rect and rect.collidepoint(mouse_pos):
            self.active_slider = key
            break
```

Add MOUSEBUTTONUP handler:
```python
if event.type == pygame.MOUSEBUTTONUP:
    self.active_slider = None
```

Add MOUSEMOTION handler with config validation:
```python
if event.type == pygame.MOUSEMOTION and self.active_slider:
    rect = getattr(self, f"slider_{self.active_slider}_rect", None)
    min_val, max_val = getattr(self, f"slider_{self.active_slider}_range", (0, 100))
    if rect:
        rel_x = max(0, min(event.pos[0] - rect.x, rect.width))
        pct = rel_x / rect.width
        new_val = min_val + pct * (max_val - min_val)
        # Round integers
        if self.active_slider in ["max_agents", "initial_food", "max_food", "predator_count", "tick_rate"]:
            new_val = int(round(new_val))
        self.control_values[self.active_slider] = new_val
        # Apply to config (live update) with validation
        if hasattr(self.sim_config, self.active_slider):
            setattr(self.sim_config, self.active_slider, new_val)
        else:
            print(f"Warning: sim_config has no attribute '{self.active_slider}'")
```

**Step 5: Test sliders**

Run: `python -m primordial.interface.cockpit_app`
Expected: World tab shows sliders, dragging updates values

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement world tab with live sliders"
```

---

### Task 3.3: Create Right Panel with Agent Table

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add _render_right_panel method**

Add this method:
```python
def _render_right_panel(self) -> None:
    """Render right agent panel."""
    if not self.right_panel_visible:
        # Draw expand button
        btn_x = self.window_width - 32
        btn_rect = pygame.Rect(btn_x, self.TOPBAR_HEIGHT + 8, 24, 24)
        pygame.draw.rect(self.screen, self.BG_PANEL, btn_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.CYAN_DIM, btn_rect, 1, border_radius=4)
        arrow = self.font_small.render("<", True, self.CYAN)
        self.screen.blit(arrow, (btn_rect.x + 8, btn_rect.y + 4))
        return

    panel_height = self.window_height - self.TOPBAR_HEIGHT - self.BOTTOMBAR_HEIGHT
    panel_x = self.window_width - self.PANEL_WIDTH
    panel_rect = pygame.Rect(panel_x, self.TOPBAR_HEIGHT, self.PANEL_WIDTH, panel_height)

    # Panel background
    pygame.draw.rect(self.screen, self.BG_PANEL, panel_rect)
    pygame.draw.line(self.screen, self.CYAN_DIM,
                    (panel_x, self.TOPBAR_HEIGHT),
                    (panel_x, self.window_height - self.BOTTOMBAR_HEIGHT))

    # Header
    header_rect = pygame.Rect(panel_x, self.TOPBAR_HEIGHT, self.PANEL_WIDTH, 36)
    pygame.draw.rect(self.screen, self.BG_DARK, header_rect)
    pygame.draw.line(self.screen, self.CYAN_DIM, (panel_x, header_rect.bottom), (panel_x + self.PANEL_WIDTH, header_rect.bottom))

    title = self.font_small.render("AGENTS", True, self.CYAN)
    self.screen.blit(title, (panel_x + 12, self.TOPBAR_HEIGHT + 10))

    # Collapse button
    btn_rect = pygame.Rect(panel_x + self.PANEL_WIDTH - 32, self.TOPBAR_HEIGHT + 6, 24, 24)
    pygame.draw.rect(self.screen, (37, 37, 48), btn_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.TEXT_DIM, btn_rect, 1, border_radius=4)
    arrow = self.font_small.render(">", True, self.TEXT_DIM)
    self.screen.blit(arrow, (btn_rect.x + 8, btn_rect.y + 4))

    # Agent table
    self._render_agent_table(panel_x + 8, self.TOPBAR_HEIGHT + 44)

def _render_agent_table(self, x: int, y: int) -> None:
    """Render agent table."""
    width = self.PANEL_WIDTH - 16

    # Column headers
    cols = ["#", "E", "H", "Age", "Gen", "Food"]
    col_widths = [24, 36, 36, 48, 36, 36]
    col_x = x

    pygame.draw.rect(self.screen, self.BG_DARK, pygame.Rect(x, y, width, 20))

    for col, cw in zip(cols, col_widths):
        header = self.font_small.render(col, True, self.TEXT_DIM)
        self.screen.blit(header, (col_x + 4, y + 2))
        col_x += cw

    y += 22

    # Get agent data sorted by alive then age
    agents_data = []
    for agent_id, wrapper in self.simulation.agents.items():
        agent = wrapper.agent
        agents_data.append({
            'id': agent_id,
            'alive': agent.is_alive,
            'energy': agent.energy / agent.genome.max_energy if agent.is_alive else 0,
            'health': agent.health / agent.genome.max_health if agent.is_alive else 0,
            'age': agent.age,
            'generation': wrapper.generation,
            'food': wrapper.lifetime_stats.get('total_food_eaten', 0),
        })

    agents_data.sort(key=lambda a: (-int(a['alive']), -a['age']))

    # Store row rects for click detection
    self.agent_table_rows = []

    # Render rows (max 10)
    for i, agent in enumerate(agents_data[:10]):
        row_rect = pygame.Rect(x, y, width, 20)
        self.agent_table_rows.append((row_rect, agent['id']))

        # Highlight selected
        if agent['id'] == self.selected_agent_id:
            pygame.draw.rect(self.screen, (42, 42, 56), row_rect)
            pygame.draw.line(self.screen, self.CYAN, (x, y), (x, y + 20), 3)

        # Dim dead agents
        text_color = self.TEXT_DIM if not agent['alive'] else self.TEXT_NORMAL

        col_x = x
        # Slot number
        slot = self.font_small.render(str(i + 1), True, text_color)
        self.screen.blit(slot, (col_x + 4, y + 2))
        col_x += col_widths[0]

        # Energy
        if agent['alive']:
            e_text = f"{int(agent['energy'] * 100)}%"
        else:
            e_text = "--"
        energy = self.font_small.render(e_text, True, text_color)
        self.screen.blit(energy, (col_x + 2, y + 2))
        col_x += col_widths[1]

        # Health
        if agent['alive']:
            h_text = f"{int(agent['health'] * 100)}"
        else:
            h_text = "DEAD"
        health = self.font_small.render(h_text, True, text_color)
        self.screen.blit(health, (col_x + 2, y + 2))
        col_x += col_widths[2]

        # Age
        if agent['alive']:
            age_text = f"{int(agent['age'])}s"
        else:
            age_text = "--"
        age = self.font_small.render(age_text, True, text_color)
        self.screen.blit(age, (col_x + 2, y + 2))
        col_x += col_widths[3]

        # Generation
        gen = self.font_small.render(str(agent['generation']), True, text_color)
        self.screen.blit(gen, (col_x + 2, y + 2))
        col_x += col_widths[4]

        # Food
        food = self.font_small.render(str(agent['food']), True, text_color)
        self.screen.blit(food, (col_x + 2, y + 2))

        y += 22

    # Selected agent detail section
    self._render_selected_agent_detail(x, y + 16)

def _render_selected_agent_detail(self, x: int, y: int) -> None:
    """Render selected agent detail panel."""
    width = self.PANEL_WIDTH - 16

    # Section header
    pygame.draw.rect(self.screen, self.BG_DARK, pygame.Rect(x - 8, y, self.PANEL_WIDTH, 28))
    pygame.draw.line(self.screen, self.CYAN_DIM, (x - 8, y), (x + width + 8, y))

    wrapper = self._get_target_agent_wrapper()
    if wrapper:
        title = self.font_small.render(f"SELECTED: {wrapper.agent_id[:8]}", True, self.CYAN)
    else:
        title = self.font_small.render("SELECTED: None", True, self.TEXT_DIM)
    self.screen.blit(title, (x, y + 6))

    y += 36

    if not wrapper:
        return

    agent = wrapper.agent

    # Status bars
    bars = [
        ("Energy", agent.energy / agent.genome.max_energy, (255, 170, 0)),
        ("Health", agent.health / agent.genome.max_health, (0, 255, 136)),
        ("Breed", agent.breeding_drive, (255, 102, 170)),
        ("Social", agent.social_connection, (170, 102, 255)),
    ]

    for label, value, color in bars:
        # Label
        lbl = self.font_small.render(label, True, self.TEXT_NORMAL)
        self.screen.blit(lbl, (x, y))

        # Value
        val = self.font_small.render(f"{int(value * 100)}%", True, self.TEXT_BRIGHT)
        self.screen.blit(val, (x + width - val.get_width(), y))

        y += 14

        # Bar track
        track_rect = pygame.Rect(x, y, width, 8)
        pygame.draw.rect(self.screen, (37, 37, 48), track_rect, border_radius=4)

        # Bar fill
        fill_width = int(width * value)
        fill_rect = pygame.Rect(x, y, fill_width, 8)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=4)

        y += 14
```

**Step 2: Add table row click handling**

Add to mouse click handling in `_handle_events`:
```python
# Agent table row clicks
if self.right_panel_visible and hasattr(self, 'agent_table_rows'):
    for row_rect, agent_id in self.agent_table_rows:
        if row_rect.collidepoint(mouse_pos):
            self.selected_agent_id = agent_id
            break
```

**Step 3: Call _render_right_panel in _render**

Add after `_render_left_panel()`:
```python
self._render_right_panel()
```

**Step 4: Test right panel**

Run: `python -m primordial.interface.cockpit_app`
Expected: Right panel shows agent table, clicking rows selects agent, detail section shows stats

**Step 5: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement right panel with agent table"
```

---

### Task 3.4: Implement _rebuild_layout and Resize Handler

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add _get_world_transform helper to eliminate DRY violation**

Add this method (used by both rendering and input):
```python
def _get_world_transform(self) -> tuple:
    """Get world-to-screen transform parameters.

    Returns:
        (world_rect, scale, offset_x, offset_y) tuple
    """
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

    return world_rect, scale, offset_x, offset_y
```

**Step 2: Implement _rebuild_layout**

Replace the placeholder `_rebuild_layout`:
```python
def _rebuild_layout(self) -> None:
    """Rebuild UI layout after panel toggle or resize."""
    # Update pygame-gui manager resolution
    self.ui_manager.set_window_resolution((self.window_width, self.window_height))

    # Clear any cached slider rects (they'll be recreated on next render)
    for key in ["max_agents", "initial_food", "max_food", "predator_count", "tick_rate"]:
        if hasattr(self, f"slider_{key}_rect"):
            delattr(self, f"slider_{key}_rect")
        if hasattr(self, f"slider_{key}_range"):
            delattr(self, f"slider_{key}_range")

    # Clear agent table rows (will be recreated on render)
    if hasattr(self, 'agent_table_rows'):
        self.agent_table_rows = []
```

**Step 3: Add resize event handler to _handle_events**

Add this to the event loop, after `pygame.QUIT` handling:
```python
if event.type == pygame.VIDEORESIZE:
    self.window_width = event.w
    self.window_height = event.h
    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
    self._rebuild_layout()
```

**Step 4: Test resize behavior**

Run: `python -m primordial.interface.cockpit_app`
Expected:
- Resizing window updates layout correctly
- Panels remain properly positioned
- TAB/SHIFT+TAB toggle panels and world view adjusts
- No errors on resize

**Step 5: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement _rebuild_layout and resize handler"
```

---

## Phase 4: Agent Selection and World Interaction

### Task 4.1: Add World Click to Select Agent

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add world click detection (uses _get_world_transform helper from Task 3.4)**

Add this method:
```python
def _screen_to_world(self, screen_pos: tuple) -> tuple:
    """Convert screen position to world position.

    Uses _get_world_transform() to reuse transform calculation (DRY).
    """
    world_rect, scale, offset_x, offset_y = self._get_world_transform()

    world_x = (screen_pos[0] - offset_x) / scale
    world_y = (screen_pos[1] - offset_y) / scale

    return world_x, world_y
```

**Step 2: REPLACE table click handler with combined handler (race condition fix)**

**IMPORTANT:** This REPLACES the handler added in Task 3.3 Step 2. Remove the old handler and use this combined version that checks table clicks FIRST with early return:

```python
# Agent table row clicks - check FIRST to prevent race condition with world click
if self.right_panel_visible and hasattr(self, 'agent_table_rows'):
    for row_rect, agent_id in self.agent_table_rows:
        if row_rect.collidepoint(mouse_pos):
            self.selected_agent_id = agent_id
            return  # Early return prevents world click from also firing

# World click to select agent (only if we didn't click the table)
world_rect, _, _, _ = self._get_world_transform()
if world_rect.collidepoint(mouse_pos):
    world_x, world_y = self._screen_to_world(mouse_pos)
    self._select_agent_at_world_pos(world_x, world_y)
```

**Step 3: Add selection method**

Add this method:
```python
def _select_agent_at_world_pos(self, world_x: float, world_y: float) -> None:
    """Select agent nearest to world position."""
    from primordial.world.geometry import Vec2

    pos = Vec2(world_x, world_y)
    nearest_id = None
    nearest_dist = float('inf')

    for agent_id, wrapper in self.simulation.agents.items():
        if not wrapper.agent.is_alive:
            continue
        dist = pos.distance_to(wrapper.agent.position)
        if dist < nearest_dist and dist < 30:
            nearest_id = agent_id
            nearest_dist = dist

    if nearest_id:
        self.selected_agent_id = nearest_id
    else:
        self.selected_agent_id = None
```

**Step 4: Test agent selection**

Run: `python -m primordial.interface.cockpit_app`
Expected: Clicking on agent in world view selects it, selection ring shows, detail panel updates

**Step 5: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add click-to-select agent in world view"
```

---

## Phase 5: Migration and Integration

### Task 5.1: Add __main__.py Entry Point

**Files:**
- Modify: `primordial/__main__.py`

**Step 1: Read current __main__.py**

Check what entry points exist.

**Step 2: Update to use CockpitApp**

Update `primordial/__main__.py` to import and run CockpitApp:
```python
"""Main entry point for primordial."""

from primordial.interface.cockpit_app import CockpitApp, main

if __name__ == "__main__":
    main()
```

**Step 3: Test entry point**

Run: `python -m primordial`
Expected: Cockpit interface launches

**Step 4: Commit**

```bash
git add primordial/__main__.py
git commit -m "feat: update main entry point to use CockpitApp"
```

---

### Task 5.2: Add Time Scale Keyboard Controls

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add time scale constants to __init__**

Add after `self.time_scale = 1.0`:
```python
self.time_scale_min = 0.25
self.time_scale_max = 4.0
```

**Step 2: Add keyboard handlers for speed control**

Add to the `if event.type == pygame.KEYDOWN` block:
```python
# Time scale controls
elif event.key == pygame.K_LEFTBRACKET:  # [ = slow down
    self.time_scale = max(self.time_scale_min, self.time_scale / 1.5)
    print(f"Time scale: {self.time_scale:.2f}x")
elif event.key == pygame.K_RIGHTBRACKET:  # ] = speed up
    self.time_scale = min(self.time_scale_max, self.time_scale * 1.5)
    print(f"Time scale: {self.time_scale:.2f}x")
elif event.key == pygame.K_BACKSLASH:  # \ = reset
    self.time_scale = 1.0
    print(f"Time scale: {self.time_scale:.2f}x (reset)")
```

**Step 3: Test time scale**

Run: `python -m primordial.interface.cockpit_app`
Expected: Press [ to slow, ] to speed up, \ to reset. Top bar speed indicator updates.

**Step 4: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add time scale keyboard controls"
```

---

### Task 5.3: Add Entity Spawn Keyboard Shortcuts

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add spawn handler method (single complete definition)**

Add this method:
```python
def _handle_spawn_keys(self, key: int, mouse_pos: tuple, mods: int) -> None:
    """Handle entity spawning keyboard shortcuts.

    Args:
        key: pygame key constant
        mouse_pos: current mouse position (screen coordinates)
        mods: pygame key modifiers (for Shift+P detection)
    """
    from primordial.world.geometry import Vec2
    from primordial.world.entities import Food, Predator, Vegetation, Water

    # Convert mouse pos to world pos
    world_x, world_y = self._screen_to_world(mouse_pos)
    pos = Vec2(world_x, world_y)

    # F = Add food
    if key == pygame.K_f:
        food = Food(
            entity_id=self.simulation.world.next_entity_id,
            position=pos,
            energy_value=50.0,
            sound_intensity=0.1,
        )
        self.simulation.world.add_entity(food)
        print(f"Added food at ({world_x:.0f}, {world_y:.0f})")

    # V = Add vegetation
    elif key == pygame.K_v:
        veg = Vegetation(
            entity_id=self.simulation.world.next_entity_id,
            position=pos,
            radius=20.0,
        )
        self.simulation.world.add_entity(veg)
        print(f"Added vegetation at ({world_x:.0f}, {world_y:.0f})")

    # W = Add water
    elif key == pygame.K_w:
        water = Water(
            entity_id=self.simulation.world.next_entity_id,
            position=pos,
            radius=30.0,
        )
        self.simulation.world.add_entity(water)
        print(f"Added water at ({world_x:.0f}, {world_y:.0f})")

    # D = Delete all vegetation
    elif key == pygame.K_d:
        for veg in list(self.simulation.world.vegetation):
            self.simulation.world.remove_entity(veg.id)
        print("Cleared all vegetation")

    # T = Heal selected agent
    elif key == pygame.K_t:
        wrapper = self._get_target_agent_wrapper()
        if wrapper:
            wrapper.agent.energy = wrapper.agent.genome.max_energy
            wrapper.agent.health = wrapper.agent.genome.max_health
            print(f"Healed agent {wrapper.agent_id}")

    # Shift+P = Add predator (NOTE: regular P is pause, handled elsewhere)
    elif key == pygame.K_p and (mods & pygame.KMOD_SHIFT):
        predator = Predator(
            entity_id=self.simulation.world.next_entity_id,
            position=pos,
            patrol_center=pos,
            patrol_radius=100.0,
        )
        self.simulation.world.add_entity(predator)
        print(f"Added predator at ({world_x:.0f}, {world_y:.0f})")
```

**Step 2: Call spawn handler in _handle_events**

Add to the KEYDOWN handling. **IMPORTANT:** Shift+P must be checked BEFORE regular P (pause):
```python
# Get mouse position and modifiers at start of KEYDOWN handling
mouse_pos = pygame.mouse.get_pos()
mods = pygame.key.get_mods()

# Shift+P for predator spawn - CHECK BEFORE regular P (pause)
if event.key == pygame.K_p and (mods & pygame.KMOD_SHIFT):
    self._handle_spawn_keys(event.key, mouse_pos, mods)
# Regular P for pause (only if Shift not held)
elif event.key == pygame.K_p:
    self.paused = not self.paused
# Other spawn keys
elif event.key in [pygame.K_f, pygame.K_v, pygame.K_w, pygame.K_d, pygame.K_t]:
    self._handle_spawn_keys(event.key, mouse_pos, mods)
```

**Step 3: Test spawn shortcuts**

Run: `python -m primordial.interface.cockpit_app`
Expected: F adds food at mouse, V adds vegetation, W adds water, D clears vegetation, T heals agent, Shift+P adds predator

**Step 4: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add entity spawn keyboard shortcuts"
```

---

### Task 5.4: Add Save/Load Agent Shortcuts

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add save agent methods**

Add these methods:
```python
def _save_selected_agent(self) -> None:
    """Save selected agent to database."""
    wrapper = self._get_target_agent_wrapper()
    if wrapper is None:
        print("No agent selected to save")
        return

    agent_id = self.agent_db.save_agent(wrapper)
    stats = wrapper.lifetime_stats
    print(f"\n=== Agent Saved ===")
    print(f"  DB ID: {agent_id}")
    print(f"  Name: {wrapper.agent_id}_gen{wrapper.generation}")
    print(f"  Generation: {wrapper.generation}")
    print(f"  Food eaten: {stats.get('total_food_eaten', 0)}")
    print(f"===================\n")

def _save_all_agents(self) -> None:
    """Save all living agents to database."""
    saved = 0
    for wrapper in self.simulation.agents.values():
        if wrapper.agent.is_alive:
            self.agent_db.save_agent(wrapper)
            saved += 1
    print(f"\n=== Saved {saved} agents ===\n")

def _list_saved_agents(self) -> None:
    """List agents in database."""
    agents = self.agent_db.list_agents(order_by='longest_life', limit=9)
    dead_slots = sum(1 for w in self.simulation.agents.values() if not w.agent.is_alive)

    print(f"\n{'='*50}")
    print(f"  SAVED AGENTS")
    print(f"{'='*50}")
    for i, a in enumerate(agents):
        print(f"  [{i+1}] {a.name[:20]:<20} Gen:{a.generation:>3}  Life:{a.longest_life:>6.1f}s")
    print(f"{'='*50}")
    print(f"  Press 1-9 to load. Dead slots: {dead_slots}")
    print(f"{'='*50}\n")

def _load_agent_by_index(self, index: int) -> None:
    """Load agent from database into dead slot."""
    # Find dead slot
    dead_wrapper = None
    for w in self.simulation.agents.values():
        if not w.agent.is_alive:
            dead_wrapper = w
            break

    if dead_wrapper is None:
        print("No dead slots available")
        return

    agents = self.agent_db.list_agents(order_by='longest_life', limit=9)
    if index >= len(agents):
        print(f"No agent at position {index + 1}")
        return

    record = agents[index]
    if self.agent_db.load_agent_into_wrapper(record.id, dead_wrapper):
        self.simulation.world.add_entity(dead_wrapper.agent)
        print(f"Loaded: {record.name} into {dead_wrapper.agent_id}")
```

**Step 2: Add keyboard handlers**

Add to KEYDOWN handling:
```python
# Save/Load shortcuts
elif event.key == pygame.K_s:
    if mods & pygame.KMOD_SHIFT:
        self._save_all_agents()
    else:
        self._save_selected_agent()
elif event.key == pygame.K_l:
    self._list_saved_agents()
elif pygame.K_1 <= event.key <= pygame.K_9:
    index = event.key - pygame.K_1
    self._load_agent_by_index(index)
```

**Step 3: Test save/load**

Run: `python -m primordial.interface.cockpit_app`
Expected: S saves selected, Shift+S saves all, L lists, 1-9 loads

**Step 4: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add save/load agent keyboard shortcuts"
```

---

### Task 5.5: Add Auto-Save/Load on Start/Exit

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add auto-save method**

Add this method:
```python
def _auto_save_agents(self) -> None:
    """Auto-save all living agents on exit."""
    saved = 0
    updated = 0
    for wrapper in self.simulation.agents.values():
        if wrapper.agent.is_alive:
            db_id = getattr(wrapper, 'db_id', None)
            self.agent_db.save_agent(wrapper, db_id=db_id)
            if db_id:
                updated += 1
            else:
                saved += 1

    if saved > 0 or updated > 0:
        print(f"\n{'='*50}")
        print(f"  AUTO-SAVED: {updated} updated, {saved} new agents")
        print(f"{'='*50}\n")
```

**Step 2: Add auto-load method**

Add this method:
```python
def _auto_load_agents(self) -> None:
    """Auto-load best agents on start."""
    stats = self.agent_db.get_stats()
    if stats['total_agents'] == 0:
        print("\n  NEW SIMULATION - No saved agents\n")
        return

    loaded = self.agent_db.auto_load_best_agents(self.simulation)
    if loaded > 0:
        print(f"\n{'='*50}")
        print(f"  LOADED {loaded} agents from previous session")
        print(f"  Total in DB: {stats['total_agents']}")
        print(f"{'='*50}\n")
```

**Step 3: Call auto-load in run() and auto-save in cleanup()**

Modify `run`:
```python
def run(self) -> None:
    """Main loop."""
    self.running = True
    self._auto_load_agents()  # Load on start

    while self.running:
        # ... existing loop ...
```

Modify `cleanup`:
```python
def cleanup(self) -> None:
    """Cleanup resources."""
    self._auto_save_agents()  # Save on exit
    pygame.quit()
```

**Step 4: Test auto-save/load**

Run: `python -m primordial.interface.cockpit_app`
Expected: Previous agents load on start, agents saved on exit

**Step 5: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add auto-save/load agents on start/exit"
```

---

### Task 5.6: Add Audio Capture Integration

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add audio capture import and initialization**

Add to imports:
```python
from primordial.interface.audio_capture import AudioCapture
```

Add to __init__, after agent database (with error handling for missing microphone):
```python
# Audio capture (with graceful fallback if mic unavailable)
try:
    self.audio_capture = AudioCapture(
        sample_rate=44100,
        channels=1,
        buffer_size=2048
    )
    self.audio_enabled = True
except Exception as e:
    print(f"Warning: Audio capture unavailable: {e}")
    self.audio_capture = None
    self.audio_enabled = False
```

**Step 2: Add microphone injection method**

Add this method:
```python
def _inject_microphone_sound(self) -> None:
    """Inject microphone as sound source at selected agent."""
    from primordial.world.sound.sound_source import SoundSource
    import numpy as np

    wrapper = self._get_target_agent_wrapper()
    if wrapper is None:
        return

    audio = self.audio_capture.get_recent(1024)
    if len(audio) == 0:
        return

    rms = np.sqrt(np.mean(audio ** 2))
    intensity = min(1.0, rms * 5.0)

    if intensity < 0.01:
        return

    source = SoundSource(
        position=wrapper.agent.position.copy(),
        frequency=300.0,
        intensity=intensity,
        is_active=True,
    )
    self.simulation.world.sound_system.add_source(source)
```

**Step 3: Add push-to-talk handling in _update (with audio_enabled check)**

Add to `_update`:
```python
# Push-to-talk: SPACE unmutes microphone (only if audio enabled)
if self.audio_enabled:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        self.audio_capture.unmute()
        self._inject_microphone_sound()
    else:
        self.audio_capture.mute()
```

**Step 4: Start/stop audio in run/cleanup (with audio_enabled check)**

Add to `run` after `_auto_load_agents`:
```python
if self.audio_enabled:
    self.audio_capture.start()
```

Add to `cleanup` before `pygame.quit()`:
```python
if self.audio_enabled and self.audio_capture:
    self.audio_capture.stop()
```

**Step 5: Test audio**

Run: `python -m primordial.interface.cockpit_app`
Expected: Hold SPACE to speak, audio visualizer in bottom bar responds (if implemented)

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add audio capture and push-to-talk"
```

---

## Phase 6: Modals (Future)

### Task 6.1: Database Browser Modal

*To be implemented* - Use pygame_gui.windows.UIWindow with UISelectionList for agent list.

### Task 6.2: Genome Editor Modal

*To be implemented* - Use pygame_gui.windows.UIWindow with sliders for each genome parameter.

### Task 6.3: Help Overlay Modal

*To be implemented* - Simple UIWindow showing keyboard shortcuts.

---

## Phase 7: Remaining Control Tabs (Future)

### Task 7.1: Agents Tab
- Default genome sliders for new agents

### Task 7.2: Learn Tab
- Learning rate, hidden dim, mixing layers

### Task 7.3: Rewards Tab
- All 12 reward values

### Task 7.4: Predators Tab
- Patrol, detection, speed, damage

### Task 7.5: Presets Tab
- Save/load configurations

---

## Summary

**Phase 1-4** creates a functional cockpit interface with:
- Sci-fi themed pygame-gui setup
- World view with entity rendering
- HUD top/bottom bars
- Collapsible left panel with World tab sliders
- Right panel with agent table and selection
- Click-to-select in world view

**Phase 5** migrates from the old interface.

**Phase 6-7** adds modals and remaining control tabs.

---

**Plan complete and saved to `docs/plans/2025-01-28-cockpit-ui-implementation.md`.**

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
