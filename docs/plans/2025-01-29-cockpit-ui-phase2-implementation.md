# Cockpit UI Phase 2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete all remaining Cockpit UI features from the brainstorming document - control tabs, modals, agent panel enhancements, and keyboard shortcuts.

**Architecture:** Extends existing CockpitApp with 5 new control tabs (reusing slider pattern from World tab), 3 pygame-gui modals, enhanced agent panel with sort/filter/actions, and additional keyboard shortcuts for control mode and map save/load.

**Tech Stack:** pygame 2.x, pygame-gui 0.6+ (for modals), existing simulation/world/lrn modules

**Reference:** `docs/plans/brainstorming-session-cockpit-ui.md` and `docs/mockups/primordial-ui-mockup.html`

---

## Pre-Implementation Setup

### Task 0: Add Required Imports and Color Constants

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add all required imports at file top**

After existing imports, add:
```python
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from primordial.agents.genome import AgentGenome
from primordial.learning.rewards import SurvivalRewards
```

**Step 2: Add color constants to __init__**

Add after existing color definitions (or create color section if not present):
```python
# UI Colors
self.CYAN = (0, 255, 255)
self.CYAN_DIM = (0, 150, 150)
self.GREEN = (0, 200, 100)
self.GREEN_DIM = (0, 100, 50)
self.BG_DARKEST = (20, 20, 28)
self.BG_DARK = (28, 28, 36)
self.BG_PANEL = (32, 32, 42)
self.TEXT_BRIGHT = (255, 255, 255)
self.TEXT_NORMAL = (200, 200, 210)
self.TEXT_DIM = (120, 120, 140)
```

**Step 3: Add user data directory helper**

```python
# User data directory for saves
self.user_data_dir = Path.home() / ".primordial"
self.user_data_dir.mkdir(exist_ok=True)
self.maps_dir = self.user_data_dir / "maps"
self.maps_dir.mkdir(exist_ok=True)
self.presets_file = self.user_data_dir / "custom_presets.json"
```

**Step 4: Commit setup changes**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "chore: add imports and constants for Phase 2 UI"
```

---

## Phase 1: Remaining Control Tabs (Tasks 1-5)

### Task 1: Implement Agents Tab (Default Genome Sliders)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add default_genome to __init__**

Add after `self.control_values` dict (around line 70):
```python
# Default genome for new agents (modifiable via Agents tab)
from primordial.agents.genome import AgentGenome
self.default_genome = AgentGenome()
```

**Step 2: Add genome slider sections to _render_left_panel_content**

Find the `else` block in `_render_left_panel_content` (around line 620) and replace with:
```python
elif self.left_panel_tab == "agents":
    # PHYSICAL section
    section = self.font_small.render("PHYSICAL", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Max Speed",
                             self.default_genome.max_speed, 50, 300, "genome_max_speed", 150.0)
    y += self._render_slider(x, y, width, "Max Angular Speed",
                             self.default_genome.max_angular_speed, 1.0, 6.0, "genome_max_angular_speed", 3.0)
    y += self._render_slider(x, y, width, "Thrust Force",
                             self.default_genome.thrust_force, 100, 1000, "genome_thrust_force", 500.0)
    y += self._render_slider(x, y, width, "Radius",
                             self.default_genome.radius, 4.0, 20.0, "genome_radius", 8.0)
    y += 8

    # SENSORY section
    section = self.font_small.render("SENSORY", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Vision Range",
                             self.default_genome.vision_range, 50, 400, "genome_vision_range", 200.0)
    y += self._render_slider(x, y, width, "Vision FOV",
                             self.default_genome.vision_fov, 60, 180, "genome_vision_fov", 120.0)
    y += self._render_slider(x, y, width, "Audio Range",
                             self.default_genome.audio_range, 50, 500, "genome_audio_range", 300.0)
    y += 8

    # METABOLIC section
    section = self.font_small.render("METABOLIC", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Base Energy Cost",
                             self.default_genome.base_energy_cost, 0.01, 0.5, "genome_base_energy_cost", 0.1)
    y += self._render_slider(x, y, width, "Movement Energy",
                             self.default_genome.movement_energy_mult, 0.1, 2.0, "genome_movement_energy_mult", 0.5)
    y += self._render_slider(x, y, width, "Eating Efficiency",
                             self.default_genome.eating_efficiency, 0.5, 1.5, "genome_eating_efficiency", 0.9)

else:
    # Placeholder for other tabs
    content_text = self.font_small.render(f"[{self.left_panel_tab.upper()} controls]", True, self.TEXT_DIM)
    self.screen.blit(content_text, (x, y + 16))
```

**Step 3: Add genome slider interaction handling**

In the MOUSEBUTTONDOWN event handler, after existing slider collision checks, add:
```python
# Genome sliders (Agents tab)
if self.left_panel_visible and self.left_panel_tab == "agents":
    genome_keys = ["genome_max_speed", "genome_max_angular_speed", "genome_thrust_force",
                   "genome_radius", "genome_vision_range", "genome_vision_fov",
                   "genome_audio_range", "genome_base_energy_cost",
                   "genome_movement_energy_mult", "genome_eating_efficiency"]
    for key in genome_keys:
        rect = getattr(self, f"slider_{key}_rect", None)
        if rect and rect.collidepoint(mouse_pos):
            self.active_slider = key
            break
```

**Step 4: Add genome slider value application**

In the MOUSEMOTION handler where other sliders are processed, add genome slider handling.
This code calculates the value from mouse position and applies it:

```python
# Handle genome slider dragging
if self.active_slider and self.active_slider.startswith("genome_"):
    # Find the slider rect and range for this key
    slider_rect = getattr(self, f"slider_{self.active_slider}_rect", None)
    if slider_rect:
        # Get min/max from slider metadata (stored during _render_slider)
        slider_meta = getattr(self, f"slider_{self.active_slider}_meta", None)
        if slider_meta:
            min_val, max_val = slider_meta
            # Calculate value from mouse position
            rel_x = max(0, min(event.pos[0] - slider_rect.x, slider_rect.width))
            pct = rel_x / slider_rect.width
            new_val = min_val + pct * (max_val - min_val)

            # Apply to default genome
            attr_name = self.active_slider.replace("genome_", "")
            if hasattr(self.default_genome, attr_name):
                setattr(self.default_genome, attr_name, new_val)
```

**Step 5: Update _render_slider to store metadata**

The `_render_slider` method needs to store min/max values for later use. Add at end of method:
```python
# Store slider metadata for interaction
setattr(self, f"slider_{key}_meta", (min_val, max_val))
```

**Step 6: Test Agents tab**

Run: `python -m primordial.interface.cockpit_app`
Expected: Click "Agents" tab, sliders for genome parameters appear, dragging updates default_genome

**Step 7: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement Agents tab with genome sliders"
```

---

### Task 2: Implement Learn Tab (Learning Parameter Sliders)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add learning control values to __init__**

Add to `self.control_values` dict (around line 72):
```python
# Learning settings
"lrn_hidden_dim": self.sim_config.lrn_hidden_dim,
"lrn_num_mixing_layers": self.sim_config.lrn_num_mixing_layers,
"learning_rate": self.sim_config.learning_rate,
"reward_modulation_scale": self.sim_config.reward_modulation_scale,
```

**Step 2: Add Learn tab content**

Add after the "agents" tab elif block:
```python
elif self.left_panel_tab == "learn":
    # ARCHITECTURE section
    section = self.font_small.render("ARCHITECTURE", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Hidden Dim",
                             self.control_values["lrn_hidden_dim"], 32, 512, "lrn_hidden_dim", 128)
    y += self._render_slider(x, y, width, "Mixing Layers",
                             self.control_values["lrn_num_mixing_layers"], 2, 12, "lrn_num_mixing_layers", 6)
    y += 8

    # TRAINING section
    section = self.font_small.render("TRAINING", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    # Learning rate needs special handling (very small values)
    lr_display = self.control_values["learning_rate"] * 10000  # Display as 0.1-10
    y += self._render_slider(x, y, width, "Learning Rate (×10⁻⁴)",
                             lr_display, 0.1, 10.0, "learning_rate_display", 1.0)
    y += self._render_slider(x, y, width, "Reward Scale",
                             self.control_values["reward_modulation_scale"], 0.1, 5.0, "reward_modulation_scale", 1.0)

    # Note about live changes
    note = self.font_small.render("⚠ Architecture changes apply", True, self.TEXT_DIM)
    self.screen.blit(note, (x, y + 16))
    note2 = self.font_small.render("   to new agents only", True, self.TEXT_DIM)
    self.screen.blit(note2, (x, y + 30))
```

**Step 3: Add Learn tab slider interaction**

Add to slider interaction handling:
```python
# Learn tab sliders
if self.left_panel_visible and self.left_panel_tab == "learn":
    learn_keys = ["lrn_hidden_dim", "lrn_num_mixing_layers", "learning_rate_display", "reward_modulation_scale"]
    for key in learn_keys:
        rect = getattr(self, f"slider_{key}_rect", None)
        if rect and rect.collidepoint(mouse_pos):
            self.active_slider = key
            break
```

**Step 4: Handle learning rate special case in MOUSEMOTION**

Add special handling for learning rate:
```python
# Special handling for learning rate (convert display value to actual)
if self.active_slider == "learning_rate_display":
    self.control_values["learning_rate"] = new_val / 10000
    if hasattr(self.sim_config, "learning_rate"):
        self.sim_config.learning_rate = new_val / 10000
elif self.active_slider in ["lrn_hidden_dim", "lrn_num_mixing_layers"]:
    new_val = int(round(new_val))
    self.control_values[self.active_slider] = new_val
    if hasattr(self.sim_config, self.active_slider):
        setattr(self.sim_config, self.active_slider, new_val)
```

**Step 5: Test Learn tab**

Run: `python -m primordial.interface.cockpit_app`
Expected: Learn tab shows architecture and training sliders

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement Learn tab with LRN parameter sliders"
```

---

### Task 3: Implement Rewards Tab (Survival Reward Sliders)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add reward values to __init__**

Add after control_values:
```python
# Reward values (modifiable - affects SurvivalRewards class)
from primordial.learning.rewards import SurvivalRewards
self.reward_values = {
    "eating_food": SurvivalRewards.EATING_FOOD,
    "taking_damage": SurvivalRewards.TAKING_DAMAGE,
    "death": SurvivalRewards.DEATH,
    "starving": SurvivalRewards.STARVING,
    "low_health": SurvivalRewards.LOW_HEALTH,
    "healthy": SurvivalRewards.HEALTHY,
    "movement_bonus": SurvivalRewards.MOVEMENT_BONUS,
    "idle_penalty": SurvivalRewards.IDLE_PENALTY,
    "breeding_drive": SurvivalRewards.HIGH_BREEDING_DRIVE,
    "loneliness": SurvivalRewards.LONELINESS_PENALTY,
    "social_bonus": SurvivalRewards.SOCIAL_BONUS,
}
```

**Step 2: Add Rewards tab content**

Add after learn tab elif:
```python
elif self.left_panel_tab == "rewards":
    # EVENT REWARDS section
    section = self.font_small.render("EVENT REWARDS", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Eating Food",
                             self.reward_values["eating_food"], 0.0, 5.0, "reward_eating_food", 1.0)
    y += self._render_slider(x, y, width, "Taking Damage",
                             self.reward_values["taking_damage"], -5.0, 0.0, "reward_taking_damage", -2.0)
    y += self._render_slider(x, y, width, "Death",
                             self.reward_values["death"], -20.0, 0.0, "reward_death", -10.0)
    y += 8

    # CONTINUOUS REWARDS section
    section = self.font_small.render("CONTINUOUS REWARDS", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Starving",
                             self.reward_values["starving"], -0.5, 0.0, "reward_starving", -0.1)
    y += self._render_slider(x, y, width, "Low Health",
                             self.reward_values["low_health"], -0.2, 0.0, "reward_low_health", -0.05)
    y += self._render_slider(x, y, width, "Healthy",
                             self.reward_values["healthy"], 0.0, 0.1, "reward_healthy", 0.01)
    y += 8

    # BEHAVIOR REWARDS section
    section = self.font_small.render("BEHAVIOR REWARDS", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Movement Bonus",
                             self.reward_values["movement_bonus"], 0.0, 0.1, "reward_movement_bonus", 0.02)
    y += self._render_slider(x, y, width, "Idle Penalty",
                             self.reward_values["idle_penalty"], -0.1, 0.0, "reward_idle_penalty", -0.01)
    y += self._render_slider(x, y, width, "Social Bonus",
                             self.reward_values["social_bonus"], 0.0, 0.1, "reward_social_bonus", 0.01)
```

**Step 3: Add Rewards tab slider interaction and application**

Add slider interaction:
```python
# Rewards tab sliders
if self.left_panel_visible and self.left_panel_tab == "rewards":
    reward_keys = ["reward_eating_food", "reward_taking_damage", "reward_death",
                   "reward_starving", "reward_low_health", "reward_healthy",
                   "reward_movement_bonus", "reward_idle_penalty", "reward_social_bonus"]
    for key in reward_keys:
        rect = getattr(self, f"slider_{key}_rect", None)
        if rect and rect.collidepoint(mouse_pos):
            self.active_slider = key
            break
```

Add value application in MOUSEMOTION:
```python
# Apply reward slider values
if self.active_slider and self.active_slider.startswith("reward_"):
    reward_key = self.active_slider.replace("reward_", "")
    self.reward_values[reward_key] = new_val
    # Update SurvivalRewards class attribute
    attr_map = {
        "eating_food": "EATING_FOOD", "taking_damage": "TAKING_DAMAGE",
        "death": "DEATH", "starving": "STARVING", "low_health": "LOW_HEALTH",
        "healthy": "HEALTHY", "movement_bonus": "MOVEMENT_BONUS",
        "idle_penalty": "IDLE_PENALTY", "social_bonus": "SOCIAL_BONUS"
    }
    if reward_key in attr_map:
        setattr(SurvivalRewards, attr_map[reward_key], new_val)
```

**Step 4: Test Rewards tab**

Run: `python -m primordial.interface.cockpit_app`
Expected: Rewards tab shows 11 sliders across 3 sections

**Step 5: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement Rewards tab with survival reward sliders"
```

---

### Task 4: Implement Predators Tab

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add predator config values to __init__**

Add:
```python
# Predator config (default values for new predators)
self.predator_config = {
    "patrol_radius": 150.0,
    "detection_radius": 250.0,
    "chase_speed": 120.0,
    "patrol_speed": 40.0,
    "damage": 20.0,
    "attack_cooldown": 1.0,
    "chase_abandon_distance": 350.0,
}
```

**Step 2: Add Predators tab content**

Add after rewards tab elif:
```python
elif self.left_panel_tab == "predators":
    # PATROL section
    section = self.font_small.render("PATROL", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Patrol Radius",
                             self.predator_config["patrol_radius"], 50, 300, "pred_patrol_radius", 150.0)
    y += self._render_slider(x, y, width, "Patrol Speed",
                             self.predator_config["patrol_speed"], 20, 80, "pred_patrol_speed", 40.0)
    y += 8

    # DETECTION section
    section = self.font_small.render("DETECTION", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Detection Radius",
                             self.predator_config["detection_radius"], 100, 400, "pred_detection_radius", 250.0)
    y += self._render_slider(x, y, width, "Chase Speed",
                             self.predator_config["chase_speed"], 60, 200, "pred_chase_speed", 120.0)
    y += self._render_slider(x, y, width, "Abandon Distance",
                             self.predator_config["chase_abandon_distance"], 200, 500, "pred_chase_abandon", 350.0)
    y += 8

    # COMBAT section
    section = self.font_small.render("COMBAT", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    y += self._render_slider(x, y, width, "Damage",
                             self.predator_config["damage"], 5, 50, "pred_damage", 20.0)
    y += self._render_slider(x, y, width, "Attack Cooldown",
                             self.predator_config["attack_cooldown"], 0.5, 3.0, "pred_attack_cooldown", 1.0)

    # Apply to existing button
    y += 16
    apply_rect = pygame.Rect(x, y, width, 28)
    pygame.draw.rect(self.screen, (37, 37, 48), apply_rect, border_radius=4)
    pygame.draw.rect(self.screen, self.CYAN_DIM, apply_rect, 1, border_radius=4)
    apply_text = self.font_small.render("Apply to All Predators", True, self.CYAN)
    self.screen.blit(apply_text, (apply_rect.centerx - apply_text.get_width() // 2, y + 6))
    self.apply_pred_btn_rect = apply_rect
```

**Step 3: Add predator slider interaction**

```python
# Predators tab sliders
if self.left_panel_visible and self.left_panel_tab == "predators":
    pred_keys = ["pred_patrol_radius", "pred_patrol_speed", "pred_detection_radius",
                 "pred_chase_speed", "pred_chase_abandon", "pred_damage", "pred_attack_cooldown"]
    for key in pred_keys:
        rect = getattr(self, f"slider_{key}_rect", None)
        if rect and rect.collidepoint(mouse_pos):
            self.active_slider = key
            break
```

**Step 4: Add predator value application and button handler**

In MOUSEMOTION:
```python
# Apply predator slider values
if self.active_slider and self.active_slider.startswith("pred_"):
    config_map = {
        "pred_patrol_radius": "patrol_radius",
        "pred_patrol_speed": "patrol_speed",
        "pred_detection_radius": "detection_radius",
        "pred_chase_speed": "chase_speed",
        "pred_chase_abandon": "chase_abandon_distance",
        "pred_damage": "damage",
        "pred_attack_cooldown": "attack_cooldown",
    }
    if self.active_slider in config_map:
        self.predator_config[config_map[self.active_slider]] = new_val
```

Add button click handler:
```python
# Apply to all predators button
if hasattr(self, 'apply_pred_btn_rect') and self.apply_pred_btn_rect.collidepoint(mouse_pos):
    self._apply_predator_config()
```

Add method:
```python
def _apply_predator_config(self) -> None:
    """Apply current predator config to all existing predators."""
    count = 0
    for predator in self.simulation.world.predators:
        predator.patrol_radius = self.predator_config["patrol_radius"]
        predator.patrol_speed = self.predator_config["patrol_speed"]
        predator.detection_radius = self.predator_config["detection_radius"]
        predator.chase_speed = self.predator_config["chase_speed"]
        predator.chase_abandon_distance = self.predator_config["chase_abandon_distance"]
        predator.damage = self.predator_config["damage"]
        predator.attack_cooldown_max = self.predator_config["attack_cooldown"]
        count += 1
    print(f"Applied config to {count} predators")
```

**Step 5: Test Predators tab**

Run: `python -m primordial.interface.cockpit_app`
Expected: Predators tab shows sliders, "Apply to All" button works

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement Predators tab with combat sliders"
```

---

### Task 5: Implement Presets Tab

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add presets storage to __init__**

```python
# Presets
self.presets = {
    "Easy": {"max_agents": 3, "predator_count": 0, "initial_food": 80},
    "Normal": {"max_agents": 5, "predator_count": 2, "initial_food": 50},
    "Hard": {"max_agents": 8, "predator_count": 4, "initial_food": 30},
    "Chaos": {"max_agents": 15, "predator_count": 6, "initial_food": 100},
}
self.current_preset = "Normal"
# Use user data directory instead of source directory
self._load_custom_presets()
```

**Step 2: Add preset methods**

```python
def _load_custom_presets(self) -> None:
    """Load custom presets from file."""
    if self.presets_file.exists():
        try:
            with open(self.presets_file) as f:
                custom = json.load(f)
                self.presets.update(custom)
        except Exception as e:
            print(f"Warning: Could not load custom presets: {e}")

def _save_current_as_preset(self, name: str) -> None:
    """Save current configuration as a preset."""
    preset = {
        "max_agents": self.control_values["max_agents"],
        "predator_count": self.control_values["predator_count"],
        "initial_food": self.control_values["initial_food"],
        "max_food": self.control_values["max_food"],
        "tick_rate": self.control_values["tick_rate"],
    }
    self.presets[name] = preset

    # Save custom presets (excluding built-ins)
    custom = {k: v for k, v in self.presets.items()
              if k not in ["Easy", "Normal", "Hard", "Chaos"]}
    try:
        with open(self.presets_file, 'w') as f:
            json.dump(custom, f, indent=2)
        print(f"Saved preset: {name}")
    except Exception as e:
        print(f"Error saving preset: {e}")

def _apply_preset(self, name: str) -> None:
    """Apply a preset to current configuration."""
    if name not in self.presets:
        return
    preset = self.presets[name]
    for key, value in preset.items():
        if key in self.control_values:
            self.control_values[key] = value
            if hasattr(self.sim_config, key):
                setattr(self.sim_config, key, value)
    self.current_preset = name
    print(f"Applied preset: {name}")
```

**Step 3: Add Presets tab content**

```python
elif self.left_panel_tab == "presets":
    # BUILT-IN PRESETS section
    section = self.font_small.render("BUILT-IN PRESETS", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    self.preset_btn_rects = {}
    for preset_name in ["Easy", "Normal", "Hard", "Chaos"]:
        btn_rect = pygame.Rect(x, y, width, 28)
        is_active = self.current_preset == preset_name
        bg_color = (42, 42, 56) if is_active else (37, 37, 48)
        border_color = self.CYAN if is_active else (42, 42, 56)

        pygame.draw.rect(self.screen, bg_color, btn_rect, border_radius=4)
        pygame.draw.rect(self.screen, border_color, btn_rect, 1, border_radius=4)

        text_color = self.CYAN if is_active else self.TEXT_NORMAL
        text = self.font_small.render(preset_name, True, text_color)
        self.screen.blit(text, (x + 12, y + 6))

        # Show key values
        p = self.presets[preset_name]
        info = f"Agents:{p.get('max_agents', '?')} Pred:{p.get('predator_count', '?')}"
        info_text = self.font_small.render(info, True, self.TEXT_DIM)
        self.screen.blit(info_text, (x + width - info_text.get_width() - 8, y + 6))

        self.preset_btn_rects[preset_name] = btn_rect
        y += 32

    y += 8

    # CUSTOM PRESETS section
    section = self.font_small.render("CUSTOM PRESETS", True, self.TEXT_DIM)
    self.screen.blit(section, (x, y))
    pygame.draw.line(self.screen, (42, 42, 56), (x, y + 16), (x + width, y + 16))
    y += 24

    custom_presets = [k for k in self.presets if k not in ["Easy", "Normal", "Hard", "Chaos"]]
    if not custom_presets:
        note = self.font_small.render("No custom presets", True, self.TEXT_DIM)
        self.screen.blit(note, (x + 12, y))
        y += 20
    else:
        for preset_name in custom_presets[:4]:  # Show max 4
            btn_rect = pygame.Rect(x, y, width, 28)
            is_active = self.current_preset == preset_name
            bg_color = (42, 42, 56) if is_active else (37, 37, 48)
            pygame.draw.rect(self.screen, bg_color, btn_rect, border_radius=4)
            text = self.font_small.render(preset_name, True, self.TEXT_NORMAL)
            self.screen.blit(text, (x + 12, y + 6))
            self.preset_btn_rects[preset_name] = btn_rect
            y += 32

    y += 8

    # Save current button
    save_rect = pygame.Rect(x, y, width, 28)
    pygame.draw.rect(self.screen, self.CYAN_DIM, save_rect, border_radius=4)
    save_text = self.font_small.render("Save Current as Preset", True, self.BG_DARKEST)
    self.screen.blit(save_text, (save_rect.centerx - save_text.get_width() // 2, y + 6))
    self.save_preset_btn_rect = save_rect
```

**Step 4: Add preset button click handling**

```python
# Preset buttons
if self.left_panel_tab == "presets" and hasattr(self, 'preset_btn_rects'):
    for name, rect in self.preset_btn_rects.items():
        if rect.collidepoint(mouse_pos):
            self._apply_preset(name)
            break

    if hasattr(self, 'save_preset_btn_rect') and self.save_preset_btn_rect.collidepoint(mouse_pos):
        # Simple name: Custom_1, Custom_2, etc.
        existing = [k for k in self.presets if k.startswith("Custom_")]
        num = len(existing) + 1
        self._save_current_as_preset(f"Custom_{num}")
```

**Step 5: Test Presets tab**

Run: `python -m primordial.interface.cockpit_app`
Expected: Preset buttons work, Save button creates Custom_N presets

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement Presets tab with save/load functionality"
```

---

## Phase 2: Agent Panel Enhancements (Tasks 6-8)

### Task 6: Add Sort and Filter Dropdowns to Agent Table

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add sort/filter state to __init__**

```python
# Agent table sort/filter
self.agent_sort_key = "age"  # age, energy, health, generation, food
self.agent_filter = "all"  # all, alive, dead
self.sort_dropdown_open = False
self.filter_dropdown_open = False
```

**Step 2: Update _render_agent_table with sort dropdown**

After the table, before selected agent section, add:
```python
# Sort/Filter row
y += 8
sort_x = x

# Sort dropdown
sort_rect = pygame.Rect(sort_x, y, 80, 22)
pygame.draw.rect(self.screen, (37, 37, 48), sort_rect, border_radius=4)
pygame.draw.rect(self.screen, (42, 42, 56), sort_rect, 1, border_radius=4)
sort_text = self.font_small.render(f"Sort: {self.agent_sort_key[:4]}", True, self.TEXT_NORMAL)
self.screen.blit(sort_text, (sort_x + 4, y + 3))
arrow = self.font_small.render("▼", True, self.TEXT_DIM)
self.screen.blit(arrow, (sort_x + 66, y + 3))
self.sort_dropdown_rect = sort_rect

# Filter dropdown
filter_x = sort_x + 88
filter_rect = pygame.Rect(filter_x, y, 80, 22)
pygame.draw.rect(self.screen, (37, 37, 48), filter_rect, border_radius=4)
pygame.draw.rect(self.screen, (42, 42, 56), filter_rect, 1, border_radius=4)
filter_text = self.font_small.render(f"Show: {self.agent_filter}", True, self.TEXT_NORMAL)
self.screen.blit(filter_text, (filter_x + 4, y + 3))
arrow2 = self.font_small.render("▼", True, self.TEXT_DIM)
self.screen.blit(arrow2, (filter_x + 66, y + 3))
self.filter_dropdown_rect = filter_rect

y += 26

# Dropdown menus (render on top if open)
if self.sort_dropdown_open:
    menu_y = self.sort_dropdown_rect.bottom + 2
    sort_options = ["age", "energy", "health", "gen", "food"]
    self.sort_option_rects = []
    for opt in sort_options:
        opt_rect = pygame.Rect(sort_x, menu_y, 80, 20)
        bg = (50, 50, 60) if opt == self.agent_sort_key else (37, 37, 48)
        pygame.draw.rect(self.screen, bg, opt_rect)
        opt_text = self.font_small.render(opt.capitalize(), True, self.TEXT_NORMAL)
        self.screen.blit(opt_text, (sort_x + 4, menu_y + 2))
        self.sort_option_rects.append((opt_rect, opt))
        menu_y += 20

if self.filter_dropdown_open:
    menu_y = self.filter_dropdown_rect.bottom + 2
    filter_options = ["all", "alive", "dead"]
    self.filter_option_rects = []
    for opt in filter_options:
        opt_rect = pygame.Rect(filter_x, menu_y, 80, 20)
        bg = (50, 50, 60) if opt == self.agent_filter else (37, 37, 48)
        pygame.draw.rect(self.screen, bg, opt_rect)
        opt_text = self.font_small.render(opt.capitalize(), True, self.TEXT_NORMAL)
        self.screen.blit(opt_text, (filter_x + 4, menu_y + 2))
        self.filter_option_rects.append((opt_rect, opt))
        menu_y += 20
```

**Step 3: Update agent sorting logic**

Update the agents_data.sort() line:
```python
# Sort based on current sort key
sort_funcs = {
    "age": lambda a: (-int(a['alive']), -a['age']),
    "energy": lambda a: (-int(a['alive']), -a['energy']),
    "health": lambda a: (-int(a['alive']), -a['health']),
    "gen": lambda a: (-int(a['alive']), -a['generation']),
    "food": lambda a: (-int(a['alive']), -a['food']),
}
agents_data.sort(key=sort_funcs.get(self.agent_sort_key, sort_funcs["age"]))

# Filter
if self.agent_filter == "alive":
    agents_data = [a for a in agents_data if a['alive']]
elif self.agent_filter == "dead":
    agents_data = [a for a in agents_data if not a['alive']]
```

**Step 4: Add dropdown click handling**

```python
# Sort dropdown toggle
if hasattr(self, 'sort_dropdown_rect') and self.sort_dropdown_rect.collidepoint(mouse_pos):
    self.sort_dropdown_open = not self.sort_dropdown_open
    self.filter_dropdown_open = False
    return

# Filter dropdown toggle
if hasattr(self, 'filter_dropdown_rect') and self.filter_dropdown_rect.collidepoint(mouse_pos):
    self.filter_dropdown_open = not self.filter_dropdown_open
    self.sort_dropdown_open = False
    return

# Sort option selection
if self.sort_dropdown_open and hasattr(self, 'sort_option_rects'):
    for rect, opt in self.sort_option_rects:
        if rect.collidepoint(mouse_pos):
            self.agent_sort_key = opt
            self.sort_dropdown_open = False
            return

# Filter option selection
if self.filter_dropdown_open and hasattr(self, 'filter_option_rects'):
    for rect, opt in self.filter_option_rects:
        if rect.collidepoint(mouse_pos):
            self.agent_filter = opt
            self.filter_dropdown_open = False
            return

# Close dropdowns on click elsewhere
if self.sort_dropdown_open or self.filter_dropdown_open:
    self.sort_dropdown_open = False
    self.filter_dropdown_open = False
```

**Step 5: Test sort/filter**

Run: `python -m primordial.interface.cockpit_app`
Expected: Dropdowns open, selecting changes sort/filter

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add sort and filter dropdowns to agent table"
```

---

### Task 7: Add Agent Action Buttons (Track, Edit, Heal, Respawn)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add tracking state to __init__**

```python
# Camera tracking
self.tracking_agent_id: Optional[str] = None
```

**Step 2: Add action buttons to _render_selected_agent_detail**

**Note on emoji buttons:** The code below uses emoji symbols for button labels. If emojis don't render properly in your pygame font, replace them with ASCII equivalents:
- `👁` → `[o]` or `Eye`
- `✏` → `[E]` or `Edit`
- `💉` → `[+]` or `Heal`
- `🔄` → `[R]` or `Resp`
- `💾` → `[S]` or `Save`
- `📂` → `[D]` or `Load`

After the status bars, add:
```python
y += 8

# ACTIONS section header
pygame.draw.rect(self.screen, self.BG_DARK, pygame.Rect(x - 8, y, self.PANEL_WIDTH, 24))
actions_title = self.font_small.render("ACTIONS", True, self.TEXT_DIM)
self.screen.blit(actions_title, (x, y + 4))
y += 28

# Button row 1: Track, Edit Genome
btn_width = (width - 8) // 2
track_rect = pygame.Rect(x, y, btn_width, 26)
is_tracking = self.tracking_agent_id == wrapper.agent_id if wrapper else False
track_bg = self.CYAN_DIM if is_tracking else (37, 37, 48)
pygame.draw.rect(self.screen, track_bg, track_rect, border_radius=4)
pygame.draw.rect(self.screen, self.TEXT_DIM, track_rect, 1, border_radius=4)
track_text = self.font_small.render("👁 Track" if not is_tracking else "👁 Stop", True,
                                     self.BG_DARKEST if is_tracking else self.TEXT_NORMAL)
self.screen.blit(track_text, (x + 8, y + 5))
self.track_btn_rect = track_rect

edit_rect = pygame.Rect(x + btn_width + 8, y, btn_width, 26)
pygame.draw.rect(self.screen, (37, 37, 48), edit_rect, border_radius=4)
pygame.draw.rect(self.screen, self.TEXT_DIM, edit_rect, 1, border_radius=4)
edit_text = self.font_small.render("✏ Edit", True, self.TEXT_NORMAL)
self.screen.blit(edit_text, (x + btn_width + 16, y + 5))
self.edit_genome_btn_rect = edit_rect

y += 30

# Button row 2: Heal, Respawn
heal_rect = pygame.Rect(x, y, btn_width, 26)
pygame.draw.rect(self.screen, (0, 100, 60), heal_rect, border_radius=4)
pygame.draw.rect(self.screen, self.GREEN, heal_rect, 1, border_radius=4)
heal_text = self.font_small.render("💉 Heal", True, self.TEXT_BRIGHT)
self.screen.blit(heal_text, (x + 8, y + 5))
self.heal_btn_rect = heal_rect

respawn_rect = pygame.Rect(x + btn_width + 8, y, btn_width, 26)
can_respawn = wrapper and not wrapper.agent.is_alive if wrapper else False
respawn_bg = (100, 60, 0) if can_respawn else (40, 40, 40)
pygame.draw.rect(self.screen, respawn_bg, respawn_rect, border_radius=4)
pygame.draw.rect(self.screen, self.TEXT_DIM, respawn_rect, 1, border_radius=4)
respawn_text = self.font_small.render("🔄 Respawn", True, self.TEXT_NORMAL if can_respawn else self.TEXT_DIM)
self.screen.blit(respawn_text, (x + btn_width + 16, y + 5))
self.respawn_btn_rect = respawn_rect

y += 30

# Button row 3: Save, Load DB
save_rect = pygame.Rect(x, y, btn_width, 26)
pygame.draw.rect(self.screen, (37, 37, 48), save_rect, border_radius=4)
pygame.draw.rect(self.screen, self.TEXT_DIM, save_rect, 1, border_radius=4)
save_text = self.font_small.render("💾 Save", True, self.TEXT_NORMAL)
self.screen.blit(save_text, (x + 8, y + 5))
self.save_agent_btn_rect = save_rect

load_db_rect = pygame.Rect(x + btn_width + 8, y, btn_width, 26)
pygame.draw.rect(self.screen, (37, 37, 48), load_db_rect, border_radius=4)
pygame.draw.rect(self.screen, self.TEXT_DIM, load_db_rect, 1, border_radius=4)
load_text = self.font_small.render("📂 Load DB", True, self.TEXT_NORMAL)
self.screen.blit(load_text, (x + btn_width + 16, y + 5))
self.load_db_btn_rect = load_db_rect
```

**Step 3: Add button click handlers**

```python
# Agent action buttons
if hasattr(self, 'track_btn_rect') and self.track_btn_rect.collidepoint(mouse_pos):
    wrapper = self._get_target_agent_wrapper()
    if wrapper:
        if self.tracking_agent_id == wrapper.agent_id:
            self.tracking_agent_id = None
            print("Stopped tracking")
        else:
            self.tracking_agent_id = wrapper.agent_id
            print(f"Tracking agent {wrapper.agent_id}")
    return

if hasattr(self, 'heal_btn_rect') and self.heal_btn_rect.collidepoint(mouse_pos):
    wrapper = self._get_target_agent_wrapper()
    if wrapper and wrapper.agent.is_alive:
        wrapper.agent.energy = wrapper.agent.genome.max_energy
        wrapper.agent.health = wrapper.agent.genome.max_health
        print(f"Healed agent {wrapper.agent_id}")
    return

if hasattr(self, 'respawn_btn_rect') and self.respawn_btn_rect.collidepoint(mouse_pos):
    wrapper = self._get_target_agent_wrapper()
    if wrapper and not wrapper.agent.is_alive:
        wrapper.agent.respawn()
        self.simulation.world.add_entity(wrapper.agent)
        print(f"Respawned agent {wrapper.agent_id}")
    return

if hasattr(self, 'save_agent_btn_rect') and self.save_agent_btn_rect.collidepoint(mouse_pos):
    self._save_selected_agent()
    return

if hasattr(self, 'edit_genome_btn_rect') and self.edit_genome_btn_rect.collidepoint(mouse_pos):
    self._open_genome_editor()
    return

if hasattr(self, 'load_db_btn_rect') and self.load_db_btn_rect.collidepoint(mouse_pos):
    self._open_database_browser()
    return
```

**Step 4: Add camera tracking to _render_world**

At the start of _render_world, after getting transform:
```python
# Camera tracking - center on tracked agent
if self.tracking_agent_id and self.tracking_agent_id in self.simulation.agents:
    wrapper = self.simulation.agents[self.tracking_agent_id]
    if wrapper.agent.is_alive:
        # Offset calculation to center on agent
        agent_screen_x = offset_x + wrapper.agent.position.x * scale
        agent_screen_y = offset_y + wrapper.agent.position.y * scale
        center_x = world_rect.centerx
        center_y = world_rect.centery
        # Adjust offset to center on agent
        offset_x += center_x - agent_screen_x
        offset_y += center_y - agent_screen_y
    else:
        # Stop tracking dead agents
        self.tracking_agent_id = None
```

**Step 5: Test action buttons**

Run: `python -m primordial.interface.cockpit_app`
Expected: Track follows agent, Heal restores stats, buttons are clickable

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add agent action buttons (track, edit, heal, respawn)"
```

---

### Task 8: Add Offspring Column to Agent Table

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Update table columns in _render_agent_table**

Update columns list. Note: If emojis don't render, use `"Fd"` and `"Ch"` (for Food/Children) instead:
```python
# Column headers (use Fd/Ch if emojis don't render)
cols = ["#", "E", "H", "Age", "Gen", "🍖", "👶"]
col_widths = [22, 32, 32, 40, 32, 32, 32]
```

**Step 2: Add offspring to agent data gathering**

```python
agents_data.append({
    'id': agent_id,
    'alive': agent.is_alive,
    'energy': agent.energy / agent.genome.max_energy if agent.is_alive else 0,
    'health': agent.health / agent.genome.max_health if agent.is_alive else 0,
    'age': agent.age,
    'generation': wrapper.generation,
    'food': wrapper.lifetime_stats.get('total_food_eaten', 0),
    'offspring': wrapper.lifetime_stats.get('offspring_count', 0),  # Add this
})
```

**Step 3: Add offspring column rendering**

After the food column rendering:
```python
# Offspring
offspring = self.font_small.render(str(agent['offspring']), True, text_color)
self.screen.blit(offspring, (col_x + 2, y + 2))
```

**Step 4: Test offspring column**

Run: `python -m primordial.interface.cockpit_app`
Expected: Table shows offspring count column

**Step 5: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: add offspring column to agent table"
```

---

## Phase 3: Modals (Tasks 9-11)

### Task 9: Implement Help Overlay Modal

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add modal state to __init__**

```python
# Modals
self.help_modal_open = False
self.genome_editor_open = False
self.database_browser_open = False
```

**Step 2: Create help modal rendering method**

```python
def _render_help_modal(self) -> None:
    """Render keyboard shortcuts help overlay."""
    if not self.help_modal_open:
        return

    # Darken background
    overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    self.screen.blit(overlay, (0, 0))

    # Modal window
    modal_w, modal_h = 600, 450
    modal_x = (self.window_width - modal_w) // 2
    modal_y = (self.window_height - modal_h) // 2
    modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)

    pygame.draw.rect(self.screen, self.BG_PANEL, modal_rect, border_radius=8)
    pygame.draw.rect(self.screen, self.CYAN_DIM, modal_rect, 2, border_radius=8)

    # Header
    header_rect = pygame.Rect(modal_x, modal_y, modal_w, 36)
    pygame.draw.rect(self.screen, self.BG_DARK, header_rect, border_top_left_radius=8, border_top_right_radius=8)
    title = self.font.render("KEYBOARD SHORTCUTS", True, self.CYAN)
    self.screen.blit(title, (modal_x + 16, modal_y + 8))

    # Close button
    close_rect = pygame.Rect(modal_x + modal_w - 36, modal_y + 6, 24, 24)
    pygame.draw.rect(self.screen, (80, 40, 40), close_rect, border_radius=4)
    close_text = self.font_small.render("✕", True, self.TEXT_BRIGHT)
    self.screen.blit(close_text, (close_rect.x + 6, close_rect.y + 4))
    self.help_close_rect = close_rect

    # Content - two columns
    y = modal_y + 50
    col1_x = modal_x + 24
    col2_x = modal_x + modal_w // 2 + 12

    shortcuts = [
        # Column 1
        ("TEACHING", [
            ("R", "Reward"),
            ("X", "Punish"),
            ("SPACE", "Push-to-Talk"),
        ]),
        ("TIME CONTROL", [
            ("[", "Slow Down"),
            ("]", "Speed Up"),
            ("\\", "Reset Speed"),
            ("P", "Pause/Play"),
        ]),
        ("SPAWNING", [
            ("F", "Add Food"),
            ("V", "Add Vegetation"),
            ("W", "Add Water"),
            ("Shift+P", "Add Predator"),
            ("D", "Delete All Veg"),
        ]),
        # Column 2
        ("AGENT CONTROL", [
            ("1-9", "Select Agent"),
            ("T", "Heal Selected"),
            ("C", "Control Mode On/Off"),
            ("Arrows", "Move (in ctrl mode)"),
        ]),
        ("SAVE/LOAD", [
            ("S", "Save Agent"),
            ("Shift+S", "Save All"),
            ("L", "List Saved"),
            ("M", "Save Map"),
            ("Shift+M", "Load Map"),
        ]),
        ("PANELS", [
            ("TAB", "Toggle Left Panel"),
            ("Shift+TAB", "Toggle Right"),
            ("Shift+H", "Toggle HUD"),
            ("H/?", "This Help"),
            ("ESC", "Close Modal/Quit"),
        ]),
    ]

    x = col1_x
    for i, (section, keys) in enumerate(shortcuts):
        if i == 3:  # Switch to column 2
            x = col2_x
            y = modal_y + 50

        # Section header
        sect_text = self.font_small.render(f"═══ {section} ═══", True, self.CYAN)
        self.screen.blit(sect_text, (x, y))
        y += 20

        for key, action in keys:
            key_text = self.font_small.render(f"{key}", True, self.TEXT_BRIGHT)
            self.screen.blit(key_text, (x, y))
            dots = "." * (12 - len(key))
            dots_text = self.font_small.render(dots, True, self.TEXT_DIM)
            self.screen.blit(dots_text, (x + key_text.get_width(), y))
            action_text = self.font_small.render(action, True, self.TEXT_NORMAL)
            self.screen.blit(action_text, (x + 100, y))
            y += 18

        y += 12

    # Got It button
    btn_rect = pygame.Rect(modal_x + modal_w // 2 - 50, modal_y + modal_h - 45, 100, 30)
    pygame.draw.rect(self.screen, self.CYAN_DIM, btn_rect, border_radius=4)
    btn_text = self.font_small.render("Got It", True, self.BG_DARKEST)
    self.screen.blit(btn_text, (btn_rect.centerx - btn_text.get_width() // 2, btn_rect.y + 7))
    self.help_gotit_rect = btn_rect
```

**Step 3: Add help modal to _render**

After `self.ui_manager.draw_ui(self.screen)`:
```python
# Render modals on top
self._render_help_modal()
```

**Step 4: Add help modal click handling**

```python
# Help modal handling
if self.help_modal_open:
    if hasattr(self, 'help_close_rect') and self.help_close_rect.collidepoint(mouse_pos):
        self.help_modal_open = False
        return
    if hasattr(self, 'help_gotit_rect') and self.help_gotit_rect.collidepoint(mouse_pos):
        self.help_modal_open = False
        return
    return  # Consume click when modal open
```

**Step 5: Add H key and ? button to open help**

In KEYDOWN handler:
```python
elif event.key == pygame.K_h:
    self.help_modal_open = not self.help_modal_open
```

Update bottom bar Help button rendering to store rect:
```python
self.help_btn_rect = help_rect
```

Add help button click:
```python
if hasattr(self, 'help_btn_rect') and self.help_btn_rect.collidepoint(mouse_pos):
    self.help_modal_open = True
    return
```

**Step 6: Test help modal**

Run: `python -m primordial.interface.cockpit_app`
Expected: H key opens help, clicking ? opens help, clicking close/Got It closes

**Step 7: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement help overlay modal with keyboard shortcuts"
```

---

### Task 10: Implement Database Browser Modal

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add database browser state**

```python
# Database browser state
self.DB_AGENTS_PER_PAGE = 8  # Constant for pagination
self.db_page = 0
self.db_selected_id = None
self.db_search_text = ""
self.db_favorites_only = False
self.db_agents_cache = []
```

**Step 2: Create database browser modal method**

```python
def _open_database_browser(self) -> None:
    """Open database browser modal."""
    self.database_browser_open = True
    self.db_page = 0
    self.db_selected_id = None
    self._refresh_db_cache()

def _refresh_db_cache(self) -> None:
    """Refresh the cached agent list from database."""
    self.db_agents_cache = self.agent_db.list_agents(
        order_by='longest_life',
        limit=100
    )

def _render_database_browser(self) -> None:
    """Render database browser modal."""
    if not self.database_browser_open:
        return

    # Darken background
    overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    self.screen.blit(overlay, (0, 0))

    # Modal window
    modal_w, modal_h = 700, 500
    modal_x = (self.window_width - modal_w) // 2
    modal_y = (self.window_height - modal_h) // 2
    modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)

    pygame.draw.rect(self.screen, self.BG_PANEL, modal_rect, border_radius=8)
    pygame.draw.rect(self.screen, self.CYAN_DIM, modal_rect, 2, border_radius=8)

    # Header
    header_rect = pygame.Rect(modal_x, modal_y, modal_w, 36)
    pygame.draw.rect(self.screen, self.BG_DARK, header_rect, border_top_left_radius=8, border_top_right_radius=8)
    title = self.font.render("AGENT DATABASE", True, self.CYAN)
    self.screen.blit(title, (modal_x + 16, modal_y + 8))

    # Close button
    close_rect = pygame.Rect(modal_x + modal_w - 36, modal_y + 6, 24, 24)
    pygame.draw.rect(self.screen, (80, 40, 40), close_rect, border_radius=4)
    close_text = self.font_small.render("✕", True, self.TEXT_BRIGHT)
    self.screen.blit(close_text, (close_rect.x + 6, close_rect.y + 4))
    self.db_close_rect = close_rect

    y = modal_y + 44

    # Agent list
    self.db_row_rects = []
    start_idx = self.db_page * self.DB_AGENTS_PER_PAGE
    page_agents = self.db_agents_cache[start_idx:start_idx + self.DB_AGENTS_PER_PAGE]

    # Table header
    pygame.draw.rect(self.screen, self.BG_DARK, pygame.Rect(modal_x + 12, y, modal_w - 24, 22))
    headers = ["⭐", "Name", "Gen", "Life", "Food", "Saved"]
    hx = modal_x + 16
    for h, w in zip(headers, [24, 180, 40, 60, 50, 80]):
        ht = self.font_small.render(h, True, self.TEXT_DIM)
        self.screen.blit(ht, (hx, y + 3))
        hx += w
    y += 26

    for agent in page_agents:
        row_rect = pygame.Rect(modal_x + 12, y, modal_w - 24, 24)
        is_selected = self.db_selected_id == agent.id
        bg = (50, 50, 60) if is_selected else (37, 37, 48)
        pygame.draw.rect(self.screen, bg, row_rect)
        if is_selected:
            pygame.draw.line(self.screen, self.CYAN, (row_rect.x, y), (row_rect.x, y + 24), 3)

        rx = modal_x + 16
        # Favorite star
        star = "★" if getattr(agent, 'is_favorite', False) else "☆"
        star_text = self.font_small.render(star, True, (255, 200, 0) if star == "★" else self.TEXT_DIM)
        self.screen.blit(star_text, (rx, y + 4))
        rx += 24

        # Name
        name_text = self.font_small.render(agent.name[:20], True, self.TEXT_NORMAL)
        self.screen.blit(name_text, (rx, y + 4))
        rx += 180

        # Gen
        gen_text = self.font_small.render(str(agent.generation), True, self.TEXT_NORMAL)
        self.screen.blit(gen_text, (rx, y + 4))
        rx += 40

        # Lifespan
        life_text = self.font_small.render(f"{agent.longest_life:.0f}s", True, self.TEXT_NORMAL)
        self.screen.blit(life_text, (rx, y + 4))
        rx += 60

        # Food
        food_text = self.font_small.render(str(agent.total_food), True, self.TEXT_NORMAL)
        self.screen.blit(food_text, (rx, y + 4))
        rx += 50

        # Saved date
        saved_text = self.font_small.render("recent", True, self.TEXT_DIM)
        self.screen.blit(saved_text, (rx, y + 4))

        self.db_row_rects.append((row_rect, agent.id))
        y += 26

    # Pagination
    y += 8
    total_pages = max(1, (len(self.db_agents_cache) + self.DB_AGENTS_PER_PAGE - 1) // self.DB_AGENTS_PER_PAGE)
    page_text = self.font_small.render(f"Page {self.db_page + 1} of {total_pages}", True, self.TEXT_DIM)
    self.screen.blit(page_text, (modal_x + modal_w // 2 - 50, y))

    prev_rect = pygame.Rect(modal_x + modal_w // 2 - 100, y - 2, 30, 22)
    next_rect = pygame.Rect(modal_x + modal_w // 2 + 60, y - 2, 30, 22)
    pygame.draw.rect(self.screen, (37, 37, 48), prev_rect, border_radius=4)
    pygame.draw.rect(self.screen, (37, 37, 48), next_rect, border_radius=4)
    prev_text = self.font_small.render("◀", True, self.TEXT_NORMAL)
    next_text = self.font_small.render("▶", True, self.TEXT_NORMAL)
    self.screen.blit(prev_text, (prev_rect.x + 8, prev_rect.y + 3))
    self.screen.blit(next_text, (next_rect.x + 8, next_rect.y + 3))
    self.db_prev_rect = prev_rect
    self.db_next_rect = next_rect

    y += 30

    # Bottom buttons
    cancel_rect = pygame.Rect(modal_x + modal_w - 220, modal_y + modal_h - 45, 100, 30)
    load_rect = pygame.Rect(modal_x + modal_w - 110, modal_y + modal_h - 45, 100, 30)

    pygame.draw.rect(self.screen, (60, 60, 70), cancel_rect, border_radius=4)
    cancel_text = self.font_small.render("Cancel", True, self.TEXT_NORMAL)
    self.screen.blit(cancel_text, (cancel_rect.centerx - cancel_text.get_width() // 2, cancel_rect.y + 7))
    self.db_cancel_rect = cancel_rect

    can_load = self.db_selected_id is not None
    load_bg = self.CYAN_DIM if can_load else (40, 40, 40)
    pygame.draw.rect(self.screen, load_bg, load_rect, border_radius=4)
    load_text = self.font_small.render("Load Agent", True, self.BG_DARKEST if can_load else self.TEXT_DIM)
    self.screen.blit(load_text, (load_rect.centerx - load_text.get_width() // 2, load_rect.y + 7))
    self.db_load_rect = load_rect
```

**Step 3: Add database browser click handling**

```python
# Database browser handling
if self.database_browser_open:
    if hasattr(self, 'db_close_rect') and self.db_close_rect.collidepoint(mouse_pos):
        self.database_browser_open = False
        return
    if hasattr(self, 'db_cancel_rect') and self.db_cancel_rect.collidepoint(mouse_pos):
        self.database_browser_open = False
        return

    # Row selection
    if hasattr(self, 'db_row_rects'):
        for rect, agent_id in self.db_row_rects:
            if rect.collidepoint(mouse_pos):
                self.db_selected_id = agent_id
                return

    # Pagination
    if hasattr(self, 'db_prev_rect') and self.db_prev_rect.collidepoint(mouse_pos):
        if self.db_page > 0:
            self.db_page -= 1
        return
    if hasattr(self, 'db_next_rect') and self.db_next_rect.collidepoint(mouse_pos):
        total_pages = max(1, (len(self.db_agents_cache) + self.DB_AGENTS_PER_PAGE - 1) // self.DB_AGENTS_PER_PAGE)
        if self.db_page < total_pages - 1:
            self.db_page += 1
        return

    # Load button
    if hasattr(self, 'db_load_rect') and self.db_load_rect.collidepoint(mouse_pos):
        if self.db_selected_id:
            self._load_agent_from_db(self.db_selected_id)
            self.database_browser_open = False
        return

    return  # Consume click

def _load_agent_from_db(self, db_id: int) -> None:
    """Load agent from database into dead slot."""
    dead_wrapper = None
    for w in self.simulation.agents.values():
        if not w.agent.is_alive:
            dead_wrapper = w
            break

    if dead_wrapper is None:
        print("No dead slots available")
        return

    if self.agent_db.load_agent_into_wrapper(db_id, dead_wrapper):
        self.simulation.world.add_entity(dead_wrapper.agent)
        self.selected_agent_id = dead_wrapper.agent_id
        print(f"Loaded agent from database")
```

**Step 4: Add to _render**

```python
self._render_database_browser()
```

**Step 5: Test database browser**

Run: `python -m primordial.interface.cockpit_app`
Expected: Load DB button opens browser, can select and load agents

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement database browser modal"
```

---

### Task 11: Implement Genome Editor Modal

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add genome editor state**

```python
# Genome editor state
self.genome_editor_agent_id: Optional[str] = None
self.genome_editor_values: Dict[str, float] = {}
self.genome_editor_tab = "physical"
```

**Step 2: Create genome editor modal method**

```python
def _open_genome_editor(self) -> None:
    """Open genome editor for selected agent."""
    wrapper = self._get_target_agent_wrapper()
    if not wrapper:
        return
    self.genome_editor_open = True
    self.genome_editor_agent_id = wrapper.agent_id
    # Copy current genome values
    self.genome_editor_values = wrapper.agent.genome.to_dict()
    self.genome_editor_tab = "physical"

def _render_genome_editor(self) -> None:
    """Render genome editor modal."""
    if not self.genome_editor_open:
        return

    # Darken background
    overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    self.screen.blit(overlay, (0, 0))

    # Modal window
    modal_w, modal_h = 550, 480
    modal_x = (self.window_width - modal_w) // 2
    modal_y = (self.window_height - modal_h) // 2
    modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)

    pygame.draw.rect(self.screen, self.BG_PANEL, modal_rect, border_radius=8)
    pygame.draw.rect(self.screen, self.CYAN_DIM, modal_rect, 2, border_radius=8)

    # Header
    agent_name = self.genome_editor_agent_id[:8] if self.genome_editor_agent_id else "Unknown"
    header_rect = pygame.Rect(modal_x, modal_y, modal_w, 36)
    pygame.draw.rect(self.screen, self.BG_DARK, header_rect, border_top_left_radius=8, border_top_right_radius=8)
    title = self.font.render(f"GENOME EDITOR: {agent_name}", True, self.CYAN)
    self.screen.blit(title, (modal_x + 16, modal_y + 8))

    # Close button
    close_rect = pygame.Rect(modal_x + modal_w - 36, modal_y + 6, 24, 24)
    pygame.draw.rect(self.screen, (80, 40, 40), close_rect, border_radius=4)
    close_text = self.font_small.render("✕", True, self.TEXT_BRIGHT)
    self.screen.blit(close_text, (close_rect.x + 6, close_rect.y + 4))
    self.ge_close_rect = close_rect

    # Tabs
    y = modal_y + 44
    tabs = ["Physical", "Sensory", "Metabolic", "Health", "Mutation"]
    tab_keys = ["physical", "sensory", "metabolic", "health", "mutation"]
    tx = modal_x + 12
    self.ge_tab_rects = {}
    for label, key in zip(tabs, tab_keys):
        tab_rect = pygame.Rect(tx, y, 90, 24)
        is_active = self.genome_editor_tab == key
        bg = (42, 42, 56) if is_active else (37, 37, 48)
        border = self.CYAN if is_active else (37, 37, 48)
        pygame.draw.rect(self.screen, bg, tab_rect, border_radius=4)
        pygame.draw.rect(self.screen, border, tab_rect, 1, border_radius=4)
        text_color = self.CYAN if is_active else self.TEXT_DIM
        text = self.font_small.render(label, True, text_color)
        self.screen.blit(text, (tx + 8, y + 5))
        self.ge_tab_rects[key] = tab_rect
        tx += 100

    y += 36

    # Tab content - sliders
    x = modal_x + 20
    width = modal_w - 40

    tab_params = {
        "physical": [
            ("max_speed", "Max Speed", 50, 300, 150.0),
            ("max_angular_speed", "Max Angular Speed", 1.0, 6.0, 3.0),
            ("thrust_force", "Thrust Force", 100, 1000, 500.0),
            ("torque_force", "Torque Force", 200, 2000, 1000.0),
            ("radius", "Radius", 4.0, 20.0, 8.0),
            ("mass", "Mass", 0.5, 3.0, 1.0),
        ],
        "sensory": [
            ("vision_range", "Vision Range", 50, 400, 200.0),
            ("vision_fov", "Vision FOV", 60, 180, 120.0),
            ("vision_rays", "Vision Rays", 8, 64, 32),
            ("audio_range", "Audio Range", 50, 500, 300.0),
            ("touch_range", "Touch Range", 5, 30, 15.0),
        ],
        "metabolic": [
            ("base_energy_cost", "Base Energy Cost", 0.01, 0.5, 0.1),
            ("movement_energy_mult", "Movement Energy", 0.1, 2.0, 0.5),
            ("vocalize_energy_mult", "Vocalize Energy", 0.5, 3.0, 1.0),
            ("eating_efficiency", "Eating Efficiency", 0.5, 1.5, 0.9),
        ],
        "health": [
            ("max_health", "Max Health", 50, 200, 100.0),
            ("max_energy", "Max Energy", 50, 200, 100.0),
            ("damage_resistance", "Damage Resistance", 0.5, 2.0, 1.0),
            ("healing_rate", "Healing Rate", 0.01, 0.5, 0.1),
        ],
        "mutation": [
            ("mutation_rate", "Mutation Rate", 0.0, 0.5, 0.1),
            ("mutation_scale", "Mutation Scale", 0.0, 0.5, 0.1),
        ],
    }

    self.ge_slider_rects = {}
    self.ge_reset_rects = {}
    for param, label, min_v, max_v, default in tab_params.get(self.genome_editor_tab, []):
        value = self.genome_editor_values.get(param, default)

        # Label
        lbl = self.font_small.render(label, True, self.TEXT_NORMAL)
        self.screen.blit(lbl, (x, y))
        def_text = self.font_small.render(f"Default: {default}", True, self.TEXT_DIM)
        self.screen.blit(def_text, (x + width - def_text.get_width(), y))
        y += 18

        # Slider track
        track_rect = pygame.Rect(x, y + 4, width - 80, 6)
        pygame.draw.rect(self.screen, (37, 37, 48), track_rect, border_radius=3)

        # Fill
        pct = (value - min_v) / (max_v - min_v) if max_v > min_v else 0
        fill_width = int(track_rect.width * max(0, min(1, pct)))
        fill_rect = pygame.Rect(x, y + 4, fill_width, 6)
        pygame.draw.rect(self.screen, self.CYAN_DIM, fill_rect, border_radius=3)

        # Thumb
        thumb_x = x + fill_width
        thumb_rect = pygame.Rect(thumb_x - 6, y, 12, 14)
        pygame.draw.rect(self.screen, self.CYAN, thumb_rect, border_radius=6)

        # Store for interaction
        slider_rect = pygame.Rect(x, y, width - 80, 14)
        self.ge_slider_rects[param] = (slider_rect, min_v, max_v)

        # Value display
        val_str = f"{value:.2f}" if isinstance(value, float) else str(int(value))
        val_text = self.font_small.render(val_str, True, self.TEXT_BRIGHT)
        self.screen.blit(val_text, (x + width - 55, y))

        # Reset button
        reset_rect = pygame.Rect(x + width - 20, y, 16, 14)
        pygame.draw.rect(self.screen, (60, 60, 70), reset_rect, border_radius=3)
        reset_text = self.font_small.render("↺", True, self.TEXT_DIM)
        self.screen.blit(reset_text, (reset_rect.x + 2, reset_rect.y))
        # Store reset button rect for click handling
        self.ge_reset_rects[param] = (reset_rect, default)

        y += 32

    # Bottom buttons
    cancel_rect = pygame.Rect(modal_x + modal_w - 220, modal_y + modal_h - 45, 100, 30)
    apply_rect = pygame.Rect(modal_x + modal_w - 110, modal_y + modal_h - 45, 100, 30)

    pygame.draw.rect(self.screen, (60, 60, 70), cancel_rect, border_radius=4)
    cancel_text = self.font_small.render("Cancel", True, self.TEXT_NORMAL)
    self.screen.blit(cancel_text, (cancel_rect.centerx - cancel_text.get_width() // 2, cancel_rect.y + 7))
    self.ge_cancel_rect = cancel_rect

    pygame.draw.rect(self.screen, self.GREEN, apply_rect, border_radius=4)
    apply_text = self.font_small.render("Apply", True, self.BG_DARKEST)
    self.screen.blit(apply_text, (apply_rect.centerx - apply_text.get_width() // 2, apply_rect.y + 7))
    self.ge_apply_rect = apply_rect
```

**Step 3: Add genome editor interaction**

```python
# Genome editor handling
if self.genome_editor_open:
    if hasattr(self, 'ge_close_rect') and self.ge_close_rect.collidepoint(mouse_pos):
        self.genome_editor_open = False
        return
    if hasattr(self, 'ge_cancel_rect') and self.ge_cancel_rect.collidepoint(mouse_pos):
        self.genome_editor_open = False
        return

    # Tab switching
    if hasattr(self, 'ge_tab_rects'):
        for key, rect in self.ge_tab_rects.items():
            if rect.collidepoint(mouse_pos):
                self.genome_editor_tab = key
                return

    # Apply button
    if hasattr(self, 'ge_apply_rect') and self.ge_apply_rect.collidepoint(mouse_pos):
        self._apply_genome_changes()
        return

    # Reset button clicks
    if hasattr(self, 'ge_reset_rects'):
        for param, (rect, default) in self.ge_reset_rects.items():
            if rect.collidepoint(mouse_pos):
                self.genome_editor_values[param] = default
                return

    # Slider interaction
    if hasattr(self, 'ge_slider_rects'):
        for param, (rect, min_v, max_v) in self.ge_slider_rects.items():
            if rect.collidepoint(mouse_pos):
                self.active_slider = f"ge_{param}"
                return

    return  # Consume click

def _apply_genome_changes(self) -> None:
    """Apply edited genome values to agent."""
    if not self.genome_editor_agent_id:
        return
    if self.genome_editor_agent_id not in self.simulation.agents:
        return

    wrapper = self.simulation.agents[self.genome_editor_agent_id]
    genome = wrapper.agent.genome

    for key, value in self.genome_editor_values.items():
        if hasattr(genome, key):
            setattr(genome, key, value)

    print(f"Applied genome changes to {self.genome_editor_agent_id[:8]}")
    self.genome_editor_open = False
```

**Step 4: Add genome editor slider motion handling**

In MOUSEMOTION handler:
```python
# Genome editor sliders
if self.active_slider and self.active_slider.startswith("ge_"):
    param = self.active_slider.replace("ge_", "")
    if hasattr(self, 'ge_slider_rects') and param in self.ge_slider_rects:
        rect, min_v, max_v = self.ge_slider_rects[param]
        rel_x = max(0, min(event.pos[0] - rect.x, rect.width))
        pct = rel_x / rect.width
        new_val = min_v + pct * (max_v - min_v)
        # Round appropriately
        if param in ["vision_rays"]:
            new_val = int(round(new_val))
        self.genome_editor_values[param] = new_val
```

**Step 5: Add to _render**

```python
self._render_genome_editor()
```

**Step 6: Test genome editor**

Run: `python -m primordial.interface.cockpit_app`
Expected: Edit button opens editor, tabs switch, sliders work, Apply saves changes

**Step 7: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement genome editor modal"
```

---

## Phase 4: Additional Keyboard Shortcuts (Tasks 12-14)

### Task 12: Implement Control Mode (C + Arrow Keys)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add control mode state**

```python
# Control mode
self.control_mode = False
```

**Step 2: Add C key handler**

```python
elif event.key == pygame.K_c:
    self.control_mode = not self.control_mode
    if self.control_mode:
        print("Control mode ON - use arrow keys to move agent")
    else:
        print("Control mode OFF")
```

**Step 3: Add arrow key handling in _update**

Add this near the start of `_update(self, dt: float)` method, where `dt` is the delta time passed by the main loop:

```python
# Control mode - arrow keys move selected agent
if self.control_mode and not self.paused:
    wrapper = self._get_target_agent_wrapper()
    if wrapper and wrapper.agent.is_alive:
        keys = pygame.key.get_pressed()
        thrust = 0.0
        torque = 0.0

        if keys[pygame.K_UP]:
            thrust = 1.0
        if keys[pygame.K_DOWN]:
            thrust = -0.5
        if keys[pygame.K_LEFT]:
            torque = -1.0
        if keys[pygame.K_RIGHT]:
            torque = 1.0

        # Apply forces directly
        if thrust != 0 or torque != 0:
            agent = wrapper.agent
            # Ensure agent has required attributes
            thrust_force = getattr(agent.genome, 'thrust_force', 500.0)
            torque_force = getattr(agent.genome, 'torque_force', 1000.0)
            mass = getattr(agent.genome, 'mass', 1.0)
            max_speed = getattr(agent.genome, 'max_speed', 200.0)

            # Thrust in facing direction
            fx = math.cos(agent.angle) * thrust * thrust_force * dt
            fy = math.sin(agent.angle) * thrust * thrust_force * dt
            agent.velocity.x += fx / mass
            agent.velocity.y += fy / mass

            # Clamp velocity to max speed
            speed = math.sqrt(agent.velocity.x**2 + agent.velocity.y**2)
            if speed > max_speed:
                scale = max_speed / speed
                agent.velocity.x *= scale
                agent.velocity.y *= scale

            # Torque
            agent.angular_velocity += torque * torque_force * dt / 100
```

**Step 4: Add control mode indicator to top bar**

In _render_topbar, after population:
```python
# Control mode indicator
if self.control_mode:
    x += 20
    pygame.draw.line(self.screen, (40, 40, 50), (x, 8), (x, self.TOPBAR_HEIGHT - 8))
    x += 10
    ctrl_text = self.font_small.render("CTRL", True, (255, 100, 100))
    self.screen.blit(ctrl_text, (x, 14))
```

**Step 5: Test control mode**

Run: `python -m primordial.interface.cockpit_app`
Expected: C toggles control, arrows move selected agent, indicator shows in top bar

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement control mode with arrow key movement"
```

---

### Task 13: Implement HUD Toggle (H Key)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add HUD visible state**

```python
# HUD visibility
self.hud_visible = True
```

**Step 2: Add H key handler (after help modal check)**

```python
elif event.key == pygame.K_h:
    if not self.help_modal_open:  # H opens help when nothing else
        self.hud_visible = not self.hud_visible
```

Wait - H already opens help. Let's use a different key or modifier:
```python
elif event.key == pygame.K_h:
    mods = pygame.key.get_mods()
    if mods & pygame.KMOD_SHIFT:
        # Shift+H toggles HUD
        self.hud_visible = not self.hud_visible
    else:
        # H toggles help
        self.help_modal_open = not self.help_modal_open
```

**Step 3: Update _render to respect HUD visibility**

Wrap HUD bar rendering:
```python
# Render HUD bars (if visible)
if self.hud_visible:
    self._render_topbar()
    self._render_bottombar()
```

**Step 4: Update layout calculations**

In `_get_world_rect` (or equivalent method that calculates the world view area), update to account for HUD visibility:
```python
def _get_world_rect(self) -> pygame.Rect:
    """Get the screen rectangle for the world view, accounting for panels and HUD."""
    top = self.TOPBAR_HEIGHT if self.hud_visible else 0
    bottom_margin = self.BOTTOMBAR_HEIGHT if self.hud_visible else 0
    left = self.PANEL_WIDTH if self.left_panel_visible else 0
    right = self.PANEL_WIDTH if self.right_panel_visible else 0

    return pygame.Rect(
        left,
        top,
        self.window_width - left - right,
        self.window_height - top - bottom_margin
    )
```

If this method doesn't exist, add it and update any code that calculates the world view area to use it.

**Step 5: Test HUD toggle**

Run: `python -m primordial.interface.cockpit_app`
Expected: Shift+H hides/shows HUD bars

**Step 6: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement HUD toggle with Shift+H"
```

---

### Task 14: Implement Map Save/Load (M and Shift+M)

**Files:**
- Modify: `primordial/interface/cockpit_app.py`

**Step 1: Add map save/load methods**

```python
def _save_map(self) -> None:
    """Save current world state to file."""
    map_data = {
        'timestamp': datetime.now().isoformat(),
        'world_width': self.simulation.world.width,
        'world_height': self.simulation.world.height,
        'food_positions': [
            {'x': f.position.x, 'y': f.position.y, 'energy': f.energy_value}
            for f in self.simulation.world.food_items if f.is_active
        ],
        'vegetation': [
            {'x': v.position.x, 'y': v.position.y, 'radius': v.radius}
            for v in self.simulation.world.vegetation
        ],
        'water': [
            {'x': w.position.x, 'y': w.position.y, 'radius': w.radius}
            for w in self.simulation.world.static_entities
            if hasattr(w, 'radius')
        ],
        'predators': [
            {
                'x': p.position.x, 'y': p.position.y,
                'patrol_x': p.patrol_center.x, 'patrol_y': p.patrol_center.y,
                'patrol_radius': p.patrol_radius
            }
            for p in self.simulation.world.predators if p.is_active
        ],
    }

    # Use user data directory (set up in Task 0)
    filename = self.maps_dir / f"map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)
        print(f"Saved map to {filename}")
    except Exception as e:
        print(f"Error saving map: {e}")

def _load_map(self) -> None:
    """Load most recent map file."""
    from primordial.world.entities import Food, Vegetation, Water, Predator
    from primordial.world.geometry import Vec2

    # Use user data directory (set up in Task 0)
    if not self.maps_dir.exists():
        print("No maps directory found")
        return

    map_files = sorted(self.maps_dir.glob("map_*.json"), reverse=True)
    if not map_files:
        print("No map files found")
        return

    try:
        with open(map_files[0]) as f:
            map_data = json.load(f)
    except Exception as e:
        print(f"Error loading map: {e}")
        return

    # Clear existing entities (except agents)
    self.simulation.world.food_items.clear()
    for v in list(self.simulation.world.vegetation):
        self.simulation.world.remove_entity(v.id)
    for p in list(self.simulation.world.predators):
        self.simulation.world.remove_entity(p.id)

    # Load food
    for fd in map_data.get('food_positions', []):
        food = Food(
            entity_id=self.simulation.world.next_entity_id,
            position=Vec2(fd['x'], fd['y']),
            energy_value=fd.get('energy', 50.0),
        )
        self.simulation.world.add_entity(food)

    # Load vegetation
    for vd in map_data.get('vegetation', []):
        veg = Vegetation(
            entity_id=self.simulation.world.next_entity_id,
            position=Vec2(vd['x'], vd['y']),
            radius=vd.get('radius', 20.0),
        )
        self.simulation.world.add_entity(veg)

    # Load predators
    for pd in map_data.get('predators', []):
        pred = Predator(
            entity_id=self.simulation.world.next_entity_id,
            position=Vec2(pd['x'], pd['y']),
            patrol_center=Vec2(pd['patrol_x'], pd['patrol_y']),
            patrol_radius=pd.get('patrol_radius', 150.0),
        )
        self.simulation.world.add_entity(pred)

    print(f"Loaded map from {map_files[0].name}")
```

**Step 2: Add M key handlers**

```python
elif event.key == pygame.K_m:
    mods = pygame.key.get_mods()
    if mods & pygame.KMOD_SHIFT:
        self._load_map()
    else:
        self._save_map()
```

**Step 3: Test map save/load**

Run: `python -m primordial.interface.cockpit_app`
Expected: M saves current map, Shift+M loads most recent map

**Step 4: Commit**

```bash
git add primordial/interface/cockpit_app.py
git commit -m "feat: implement map save/load with M and Shift+M"
```

---

## Summary

**Phase 1** (Tasks 1-5): Complete the 5 remaining control tabs
- Agents tab: 10 genome sliders
- Learn tab: LRN architecture and training sliders
- Rewards tab: 11 survival reward sliders
- Predators tab: 7 combat parameter sliders
- Presets tab: Built-in and custom preset management

**Phase 2** (Tasks 6-8): Enhance agent panel
- Sort dropdown (5 options)
- Filter dropdown (3 options)
- Action buttons (Track, Edit, Heal, Respawn, Save, Load DB)
- Offspring column in table

**Phase 3** (Tasks 9-11): Add modals
- Help overlay with keyboard shortcuts
- Database browser with pagination
- Genome editor with 5 tabs

**Phase 4** (Tasks 12-14): Additional shortcuts
- Control mode (C + arrows)
- HUD toggle (Shift+H)
- Map save/load (M, Shift+M)

---

**Plan complete and saved to `docs/plans/2025-01-29-cockpit-ui-phase2-implementation.md`.**

Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
