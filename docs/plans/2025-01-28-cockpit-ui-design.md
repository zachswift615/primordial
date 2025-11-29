# Cockpit UI Design Document

**Date:** 2025-01-28
**Status:** Approved
**Mockup:** `docs/mockups/primordial-ui-mockup.html`

## Overview

A sci-fi/neon themed "cockpit" interface for the Primordial simulation that provides comprehensive control over all simulation parameters while maximizing world view visibility.

## Design Decisions

### Layout Architecture: Cockpit Layout
- **Decision:** Full-screen world view with overlay panels (not side-by-side)
- **Reasoning:** Maximizes simulation visibility, panels can collapse for immersive viewing, scales well to any resolution, achieves sci-fi aesthetic

### Visual Style: Sci-Fi/Neon with Opaque Panels
- **Decision:** Dark backgrounds (#0a0a12, #1a1a24), cyan accents (#00ffff), opaque panels (no transparency)
- **Reasoning:** User preference for readability over semi-transparent aesthetic

### Parameter Changes: Live/Real-time
- **Decision:** All slider changes apply immediately as user drags
- **Reasoning:** Enables rapid experimentation without commit/apply friction

### Control Organization: Tabbed Panels
- **Decision:** 6 tabs in left panel (World, Agents, Learn, Rewards, Predators, Presets)
- **Reasoning:** Groups related controls logically, keeps panel width manageable

### Use Cases: All-Purpose
- **Decision:** Design supports scientific research, education/demo, and rapid prototyping
- **Reasoning:** User wants flexibility for multiple scenarios

## Component Specifications

### HUD Top Bar (44px height)
- Logo/title
- FPS counter
- Simulation speed controls (◀ [1.0x] ▶)
- Day/night indicator with icon
- Generation counter
- Population display (current/max)
- Settings gear button

### Left Control Panel (280px width)
Collapsible panel with tabs:

| Tab | Contents |
|-----|----------|
| **World** | max_agents, initial_food, max_food, predator_count, world dimensions, tick_rate, vegetation_clusters, water_bodies, day/night cycle params |
| **Agents** | Default genome values for new agents (speed, vision, energy costs, etc.) |
| **Learn** | learning_rate, hidden_dim, mixing_layers, reward_modulation_scale |
| **Rewards** | All 12 reward values (EATING_FOOD, DEATH, STARVING, SOCIAL_BONUS, etc.) |
| **Predators** | patrol_radius, detection_radius, chase_speed, damage, attack_cooldown |
| **Presets** | Save/load named configurations, built-in presets |

Each control has:
- Label with default value shown
- Horizontal slider
- Numeric input for precise values
- Reset button (↺)

### Right Agent Panel (280px width)
Collapsible panel with:

**Agent Table (10 rows)**
- Columns: #, Energy, Health, Age, Generation, Food Eaten, Offspring
- Click column header to sort
- Click row to select
- Dead agents shown grayed out
- Filter dropdown: All, Alive, Dead, Male, Female

**Selected Agent Detail**
- Status bars: Energy, Health, Breeding Drive, Social Connection
- Info grid: Age, Gender, Generation, Food, Offspring, Deaths
- Action buttons: Track, Edit Genome, Heal, Respawn, Save, Load DB

### HUD Bottom Bar (50px height)
- Teaching buttons: Reward (R), Punish (X) with glow effects
- Audio visualizer (animated bars)
- System buttons: Pause, Record, Help

### Modals

**Database Browser**
- Search by name
- Favorites filter
- Sortable table with pagination
- Preview pane (genome + lifetime stats)
- Notes field
- Actions: Favorite, Rename, Delete, Load

**Genome Editor**
- 5 tabs: Physical, Sensory, Metabolic, Survival, Mutation
- Slider for each genome parameter with min/max range
- Reset individual or reset all

**Help Overlay**
- Two-column grid of keyboard shortcuts
- Grouped by function

## Color Palette

```css
--bg-darkest: #0a0a12;
--bg-dark: #12121a;
--bg-panel: #1a1a24;
--bg-input: #252530;
--bg-hover: #2a2a38;

--cyan: #00ffff;
--cyan-dim: #00aaaa;
--green: #00ff88;
--red: #ff4466;
--yellow: #ffaa00;
--purple: #aa66ff;
--pink: #ff66aa;

--text-bright: #ffffff;
--text-normal: #ccccdd;
--text-dim: #888899;
```

## Icons

Using Unicode/Emoji for simplicity:
- ⚡ Energy
- ❤ Health
- 💕 Breeding
- 👥 Social
- 🧬 Generation
- 🍖 Food
- 👶 Offspring
- 🏆 Reward
- ❌ Punish
- 🎤 Audio
- ⚙ Settings

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| R | Reward |
| X | Punish |
| SPACE | Push-to-Talk |
| [ | Slow Down |
| ] | Speed Up |
| \ | Reset Speed |
| P | Pause/Play |
| F | Add Food |
| V | Add Vegetation |
| W | Add Water |
| SHIFT+P | Add Predator |
| D | Delete All Vegetation |
| 1-9 | Select Agent |
| T | Heal Selected |
| C | Control Mode |
| Arrows | Move (in control mode) |
| S | Save Agent |
| SHIFT+S | Save All |
| L | List Saved |
| M | Save Map |
| SHIFT+M | Load Map |
| TAB | Toggle Left Panel |
| SHIFT+TAB | Toggle Right Panel |
| H | Toggle HUD |
| ESC | Close Modal / Quit |

## pygame-gui Widget Mapping

| UI Component | pygame-gui Widget |
|--------------|-------------------|
| Panels | UIPanel |
| Tabs | UIButton (styled as tabs) |
| Sliders | UIHorizontalSlider |
| Numeric inputs | UITextEntryLine |
| Dropdowns | UIDropDownMenu |
| Tables | UISelectionList or custom |
| Modals | UIWindow |
| Buttons | UIButton |
| Status bars | UIStatusBar or custom draw |
| Labels | UILabel |

## Responsive Behavior

- Panels collapse via button click or keyboard shortcut
- World view expands to fill available space
- Minimum window size: 1024x768
- Panels scroll internally if content exceeds height
