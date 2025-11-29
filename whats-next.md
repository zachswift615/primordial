# Session Handoff: Cockpit UI Design & Implementation Plan

**Created:** 2025-11-28
**Purpose:** Enable continuation in a fresh context with complete precision

---

<original_task>
Design and plan a comprehensive pygame-gui based "cockpit" interface for the Primordial simulation. User requirements:
- Beautiful sci-fi/neon themed UI achievable with pygame-gui
- Control over nearly every aspect of agents and world (stat depletion rates, reward contributions, predator reproduction, food/water stat boosts, social HP reduction, etc.)
- Live updating agent table with 10 rows, scrollable, with ability to select and load agents from database
- HTML/CSS mockup first, then implementation plan for converting to pygame-gui
</original_task>

<work_completed>
## Design Exploration (Brainstorming Skill)
- Gathered requirements through structured questions
- User preferences established:
  - **Use case**: All-purpose (research, education, rapid prototyping)
  - **Apply mode**: Live/real-time changes
  - **Organization**: Tabbed panels
  - **Style**: Sci-fi/Neon with **opaque panels** (user explicitly rejected semi-transparency)
  - **Resolution**: Flexible/resizable

## Architecture Decision
- Evaluated 3 layout approaches:
  - A: Command Center (fixed panels)
  - B: Mission Control (three-column)
  - C: Cockpit (full-screen world with overlay panels) - **SELECTED**
- Cockpit layout chosen for maximum world view visibility and collapsible panels

## Design Document Created
**File**: `docs/plans/2025-01-28-cockpit-ui-design.md`
- Complete color palette (CSS variables)
- Component specifications for all 7 UI elements
- pygame-gui widget mapping
- Keyboard shortcuts reference
- All design decisions documented

## HTML/CSS Mockup Created
**File**: `docs/mockups/primordial-ui-mockup.html`
- Fully interactive mockup with:
  - HUD top bar (FPS, speed, day/night, gen, population)
  - Left control panel with 6 tabs (World, Agents, Learn, Rewards, Predators, Presets)
  - Right agent panel with 10-row table and selected agent detail
  - HUD bottom bar (teaching buttons, audio visualizer, system controls)
  - Database browser modal
  - Genome editor modal
  - Help/keyboard shortcuts modal
- All styling matches sci-fi/neon theme
- Tab switching, row selection, modal open/close all functional

## Implementation Plan Created & Refined
**File**: `docs/plans/2025-01-28-cockpit-ui-implementation.md`

Plan reviewed twice by superpowers:code-reviewer agent. Issues identified and fixed:
- MVP scope clearly defined (Phase 1-5 vs Phase 6-7 future)
- pygame-gui usage clarified (modals only, raw pygame for main UI)
- `_rebuild_layout()` fully implemented (Task 3.4)
- Resize handler added (pygame.VIDEORESIZE)
- `_get_world_transform()` helper extracts duplicate code (DRY fix)
- `_screen_to_world()` uses the helper
- Task 5.2 broken into 5 subtasks (5.2-5.6) with complete code
- Race condition fixed with early return on table click
- Config validation with hasattr() check before setattr()
- Audio error handling with graceful fallback
- Theme.json error handling with defaults fallback
</work_completed>

<current_state>
## Deliverables Status
| Deliverable | Status | Location |
|-------------|--------|----------|
| Design document | Complete | `docs/plans/2025-01-28-cockpit-ui-design.md` |
| HTML/CSS mockup | Complete | `docs/mockups/primordial-ui-mockup.html` |
| Implementation plan | Complete | `docs/plans/2025-01-28-cockpit-ui-implementation.md` |
| CockpitApp implementation | Not started | `primordial/interface/cockpit_app.py` |
| theme.json | Not started | `primordial/interface/theme.json` |
| pygame-gui in requirements | Not started | `requirements.txt` |

## Plan Structure
17 tasks across 5 MVP phases + 2 future phases:

**Phase 1: Setup and Core Layout** (Tasks 1.1-1.3)
**Phase 2: HUD Bars** (Tasks 2.1-2.2)
**Phase 3: Side Panels** (Tasks 3.1-3.4)
**Phase 4: Agent Selection** (Task 4.1)
**Phase 5: Migration and Integration** (Tasks 5.1-5.6)
**Phase 6-7: Future** (Modals, remaining tabs)
</current_state>

<next_steps>
## Execute the Implementation Plan

### Quick Start
```
Read these files:
1. docs/plans/2025-01-28-cockpit-ui-implementation.md (the plan)
2. docs/plans/2025-01-28-cockpit-ui-design.md (design decisions)
3. docs/mockups/primordial-ui-mockup.html (open in browser for visual reference)
```

### Execution Options
1. **Use executing-plans skill**:
   ```
   Use superpowers:executing-plans to implement docs/plans/2025-01-28-cockpit-ui-implementation.md
   ```

2. **Use subagent-driven-development skill**:
   ```
   Use superpowers:subagent-driven-development
   ```

### First Task
Task 1.1: Install pygame-gui and create theme.json
- Add `pygame-gui>=0.6.0` to requirements.txt
- Create `primordial/interface/theme.json` with sci-fi color palette
</next_steps>

<gotchas>
## Critical Technical Notes

1. **Opaque panels** - User explicitly rejected semi-transparency for readability

2. **pygame-gui is for modals only** - Main UI uses raw pygame drawing for performance

3. **`_get_world_transform()` helper** - Returns `(world_rect, scale, offset_x, offset_y)` tuple, used by both rendering and input

4. **Click handler race condition** - Table clicks must be checked FIRST with early return before world clicks

5. **Shift+P precedence** - Must check Shift+P (spawn predator) BEFORE regular P (pause)

6. **Audio capture fallback** - Wrapped in try/catch with `audio_enabled` flag for graceful degradation

7. **Config validation** - Use hasattr() before setattr() when applying slider values

## Files to Reference
- Existing interface: `primordial/interface/integrated_app.py`
- Existing renderer: `primordial/interface/renderer.py`
- Existing config: `primordial/interface/config.py`
- Agent database: `primordial/simulation/agent_database.py`
</gotchas>
