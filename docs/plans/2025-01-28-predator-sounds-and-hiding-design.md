# Predator Sounds and Hiding Mechanics Design

**Date:** 2025-01-28
**Status:** Approved for implementation

## Overview

Three improvements to the pygame simulation to make predator-agent interactions more realistic:

1. **Predator footstep sounds** - Pulsing sounds while patrolling that contrast with chase growl
2. **Line-of-sight hiding** - Agents can hide behind vegetation to avoid detection
3. **Vegetation rendering** - Make vegetation visible in the UI

## Feature 1: Predator Footstep Sound System

### Current Behavior
- Predators emit continuous 100Hz growl only when in CHASING state
- Silent while PATROLLING or RETURNING

### New Behavior

| State | Sound Frequency | Intensity | Pattern |
|-------|-----------------|-----------|---------|
| PATROLLING | 100Hz | 0.3 | Pulsing at rate proportional to speed |
| CHASING | 100Hz | 0.5 | Continuous (existing behavior) |
| RETURNING | 100Hz | 0.3 | Pulsing at rate proportional to speed |

### Pulse Rate Formula
```
pulse_rate = speed / 15  (Hz)
```
- Patrol speed 30 → ~2 taps/second
- Chase speed 80 → continuous (no pulsing)

### Intensity Modulation
```python
# For patrolling/returning states:
pulse_phase += pulse_rate * dt * 2 * pi
intensity = base_intensity * (0.5 + 0.5 * sin(pulse_phase))

# For chasing state:
intensity = chase_intensity  # constant, no modulation
```

### Implementation Changes

**File:** `primordial/world/entities/predator.py`

1. Add instance variables:
   - `pulse_phase: float = 0.0`

2. Modify `update()` method:
   - Calculate pulse_rate from current speed
   - Update pulse_phase each tick
   - Emit sound in all states (not just CHASING)
   - Modulate intensity for non-chase states

3. Modify `get_sound_sources()`:
   - Return pulsing sound for PATROLLING/RETURNING
   - Return continuous sound for CHASING

## Feature 2: Line-of-Sight Hiding Mechanics

### Current Behavior
- Predators detect agents within `detection_radius` (200 units) unconditionally
- Vegetation blocks agent vision rays but predators ignore it

### New Behavior
- Predators must have clear line-of-sight to detect agents
- Vegetation blocks predator detection rays
- 1-second grace period when chasing target goes behind cover (prevents flicker)

### Detection Logic
```python
def can_detect_agent(predator, agent, world):
    distance = (agent.position - predator.position).magnitude()
    if distance > predator.detection_radius:
        return False

    # NEW: Check line of sight
    if not world.has_line_of_sight(predator.position, agent.position):
        return False

    return True
```

### Line-of-Sight Check
```python
def has_line_of_sight(self, from_pos: Vec2, to_pos: Vec2) -> bool:
    """Check if there's clear line of sight between two positions."""
    for veg in self.vegetation:
        if self._ray_intersects_circle(from_pos, to_pos, veg.position, veg.radius):
            return False
    return True
```

### Chase State Handling
- When chasing, if LOS is lost:
  - Start a 1-second timer (`los_lost_time`)
  - Continue moving toward last known position
  - If LOS regained within 1 second, continue chase
  - If LOS still blocked after 1 second, transition to RETURNING

### Implementation Changes

**File:** `primordial/world/world.py`

1. Add method:
   - `has_line_of_sight(from_pos, to_pos) -> bool`
   - `_ray_intersects_circle(ray_start, ray_end, circle_center, circle_radius) -> bool`

**File:** `primordial/world/entities/predator.py`

1. Add instance variable:
   - `los_lost_time: float = 0.0`
   - `last_known_target_pos: Vec2 | None = None`

2. Modify `_check_for_agents()`:
   - Add LOS check before transitioning to CHASING

3. Modify `_update_chasing()`:
   - Track LOS status
   - Implement 1-second grace period
   - Transition to RETURNING if target hidden too long

## Feature 3: Vegetation Rendering

### Current Behavior
- Vegetation exists in world but not passed to renderer
- Users cannot see vegetation in the pygame UI

### New Behavior
- Vegetation rendered as dark green irregular polygons
- Distinct from bright green food circles
- Legend updated to explain vegetation

### Visual Design

| Entity | Color RGB | Shape |
|--------|-----------|-------|
| Food | (100, 255, 100) | Small filled circle |
| Vegetation | (30, 80, 30) | Irregular 7-point polygon |
| Agent | (100, 150, 255) | Circle with direction line |
| Predator | (255, 100, 100) | Triangle pointing in direction |

### Polygon Generation
```python
num_points = 7
points = []
for i in range(num_points):
    angle = (i / num_points) * 2 * math.pi
    # Deterministic variation for consistent shape
    variation = 0.7 + 0.6 * ((i * 3) % 5) / 5
    r = radius * variation
    px = screen_x + int(math.cos(angle) * r)
    py = screen_y + int(math.sin(angle) * r)
    points.append((px, py))
pygame.draw.polygon(surface, (30, 80, 30), points)
```

### Implementation Changes

**File:** `primordial/interface/integrated_app.py`

1. Modify `_get_world_state()`:
   - Add vegetation to entities list

**File:** `primordial/interface/ui_panels.py`

1. Modify `WorldViewPanel.render()`:
   - Add `elif entity_type == "vegetation":` rendering case

2. Modify draw order to render vegetation behind other entities

**File:** `primordial/interface/config.py`

1. Add color constant:
   - `VEGETATION = (30, 80, 30)`

## Files to Modify

| File | Changes |
|------|---------|
| `primordial/world/entities/predator.py` | Footstep sounds, LOS detection, chase grace period |
| `primordial/world/world.py` | Add `has_line_of_sight()` method |
| `primordial/interface/integrated_app.py` | Add vegetation to entity list |
| `primordial/interface/ui_panels.py` | Render vegetation as irregular polygons |
| `primordial/interface/config.py` | Add VEGETATION color constant |

## Testing

1. **Sound test:** Run simulation, observe predator sound changes with speed
2. **Hiding test:** Position agent behind vegetation, verify predator doesn't detect
3. **Visual test:** Confirm vegetation appears as dark green irregular shapes
4. **Chase break test:** While being chased, move behind vegetation, verify 1-sec delay before predator gives up

## Success Criteria

- [ ] Predators make audible footstep-like sounds while patrolling
- [ ] Sound pulse rate increases with predator speed
- [ ] Sound becomes continuous growl when chasing
- [ ] Agents behind vegetation are not detected by predators
- [ ] Chasing predators lose target after 1 second behind cover
- [ ] Vegetation is visible as dark green irregular shapes
- [ ] Food and vegetation are visually distinct
