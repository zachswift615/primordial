# Single Player Training Mode - Design Document

> "Stats are autobiography, not biography" - An agent's stats tell the story of what it survived, not what you assigned it.

## Core Philosophy

The player is a **trainer**, not a creator. You don't build agents by tweaking sliders - you develop them through experience. Every stat reflects something the agent actually did. Every specialist earned their title through survival.

---

## Table of Contents

1. [Core Game Loop](#core-game-loop)
2. [Trait Progression System](#trait-progression-system)
3. [Stat Decay Model](#stat-decay-model)
4. [Behavior Detection](#behavior-detection)
5. [Agent Identity & Titles](#agent-identity--titles)
6. [Persistence & Checkpoints](#persistence--checkpoints)
7. [Cloning System](#cloning-system)
8. [Environmental Objects](#environmental-objects)
9. [Training Environments](#training-environments)
10. [Starting Agents](#starting-agents)
11. [UI Requirements](#ui-requirements)

---

## Core Game Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  1. START                                                       │
│     └─► Begin with randomly-initialized agents, name them       │
│                                                                 │
│  2. TRAIN                                                       │
│     └─► Run agents through environments that apply pressure     │
│                                                                 │
│  3. WATCH                                                       │
│     └─► Stats grow organically from behavior                    │
│     └─► Agents become who they are through experience           │
│                                                                 │
│  4. CLONE                                                       │
│     └─► Branch development at interesting moments               │
│     └─► Create parallel experiments                             │
│                                                                 │
│  5. CHECKPOINT                                                  │
│     └─► Protect against bad learning                            │
│     └─► Bookmark promising states                               │
│                                                                 │
│  6. SPECIALIZE                                                  │
│     └─► Different clones through different training regimens    │
│     └─► Build a stable of specialists                           │
│                                                                 │
│  7. CHALLENGE                                                   │
│     └─► Tackle training levels with the right agent             │
│     └─► Match specialist to challenge type                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Trait Progression System

Stats grow from behavior. The more an agent does something, the better it gets at it.

### Behavior → Trait Mapping

| Behavior | Trait Gained | Detection Method |
|----------|--------------|------------------|
| Sustained running | **Max Speed** | Velocity > 70% of current max for N consecutive ticks |
| Narrow predator escapes | **Reaction Time** | Was within predator danger zone, successfully escaped |
| Spotting distant food | **Vision Range** | Acquired food target from distance > 80% of current vision |
| Reacting to sounds | **Hearing Range** | Changed direction toward sound source before visual contact |
| Low-energy survival | **Energy Efficiency** | Time spent moving while below 30% energy |
| Quick direction changes | **Agility** | High rotation rate during active evasion |
| Absorbing damage | **Max Health** | Damage taken → survived → recovered |
| Long survival streaks | **Stamina** | Continuous time alive without entering critical state |
| Object manipulation | **Tactics** | Successfully using objects for cover, barriers, or escape |

### Stat Growth Formula

```python
# When qualifying behavior is detected:
stat_gain = base_gain * difficulty_multiplier * diminishing_returns_factor

# Example values:
base_gain = 0.1  # Very small per-event
difficulty_multiplier = 1.0 - 3.0  # Harder situations = more growth
diminishing_returns_factor = 1.0 / (1.0 + current_stat_level * 0.1)  # Slower growth at high levels
```

---

## Stat Decay Model

Stats have two components: **permanent gains** and **active gains**.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   TOTAL STAT = BASE + PERMANENT GAINS + ACTIVE GAINS            │
│                                                                 │
│   ┌─────────┬────────────────────┬─────────────────────┐        │
│   │  BASE   │  PERMANENT (20%)   │  ACTIVE (80%)       │        │
│   │         │  Never decays      │  Decays slowly      │        │
│   │   10    │      +2            │     +8              │        │
│   └─────────┴────────────────────┴─────────────────────┘        │
│                                                                 │
│   Example: Agent trained Speed from 10 → 20                     │
│   - Base: 10 (starting value)                                   │
│   - Permanent: +2 (20% of gains, locked in forever)             │
│   - Active: +8 (80% of gains, will decay if not maintained)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Decay Rules

1. **Permanent gains (20%)**: Never decay. This is the "muscle memory" that sticks.
2. **Active gains (80%)**: Decay very slowly when behavior is not exercised.
3. **Decay rate**: ~1% of active gains per minute of non-use (tunable)
4. **Decay floor**: Active gains cannot decay below 0 (permanent gains protected)

### Example Decay Scenario

```
Agent "Scout" has Vision Range: 10 (base) + 4 (permanent) + 16 (active) = 30

Scout stops doing long-range spotting for 10 minutes:
- Active gains decay: 16 * 0.99^10 ≈ 14.5
- New total: 10 + 4 + 14.5 = 28.5

Scout stops for 100 minutes:
- Active gains decay: 16 * 0.99^100 ≈ 5.9
- New total: 10 + 4 + 5.9 = 19.9

Scout never uses vision again (1000 minutes):
- Active gains decay: 16 * 0.99^1000 ≈ 0.0007 → floors at 0
- New total: 10 + 4 + 0 = 14 (permanent gains preserved!)
```

### Why This Model

- **Training matters**: You can build stats through dedicated practice
- **Specialization is maintained**: Can't be max-level at everything simultaneously
- **Progress isn't lost**: Permanent gains mean training always leaves a mark
- **Feels fair**: Very slow decay doesn't punish you for trying other things

---

## Behavior Detection

### Max Speed Training
```python
def detect_speed_training(agent, ticks_at_high_speed):
    """Detect sustained high-speed running"""
    threshold = agent.max_speed * 0.70
    if agent.current_velocity >= threshold:
        ticks_at_high_speed += 1
        if ticks_at_high_speed >= SPEED_TRAINING_THRESHOLD:  # e.g., 60 ticks
            grant_stat_increase(agent, "max_speed")
            ticks_at_high_speed = 0
    else:
        ticks_at_high_speed = 0  # Reset if they slow down
    return ticks_at_high_speed
```

### Reaction Time Training
```python
def detect_escape_training(agent, predator):
    """Detect narrow escapes from predators"""
    danger_zone = predator.attack_range * 1.5
    was_in_danger = distance(agent, predator) < danger_zone

    # On successful escape (was in danger, now safe, not damaged)
    if was_in_danger and not agent.in_danger and not agent.took_damage:
        escape_margin = calculate_escape_margin(agent, predator)
        if escape_margin < NARROW_ESCAPE_THRESHOLD:  # Close call
            grant_stat_increase(agent, "reaction_time", multiplier=2.0)  # Bonus for close calls
        else:
            grant_stat_increase(agent, "reaction_time", multiplier=1.0)
```

### Vision Range Training
```python
def detect_vision_training(agent, food):
    """Detect long-distance food spotting"""
    spot_distance = distance(agent.position, food.position)
    spot_ratio = spot_distance / agent.vision_range

    if spot_ratio > 0.80:  # Spotted at edge of vision
        grant_stat_increase(agent, "vision_range")
```

### Hearing Range Training
```python
def detect_hearing_training(agent, sound_source):
    """Detect reaction to sound before visual confirmation"""
    heard_distance = distance(agent.position, sound_source.position)
    can_see = heard_distance <= agent.vision_range

    if not can_see and agent.changed_direction_toward(sound_source):
        grant_stat_increase(agent, "hearing_range")
```

### Energy Efficiency Training
```python
def detect_efficiency_training(agent, ticks_at_low_energy):
    """Detect survival while low on energy"""
    if agent.energy < agent.max_energy * 0.30 and agent.is_moving:
        ticks_at_low_energy += 1
        if ticks_at_low_energy >= EFFICIENCY_TRAINING_THRESHOLD:
            grant_stat_increase(agent, "energy_efficiency")
            ticks_at_low_energy = 0
    else:
        ticks_at_low_energy = 0
    return ticks_at_low_energy
```

### Agility Training
```python
def detect_agility_training(agent):
    """Detect quick direction changes during evasion"""
    if agent.is_evading and agent.rotation_rate > AGILITY_THRESHOLD:
        grant_stat_increase(agent, "agility")
```

### Tactics Training
```python
def detect_tactics_training(agent, objects, predators):
    """Detect successful use of objects for tactical advantage"""

    # Check if agent put object between self and predator
    for obj in agent.recently_moved_objects:
        for predator in predators:
            if object_blocks_line_of_sight(obj, agent, predator):
                grant_stat_increase(agent, "tactics", multiplier=2.0)
                return

    # Check if agent used object as cover during danger
    if agent.is_behind_cover and agent.was_in_danger:
        grant_stat_increase(agent, "tactics", multiplier=1.0)
```

---

## Agent Identity & Titles

Agents receive dynamic titles based on their stat distribution and behaviors. Titles are calculated:
- Periodically during free play (every N minutes)
- At the end of training levels
- When all agents in a session die

### Title Definitions

| Title | Primary Condition | Secondary Conditions |
|-------|-------------------|---------------------|
| **Scout** | Vision OR Hearing in top 20% | - |
| **Sprinter** | Speed in top 20% | Efficiency below average |
| **Marathon Runner** | Speed + Efficiency both above average | High stamina |
| **Ghost** | High evasion rate (>80%) | Low damage taken |
| **Tank** | Max Health in top 20% | High damage survived |
| **Survivor** | Efficiency in top 20% | Long average lifespan |
| **Wanderer** | Distance traveled in top 20% | - |
| **Homebody** | Territory size in bottom 20% | High survival rate |
| **Glutton** | Food consumption rate in top 20% | - |
| **Minimalist** | Food consumption rate in bottom 20% | Above average survival |
| **Tactician** | Tactics in top 20% | Object interactions > threshold |
| **Elder** | Longest cumulative survival time | - |
| **Prodigy** | Total stats in top 20% | Age below average |
| **Acrobat** | Agility in top 20% | High direction change rate |
| **Ninja** | Reaction Time in top 20% | Many narrow escapes |
| **Generalist** | All stats within 20% of each other | No stat in top/bottom 20% |

### Title Display

Agents can have:
- **Primary Title**: Strongest distinguishing characteristic
- **Secondary Title**: Second strongest (optional, shown as "Sprinter-Scout")

```
┌─────────────────────────────────────┐
│  ZEPHYR                             │
│  ══════════════════════════════     │
│  Title: Ghost-Scout                 │
│  "The one that sees without         │
│   being seen"                       │
└─────────────────────────────────────┘
```

### Title Calculation Algorithm

```python
def calculate_titles(agent, all_agents):
    """Calculate agent's titles based on stats relative to population"""
    percentiles = {}

    for stat in TRACKED_STATS:
        all_values = [a.get_stat(stat) for a in all_agents]
        agent_value = agent.get_stat(stat)
        percentiles[stat] = calculate_percentile(agent_value, all_values)

    # Check each title's conditions
    qualifying_titles = []
    for title, conditions in TITLE_CONDITIONS.items():
        if meets_conditions(agent, percentiles, conditions):
            qualifying_titles.append((title, calculate_title_strength(agent, conditions)))

    # Sort by strength, return top 1-2
    qualifying_titles.sort(key=lambda x: x[1], reverse=True)

    primary = qualifying_titles[0][0] if qualifying_titles else "Novice"
    secondary = qualifying_titles[1][0] if len(qualifying_titles) > 1 else None

    return primary, secondary
```

---

## Persistence & Checkpoints

### No Permadeath

- Agents **do not die permanently**
- On death, agent respawns from last state
- Death is a **learning experience**: the moments before death are training data
- Neural net learns "that was bad, don't do that"

### User Checkpoints

Each agent has **5 checkpoint slots** that the user controls.

```
┌─────────────────────────────────────────────────────────────────┐
│  CHECKPOINTS FOR: Zephyr                                        │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  [1] "Before Gauntlet" - Saved 2 hours ago                      │
│      Stats: Speed 15, Vision 22, Tactics 8                      │
│      Title: Scout                                               │
│                                                                 │
│  [2] "Post-Maze Master" - Saved 45 min ago                      │
│      Stats: Speed 15, Vision 24, Tactics 14                     │
│      Title: Scout-Tactician                                     │
│                                                                 │
│  [3] Empty                                                      │
│  [4] Empty                                                      │
│  [5] Empty                                                      │
│                                                                 │
│  [ Save Current State ]  [ Load Selected ]  [ Delete ]          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Checkpoint Data Stored

- Full genome (stats with permanent/active breakdown)
- Neural network weights
- Current title(s)
- Cumulative statistics (time alive, food eaten, escapes, etc.)
- Timestamp and user-provided label

### Auto-Snapshots (Optional/Future)

System could auto-save at milestones:
- Completed a training level
- Survived 5 minutes continuous
- Achieved a new title
- Set a personal record

These would be separate from user checkpoints (non-destructive).

---

## Cloning System

Cloning creates an exact copy of an agent at its current state.

### What Gets Cloned

- Full genome (base + permanent + active gains)
- Neural network weights (learned behaviors)
- Current title(s)
- Cumulative statistics

### What Doesn't Get Cloned

- Checkpoints (clone starts with empty checkpoint slots)
- Name (user must provide new name)

### Clone Metadata

```python
class CloneInfo:
    source_agent: str      # Name of original agent
    source_id: UUID        # ID of original agent
    clone_timestamp: datetime
    clone_generation: int  # How many clones deep (original=0, clone=1, clone-of-clone=2)
```

### Clone Naming Suggestions

When cloning, suggest names like:
- "{OriginalName}-Alpha", "{OriginalName}-Beta"
- "{OriginalName} II", "{OriginalName} III"
- Or completely custom

### Lineage Tracking

```
Original: "Zephyr" (Gen 0)
    ├── Clone: "Zephyr-Alpha" (Gen 1) - Gauntlet specialist
    │       └── Clone: "Zephyr-Alpha-2" (Gen 2) - Extreme evasion
    └── Clone: "Zephyr-Beta" (Gen 1) - Maze specialist
            └── Clone: "Zephyr-Beta-Scout" (Gen 2) - Vision focused
```

---

## Environmental Objects

Agents can interact with objects in the world to gain tactical advantages.

### Object Types

| Object | Weight | Interactions | Properties |
|--------|--------|--------------|------------|
| **Boulder** | Heavy | Push (slow), Pull (slow) | Blocks vision, Blocks movement |
| **Rock** | Medium | Push, Pull, Pick up, Place | Blocks vision (partial), Blocks movement |
| **Log** | Medium | Push, Pull, Pick up, Place, Rotate | Blocks movement, Can bridge gaps |
| **Branch** | Light | Push, Pull, Pick up, Place, Rotate | Partial cover |
| **Bush** | Static | Hide inside | Concealment (not full cover) |

### Object Interactions

```python
class ObjectInteraction:
    PUSH = "push"       # Move object away from agent
    PULL = "pull"       # Move object toward agent
    PICKUP = "pickup"   # Agent carries object (slows movement)
    PLACE = "place"     # Put carried object down
    ROTATE = "rotate"   # Change object orientation
```

### Interaction Costs

| Action | Energy Cost | Speed Penalty | Time |
|--------|-------------|---------------|------|
| Push (light) | Low | 20% | Instant |
| Push (medium) | Medium | 40% | Short |
| Push (heavy) | High | 60% | Long |
| Pull | Same as push | Same | Same |
| Pick up | Medium | - | Short |
| Carry | Ongoing drain | 50% | Continuous |
| Place | Low | - | Instant |
| Rotate | Low | 10% | Short |

### Tactical Uses

1. **Barrier Creation**: Push objects to block predator paths
2. **Cover**: Hide behind objects to break line of sight
3. **Escape Routes**: Pre-position objects for quick escapes
4. **Forts**: Build defensive positions near food sources
5. **Traps**: (Advanced) Funnel predators into dead ends

### Detecting Tactical Success

```python
def evaluate_tactical_success(agent, action, objects, predators):
    """Determine if an object interaction was tactically successful"""

    success_conditions = [
        # Created cover that blocked predator vision
        object_now_blocks_los_to_predator(action.object, agent, predators),

        # Created barrier that blocked predator path
        object_now_blocks_predator_path(action.object, predators),

        # Used object as cover while in danger
        agent.in_danger and agent.behind_cover_of(action.object),

        # Escaped using object as obstacle
        agent.just_escaped and predator_blocked_by(action.object, predators),
    ]

    return any(success_conditions)
```

---

## Training Environments

Pre-designed environments that apply specific evolutionary pressures.

### Level 1: Boot Camp
```
┌─────────────────────────────────────────┐
│  BOOT CAMP                              │
│  ═══════════════════════════════════    │
│  Pressure: Basic foraging               │
│  Predators: None                        │
│  Features: Open space, abundant food    │
│  Goal: Learn to find and eat food       │
│  Trains: Vision Range, basic movement   │
└─────────────────────────────────────────┘
```

### Level 2: The Gauntlet
```
┌─────────────────────────────────────────┐
│  THE GAUNTLET                           │
│  ═══════════════════════════════════    │
│  Pressure: Evasion                      │
│  Predators: 1 slow predator             │
│  Features: Narrow corridor              │
│  Goal: Learn predator avoidance         │
│  Trains: Reaction Time, Speed           │
└─────────────────────────────────────────┘
```

### Level 3: The Maze
```
┌─────────────────────────────────────────┐
│  THE MAZE                               │
│  ═══════════════════════════════════    │
│  Pressure: Navigation                   │
│  Predators: None                        │
│  Features: Complex walls, food at end   │
│  Goal: Learn pathfinding                │
│  Trains: Vision Range, Memory?          │
└─────────────────────────────────────────┘
```

### Level 4: The Arena
```
┌─────────────────────────────────────────┐
│  THE ARENA                              │
│  ═══════════════════════════════════    │
│  Pressure: Multi-threat evasion         │
│  Predators: 2-3 varied predators        │
│  Features: Open with scattered cover    │
│  Goal: Handle multiple threats          │
│  Trains: Agility, Reaction Time         │
└─────────────────────────────────────────┘
```

### Level 5: The Drought
```
┌─────────────────────────────────────────┐
│  THE DROUGHT                            │
│  ═══════════════════════════════════    │
│  Pressure: Resource scarcity            │
│  Predators: 1 patrolling                │
│  Features: Sparse food, one water       │
│  Goal: Energy management                │
│  Trains: Energy Efficiency, Stamina     │
└─────────────────────────────────────────┘
```

### Level 6: The Fortress
```
┌─────────────────────────────────────────┐
│  THE FORTRESS                           │
│  ═══════════════════════════════════    │
│  Pressure: Object manipulation          │
│  Predators: 2 aggressive                │
│  Features: Many movable objects         │
│  Goal: Use environment defensively      │
│  Trains: Tactics                        │
└─────────────────────────────────────────┘
```

### Level 7: The Pack Hunt
```
┌─────────────────────────────────────────┐
│  THE PACK HUNT                          │
│  ═══════════════════════════════════    │
│  Pressure: Coordinated predators        │
│  Predators: 3+ that work together       │
│  Features: Mixed terrain                │
│  Goal: Survive coordinated attacks      │
│  Trains: All evasion stats              │
└─────────────────────────────────────────┘
```

### Level 8: The Ultimate Challenge
```
┌─────────────────────────────────────────┐
│  THE ULTIMATE CHALLENGE                 │
│  ═══════════════════════════════════    │
│  Pressure: Everything                   │
│  Predators: Many, varied                │
│  Features: All hazards combined         │
│  Goal: Prove mastery                    │
│  Trains: Tests all stats                │
└─────────────────────────────────────────┘
```

---

## Starting Agents

### Initial Agent Generation

When starting a new game, player receives 3-5 starter agents.

```python
def generate_starter_agents(count=4):
    agents = []
    for i in range(count):
        agent = Agent(
            name=None,  # User must name
            genome=generate_random_genome(),
            neural_net=initialize_random_weights(),
        )
        agents.append(agent)
    return agents
```

### Random Genome Generation

```python
def generate_random_genome():
    """Generate starting genome with slight variations"""
    base_stats = {
        "max_speed": 10,
        "vision_range": 10,
        "hearing_range": 10,
        "reaction_time": 10,
        "energy_efficiency": 10,
        "agility": 10,
        "max_health": 10,
        "stamina": 10,
        "tactics": 5,  # Starts lower, must be earned
    }

    # Add random variation (-2 to +2 for each stat)
    genome = {}
    for stat, base in base_stats.items():
        variation = random.uniform(-2, 2)
        genome[stat] = max(1, base + variation)  # Minimum of 1

    return genome
```

### Naming Ceremony

Before any training begins, player must name their starter agents:

```
┌─────────────────────────────────────────────────────────────────┐
│  NAME YOUR AGENTS                                               │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  These agents are ready to begin their journey.                 │
│  Give them names that will echo through history.                │
│                                                                 │
│  Agent 1: [________________]  Stats preview: Fast, Low vision   │
│  Agent 2: [________________]  Stats preview: Balanced           │
│  Agent 3: [________________]  Stats preview: High efficiency    │
│  Agent 4: [________________]  Stats preview: High vision        │
│                                                                 │
│                    [ Begin Training ]                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## UI Requirements

### Agent Panel (Real-time)

```
┌─────────────────────────────────────────┐
│ ZEPHYR                    [Ghost-Scout] │
├─────────────────────────────────────────┤
│ Speed:      ████████████░░░░  15.2      │
│ Vision:     ██████████████████ 22.1     │
│ Hearing:    ██████████░░░░░░  12.4      │
│ Reaction:   ████████████████░ 18.7      │
│ Efficiency: ████████░░░░░░░░  10.2      │
│ Agility:    ██████████████░░  16.5      │
│ Health:     ██████████░░░░░░  11.0      │
│ Stamina:    ████████████░░░░  14.3      │
│ Tactics:    ██████░░░░░░░░░░   7.8      │
├─────────────────────────────────────────┤
│ Energy: ████████████████░░░░ 78%        │
│ Time Alive: 4m 32s                      │
│ Food Eaten: 23                          │
│ Escapes: 12                             │
├─────────────────────────────────────────┤
│ [Checkpoint] [Clone] [Respawn]          │
└─────────────────────────────────────────┘
```

### Stable View (All Agents)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ YOUR STABLE                                                   [4 Agents]│
├──────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│ Zephyr       │ Bolt         │ Tank         │ Whisper      │             │
│ Ghost-Scout  │ Sprinter     │ Survivor     │ Tactician    │   [ + ]     │
│              │              │              │              │   Clone     │
│ ⭐⭐⭐       │ ⭐⭐         │ ⭐⭐⭐⭐     │ ⭐⭐⭐       │             │
│ Gen 0        │ Gen 1        │ Gen 0        │ Gen 2        │             │
│ Time: 2h 15m │ Time: 45m    │ Time: 3h 02m │ Time: 1h 30m │             │
├──────────────┴──────────────┴──────────────┴──────────────┴─────────────┤
│ [ Deploy Selected ]  [ View Details ]  [ Compare ]  [ Delete ]          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training Level Selection

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TRAINING LEVELS                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ BOOT    │  │ THE     │  │ THE     │  │ THE     │                     │
│  │ CAMP    │  │ GAUNTLET│  │ MAZE    │  │ ARENA   │                     │
│  │ ⭐⭐⭐  │  │ ⭐⭐    │  │ ⭐      │  │ 🔒      │                     │
│  │ Cleared │  │ Cleared │  │ Working │  │ Locked  │                     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                     │
│                                                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │ THE     │  │ THE     │  │ THE PACK│  │ ULTIMATE│                     │
│  │ DROUGHT │  │ FORTRESS│  │ HUNT    │  │CHALLENGE│                     │
│  │ 🔒      │  │ 🔒      │  │ 🔒      │  │ 🔒      │                     │
│  │ Locked  │  │ Locked  │  │ Locked  │  │ Locked  │                     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘                     │
│                                                                         │
│ Selected: The Maze                                                      │
│ Recommended Agent: Zephyr (Scout)                                       │
│ Best Time: 2m 34s by Tank                                               │
│                                                                         │
│ [ Start Training ]  [ Agent: Zephyr ▼ ]                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Priority

### Phase 1: Core Systems
1. Stat growth from behavior (detection + formulas)
2. Decay system (permanent + active gains)
3. Basic title calculation

### Phase 2: Persistence
4. Checkpoint system (save/load agent states)
5. Cloning system
6. Lineage tracking

### Phase 3: Environment
7. Environmental objects (push/pull/pickup/place)
8. Tactics stat detection
9. First 3 training levels

### Phase 4: Polish
10. Full title system with all titles
11. Remaining training levels
12. UI improvements

---

## Open Questions

1. **Memory stat?** - Should agents have a memory stat that helps with navigation/pattern recognition?
2. **Team dynamics?** - How do multiple agents interact in same environment? Do they help or compete?
3. **Sound system** - How sophisticated should hearing/sound detection be?
4. **Predator diversity** - What different predator behaviors should exist?
5. **Object physics** - How realistic should object interactions be?

---

## Appendix: Stat Caps

| Stat | Base | Soft Cap | Hard Cap |
|------|------|----------|----------|
| Max Speed | 10 | 30 | 50 |
| Vision Range | 10 | 30 | 50 |
| Hearing Range | 10 | 30 | 50 |
| Reaction Time | 10 | 30 | 50 |
| Energy Efficiency | 10 | 30 | 50 |
| Agility | 10 | 30 | 50 |
| Max Health | 10 | 30 | 50 |
| Stamina | 10 | 30 | 50 |
| Tactics | 5 | 25 | 40 |

Soft cap: Growth slows significantly
Hard cap: Cannot exceed this value

---

*Document created from brainstorming session - November 2024*
*Core philosophy: Stats are autobiography, not biography*
