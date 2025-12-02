# Single Player Training Mode - Next Session

## Quick Context

We're building a single-player training mode where agents develop stats through behavior ("use it, build it"). Stats grow from what agents actually do - running builds speed, escaping predators builds reaction time, etc.

**Core philosophy:** "Stats are autobiography, not biography" - an agent's stats tell the story of what it survived.

## What's Already Done

### 1. Design Doc
`docs/gamification/single-player-training-design.md`
- Full spec for behavior → stat mapping
- Permanent (20%) + Active (80%) gain split with decay
- 8 training levels defined
- Clone/checkpoint systems

### 2. Training System (Working Spike)
`primordial/agents/training.py`
- `AgentTrainer` class detects behaviors and applies stat gains
- Integrated into simulation loop (`primordial/simulation/simulation.py:172`)
- Console logs: 🏃 speed, ⚡ reaction, 👁️ vision

**Key changes to genome:**
- `primordial/agents/genome.py` - Added `permanent_gains`, `active_gains` dicts
- `get_effective_stat()` returns base + permanent + active
- `apply_training_gain()` with diminishing returns
- `decay_active_gains()` for slow decay

### 3. UI Mockup
`docs/mockups/primordial-training-mockup.html`
- Agent Stable, Training Levels, Agent Profile, Checkpoints, Training System
- Open in browser to see the design

## Next Tasks (Priority Order)

### 1. Save/Load Persistence
Training gains need to persist when saving/loading agents.

**Files to check:**
- `primordial/agents/genome.py` - `to_dict()` and `from_dict()` already use `fields()` so the new dicts should serialize
- `primordial/agents/body.py` - `save()` and `load()` methods
- Verify with: save agent, reload, check `genome.get_training_summary()`

### 2. In-Game Training Stats UI
Add visual stat bars to the existing UI showing base/permanent/active breakdown.

**Mockup reference:** See "Agent Profile" tab in mockup, specifically the "Training Stats" card

### 3. Boot Camp Level
First training level - open space, lots of food, no predators. Goal: learn basic foraging.

**See:** `docs/gamification/single-player-training-design.md` → Training Environments section

## Prompt for Next Session

```
I'm continuing work on the single-player training mode for Primordial.

Read docs/gamification/NEXT-SESSION.md for context on what's been built and what's next.

The training system spike is working (primordial/agents/training.py). Next priority is ensuring training gains persist through save/load. Can you verify this works and fix if needed?
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `docs/gamification/single-player-training-design.md` | Full design spec |
| `docs/mockups/primordial-training-mockup.html` | UI mockup |
| `primordial/agents/training.py` | Behavior detection & stat growth |
| `primordial/agents/genome.py` | Genome with training gains |
| `primordial/simulation/simulation.py` | Trainer integration (line ~172) |

## Testing the Training System

```python
# Quick test that training works
python -c "
from primordial.agents.genome import AgentGenome

genome = AgentGenome()
print('Before:', genome.get_effective_stat('max_speed'))

genome.apply_training_gain('max_speed', 5.0)
print('After training:', genome.get_effective_stat('max_speed'))
print('Permanent:', genome.permanent_gains['max_speed'])
print('Active:', genome.active_gains['max_speed'])
"
```

---
*Created: Dec 2024*
