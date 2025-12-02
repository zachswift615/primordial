# MineDojo Integration Design

**Date:** 2025-12-01
**Status:** Approved
**Goal:** Enable LRN to learn in Minecraft via MineDojo while maintaining backward compatibility with Primordial simulation.

## Overview

Extend the LRN architecture to support Minecraft as an alternative training environment. The Fourier mixing core remains unchanged - only the encoders become configurable based on environment type.

## Architecture Decision

**Approach:** Hybrid Configurable LRN (Option 3)

- Extend `LRNConfig` with environment presets
- Create pluggable encoders for different input modalities
- Same Fourier mixing layers work for both environments
- Full backward compatibility with existing Primordial simulation

## Observation Mapping

| Primordial | MineDojo | Notes |
|------------|----------|-------|
| Vision (32 rays × 4) | RGB frames (64×64×3) | CNN encoder outputs same (32, 128) shape |
| Audio (100 × 2) | Not used initially | Can add sound events later |
| Proprioception (7) | life_stats (10) | health, food, oxygen, armor, etc. |
| Touch (8 dirs) | damage_source | damage direction and type |

## Action Mapping

| LRN Output | Minecraft Action |
|------------|------------------|
| action[0] | forward (binary, threshold 0.5) |
| action[1] | back (binary) |
| action[2] | left (binary) |
| action[3] | right (binary) |
| action[4] | jump (binary) |
| action[5] | camera pitch (continuous, scaled) |
| action[6] | camera yaw (continuous, scaled) |
| action[7] | attack (binary) |

## Reward Function

```python
reward = 0.0

# Health changes (survival signal)
reward += (current_health - prev_health) * 2.0

# Food changes
reward += (current_food - prev_food) * 0.5

# Navigation (for NavigateDense task)
reward += (prev_distance - current_distance) * 1.0  # Closer = positive

# Movement bonus (anti-idle)
reward += 0.01 if moved else -0.01

# Death penalty
reward += -10.0 if died else 0.0
```

## File Structure

```
primordial/
├── lrn/
│   ├── lrn_config.py          # MODIFY: Add environment, mc_* fields
│   ├── encoders.py            # MODIFY: Add MinecraftVisionEncoder
│   └── architecture.py        # MODIFY: Select encoder by config
│
├── minecraft/                  # NEW
│   ├── __init__.py
│   ├── wrapper.py             # MinecraftAgentWrapper
│   ├── observation.py         # ObservationProcessor
│   ├── actions.py             # ActionConverter
│   ├── rewards.py             # MinecraftRewardComputer
│   └── config.py              # MinecraftConfig
│
└── scripts/
    └── run_minecraft.py       # Entry point for gaming PC
```

## LRNConfig Changes

```python
@dataclass
class LRNConfig:
    # Existing fields (unchanged defaults)
    hidden_dim: int = 128
    num_mixing_layers: int = 6
    vision_rays: int = 32
    vision_features: int = 4
    audio_samples: int = 100
    audio_channels: int = 2
    proprio_dim: int = 7
    touch_dim: int = 8
    action_dim: int = 5

    # NEW: Environment selection
    environment: str = "primordial"  # "primordial" | "minecraft"

    # NEW: Minecraft-specific (only used if environment="minecraft")
    mc_rgb_size: int = 64
    mc_rgb_channels: int = 3
    mc_proprio_dim: int = 10
    mc_touch_dim: int = 8  # damage_source features
    mc_action_dim: int = 8
```

## Encoder Architecture

### MinecraftVisionEncoder

```
Input: (batch, 3, 64, 64)
    │
    ▼
Conv2d(3→32, 4×4, stride=2) + ReLU    # → (batch, 32, 31, 31)
    │
    ▼
Conv2d(32→64, 4×4, stride=2) + ReLU   # → (batch, 64, 14, 14)
    │
    ▼
Conv2d(64→128, 4×4, stride=2) + ReLU  # → (batch, 128, 6, 6)
    │
    ▼
Flatten → (batch, 4608)
    │
    ▼
Linear(4608 → 32 * 128) → Reshape     # → (batch, 32, 128)

Output: (batch, 32, 128)  # Same as PrimordialVisionEncoder!
```

## Entry Point Usage

```bash
# On gaming PC after git pull:
pip install minedojo

# Run with visualization
python -m primordial.scripts.run_minecraft --render

# Run headless (faster training)
python -m primordial.scripts.run_minecraft --no-render --episodes 1000

# Resume from checkpoint
python -m primordial.scripts.run_minecraft --checkpoint ./checkpoints/mc_agent.pt
```

## Success Criteria

1. Existing Primordial tests pass unchanged
2. Agent can run in MineDojo NavigateDense environment
3. Loss decreases over first 1000 steps
4. Agent visibly learns to move toward goal over ~30 minutes
5. Render mode shows real-time Minecraft gameplay

## Dependencies

```
minedojo>=0.1.0
```

Note: MineDojo requires Java 8 and has specific GPU requirements. See MineDojo docs for setup.
