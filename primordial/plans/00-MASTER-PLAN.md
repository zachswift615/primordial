# Primordial: Living Resonance Network
## Master Implementation Plan

**Vision**: Build an alternative to the transformer paradigm — a Fourier-based neural architecture that learns through continuous sensory experience, survival pressure, and human teaching, running on consumer hardware.

**Core Thesis**: Can a Fourier-based architecture with continuous sensory input learn survival behaviors through online learning and human teaching, without batch training or labeled datasets?

---

## Executive Summary

| Component | Files | Est. Time | Dependencies |
|-----------|-------|-----------|--------------|
| 1. World & Physics | ~20 files | 6-7 days | None |
| 2. Agent Body & Senses | ~10 files | 5-6 days | World |
| 3. LRN Architecture | ~10 files | 12-15 days | Agent |
| 4. Online Learning | ~10 files | 4-5 weeks | LRN |
| 5. Human Interface | ~15 files | 10 tasks | World, Agent |

**Total**: ~65 files, ~8-10 weeks for MVP

**Target Specs**:
- ~800K parameters (laptop-runnable)
- 60 FPS simulation
- <10ms forward pass
- Online single-sample learning

---

## Project Structure

```
primordial/
├── world/                      # Component 1: World & Physics
│   ├── __init__.py
│   ├── world.py               # Main World class
│   ├── physics.py             # Physics engine
│   ├── spatial_grid.py        # Spatial partitioning
│   ├── entities/
│   │   ├── base.py            # Entity base class
│   │   ├── food.py
│   │   ├── predator.py
│   │   └── vegetation.py
│   ├── sound/
│   │   ├── sound_system.py
│   │   └── sound_source.py
│   └── environment.py         # Day/night cycle
│
├── agent/                      # Component 2: Agent Body
│   ├── __init__.py
│   ├── body.py                # AgentState, AgentBody
│   ├── sensors.py             # Vision, Audio, Proprio, Touch
│   ├── actions.py             # AgentAction
│   └── genome.py              # Heritable hyperparameters
│
├── lrn/                        # Component 3: Neural Architecture
│   ├── __init__.py
│   ├── config.py              # LRNConfig
│   ├── encoders.py            # Wavelet encoders per modality
│   ├── mixing.py              # FourierMixingLayer
│   ├── heads.py               # Prediction + Reward + Action heads
│   ├── architecture.py        # Main LRN model
│   └── genome.py              # GenomeModulator
│
├── learning/                   # Component 4: Online Learning
│   ├── __init__.py
│   ├── losses.py              # Multi-task prediction losses
│   ├── rewards.py             # Survival + teaching rewards
│   ├── reward_buffer.py       # Reward history for prediction targets
│   ├── optimizer.py           # Reward-modulated optimizer
│   ├── learning_loop.py       # Main online training loop
│   ├── stability.py           # Gradient clipping, EMA
│   ├── metrics.py             # Learning metrics
│   └── checkpointing.py       # Death handling
│
├── interface/                  # Component 5: Human Teaching
│   ├── __init__.py
│   ├── app.py                 # Main application
│   ├── renderer.py            # Multi-panel renderer
│   ├── input_handler.py       # Keyboard/mouse/controller
│   ├── teaching_signals.py    # Signal system
│   ├── demo_mode.py           # Demonstration controller
│   ├── audio_capture.py       # Microphone capture
│   └── ui_panels.py           # Individual panels
│
├── config/
│   ├── world_config.yaml
│   ├── lrn_config.yaml
│   └── learning_config.yaml
│
├── experiments/
│   ├── baseline_survival.py
│   ├── teaching_impact.py
│   └── fourier_vs_transformer.py
│
├── tests/
│   ├── world/
│   ├── agent/
│   ├── lrn/
│   ├── learning/
│   └── interface/
│
└── main.py                    # Entry point
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

**Goal**: World simulation running at 60 FPS with entities

#### Week 1: Core World
- [ ] Vec2, Circle, AABB geometry primitives
- [ ] Entity base class
- [ ] Physics engine (semi-implicit Euler)
- [ ] Collision detection (spatial grid)
- [ ] Food, Vegetation, Water entities

#### Week 2: Dynamic World
- [ ] Predator entity with AI state machine
- [ ] Sound propagation system
- [ ] Day/night cycle
- [ ] World integration tests
- [ ] Performance optimization (60 FPS target)

**Milestone**: World runs with predators, food, physics at 60 FPS

---

### Phase 2: Agent Body (Week 3)

**Goal**: Agent with continuous sensory streams

#### Tasks
- [ ] AgentState with physics
- [ ] Vision system (32-ray raycasting)
- [ ] Audio system (stereo mixing)
- [ ] Proprioception (internal state)
- [ ] Touch sensors (8 directions)
- [ ] Action application
- [ ] Energy/health mechanics
- [ ] Genome system (heritable params)

**Milestone**: Agent survives/dies in world, outputs tensor observations

---

### Phase 3: LRN Architecture (Weeks 4-6)

**Goal**: Fourier-based neural network processing sensory streams

#### Week 4: Encoders
- [ ] LRNConfig dataclass
- [ ] VisionEncoder, AudioEncoder
- [ ] ProprioEncoder, TouchEncoder
- [ ] Shape tests for all encoders

#### Week 5: Core Mixing
- [ ] FourierMixingLayer with learnable spectral filters
- [ ] FFT/iFFT operations
- [ ] Gradient flow tests
- [ ] Stack 3-6 mixing layers

#### Week 6: Output & Integration
- [ ] PredictionHead (next sensory state)
- [ ] ActionHead (motor commands)
- [ ] GenomeModulator (evolution support)
- [ ] Complete forward pass
- [ ] Parameter count validation (~800K)

**Milestone**: LRN processes sensory input, outputs actions + predictions

---

### Phase 4: Online Learning (Weeks 7-9)

**Goal**: Agent learns from experience in real-time

#### Week 7: Loss & Rewards
- [ ] Prediction loss (MSE)
- [ ] Survival rewards computation
- [ ] Human teaching signal integration
- [ ] Reward combiner

#### Week 8: Training Loop
- [ ] RewardModulatedOptimizer
- [ ] Gradient clipping
- [ ] EMA for stability
- [ ] Online learning loop

#### Week 9: Robustness
- [ ] Death handling (checkpoint, reset)
- [ ] Learning rate scheduling
- [ ] Metrics and logging
- [ ] TensorBoard integration

**Milestone**: Agent learns to survive longer over time

---

### Phase 5: Human Interface (Weeks 8-10, parallel)

**Goal**: Interactive teaching through reward/punishment/demonstration

#### Tasks (can parallelize with Phase 4)
- [ ] Pygame window with multi-panel layout
- [ ] World view (top-down, zoomable)
- [ ] Agent POV (first-person)
- [ ] Status panels (energy, health, metrics)
- [ ] Input handler (keyboard, mouse, controller)
- [ ] Teaching signals (reward, punish, point, demonstrate)
- [ ] Microphone capture for voice
- [ ] Demonstration mode (human controls agent)
- [ ] Save/load agent state

**Milestone**: Human can interact with and teach agent in real-time

---

## Key Technical Decisions

### 1. Fourier Mixing (replaces Attention)

```python
# Core operation
x_fft = torch.fft.rfft(x, dim=1)          # O(n log n)
x_filtered = x_fft * learnable_filter      # Learned "resonance"
x_out = torch.fft.irfft(x_filtered, ...)   # Back to time domain
```

**Why**: O(n log n) vs O(n²), hardware-optimized, natural for continuous signals

### 2. Multi-Task Prediction (Sensory + Reward)

```python
# Two prediction heads, two losses
sensory_loss = MSE(predicted_next_senses, actual_next_senses)
reward_loss = MSE(predicted_rewards, actual_rewards)  # NEW!
total_loss = sensory_loss + reward_loss
```

**Why**:
- Sensory prediction teaches world dynamics ("what will I sense?")
- Reward prediction teaches survival value ("will this hurt or help?")
- Creates DIRECT gradient toward survival (solves conceptual gap)
- Biologically plausible (dopamine = reward prediction error)

### 3. Reward Modulation (augments learning rate)

```python
effective_lr = base_lr * (1.0 + reward_scale * reward)
```

**Why**: Amplifies learning during important moments (survival events, human teaching)

### 4. Continuous Sensory Streams (no tokenization)

**Why**: Humans don't tokenize reality; continuous input enables richer representations

### 5. Genome System (for future evolution)

```python
class AgentGenome:
    vision_range: float = 200.0
    hidden_dim: int = 128
    learning_rate: float = 0.001
    # ... heritable hyperparameters
```

**Why**: Enables Phase 2 breeding/evolution without rewrite

---

## Success Criteria

### Phase 1 Complete When:
1. [ ] Agent survives >5x longer with learning ON vs OFF
2. [ ] Human teaching accelerates learning >2x
3. [ ] LRN performs within 80% of transformer at 3x speed
4. [ ] No catastrophic forgetting over 1 hour
5. [ ] Runs at 60 FPS on laptop GPU

### Stretch Goals (Phase 2):
- Agent responds to human "words" (sound associations)
- Multiple agents with breeding
- Evolved architectures outperform hand-designed
- Agent exhibits curiosity/exploration

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Fourier mixing doesn't learn | Fall back to FNet (proven), add attention layers |
| Online learning unstable | Gradient accumulation, lower LR, EMA |
| Performance too slow | Profile early, optimize FFT, reduce model size |
| Audio capture issues | Make optional, mock for testing |
| Catastrophic forgetting | EWC, experience replay, accept and measure |

---

## Dependencies

```
# requirements.txt
numpy>=1.24.0
torch>=2.0.0
pygame>=2.5.0
sounddevice>=0.4.0
pyyaml>=6.0
tensorboard>=2.13.0
pytest>=7.4.0
```

---

## How to Start

1. **Read the detailed plans** in `plans/01-05-*.md`
2. **Start with Component 1** (World) — no dependencies
3. **Use TDD**: Write test, see fail, implement, see pass
4. **Commit after each task** passes tests
5. **Run integration tests** after each component

```bash
# Run all tests
pytest tests/ -v

# Run specific component
pytest tests/world/ -v

# Run with coverage
pytest tests/ --cov=primordial
```

---

## Detailed Plans

| Plan | Lines | Focus |
|------|-------|-------|
| [01-world-system.md](01-world-system.md) | ~1800 | Physics, entities, sound |
| [02-agent-body.md](02-agent-body.md) | ~1100 | Sensors, actions, genome |
| [03-lrn-architecture.md](03-lrn-architecture.md) | ~1600 | Fourier mixing, encoders |
| [04-learning-system.md](04-learning-system.md) | ~1500 | Online learning, rewards |
| [05-human-interface.md](05-human-interface.md) | ~2600 | Pygame UI, teaching |

---

## Research References

- **FNet** (Google, 2021): https://arxiv.org/abs/2105.03824
- **FFTNet** (2025): https://arxiv.org/html/2502.18394v4
- **AI Habitat**: https://aihabitat.org/
- **Developmental Robotics**: https://mitpress.mit.edu/9780262028011/developmental-robotics/
- **Continual Learning**: https://arxiv.org/html/2403.05175v1

---

**Let's build the future of AI — one that learns like a living thing.**
