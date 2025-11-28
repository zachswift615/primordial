# Primordial: Living Resonance Networks

**An alternative to the transformer paradigm.**

## The Vision

What if we could build AI that learns the way living things do — not through massive batch training on static datasets, but through continuous sensory experience, survival pressure, and real-time human teaching?

Primordial explores a radically different approach:

- **Fourier-based architecture** instead of attention (O(n log n) vs O(n²))
- **Continuous sensory streams** instead of tokenization
- **Online learning** instead of batch training
- **Survival pressure** instead of task rewards
- **Human teaching** through direct interaction, not labeled data

The goal: An AI that runs on consumer hardware and learns like a living thing.

## The Core Insight

### Transformers ask the wrong question

Transformers ask: "Which tokens should attend to which?" — an O(n²) question that requires massive compute.

**What if we asked instead:** "How do patterns naturally resonate and interfere?" — leveraging physics we can compute cheaply via FFT.

### Living things learn differently

Current AI learns *about* the world from human descriptions.
An embodied agent learns *in* a world through direct experience.

A transformer trained on text knows "fire is hot" because those words co-occur.
An embodied agent knows fire is hot because it got burned.

Is that difference just poetic, or does it unlock something?

## The Architecture: Living Resonance Networks (LRN)

```
CONTINUOUS SENSORY STREAMS
         │
         ▼
┌─────────────────────────────┐
│   WAVELET DECOMPOSITION     │  Each sense → time-frequency representation
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   FOURIER MIXING LAYERS     │  O(n log n) replacement for attention
│   (learnable spectral       │  Related concepts amplify (resonance)
│    filters)                 │  Unrelated concepts cancel (interference)
└──────────────┬──────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│Sensory │ │Reward  │ │Action  │
│Predict │ │Predict │ │Head    │
│Head    │ │Head    │ │        │
└────────┘ └────────┘ └────────┘
    │          │          │
    ▼          ▼          ▼
"What will  "Will this  "What should
 I sense?"  help/hurt?"  I do?"
```

### Multi-Task Prediction (The Key Innovation)

The agent learns by predicting *two* things:

1. **Sensory prediction**: "What will I sense next?" → Learns world dynamics
2. **Reward prediction**: "Will this help or hurt me?" → Learns survival value

This creates a **direct gradient toward survival**. No reinforcement learning machinery needed — just prediction error, like how dopamine neurons actually work.

## The World: "Primordial"

A 2D continuous survival simulation where the agent must:

- Find food to maintain energy
- Avoid predators that cause damage
- Navigate obstacles and terrain
- Respond to a human teacher in real-time

```
┌────────────────────────────────────────────┐
│                                            │
│    🌿  🌿        ~~~~~          🌿         │
│         🌿    ~~~ 💧 ~~~    🔴             │
│                 ~~~~~     (predator)       │
│    🍎                                🌿    │
│              ◉ ←── AGENT                   │
│    🌿    🍎                                │
│                                  🔴        │
│         🌿  🌿           🍎        🌿      │
│                                            │
└────────────────────────────────────────────┘
```

### Agent Senses (All Continuous)

| Sense | Description |
|-------|-------------|
| **Vision** | 32 rays, 120° FOV, returns distance + RGB |
| **Audio** | Stereo waveform, all sounds mixed by distance |
| **Proprioception** | Energy, health, velocity, hunger, pain |
| **Touch** | 8 directional contact sensors |

### Human Teaching Interface

No datasets. No labels. Just be there:

- **Reward/Punish**: Real-time feedback buttons
- **Point**: Direct agent's attention to objects
- **Demonstrate**: Take control, show how it's done
- **Speak**: Associate sounds with objects (proto-language)

Like raising a child, not training a model.

## Why This Might Work on Consumer Hardware

| Property | Transformers | LRN |
|----------|--------------|-----|
| Core operation | MatMul (dense) | FFT (sparse, hardware-optimized) |
| Memory scaling | O(n²) | O(n log n) |
| Parallelism | Good | Excellent (embarrassingly parallel) |
| Hardware | Needs tensor cores | Runs well on CPU, DSP, FPGA |
| Quantization | Tricky | Phase is naturally robust |

FFTs have 50+ years of optimization. Every phone has dedicated FFT silicon.

## Research Foundations

This project builds on real, proven research:

- **[FNet](https://arxiv.org/abs/2105.03824)** (Google, 2021): Replaced attention with FFT, achieved 92-97% of BERT's accuracy at 7x speed
- **[FFTNet](https://arxiv.org/html/2502.18394v4)** (2025): Learnable spectral filters, competitive accuracy with better efficiency
- **[AI Habitat](https://aihabitat.org/)**: Photo-realistic 3D embodied AI simulation
- **[Developmental Robotics](https://mitpress.mit.edu/9780262028011/developmental-robotics/)**: Building robots that learn like babies
- **[Continual Learning](https://arxiv.org/html/2403.05175v1)**: Addressing catastrophic forgetting

The gap: **Nobody has combined** Fourier mixing + continuous input + online learning + embodied survival.

## What Success Looks Like

Phase 1 proves the thesis if:

1. **Learning works**: Trained agents survive >5x longer than untrained
2. **Teaching helps**: Human-taught agents learn >2x faster
3. **Architecture is viable**: LRN performs within 80% of transformer at 3x speed
4. **Continuous input works**: No tokenization needed
5. **Online learning is stable**: No catastrophic forgetting over 1 hour

If these hold → Breeding, evolution, scaling.

## The Bigger Vision (Future Phases)

**Phase 2: Evolution**
- Multiple agents in same world
- Breeding: survivors reproduce
- Genome mutation and crossover
- Social behaviors emerge?

**Phase 3: Scale**
- Larger models (10M+ params)
- Multiple human teachers
- Language emergence experiments
- Persistent worlds

**Phase 4: ???**
- This is where it gets philosophically interesting
- At what scale do we need to worry about suffering?
- When does "pain signal" become pain?

## Project Structure

```
primordial/
├── world/          # Physics, entities, sound propagation
├── agent/          # Body, sensors, actions, genome
├── lrn/            # Fourier mixing, encoders, prediction heads
├── learning/       # Online training, rewards, stability
├── interface/      # Pygame UI, human teaching
├── experiments/    # Validation experiments
└── plans/          # Detailed implementation plans (~9000 lines)
```

## Getting Started

```bash
# Clone and setup
git clone <repo>
cd primordial
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run simulation (once implemented)
python main.py
```

## The Honest Uncertainty

This is speculative research. We don't know:

- If Fourier mixing scales to frontier capability
- If online learning can match batch training quality
- If survival pressure creates meaningful representations
- If this paradigm leads anywhere transformers can't go

But the components are proven individually. The combination is novel. The compute is cheap enough to actually try.

**Worth exploring? We think so.**

---

## Status

**Current**: Implementation planning complete. ~9000 lines of detailed specs across 6 plan documents.

**Next**: Build Phase 1 (World + Agent + LRN + Learning + Interface)

**Timeline**: ~8-10 weeks to MVP

---

*"The question is not whether machines can think, but whether they can learn to think by living."*
