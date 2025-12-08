# Primordial: Living Resonance Networks

**Exploring what happens when AI learns through sensory experience instead of text.**

## What We're Building

An AI that learns like a living thing:
- Continuous sensory streams (vision, audio, proprioception, touch)
- Fourier-based neural architecture (O(n log n) instead of O(n²))
- Online learning from experience, not batch training
- Self-supervised through prediction: "What will I sense next?" + "Will this help or hurt?"

The goal: See how far we can push embodied learning on consumer hardware.

## Current Progress

### Speech Learning (Active Development)

#### Previous Work (Piper-based)
- 99.4% phoneme perception accuracy
- 98% accuracy on multi-word phrases with synthetic TTS
- Self-listening validation with acoustic similarity scoring

#### Current Direction: SPARC Integration (December 2025)

We're migrating from Piper TTS to **Berkeley's SPARC** (Speech Articulatory Coding) for interpretable, differentiable speech production.

**Why SPARC?**
- **Interpretable control**: 12D articulatory features (tongue, lips, jaw positions) vs opaque phoneme tokens
- **Differentiable**: Can train end-to-end with audio reconstruction loss
- **Embodied**: Mirrors actual vocal tract kinematics from EMA data

**The Architecture:**
```
┌─────────────────────────────────────────────────────┐
│  PRIMORDIAL (Brain)         SPARC (Mouth)           │
│                                                     │
│  Intent/Target ──> Articulatory ──> SPARC ──> Audio │
│                    Head (14D)       Decoder         │
│                                                     │
│  Learns HOW to     EMA (12D)        Executes the    │
│  move articulators + Pitch (1D)     movements,      │
│                    + Loudness (1D)  makes sound     │
└─────────────────────────────────────────────────────┘
```

**Primordial = Brain** (decides articulator movements)
**SPARC = Mouth** (converts movements to sound)

See `primordial/speech/ARCHITECTURE.md` for full details.

**Training Pipeline:**
1. **Supervised**: Predict SPARC articulatory features from mel spectrograms
2. **Self-listening**: Generate audio, compare to target (differentiable)
3. **RL (future)**: Babbling curriculum, learn from reward

**What this enables:**
- Model learns to control a virtual vocal tract
- Natural prosody emerges from learning pitch/loudness patterns
- Foundation for true embodied speech learning (like a baby babbling)

### Architecture: Living Resonance Networks

```
SENSORY INPUT
     ↓
CNN/Wavelet Encoders
     ↓
Fourier Mixing Layers (6 layers, 128-dim hidden)
  - Learnable spectral filters
  - O(n log n) complexity
  - Natural frequency-domain reasoning
     ↓
  ┌──────┬──────┬──────┐
  ↓      ↓      ↓      ↓
Sensory Reward Action Sequence
Predict Predict Head  Decoder
```

**Key innovations:**
- **Spectral filtering** instead of attention: patterns resonate/interfere in frequency space
- **Multi-task prediction**: world dynamics + survival value learned simultaneously
- **SPARC integration**: interpretable articulatory control (tongue, lips, jaw)
- **Self-listening training**: agent hears its own outputs and adjusts (differentiable with SPARC)

## Parameter Efficiency

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| Speech Encoder (CNN + 6 Fourier layers) | ~800K | Audio → phoneme features |
| Sequence Decoder (3 transformer layers) | ~610K | Autoregressive generation |
| Total | ~1.4M | Full speech production system |

Runs on CPU. Forward pass <10ms. Online learning with batch_size=1.

## The Fourier Advantage

**Why Fourier transforms instead of attention?**

1. **Speed**: FFT is O(n log n), heavily hardware-optimized (50+ years)
2. **Natural for signals**: Audio, vision, sensory streams are continuous
3. **Frequency reasoning**: Patterns like rhythm, periodicity emerge naturally
4. **Parameter efficient**: Spectral filters vs. full Q/K/V projections
5. **Interpretable**: Can visualize what frequencies the network attends to

FFTNet (2025) and FNet (Google, 2021) proved pure FFT can replace attention. We're pushing it further with continuous sensory input and online learning.

## What's Next

### Immediate Goals
- Add visual grounding: show object → say word → learn association
- Cross-modal fusion: correlate sounds with visual patterns
- Sentence-level prosody (intonation, rhythm)

### Medium-term Vision
- Full sensorimotor loop in 2D simulation
- Human teaching interface (reward/punish, pointing, demonstration)
- Survival-driven learning (find food, avoid predators)
- Proto-language emergence from embodied experience

### Long-term Questions
- How far can Fourier mixing scale? (10M params? 100M?)
- Can survival pressure create representations as rich as language pretraining?
- Does embodied learning unlock capabilities text-only models can't reach?
- At what scale does "pain signal" become something we need to care about?

## Research Foundations

Built on proven components:
- **FNet** (Google, 2021): FFT replaces attention at 92-97% accuracy
- **FFTNet** (2025): Learnable spectral filters for adaptive frequency response
- **Developmental Robotics**: Learning through sensorimotor experience
- **Self-supervised learning**: Prediction as the learning signal

The novel combination: Fourier + continuous input + online learning + embodied self-supervision.

## File Structure

```
primordial/
├── speech/               # Speech perception and production
│   ├── ARCHITECTURE.md   # Brain vs Mouth architecture docs
│   ├── encoders.py       # CNN encoder, mel spectrogram (with CMVN)
│   ├── sparc_integration.py  # SPARC wrapper, ArticulatoryHead (planned)
│   ├── sequence_decoder.py   # Transformer decoder
│   ├── training.py       # Training loops
│   └── config.py         # Speech configuration
├── docs/
│   ├── sparc_integration_plan.md  # Full training pipeline plan
│   └── primordial-ebook.md        # Project history and vision
├── lrn/                  # Core architecture (Fourier mixing)
├── world/                # 2D survival simulation
├── agent/                # Embodied agent
└── your_voice_embedding.npy  # Model's voice identity (64D SPARC embedding)
```

## Running the Code

```bash
# Install dependencies
pip install -r requirements.txt

# Train phoneme perception (Phase 1)
python -m primordial.scripts.run_speech --phase classification --epochs 20

# Train phoneme production (Phase 2)
python -m primordial.scripts.train_production_interactive --epochs 50

# Train sequence generation (Phase 3)
python -m primordial.scripts.train_sequence \
  --encoder-checkpoint checkpoints/production/curriculum_best.pt \
  --epochs 200

# Run tests
pytest tests/ -v
```

## Results So Far

**Speech perception:** 99.4% accuracy (40 phonemes)
**Speech production:** 100% accuracy (individual phonemes)
**Sequence generation:** 98% accuracy (words + phrases)
**Multi-word phrases:** "look here" → ['L', 'UH', 'K', 'HH', 'IH', 'R'] ✓
**Self-listening validation:** 0.95 acoustic similarity on correct productions
**Progressive curriculum:** Dual-gated (token accuracy + acoustic match)
**Online learning:** Stable with single-sample updates

## The Experiment

This is fundamentally an exploration: **How far can we push Fourier-based architectures with embodied, sensory learning?**

We don't know the limits yet. The speech results—98% accuracy on multi-word phrases, acoustic validation confirming productions sound correct, proper sequence termination—suggest there's something here worth pursuing.

The components are individually proven. The combination is novel. The compute requirements are manageable (runs on laptop CPU).

Let's see where this goes.

---

*"The question is not whether machines can think, but whether they can learn to think by living."*
