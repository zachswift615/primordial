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

### Speech Learning (Complete)

The agent learned to speak through self-listening:

**Perception (Phase 1)**
- 99.4% phoneme classification accuracy
- CNN encoder → Fourier mixing → phoneme recognition
- 40 phonemes mastered

**Production (Phase 2)**
- 100% accuracy producing individual phonemes
- 6D articulatory latent space (front-back, high-low, rounded, voiced, manner, vowel/consonant)
- Self-listening loop: produce → synthesize via TTS → hear → adjust

**Autoregressive Sequences (Phase 3)**
- 96% accuracy on word-level phoneme sequences
- 3-layer transformer decoder with dual heads (discrete tokens + continuous latents)
- Progressive curriculum: syllables → short words → long words
- Proper sequence termination with EOS tokens

**Multi-Word Phrases (Phase 4)** *(New!)*
- **98% accuracy** on multi-word phrases and sentences
- 106 entries: single words + 55 hand-curated phrases
- Dual-gated curriculum: token accuracy AND acoustic similarity must pass
- Self-listening validation confirms productions actually sound correct
- Phrases like "look here", "I love you", "the cat is sleeping" generated correctly

**What this proves:** Fourier mixing scales to longer sequences (up to 30 phonemes), acoustic validation catches quality issues that token accuracy misses, and the architecture handles multi-word generation without modification.

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
- **Articulatory latent space**: phonemes positioned by physical production features
- **Self-listening training**: agent hears its own outputs and adjusts

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
├── speech/           # Phoneme perception, production, sequence generation
│   ├── encoders.py   # CNN, Fourier mixing
│   ├── heads.py      # Perception, production heads
│   ├── latent.py     # 6D articulatory space
│   ├── sequence_decoder.py  # Transformer decoder
│   └── training.py   # Self-listening loop
├── lrn/              # Core architecture (Fourier mixing, encoders, heads)
├── world/            # 2D simulation (planned)
├── agent/            # Embodied agent (planned)
└── tests/            # 460 tests passing
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
