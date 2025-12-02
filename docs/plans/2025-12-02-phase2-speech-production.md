# Phase 2: Speech Production Loop Design

**Date:** 2025-12-02
**Status:** Approved
**Prerequisites:** Phase 1 perception complete (99.4% accuracy with CNN encoder)

## Overview

Teach the agent to *produce* phonemes by navigating a linguistically-structured latent space, synthesizing audio via TTS, and learning from self-listening feedback.

**Core insight:** The agent uses its own trained perception to judge its production—like a baby who can recognize "mama" before they can say it clearly.

## Architecture

### Latent Phoneme Space (6D)

Phonemes are positioned in a 6-dimensional space based on articulatory features:

| Dim | Feature | Range | Description |
|-----|---------|-------|-------------|
| 0 | Front-Back | -1 to 1 | Vowel frontness / Consonant place |
| 1 | High-Low | -1 to 1 | Vowel height |
| 2 | Rounded | -1 to 1 | Lip rounding |
| 3 | Voiced | -1 to 1 | Voicing (P↔B) |
| 4 | Manner | -1 to 1 | Stop↔Fricative↔Nasal↔Approximant |
| 5 | Vowel/Cons | -1 to 1 | Vowel (-1) ↔ Consonant (1) |

**Key property:** Nearby points = similar sounds. The path from P→B is a single dimension flip (voicing).

### Dual-Head Architecture

```
                         ┌─────────────────────┐
Mel Spectrogram ────────→│ CNN Encoder         │
(80 x n_frames)          └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ 6x Fourier Mixing   │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ Pooling → 384 dim   │
                         └──────────┬──────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Perception Head │      │ Production Head │      │ Audio Recon     │
│ (existing)      │      │ (NEW)           │      │ (existing)      │
├─────────────────┤      ├─────────────────┤      └─────────────────┘
│ phoneme: 41     │      │ latent: 6       │
│ duration: 1     │      │ duration: 1     │
│ pitch: 1        │      │ pitch: 1        │
└─────────────────┘      └─────────────────┘
```

### Self-Listening Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                     SELF-LISTENING LOOP                         │
│                                                                 │
│   ┌──────────┐    latent     ┌──────────┐    audio             │
│   │ Production│─────────────→│  Piper   │──────────┐           │
│   │ Head     │               │  TTS     │          │           │
│   └────▲─────┘               └──────────┘          │           │
│        │                                           │           │
│        │ gradient                                  ▼           │
│        │                                    ┌──────────┐       │
│   ┌────┴─────┐                              │  Mel     │       │
│   │  Loss    │◄─────────────────────────────│Spectrogram│      │
│   │ Compute  │         comparison           └────┬─────┘       │
│   └────▲─────┘                                   │             │
│        │                                         ▼             │
│        │              perceived            ┌──────────┐        │
│        └───────────────────────────────────│Perception│        │
│                     phoneme                │ Head     │        │
│                                            └──────────┘        │
│                                                                 │
│   "I tried to say X → I heard Y → adjust to make X = Y"        │
└─────────────────────────────────────────────────────────────────┘
```

## Training Curriculum

### Phase 2a: Babbling (Exploration)

Agent explores random latent positions, learns what sounds map to what regions.

```python
def babbling_step():
    random_latent = torch.randn(batch, 6).tanh()
    nearest_phoneme = find_nearest_anchor(random_latent)
    audio = tts.synthesize_phonemes([nearest_phoneme])
    mel = compute_mel_spectrogram(audio)
    perceived = model.perception_head(model.encode(mel))
    loss = MSE(random_latent, ANCHORS[perceived_phoneme])
```

### Phase 2b: Imitation (Refinement)

Agent tries to reproduce specific target phonemes.

```python
def imitation_step(target_phoneme):
    latent, dur, pitch = model.production_head(model.encode(context))
    produced_phoneme = find_nearest_anchor(latent)
    audio = tts.synthesize_phonemes([produced_phoneme])
    mel = compute_mel_spectrogram(audio)
    perceived = model.perception_head(model.encode(mel)).argmax()

    latent_loss = MSE(latent, ANCHORS[target_phoneme])
    embed_loss = MSE(mel_embedding, target_mel_embedding)
    match_reward = 1.0 if perceived == target_phoneme else 0.0

    total_loss = latent_loss + embed_loss - match_reward * 0.1
```

### Training Schedule

| Epochs | Babbling | Imitation | Focus |
|--------|----------|-----------|-------|
| 1-10 | 80% | 20% | Explore the space |
| 11-30 | 50% | 50% | Balance |
| 31+ | 20% | 80% | Refine production |

**Decay formula:** `babbling_ratio = max(0.2, 0.8 - epoch * 0.02)`

## Loss Functions

1. **Latent anchor loss:** Pull production toward target phoneme's anchor
   ```python
   latent_loss = MSE(predicted_latent, ANCHORS[target])
   ```

2. **Embedding similarity:** Continuous gradient for audio quality
   ```python
   embed_loss = MSE(produced_mel_embedding, target_mel_embedding)
   ```

3. **Match reward:** Discrete bonus for correct perception
   ```python
   match_reward = 1.0 if perceived == target else 0.0
   ```

## TTS Integration

**Current approach:** Snap to nearest anchor (Piper doesn't support continuous phonemes)

```python
def snap_to_nearest_anchor(latent: Tensor) -> str:
    distances = {p: torch.dist(latent, anchor) for p, anchor in ANCHORS.items()}
    return min(distances, key=distances.get)
```

**Future option:** Audio blending or custom continuous decoder.

## Decode & Visualization Tools

### decode_latent.py

Interpret what the agent intended to say:

```bash
python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5"

Output:
  Nearest anchor: IH (distance: 0.23)
  Runner-up: EH (distance: 0.41)
  Feature interpretation:
    Front-Back:  0.30 (slightly front)
    High-Low:   -0.70 (mid-low)
    ...
```

### Training Logs

```
Target: IY  | Intended: IY  (d=0.12) | Heard: IY  | ✓
Target: B   | Intended: P   (d=0.34) | Heard: P   | ✗  (voicing confusion)
```

### visualize_latent.py

2D projection of latent space showing:
- All 41 phoneme anchors
- Agent's recent production attempts
- Lines from attempts to targets

## File Changes

### New Files
- `primordial/speech/latent.py` - PHONEME_ANCHORS, snap_to_nearest, distance functions
- `primordial/scripts/decode_latent.py` - CLI for interpreting latents
- `primordial/scripts/visualize_latent.py` - Latent space visualization

### Modified Files
- `primordial/speech/config.py` - Add latent_dim, babbling_ratio, curriculum settings
- `primordial/speech/heads.py` - Add ProductionHead class
- `primordial/speech/training.py` - Add ProductionTrainer, babbling/imitation loops
- `primordial/speech/tts.py` - Add snap_to_nearest_anchor()
- `primordial/scripts/run_speech.py` - Add --phase production support

## Config Additions

```python
@dataclass
class SpeechConfig:
    # ... existing ...

    # Phase 2: Production
    latent_dim: int = 6
    babbling_ratio: float = 0.8
    babbling_decay: float = 0.02
    min_babbling_ratio: float = 0.2
```

## CLI Usage

```bash
# Phase 1: Perception (existing)
python -m primordial.scripts.run_speech --phase classification --encoder cnn --epochs 20

# Phase 2: Production (new)
python -m primordial.scripts.run_speech --phase production --encoder cnn --epochs 50

# Decode a latent
python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5"

# Visualize latent space
python -m primordial.scripts.visualize_latent --output latent_space.png
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Latent → Perceived match rate | >80% |
| Average anchor distance | <0.2 |
| Vowel accuracy | >85% |
| Voiced/unvoiced accuracy | >90% |

## Future Extensions

1. **Continuous TTS:** Train decoder for smooth inter-phoneme sounds
2. **Sequence production:** Babble → syllables → words
3. **Prosody learning:** Intonation, rhythm, stress patterns
4. **Cross-modal grounding:** Associate sounds with meanings
