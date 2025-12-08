# Primordial Speech Architecture

## The Big Picture

Primordial is building an **embodied speech learning system** - a model that learns to speak the way humans do, by controlling articulators (tongue, lips, jaw) to produce sound.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    "I want to       ┌──────────────┐      ┌──────────────┐     │
│     say hello"  ───>│  PRIMORDIAL  │─────>│    SPARC     │───> │
│                     │   (Brain)    │      │   (Mouth)    │     │
│    Intent/Goal      │              │      │              │     │
│                     │ Decides HOW  │      │ Executes the │     │
│                     │ to move the  │      │ movements,   │     │
│                     │ articulators │      │ makes sound  │     │
│                     └──────────────┘      └──────────────┘     │
│                                                                 │
│                     Articulatory          Audio                 │
│                     Trajectory            Waveform              │
│                     (12D EMA +                                  │
│                      pitch +                                    │
│                      loudness)                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Two Systems, Two Roles

### SPARC (Berkeley's Speech Articulatory Coding)

**What it is**: An analysis-synthesis system for speech

**What it does**:
- **Encoder**: Analyzes existing audio → extracts articulatory features
- **Decoder**: Takes articulatory features → synthesizes audio

**What it CANNOT do**:
- Generate novel speech
- Decide what to say
- Produce utterances without input audio to analyze

**Analogy**: SPARC is like a **robotic mouth** or a **motion-capture playback system**. It can record how someone moved their mouth and replay those movements (with a different voice). But it can't decide on its own what movements to make.

### Primordial (This Project)

**What it is**: The motor control system that decides how to move articulators

**What it does**:
- Takes some input (target phoneme, audio to imitate, visual scene, internal thought)
- Generates articulatory trajectory (how to move tongue, lips, jaw over time)
- Learns speech production through experience (like a baby learning to talk)

**What it CANNOT do** (yet):
- Turn articulator movements into actual sound (that's SPARC's job)

**Analogy**: Primordial is the **brain** that controls the mouth. It's the puppeteer, the choreographer, the motor cortex.

---

## Why We Need Both

```
SPARC alone:
    Audio (must exist) → Encoder → Articulation → Decoder → Audio

    Problem: Need input audio. Can't generate novel speech.

Primordial alone:
    Intent → Model → Articulation → ???

    Problem: Can decide what movements to make, but can't produce sound.

Together:
    Intent → Primordial → Articulation → SPARC Decoder → Audio

    Solution: Primordial decides, SPARC executes.
```

---

## The Articulatory Control Space

SPARC defines a 14-dimensional control space based on real vocal tract measurements:

### EMA Features (12D) - Articulator Positions
| Channel | Name | What it controls |
|---------|------|------------------|
| 0 | TDX | Tongue Dorsum X (back of tongue, horizontal) |
| 1 | TDY | Tongue Dorsum Y (back of tongue, vertical) |
| 2 | TBX | Tongue Body X (middle of tongue, horizontal) |
| 3 | TBY | Tongue Body Y (middle of tongue, vertical) |
| 4 | TTX | Tongue Tip X (front of tongue, horizontal) |
| 5 | TTY | Tongue Tip Y (front of tongue, vertical) |
| 6 | LIX | Lower Incisor X (jaw, horizontal) |
| 7 | LIY | Lower Incisor Y (jaw, vertical) |
| 8 | ULX | Upper Lip X (horizontal) |
| 9 | ULY | Upper Lip Y (vertical) |
| 10 | LLX | Lower Lip X (horizontal) |
| 11 | LLY | Lower Lip Y (vertical) |

### Source Features (2D) - Voice characteristics
| Channel | Name | What it controls |
|---------|------|------------------|
| 12 | Pitch | Fundamental frequency (F0) in Hz |
| 13 | Loudness | Energy/volume |

### Speaker Identity (64D) - Who's speaking
- Separate embedding that captures voice quality/timbre
- Fixed for "the model's voice" - not learned, chosen once
- Allows same articulation to sound like different speakers

---

## How Learning Works

### Phase 1: Imitation (Supervised)
```
Hear speech → Primordial learns to predict what articulation produced it

Training:
    Audio → Mel spectrogram → Primordial → Predicted articulation
                                                    ↓
    Audio → SPARC Encoder → True articulation ← Compare (MSE loss)
```

### Phase 2: Self-Listening (End-to-End)
```
Primordial generates articulation → SPARC makes sound → Compare to target

Training:
    Target audio → Mel → Primordial → Articulation → SPARC Decoder → Generated audio
                                                                            ↓
    Target audio ←─────────────────────────────────────────────── Compare (audio loss)
```

### Phase 3: Babbling (Reinforcement Learning)
```
Primordial tries to say something → Hears result → Gets reward → Improves

Training:
    Target: "say 'ba'"
         ↓
    Primordial generates trajectory (exploration)
         ↓
    SPARC Decoder → Audio
         ↓
    Reward: "Did it sound like 'ba'?" (phoneme classifier)
         ↓
    Policy update (RL algorithm)
```

---

## The Developmental Analogy

Human speech development:

| Age | Stage | Primordial Equivalent |
|-----|-------|----------------------|
| 0-2 mo | Cooing, random vocalizations | Random articulation exploration |
| 4-6 mo | Vocal play, discovering sounds | Learning articulation → sound mapping |
| 6-10 mo | Babbling ("bababa", "dadada") | Phase 3: RL on simple syllables |
| 10-12 mo | Proto-words | Connecting sounds to meanings |
| 12+ mo | First words | Multimodal grounding (future) |

Primordial follows this progression:
1. First learns the mechanics (how articulation → sound)
2. Then explores and refines through self-listening
3. Eventually connects to meaning through multimodal learning

---

## Future: Multimodal Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIMORDIAL v3 (Future)                       │
│                                                                 │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐       │
│   │ Vision  │   │  Audio  │   │  Touch  │   │ Intent  │       │
│   │ Encoder │   │ Encoder │   │ Encoder │   │ Module  │       │
│   └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘       │
│        │             │             │             │              │
│        └─────────────┴─────────────┴─────────────┘              │
│                            │                                    │
│                   ┌────────▼────────┐                          │
│                   │  Fusion Layer   │                          │
│                   └────────┬────────┘                          │
│                            │                                    │
│                   ┌────────▼────────┐                          │
│                   │  Articulatory   │                          │
│                   │     Head        │                          │
│                   └────────┬────────┘                          │
│                            │                                    │
│                   ┌────────▼────────┐                          │
│                   │  SPARC Decoder  │                          │
│                   └────────┬────────┘                          │
│                            │                                    │
│                         Audio                                   │
└─────────────────────────────────────────────────────────────────┘

See a cat → Fusion → "I should say 'cat'" → Articulate → Sound
```

---

## Key Files

| File | Purpose |
|------|---------|
| `sparc_integration.py` | SPARC wrapper, ArticulatoryHead |
| `encoders.py` | Mel spectrogram encoder (input processing) |
| `training.py` | Training loops and losses |
| `config.py` | Model configuration |
| `your_voice_embedding.npy` | Fixed speaker identity for model's voice |

---

## Summary

| Component | Role | Analogy |
|-----------|------|---------|
| **Primordial** | Decides articulator movements | Brain / Motor cortex |
| **SPARC Decoder** | Converts movements to sound | Mouth / Vocal tract |
| **SPARC Encoder** | Analyzes speech (training only) | Ears / Perception |
| **Speaker Embedding** | Voice identity | Whose mouth it is |

**Primordial is learning to control a mouth. SPARC is the mouth.**
