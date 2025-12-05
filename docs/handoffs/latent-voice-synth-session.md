# Latent Voice Synthesizer - Session Handoff

**Date:** 2025-12-04
**Status:** Prototype complete, needs validation

## Goal

Build a real-time voice synthesizer controlled by 8D articulatory latent vectors, enabling an AI model to "speak" by outputting continuous latent positions rather than discrete phoneme tokens.

## What We Built This Session

### 1. Formant Synthesizer (`primordial/speech/formant_synth.py`)
- Basic source-filter model with biquad resonators
- Works for vowels, but consonants sound synthetic
- **Verdict:** Vowels distinguishable but overall quality poor

### 2. Vocal Synthesizer (`primordial/speech/vocal_synth.py`)
- Improved biquad formant filters
- Uses recorded consonant samples from `samples:consonants/` (24 recordings)
- Crossfading between consonants and vowels
- **Verdict:** Better than formant synth, but still "stitched together" sounding

### 3. Pink Trombone Port (`primordial/speech/pink_trombone.py`)
- Python port of Neil Thapen's Kelly-Lochbaum waveguide synthesizer
- 700 lines, includes Glottis (LF model) and Tract (44-segment waveguide)
- Includes `latent_to_pink_trombone()` function mapping 8D → PT params
- **Verdict:** Runs, produces sound, but not validated against browser version

### 4. Word Synthesis (`primordial/speech/speak_words.py`)
- Converts text → phonemes → audio
- Simple phoneme dictionary for common words
- Works with any of the synth backends

## The 8D Latent Space

```
Dim | Name        | Range    | Description
----|-------------|----------|------------------------------------------
0   | Front-Back  | -1 to +1 | Place: lips (-1) to throat (+1)
1   | High-Low    | -1 to +1 | Tongue height: low (-1) to high (+1)
2   | Rounded     | -1 to +1 | Lip shape: spread (-1) to rounded (+1)
3   | Voice-Breath| -1 to +1 | Voiced (-1) → silence (0) → unvoiced (+1)
4   | Manner      | -1 to +1 | Stop (-1) → fricative (0.5) → approximant (+1)
5   | Vowel-Cons  | -1 to +1 | Vowel (-1) to consonant (+1)
6   | Pitch       | -1 to +1 | F0: low (-1) to high (+1)
7   | Intensity   | 0 to +1  | Volume: silent (0) to loud (+1)
```

Phoneme anchors defined in `primordial/speech/formant_synth.py:PHONEME_ANCHORS_8D`

## Two Paths Forward

### Option A: Validate/Fix Python Pink Trombone Port

**Files:** `primordial/speech/pink_trombone.py`

The port is complete but may have subtle bugs in the waveguide math. To validate:

1. Generate identical test sequences in both browser and Python
2. Compare waveforms/spectrograms
3. Debug any differences

**8D Integration:** Already has `latent_to_pink_trombone()` mapping:
```python
from primordial.speech.pink_trombone import PinkTrombone, latent_to_pink_trombone

synth = PinkTrombone()
params = latent_to_pink_trombone(latent_8d)
audio = synth.synthesize_samples(
    num_samples,
    tongue_index=params['tongue_index'],
    tongue_diameter=params['tongue_diameter'],
    constrictions=params['constrictions']
)
```

### Option B: Node.js Pink Trombone (Recommended)

**Repo:** `/Users/zachswift/projects/Pink-Trombone` (cloned from github.com/zakaton/Pink-Trombone)

The JavaScript DSP code is 100% browser-independent. Core files:
- `script/audio/nodes/pinkTrombone/processors/Glottis.js`
- `script/audio/nodes/pinkTrombone/processors/Tract.js`
- `script/audio/nodes/pinkTrombone/processors/Nose.js`
- `script/audio/nodes/pinkTrombone/processors/Processor.js`

**What needs replacing:**
| Component | Solution |
|-----------|----------|
| AudioWorklet | Inline the processor directly |
| AudioContext | Minimal mock or `node-web-audio-api` |
| DOM elements | Remove (not needed) |
| Audio output | Write WAV or use `node-speaker` |

**8D Integration approach:**
```javascript
// Create a simple API
const synth = new PinkTromboneSynth();

function latentToParams(latent8d) {
    return {
        tongueIndex: 12 + (latent8d[0] + 1) / 2 * 17,      // front-back
        tongueDiameter: 2.05 + (1 - latent8d[1]) / 2 * 1.45, // high-low
        frequency: 80 + (latent8d[6] + 1) / 2 * 220,        // pitch
        tenseness: latent8d[3] < -0.3 ? 0.6 : 0.0,          // voice-breath
        // ... etc
    };
}

// Generate audio
const params = latentToParams(latent);
synth.setParams(params);
const samples = synth.generateSamples(numSamples);
```

**Calling from Python:**
- Option 1: subprocess + stdin/stdout (JSON params → WAV bytes)
- Option 2: HTTP server (FastAPI ↔ Express)
- Option 3: Socket for real-time streaming

## Key Insight from Research

See `/docs/research/vocal-synth-tech-options.md` for detailed analysis.

**Why formant synth failed:** Only had cascade (poles) for vowels. Consonants need parallel branch with noise excitation at constriction point, not glottis.

**Why Pink Trombone works:** Kelly-Lochbaum waveguide naturally handles both - the constriction creates the turbulence source at the right position.

## Test Commands

```bash
# Test Python Pink Trombone
python -m primordial.speech.pink_trombone
afplay pink_trombone_test.wav

# Test word synthesis (uses vocal_synth by default)
python -m primordial.speech.speak_words "hello world"

# Test 8D latent mapping
python3 -c "
from primordial.speech.pink_trombone import PinkTrombone, latent_to_pink_trombone
import numpy as np
latent = np.array([0.5, 0.3, -0.5, -1.0, 0.0, -1.0, 0.0, 0.7])
params = latent_to_pink_trombone(latent)
print(params)
"
```

## Recorded Consonant Samples

Location: `samples:consonants/` (note the colon - macOS allows this)

24 samples recorded by user, ~100-200ms each:
- Stops: P, B, T, D, K, G
- Fricatives: F, V, TH, DH, S, Z, SH, ZH, HH
- Affricates: CH, JH
- Nasals: M, N, NG
- Approximants: L, R, W, Y

All at 22050Hz except D and T at 48kHz (auto-resampled on load).

## Next Steps

1. **Decide: Python port vs Node.js** - Python port exists but unvalidated. Node.js is safer but needs setup.

2. **Real-time streaming** - Current synths are offline. Need audio callback integration for live control.

3. **Playground with 8 sliders** - User requested UI with 8 sliders controlling each latent dimension.

4. **Neural vocoder upgrade path** - Research doc suggests DDSP-SVC or HiFi-Glot as eventual upgrade for higher quality.
