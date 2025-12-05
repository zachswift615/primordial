# Pink Trombone Node.js Port - Abandoned

**Date:** 2025-12-04
**Status:** Abandoned - library not suitable for programmatic speech synthesis
**Previous Session:** `latent-voice-synth-session.md`

## Goal

Port the Pink Trombone vocal tract synthesizer to Node.js and build a sequencer UI to create intelligible speech by positioning phoneme "nodes" on a timeline.

## What We Tried

### 1. Node.js DSP Port

Built a complete Node.js port of Pink Trombone in `primordial/speech/pink-trombone-node/`:
- `synth.js` - ~850 lines porting Glottis, Tract, Nose, and Processor classes
- `cli.js` - Command-line interface with WAV output
- `speak.js` - Phoneme sequencing attempt

**Result:** The port ran and produced sound, but we couldn't validate if the DSP was correct because even the browser version had the same issues with sequencing.

### 2. Browser Parameter Testing

Created `test-controls.html` with sliders for all Pink Trombone parameters:
- Frequency, tenseness, intensity, loudness
- Tongue index and diameter (vowel control)
- 4 constriction points (consonant control)
- Vibrato controls

**Result:** Individual vowel sounds worked when clicking presets. But sequencing multiple sounds into words produced "hell-<NOISE>" instead of "hello".

### 3. Timeline Sequencer UI

Built `sequencer.html` - a full sequencer with:
- All 19 parameters exposed as sliders
- Vowel and consonant preset buttons
- Timeline where you click to add nodes
- Nodes draggable (X = time, Y = pitch)
- Linear interpolation between nodes during playback
- Preview mode to hear while tweaking
- Export/import JSON

**Result:** The UI worked, but the output still didn't sound like speech. The core problem remained unsolved.

## Why It Failed

### The Core Problem

Pink Trombone is designed for **interactive exploration** - dragging your finger around the mouth visualization. It's NOT designed for programmatic phoneme sequencing.

**What works:** Holding a steady vowel sound while manually moving the tongue position.

**What doesn't work:** Transitioning between phonemes fast enough to form words. The transitions create pops, clicks, and unnatural sounds regardless of:
- Smoothing window size
- Interpolation method
- Parameter update rate

### Documentation Issues

The zakaton/Pink-Trombone repo has a README with phoneme parameter values, but no guidance on:
- How fast to transition between phonemes
- How to handle consonant-vowel coarticulation
- What parameter trajectories produce natural speech
- Whether the library was ever intended for this use case

### What We Learned

1. **Waveguide synths need careful trajectory design** - You can't just lerp between phoneme targets. Real speech has complex coarticulation patterns.

2. **The browser version has the same problem** - It's not a bug in our port. The library itself isn't suited for sequenced speech.

3. **19 dimensions is too many to tune by hand** - Even with the sequencer UI, finding parameter trajectories that sound natural is extremely difficult.

## Files Deleted

- `primordial/speech/pink-trombone-node/` (entire directory)
- `Pink-Trombone/sequencer.html`
- `Pink-Trombone/test-controls.html`

## Alternative Approaches to Consider

If you want AI-controlled voice synthesis, consider:

1. **eSpeak-NG** - Open-source formant synth with working phoneme sequencing. Has its own phoneme timing rules.

2. **Festival/Flite** - Concatenative synthesis with diphone databases. More natural than formant synths.

3. **Coqui TTS / VITS** - Neural TTS. Give it text, get natural speech. Less control over articulation but sounds good.

4. **DDSP (Differentiable DSP)** - Google's approach. Trains a neural network to control a DSP synth. Could potentially learn the right trajectories.

5. **Record and concatenate** - Use the recorded consonant samples in `samples:consonants/` with a simpler stitching approach.

## Conclusion

Pink Trombone is a beautiful educational tool for understanding vocal tract acoustics, but it's not the right foundation for an AI-controlled speech synthesizer. The problem of sequencing articulatory parameters into natural speech is much harder than it appears, and this library doesn't solve that problem - it just exposes the raw physics.

If the goal is latent-controlled speech, a neural approach (DDSP, VITS) that learns the mapping from latent space to audio is probably more tractable than trying to hand-engineer parameter trajectories through a waveguide model.
