# Latent Voice Synthesizer Design

**Date:** 2025-12-03
**Status:** Design phase
**Goal:** Real-time synthesizer controlled directly by 8D articulatory latent vectors

---

## Motivation

The current speech pipeline uses discrete phoneme tokens:

```
Model → discrete phoneme → Piper TTS → audio
```

This breaks gradient flow and prevents true self-listening. We want:

```
Model → continuous 8D latent → Latent Synth → audio
                                     ↓
                              Model hears itself
```

The model controls its "mouth" continuously. Sound emerges from the physics of the latent position, not from looking up phoneme strings.

---

## The 8D Articulatory Latent Space

### Dimensions

| Dim | Name | Range | Description |
|-----|------|-------|-------------|
| 0 | **Front-Back** | -1 to +1 | Place of articulation: lips (-1) ↔ throat (+1) |
| 1 | **High-Low** | -1 to +1 | Tongue height: low (-1) ↔ high (+1) |
| 2 | **Rounded** | -1 to +1 | Lip shape: spread (-1) ↔ rounded (+1) |
| 3 | **Voice-Breath** | -1 to +1 | Voiced (-1) ↔ silence (0) ↔ unvoiced airflow (+1) |
| 4 | **Manner** | -1 to +1 | Airflow type: stop (-1) ↔ fricative (0.5) ↔ approximant (+1) |
| 5 | **Vowel-Cons** | -1 to +1 | Sound type: vowel (-1) ↔ consonant (+1) |
| 6 | **Pitch** | -1 to +1 | Fundamental frequency: low (-1) ↔ high (+1) |
| 7 | **Intensity** | 0 to +1 | Volume: silent (0) ↔ loud (+1) |

### Key Design Decisions

**Voice-Breath as single dimension:**
```
-1.0          0.0              +1.0
  │            │                 │
  ▼            ▼                 ▼
VOICED ──── SILENCE ──── UNVOICED AIRFLOW
 "B"        (nothing)           "P"
 "Z"          EOS               "S"
 "M"                            "F"
```

This eliminates the need for a special EOS token. Silence is the physical state where voice-breath = 0 and intensity = 0.

**Duration is time, not a dimension:**
Duration emerges from how long the model maintains a latent position, not from a separate dimension. The synth renders whatever latent it receives, for as long as it receives it.

---

## Synthesizer Architecture Options

### Option A: Formant Synthesizer

Classic speech synthesis using formant frequencies.

**Mapping:**
```
Dim 0-2 (front-back, high-low, rounded) → F1, F2, F3 formant frequencies
Dim 3 (voice-breath) →
    if < 0: voiced source (pulse train at F0)
    if = 0: silence
    if > 0: noise source (unvoiced)
Dim 4 (manner) → filter characteristics, noise bandwidth
Dim 5 (vowel-cons) → formant bandwidth, transition speed
Dim 6 (pitch) → F0 fundamental frequency (80-300 Hz typical)
Dim 7 (intensity) → output amplitude
```

**Pros:**
- Simple, interpretable
- Fast, definitely real-time capable
- Direct physical correspondence

**Cons:**
- Sounds robotic
- Limited consonant quality
- Formant mapping requires tuning

**Implementation approach:**
- Use existing formant synth library or build minimal one
- Lookup tables: latent region → formant values
- Real-time: receive latent at 100Hz, output audio at 22050Hz

### Option B: Neural Vocoder

Train a small neural network: 8D vector → audio waveform.

**Architecture:**
```
8D latent → Upsampling layers → WaveRNN/HiFi-GAN style → audio samples
```

**Training data:**
- Generate (latent, audio) pairs from existing phoneme anchors + Piper
- Interpolate between anchors for smooth coverage
- Need ~10-100 hours of aligned data

**Pros:**
- Higher quality audio
- Learns complex acoustic patterns
- End-to-end differentiable (!)

**Cons:**
- Requires training
- May not be real-time on CPU
- Black box mapping

### Option C: Hybrid Approach

Formant synth for prototyping, neural vocoder as upgrade path.

**Phase 1:** Build formant synth, get playground working
**Phase 2:** Collect (latent, audio) pairs during use
**Phase 3:** Train neural vocoder on collected data
**Phase 4:** Swap in neural vocoder, keep same interface

---

## Real-Time Playground Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     REAL-TIME PLAYGROUND                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐     │
│  │   MIC    │────────►│  MODEL   │────────►│  SYNTH   │     │
│  │  INPUT   │  audio  │ ENCODER  │ 8D lat  │  OUTPUT  │     │
│  └──────────┘         └────┬─────┘         └────┬─────┘     │
│                            │                    │            │
│                            ▼                    ▼            │
│                     ┌──────────┐         ┌──────────┐       │
│                     │  LATENT  │         │ SPEAKERS │       │
│                     │VISUALIZER│         │          │       │
│                     └──────────┘         └────┬─────┘       │
│                                               │              │
│                            ┌──────────────────┘              │
│                            ▼                                 │
│                     ┌──────────┐                            │
│                     │  MODEL   │◄─── model hears itself     │
│                     │ DECODER  │                            │
│                     └──────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components

**1. Audio Input (Mic)**
- Capture at 22050 Hz
- Convert to mel spectrogram in real-time
- Buffer: ~100ms chunks

**2. Model Encoder**
- Existing CNN + Fourier mixing encoder
- Input: mel spectrogram
- Output: audio embedding (384D)

**3. Model Decoder**
- Pure-latent sequence decoder (new)
- Input: audio embedding + previous latent
- Output: next 8D latent vector
- Runs at ~100 Hz (one latent per 10ms)

**4. Latent Synth**
- Input: 8D latent vector stream at 100 Hz
- Output: audio stream at 22050 Hz
- Must be real-time capable

**5. Audio Output (Speakers)**
- Play synth output
- Route back to model's "ears" for self-listening

**6. Latent Visualizer (Optional)**
- 2D/3D projection of 8D space
- Show current position, trajectory
- Helpful for debugging and intuition

### Timing Requirements

```
Latent rate:  100 Hz (10ms per latent)
Audio rate:   22050 Hz
Buffer size:  ~50-100ms for stability
Total latency target: <200ms (usable for interaction)
```

---

## Migration Path from Current 6D to 8D

### Current 6D Anchors
```python
PHONEME_ANCHORS = {
    'IY': [1.0, 1.0, -1.0, 1.0, 0.0, -1.0],  # voiced=1.0
    'P':  [-1.0, 0.0, 0.0, -1.0, -1.0, 1.0], # voiced=-1.0 (unvoiced)
    ...
}
```

### New 8D Anchors
```python
PHONEME_ANCHORS_8D = {
    # Voiced sounds: voice-breath = -1.0
    'IY': [1.0, 1.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.7],
    #                       ↑ voiced        ↑pitch ↑intensity

    # Unvoiced sounds: voice-breath = +1.0
    'P':  [-1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.7],
    #                      ↑ unvoiced airflow

    # Silence: voice-breath = 0, intensity = 0
    'SIL': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'EOS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Same as silence
    ...
}
```

### Conversion Function
```python
def convert_6d_to_8d(latent_6d):
    """Convert old 6D latent to new 8D format."""
    front_back, high_low, rounded, voiced, manner, vowel_cons = latent_6d

    # Convert voiced dimension to voice-breath
    # Old: voiced=1.0 means voiced, voiced=-1.0 means unvoiced
    # New: voice_breath=-1.0 means voiced, voice_breath=+1.0 means unvoiced airflow
    voice_breath = -voiced  # Flip sign

    # Default pitch and intensity
    pitch = 0.0       # Neutral pitch
    intensity = 0.7   # Medium volume (not silent, not max)

    return [front_back, high_low, rounded, voice_breath, manner, vowel_cons, pitch, intensity]
```

---

## Formant Synth Implementation Sketch

```python
class FormantSynth:
    """Real-time formant synthesizer controlled by 8D latent vectors."""

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.phase = 0.0  # For continuous waveform

        # Formant frequency lookup (approximate, needs tuning)
        # Based on standard acoustic phonetics data
        self.vowel_formants = {
            # (F1, F2, F3) in Hz for cardinal vowels
            # Indexed by (front_back, high_low) region
        }

    def latent_to_formants(self, latent):
        """Convert 8D latent to formant frequencies and source parameters."""
        front_back, high_low, rounded, voice_breath, manner, vowel_cons, pitch, intensity = latent

        # Base formant frequencies from tongue position
        F1 = self._compute_f1(high_low)        # ~300-800 Hz
        F2 = self._compute_f2(front_back)      # ~800-2500 Hz
        F3 = self._compute_f3(rounded)         # ~2500-3500 Hz

        # Fundamental frequency from pitch
        F0 = self._pitch_to_f0(pitch)          # ~80-300 Hz

        # Source type from voice-breath
        if voice_breath < -0.3:
            source = 'voiced'
        elif voice_breath > 0.3:
            source = 'unvoiced'
        else:
            source = 'silent'

        # Amplitude from intensity
        amplitude = intensity

        return F0, F1, F2, F3, source, amplitude

    def _pitch_to_f0(self, pitch):
        """Map pitch [-1, +1] to fundamental frequency."""
        # Typical range: 80 Hz (low male) to 300 Hz (high female/child)
        f0_min, f0_max = 80, 300
        return f0_min + (pitch + 1) / 2 * (f0_max - f0_min)

    def _compute_f1(self, high_low):
        """F1 inversely related to tongue height."""
        # High vowels (~300 Hz) to low vowels (~800 Hz)
        f1_high, f1_low = 300, 800
        return f1_high + (-high_low + 1) / 2 * (f1_low - f1_high)

    def _compute_f2(self, front_back):
        """F2 related to tongue frontness."""
        # Back vowels (~800 Hz) to front vowels (~2500 Hz)
        f2_back, f2_front = 800, 2500
        return f2_back + (front_back + 1) / 2 * (f2_front - f2_back)

    def _compute_f3(self, rounded):
        """F3 lowered by lip rounding."""
        # Unrounded (~3500 Hz) to rounded (~2500 Hz)
        f3_unrounded, f3_rounded = 3500, 2500
        return f3_unrounded + (rounded + 1) / 2 * (f3_rounded - f3_unrounded)

    def synthesize_frame(self, latent, num_samples=220):
        """Generate audio samples for one latent frame (10ms at 22050 Hz)."""
        F0, F1, F2, F3, source, amplitude = self.latent_to_formants(latent)

        if source == 'silent' or amplitude < 0.01:
            return np.zeros(num_samples)

        # Generate source signal
        if source == 'voiced':
            # Pulse train at F0
            samples = self._generate_pulse_train(F0, num_samples)
        else:
            # White noise for unvoiced
            samples = np.random.randn(num_samples)

        # Apply formant filters (simplified: could use proper IIR filters)
        samples = self._apply_formants(samples, F1, F2, F3)

        # Apply amplitude
        samples = samples * amplitude

        return samples

    def _generate_pulse_train(self, f0, num_samples):
        """Generate glottal pulse train."""
        period_samples = self.sample_rate / f0
        samples = np.zeros(num_samples)

        t = 0
        while t < num_samples:
            idx = int(t)
            if idx < num_samples:
                samples[idx] = 1.0
            t += period_samples

        return samples

    def _apply_formants(self, samples, f1, f2, f3):
        """Apply formant resonances (simplified)."""
        # In practice, use scipy.signal for proper bandpass filters
        # This is a placeholder for the concept
        from scipy.signal import butter, lfilter

        for formant_freq in [f1, f2, f3]:
            # Bandpass around formant frequency
            bandwidth = formant_freq * 0.1  # 10% bandwidth
            low = (formant_freq - bandwidth/2) / (self.sample_rate/2)
            high = (formant_freq + bandwidth/2) / (self.sample_rate/2)
            low = max(0.01, min(0.99, low))
            high = max(0.01, min(0.99, high))
            if low < high:
                b, a = butter(2, [low, high], btype='band')
                samples = lfilter(b, a, samples)

        return samples
```

---

## Open Questions

1. **Consonant synthesis:** Formant synth works well for vowels. Stops, fricatives, and nasals need additional modeling (noise bursts, nasal resonances, etc.)

2. **Coarticulation:** Real speech has smooth transitions. Should the synth handle interpolation, or does the model output smooth latent trajectories?

3. **Real-time performance:** Can formant synth run fast enough? Target: <5ms to synthesize 10ms of audio.

4. **Training the model:** With 8D latents, do we retrain from scratch or adapt the existing 6D model?

5. **Differentiable synth:** For true gradient flow through self-listening, the synth would need to be differentiable. Formant synth could be made differentiable with some work.

---

## Next Steps

1. **Build minimal formant synth** — vowels only, ~100 lines of code
2. **Test with static latents** — verify latent positions produce expected sounds
3. **Add real-time streaming** — latent input at 100Hz, audio output at 22050Hz
4. **Build playground UI** — mic input, visualization, speaker output
5. **Connect to model** — encoder processes mic, decoder outputs latents
6. **Close the loop** — model hears its own output

---

## References

- Klatt, D. (1980). "Software for a cascade/parallel formant synthesizer"
- Standard acoustic phonetics: formant frequencies for vowels
- HiFi-GAN, WaveRNN for neural vocoder approaches
- Existing code: `primordial/speech/latent.py` for current 6D space
