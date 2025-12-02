# Speech Learning Experiment: Emergent Speech Acquisition

**Date:** 2025-12-01
**Status:** Planning
**Goal:** Teach LRN to speak through sensorimotor learning - hearing and imitating, like an infant.

## Original Conversation Context

This session started with exploring the LRN continuous learning architecture and asking:
> "Is there a more direct approach we could take to train this type of agent/model? What's the most ambitious thing we could try?"

We discussed three approaches:
1. **Webcam world model** - predict real video frames
2. **Audio-visual teaching** - human rewards on real sensory input
3. **Game agent (Minecraft)** - rich simulation with survival signals

We implemented #3 (Minecraft/MineDojo integration) but hit Windows installation issues. During that discussion, a more interesting idea emerged:

> "Could we teach it to speak? It has what amounts to ears and could hear voice recordings. Couldn't we give it the ability to control phonemes and let it try to speak?"

This is potentially more novel and publishable than the Minecraft work.

## Why This Is Interesting

**Current speech AI paradigm:**
- Text → LLM → Text → TTS (language is symbolic, speech is output layer)
- Trained on massive text corpora offline
- No embodiment, no sensorimotor loop

**Our proposed paradigm:**
- Audio → LRN → Audio (speech is primary, not derived from text)
- Learns through imitation (hear → try to reproduce → compare → learn)
- Embodied: the agent "hears itself speak" and adjusts
- Online learning from experience, not batch training

This is closer to how infants acquire speech - babbling, hearing themselves, matching adult sounds.

## Architecture Design

### Current LRN Audio Path
```
Audio Input: (100, 2) stereo samples
    → AudioEncoder → (100, 128) sequence
    → Fourier Mixing (6 layers)
    → ActionHead → vocalize (frequency, amplitude) - just a tone
```

### Proposed Speech-Capable LRN
```
Audio Input: Mel spectrogram (80 mel bins × 100 frames)
    → MelSpectrogramEncoder → (100, 128) sequence
    → Fourier Mixing (6 layers)
    → SpeechHead → phoneme logits (44) + duration + pitch

TTS Synthesis: phonemes → Piper/Sherpa-ONNX → audio waveform
    → played through speaker
    → captured by microphone (or looped back digitally)
    → becomes next Audio Input (agent hears itself)
```

### Components Needed

1. **MelSpectrogramEncoder** (replaces simple AudioEncoder)
   - Input: (batch, 80, 100) mel spectrogram
   - Uses small CNN or linear projection
   - Output: (batch, 100, 128) sequence for Fourier mixing

2. **SpeechHead** (replaces simple vocalize output)
   - Input: (batch, 384) pooled features
   - Output:
     - phoneme_logits: (batch, 44) - probability over phonemes
     - duration: (batch, 1) - how long to hold phoneme
     - pitch: (batch, 1) - prosody control
   - Sampling: argmax or temperature-scaled sampling

3. **TTS Synthesis Module**
   - Takes phoneme sequence from SpeechHead
   - Generates audio waveform via Piper or Sherpa-ONNX
   - Returns waveform for playback and self-listening

4. **Audio Similarity Loss**
   - Compare agent's produced audio to target audio
   - Options:
     - Mel spectrogram MSE (simple, may work)
     - Wav2Vec embedding cosine similarity (better, pretrained)
     - CTC loss on phoneme alignment (linguistic)

## English Phoneme Inventory (ARPABET, 39 phonemes)

Vowels (15):
- AA (father), AE (cat), AH (but), AO (dog), AW (cow)
- AY (my), EH (bed), ER (bird), EY (say), IH (bit)
- IY (bee), OW (go), OY (boy), UH (book), UW (food)

Consonants (24):
- B, CH, D, DH (the), F, G, HH, JH (judge), K, L
- M, N, NG (sing), P, R, S, SH, T, TH (think), V
- W, Y, Z, ZH (measure)

## Training Curriculum

### Phase 1: Phoneme Imitation (Week 1-2)
- Play single phoneme recordings
- Agent outputs single phoneme prediction
- Loss: cross-entropy on phoneme classification
- Success metric: >80% accuracy on phoneme identification

### Phase 2: Phoneme Production (Week 2-3)
- Agent tries to produce heard phoneme via TTS
- Hears its own output
- Loss: audio similarity between input and self-output
- Success metric: distinguishable phoneme production

### Phase 3: Phoneme Sequences (Week 3-4)
- Play 2-3 phoneme sequences ("ba-ba", "ma-ma")
- Agent learns temporal patterns
- Loss: sequence-level audio similarity
- Success metric: can reproduce simple babbling

### Phase 4: Word Imitation (Week 4+)
- Play simple words
- Agent echoes them
- This is where emergent behavior gets interesting

## TTS Stack Decision

**Piper** (user has experience with this):
- Fast, lightweight, runs locally
- ONNX-based, good for real-time
- Phoneme input supported
- MIT license

**Sherpa-ONNX** (user also has experience):
- Similar benefits
- Good Python bindings

**Recommendation:** Start with Piper since user is familiar. Can swap later.

## Hardware Requirements

- MacBook Pro: sufficient for training (no GPU needed for small model)
- Microphone: for recording training data (optional if using existing datasets)
- Speaker/headphones: for hearing agent output (optional if using digital loopback)

## Datasets

**For phoneme training:**
- TIMIT (classic, phoneme-aligned)
- LibriSpeech (large, but needs alignment)
- Common Voice (multilingual option)

**Simplest start:**
- Record yourself saying each phoneme
- 10 examples per phoneme = 440 recordings
- Or use existing phoneme datasets

## Research Paper Angle

**Title ideas:**
- "Emergent Speech Acquisition Through Sensorimotor Learning"
- "Babbling to Words: Speech Learning Without Text in Embodied Agents"
- "Fourier-Based Continuous Learning for Speech Imitation"

**Key claims to prove:**
1. Agent learns to distinguish phonemes (classification accuracy)
2. Agent learns to produce phonemes (audio similarity metric)
3. Agent shows improvement over time (learning curve)
4. Comparison to baseline (random, or simple RNN)

**Novel contributions:**
- First (?) application of Fourier mixing to speech learning
- Embodied speech acquisition without text/language model
- Online learning from sensorimotor experience

## Implementation Plan

### Step 1: MelSpectrogramEncoder
- Add to `primordial/lrn/encoders.py`
- Test with existing LRN architecture

### Step 2: SpeechHead
- Add to `primordial/lrn/lrn_heads.py`
- Output phonemes + duration + pitch

### Step 3: Piper Integration
- Create `primordial/speech/` module
- Phoneme-to-audio synthesis
- Audio capture/playback utilities

### Step 4: Training Loop
- Create `primordial/speech/training.py`
- Phoneme classification training
- Audio similarity training

### Step 5: Experiments
- Run Phase 1-4 curriculum
- Collect metrics
- Generate figures for paper

## Next Session TODO

1. [ ] Create `primordial/speech/` module structure
2. [ ] Implement MelSpectrogramEncoder
3. [ ] Implement SpeechHead
4. [ ] Integrate Piper for TTS
5. [ ] Create phoneme training dataset (or find existing)
6. [ ] Implement training loop with audio similarity loss
7. [ ] Run Phase 1 experiment (phoneme imitation)

## Open Questions

1. Should the agent hear its own output in real-time (true sensorimotor loop) or compare after generation (simpler)?
2. How to handle phoneme duration - fixed length or variable?
3. Should we include prosody (pitch contour) or just phonemes?
4. What's the right audio similarity metric?
