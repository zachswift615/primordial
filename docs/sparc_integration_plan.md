# Primordial v2: SPARC Integration Plan

## Overview

Migrate from Piper-based phoneme output to SPARC articulatory control, enabling:
- Interpretable articulator positions (tongue, lips, jaw)
- Differentiable end-to-end training
- Natural prosody learning
- Foundation for embodied speech learning

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PRIMORDIAL v2                              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Audio     │───>│   Encoder   │───>│  Articulatory Head  │ │
│  │   Input     │    │  (existing) │    │       (new)         │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                                          ┌────────▼────────┐   │
│                                          │  EMA (12D)      │   │
│                                          │  Pitch (1D)     │   │
│                                          │  Loudness (1D)  │   │
│                                          └────────┬────────┘   │
│                                                   │             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SPARC Decoder (frozen)                      │   │
│  │         articulatory features + speaker_emb → audio      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Model Voice: Fixed 64D speaker embedding (your_voice.npy)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation (Current Sprint)

### 1.1 SPARC Integration Module
**Status**: In Progress
**Files to create**: `primordial/speech/sparc_integration.py`

```python
# Key components:
- SPARCWrapper: Thin wrapper around SPARC for encode/decode
- ArticulatoryHead: New output head (12 EMA + pitch + loudness)
- VoiceIdentity: Manages fixed speaker embedding
- preprocess_training_data(): Batch encode LibriSpeech with SPARC
```

**Tasks**:
- [ ] Create SPARCWrapper class that handles model loading
- [ ] Create ArticulatoryHead module (replaces phoneme decoder)
- [ ] Create VoiceIdentity class to manage speaker embedding
- [ ] Test roundtrip: mel → model → SPARC → audio

### 1.2 Data Preprocessing Pipeline
**Purpose**: Pre-encode LibriSpeech with SPARC to get training targets

```python
# For each LibriSpeech utterance:
{
    'audio_path': str,
    'mel': (n_mels, T_mel),           # Input to model
    'ema': (T_sparc, 12),             # Target: articulator positions
    'pitch': (T_sparc, 1),            # Target: F0
    'loudness': (T_sparc, 1),         # Target: energy
    'speaker_id': str,                # For analysis (not used in training)
}
```

**Tasks**:
- [ ] Create `scripts/preprocess_sparc.py` to batch-encode LibriSpeech
- [ ] Handle frame rate alignment (mel frames vs SPARC 50Hz)
- [ ] Save preprocessed data in efficient format (HDF5 or similar)
- [ ] Estimate storage requirements (~10GB for train-clean-100)

### 1.3 Training Script v2
**File**: `primordial/scripts/train_sparc.py`

**Tasks**:
- [ ] New training script using SPARC targets
- [ ] Loss function: MSE on EMA + pitch + loudness
- [ ] Frame alignment layer (mel rate → SPARC 50Hz)
- [ ] Validation: decode predictions, measure reconstruction quality
- [ ] Checkpoint saving with new architecture

---

## Phase 2: Supervised Training (Option A)

### 2.1 Basic Supervised Pipeline
**Goal**: Model learns to predict articulatory features from mel spectrograms

```python
# Training loop (simplified)
for mel, ema_target, pitch_target, loudness_target in dataloader:
    # Forward pass
    pred = model(mel)  # Returns {'ema': ..., 'pitch': ..., 'loudness': ...}

    # Loss
    loss = (
        mse_loss(pred['ema'], ema_target) * lambda_ema +
        mse_loss(pred['pitch'], pitch_target) * lambda_pitch +
        mse_loss(pred['loudness'], loudness_target) * lambda_loudness
    )

    # Backward
    loss.backward()
    optimizer.step()
```

**Loss weights to tune**:
- `lambda_ema`: Articulation accuracy (most important)
- `lambda_pitch`: Prosody melody
- `lambda_loudness`: Emphasis patterns

### 2.2 Validation & Metrics

| Metric | What it measures | How to compute |
|--------|------------------|----------------|
| EMA MSE | Articulation accuracy | Direct comparison |
| Pitch correlation | Prosody pattern match | Pearson correlation |
| Synthesis MOS | Perceptual quality | Decode & listen / crowdsource |
| WER (optional) | Intelligibility | ASR on decoded audio |

### 2.3 Curriculum (Supervised Phase)

| Stage | Data | Duration | Goal |
|-------|------|----------|------|
| 1 | Short utterances (<3s) | ~10 epochs | Learn basic articulation |
| 2 | Medium utterances (3-8s) | ~20 epochs | Learn prosodic patterns |
| 3 | Full LibriSpeech | ~50+ epochs | Generalization |

---

## Phase 3: Self-Listening Refinement (Option B)

### 3.1 End-to-End Training with Audio Loss
**Goal**: Model generates audio, "hears" it, compares to target

```python
# Self-listening training loop
for mel, target_audio in dataloader:
    # Model predicts articulation
    pred = model(mel)

    # Synthesize through SPARC (differentiable)
    generated_audio = sparc.decode(
        ema=pred['ema'],
        pitch=pred['pitch'],
        loudness=pred['loudness'],
        spk_emb=model_voice,
    )

    # Compare to target (multiple loss options)
    loss = audio_loss(generated_audio, target_audio)

    loss.backward()  # Gradients flow through SPARC
    optimizer.step()
```

### 3.2 Audio Loss Options

| Loss | Pros | Cons |
|------|------|------|
| Mel MSE | Simple, stable | Doesn't capture phase |
| Multi-scale spectral | Better perceptual match | More compute |
| Perceptual (wav2vec) | Semantic similarity | Requires pretrained model |
| Adversarial (GAN) | Sharp output | Training instability |

**Recommended**: Start with mel MSE, add multi-scale spectral if needed.

### 3.3 Curriculum (Self-Listening Phase)

| Stage | What | Why |
|-------|------|-----|
| 1 | Initialize from Phase 2 checkpoint | Don't start from scratch |
| 2 | Short utterances (<2s) | Memory constraints |
| 3 | Gradually increase length | Build up capability |
| 4 | Mix supervised + self-listening loss | Stability |

---

## Phase 4: RL Exploration (Option C - Future)

### 4.1 Reinforcement Learning Setup
**Goal**: Model explores articulation space, learns from reward

```python
# RL training loop (conceptual)
for target_phoneme in curriculum:
    # Agent produces articulation trajectory
    trajectory = policy(target_phoneme_embedding)

    # Environment: synthesize
    audio = sparc.decode(trajectory, model_voice)

    # Reward: multiple options
    reward = compute_reward(audio, target_phoneme)

    # RL update (PPO, SAC, etc.)
    policy.update(trajectory, reward)
```

### 4.2 Reward Function Options

| Reward | Description | Pros/Cons |
|--------|-------------|-----------|
| ASR confidence | Run ASR, check if recognized | End-to-end, but noisy |
| Phoneme classifier | Trained classifier on output | More direct signal |
| Human feedback | Rate quality | Gold standard, expensive |
| Self-supervised | Cycle consistency | No external model needed |

### 4.3 Curriculum (RL Phase)

```
Level 1: Single phonemes
    - "aa", "ee", "oo" (simple vowels)
    - Reward: phoneme classifier confidence

Level 2: CV syllables
    - "ba", "da", "ga", "ma", "na"
    - Reward: syllable recognition

Level 3: CVC words
    - "cat", "dog", "run", "big"
    - Reward: word recognition

Level 4: Multi-syllable words
    - "water", "happy", "computer"
    - Reward: word recognition + prosody

Level 5: Phrases
    - "hello world", "good morning"
    - Reward: phrase recognition + naturalness

Level 6: Sentences
    - Full LibriSpeech utterances
    - Reward: WER + MOS-like score
```

---

## Phase 5: Multimodal Integration (Future)

### 5.1 Architecture Extension

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRIMORDIAL v3 (Future)                       │
│                                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  Vision   │  │  Audio    │  │Propriocep │  │  Intent   │   │
│  │  Encoder  │  │  Encoder  │  │  Encoder  │  │  Module   │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        │              │              │              │          │
│        └──────────────┴──────────────┴──────────────┘          │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │   Fusion Layer    │                       │
│                    │  (cross-attention │                       │
│                    │   or similar)     │                       │
│                    └─────────┬─────────┘                       │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │  Articulatory     │                       │
│                    │  Head (SPARC)     │                       │
│                    └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Vision Integration Ideas

| Approach | Description |
|----------|-------------|
| Image captioning | See image → describe in speech |
| Lip reading fusion | See lips + hear audio → better recognition |
| Object naming | See object → say its name |
| Action description | See action → describe it |

### 5.3 Grounding Through Association

```python
# Conceptual multimodal training
for image, audio, text in multimodal_dataset:
    # Encode all modalities
    vision_emb = vision_encoder(image)
    audio_emb = audio_encoder(audio)

    # Fuse
    fused = fusion_layer(vision_emb, audio_emb)

    # Generate speech describing what's seen
    articulation = articulatory_head(fused)
    generated = sparc.decode(articulation, model_voice)

    # Loss: generated speech should match text/audio description
    loss = description_loss(generated, text)
```

### 5.4 Internal State Ideas (Future)

| State | What it represents | Effect on speech |
|-------|-------------------|------------------|
| Energy level | How "tired" the model is | Slower, lower pitch |
| Attention | What it's focused on | Emphasis patterns |
| Confidence | Certainty of utterance | Prosody, hesitation |
| Emotional valence | Positive/negative | Tone, pitch range |

---

## Implementation Checklist

### Immediate (This Week)
- [ ] Create `sparc_integration.py` module
- [ ] Create `ArticulatoryHead` class
- [ ] Test SPARC decode with model's voice embedding
- [ ] Verify gradients flow through SPARC decoder

### Short-term (Next 2 Weeks)
- [ ] Preprocess LibriSpeech with SPARC encoding
- [ ] Create `train_sparc.py` training script
- [ ] Run Phase 2 supervised training
- [ ] Evaluate: decode outputs, listen, measure metrics

### Medium-term (Month 1-2)
- [ ] Implement self-listening loop (Phase 3)
- [ ] Experiment with audio losses
- [ ] Fine-tune with self-listening
- [ ] Evaluate improvement over supervised-only

### Future (Month 2+)
- [ ] Explore RL training (Phase 4)
- [ ] Design multimodal architecture (Phase 5)
- [ ] Collect/prepare multimodal training data
- [ ] Implement vision encoder integration

---

## Technical Notes

### Frame Rate Alignment
- Mel spectrogram: depends on hop_length (default 160 @ 16kHz = 100Hz)
- SPARC features: 50Hz
- Need interpolation layer to align

```python
# Example alignment
mel_frames = mel.shape[-1]  # e.g., 200 frames at 100Hz = 2s
sparc_frames = int(mel_frames * 50 / 100)  # 100 frames at 50Hz = 2s
aligned_mel = F.interpolate(mel, size=sparc_frames, mode='linear')
```

### Memory Considerations
- SPARC decoder is ~400M parameters
- Keep frozen during training to save memory
- Batch size may need to be smaller than before
- Consider gradient checkpointing if needed

### Model Voice
- File: `your_voice_embedding.npy` (64D)
- Pitch stats: mean=139.4Hz, std=22.4Hz
- Use as fixed speaker identity for all inference

---

## Success Criteria

### Phase 2 (Supervised)
- [ ] EMA MSE < 0.5 (normalized)
- [ ] Decoded audio is intelligible
- [ ] Prosody patterns recognizable (questions rise, etc.)

### Phase 3 (Self-Listening)
- [ ] Improvement in perceptual quality over Phase 2
- [ ] Model can self-correct obvious errors
- [ ] Natural-sounding output

### Phase 4 (RL)
- [ ] Model can produce target phonemes from scratch
- [ ] Emergent exploration behavior
- [ ] Progressive curriculum completion

### Phase 5 (Multimodal)
- [ ] Model can describe simple images
- [ ] Cross-modal associations learned
- [ ] Grounded word meanings

---

## References

- SPARC Paper: "Coding Speech through Vocal Tract Kinematics" (Berkeley, 2024)
- Teaching Machines to Speak: Berkeley RL approach
- Primordial v1: Current Fourier mixing architecture
- Your voice embedding: `your_voice_embedding.npy`
