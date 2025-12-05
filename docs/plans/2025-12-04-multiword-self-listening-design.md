# Multi-Word Sequences with Self-Listening Validation

**Date:** 2025-12-04
**Status:** Design approved, ready for implementation

---

## Goal

Extend the sequence decoder from single words to multi-word phrases and sentences, using self-listening as a validation metric to ensure generated sequences sound correct.

---

## Current State

- 96% accuracy on single-word phoneme sequences (up to 12 phonemes)
- 51 words in `WORD_PHONEMES` dataset
- `SequenceDecoder` with dual heads (discrete tokens + 6D latents)
- Self-listening exists for production (single phonemes) but not sequences

---

## Design Decisions

### 1. Self-Listening as Validation, Not Training Loss

**Problem:** TTS synthesis is non-differentiable. The chain `phonemes → TTS → audio` breaks gradients.

**Solution:** Use self-listening purely for monitoring and curriculum gating:

| Component | Purpose |
|-----------|---------|
| Supervised loss (tokens + latents) | Training signal, gradients |
| Acoustic match score | Validation, curriculum gating, debugging |

The acoustic match score compares encoder features of:
- Generated sequence → TTS → audio → encoder → features
- Target audio → encoder → features

Using cosine similarity for interpretable 0-1 scores.

### 2. Dual Curriculum Gating

Advance to next phase only when BOTH conditions met:
- Token accuracy > threshold
- Acoustic match > threshold

This prevents the failure mode where discrete metrics look good but production sounds wrong.

### 3. Hand-Curated Phrase Data

50-100 manually written phrases with phoneme sequences, organized by length:
- Phase 1: 2-word phrases (8-12 phonemes)
- Phase 2: 3-word phrases (12-18 phonemes)
- Phase 3: Short sentences (18-25 phonemes)

Hand-curation ensures natural phrases vs. odd combinations like "banana elephant".

### 4. Progressive Curriculum

| Phase | Epochs | Max Phonemes | Data | Token Gate | Acoustic Gate |
|-------|--------|--------------|------|------------|---------------|
| 1 | 1-40 | 8 | Single words + 2-word phrases | 0.85 | 0.80 |
| 2 | 41-80 | 15 | + 3-word phrases | 0.85 | 0.80 |
| 3 | 81-130 | 22 | + short sentences | 0.80 | 0.75 |
| 4 | 131-200 | 30 | All data | — | — |

Relaxed thresholds for longer sequences (one wrong phoneme in 25 matters less than in 4).

---

## Implementation

### Acoustic Match Function

```python
def compute_acoustic_match(model, tts, generated_phonemes, target_audio):
    """
    Returns acoustic similarity score (0-1) for logging/curriculum gating.
    No gradients — purely diagnostic.
    """
    # Synthesize generated sequence
    produced_audio = tts.synthesize_phonemes(generated_phonemes)

    # Encode both
    produced_mel = audio_to_mel(produced_audio)
    target_mel = audio_to_mel(target_audio)

    with torch.no_grad():
        produced_features = model.encoder(produced_mel).squeeze()
        target_features = model.encoder(target_mel).squeeze()

    # Cosine similarity as interpretable 0-1 score
    similarity = F.cosine_similarity(
        produced_features.unsqueeze(0),
        target_features.unsqueeze(0)
    ).item()

    return similarity  # 1.0 = perfect match
```

### Training Loop Integration

```python
for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        # Supervised training (unchanged)
        loss = supervised_loss(discrete_logits, latent_preds, targets)
        loss.backward()
        optimizer.step()

        # Periodic acoustic validation (every N batches)
        if batch_idx % acoustic_check_interval == 0:
            with torch.no_grad():
                generated = model.generate(batch.audio)

                # Handle failed generations
                if generated.eos_reached and len(generated.phonemes) > 0:
                    score = compute_acoustic_match(
                        model, tts, generated.phonemes, batch.audio
                    )
                else:
                    score = 0.0  # Failed generation = 0 score

                acoustic_scores.append(score)

    # Log both metrics for divergence detection
    log("token_accuracy", accuracy)
    log("acoustic_match", np.mean(acoustic_scores))

    if accuracy > 0.9 and acoustic_match < 0.5:
        log("WARNING: high token accuracy but low acoustic match")

    # Dual-gated curriculum advancement
    phase_config = CURRICULUM_PHASES[current_phase]
    if (accuracy > phase_config["token_threshold"] and
        acoustic_match > phase_config["acoustic_threshold"]):
        advance_to_next_phase()
```

### Curriculum Configuration

```python
CURRICULUM_PHASES = {
    1: {"max_phonemes": 8, "token_threshold": 0.85, "acoustic_threshold": 0.80},
    2: {"max_phonemes": 15, "token_threshold": 0.85, "acoustic_threshold": 0.80},
    3: {"max_phonemes": 22, "token_threshold": 0.80, "acoustic_threshold": 0.75},
    4: {"max_phonemes": 30, "token_threshold": 0.0, "acoustic_threshold": 0.0},
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `primordial/speech/word_dataset.py` | Add `PHRASE_PHONEMES` dict, merge with `WORD_PHONEMES` |
| `primordial/speech/training.py` | Add `compute_acoustic_match()`, modify `SequenceTrainer` |
| `primordial/scripts/train_sequence.py` | Update curriculum, add TTS init, dual gating |
| `tests/speech/test_word_dataset.py` | Tests for phrase loading, phoneme filtering |
| `tests/speech/test_sequence_trainer.py` | Tests for acoustic match, dual gating |

## Files Unchanged

- `sequence_decoder.py` — already handles longer sequences
- `tts.py` — already has `synthesize_phonemes()` for lists
- `latent.py`, `encoders.py` — no changes needed

---

## Testing Requirements

1. **Dataset tests:**
   - Phrases concatenate phonemes correctly
   - Dataset filters by `max_phonemes` correctly
   - Merged dataset includes both words and phrases

2. **Acoustic match tests:**
   - Returns values in 0-1 range
   - Identical audio → score ≈ 1.0
   - Different phonemes → score < 0.5

3. **Curriculum tests:**
   - Phase advances only when both gates pass
   - Phase doesn't advance if only one gate passes
   - Failed generations score 0.0

---

## Future Extensions (Deferred)

1. **Policy gradient training** — use acoustic match as reward signal with REINFORCE
2. **Differentiable vocoder** — end-to-end gradients through synthesis
3. **Hierarchical generation** — word-level → phoneme-level decoder
4. **Prosody modeling** — duration/pitch contours over sequences

---

## Success Criteria

- [ ] 85%+ token accuracy on 2-word phrases
- [ ] 80%+ acoustic match on 2-word phrases
- [ ] 80%+ token accuracy on 3-word phrases
- [ ] 75%+ acoustic match on short sentences
- [ ] No catastrophic forgetting of single words
