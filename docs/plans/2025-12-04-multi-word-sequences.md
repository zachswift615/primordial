# Multi-Word Sequence Generation

**Date:** 2025-12-04
**Status:** Design complete, ready for implementation
**Prerequisites:** Existing sequence decoder at 96% single-word accuracy

---

## Goal

Extend the sequence decoder from single words to multi-word phrases and short sentences, while adding acoustic validation to ensure generated sequences actually sound correct.

---

## Design Summary

### Approach: Sequential Extension with Acoustic Validation

1. Add hand-curated phrase data to the existing dataset
2. Extend curriculum to handle longer sequences (up to 30 phonemes)
3. Add acoustic match validation (self-listening as metric, not training signal)
4. Gate curriculum advancement on BOTH token accuracy AND acoustic match

### Key Insight: Self-Listening as Validation

Self-listening cannot provide training gradients because:
```
generated_phonemes → TTS → audio → encoder → loss
        ↑
   (discrete, non-differentiable)
```

Instead, we use acoustic match as:
- Validation metric ("does it sound right?")
- Curriculum gate ("don't advance until productions sound good")
- Debugging signal ("high token accuracy + low acoustic match = problem")

The supervised loss (cross-entropy on tokens + MSE on latents) provides the actual training gradients.

---

## Architecture

No changes to model architecture. The existing `SequenceDecoder` already handles variable-length sequences up to 30+ phonemes.

```
Audio → CNN Encoder → Fourier Mixing → Pooled (384)
                                           ↓
                              Memory Projection (384→128)
                                           ↓
                              Transformer Decoder (3 layers)
                                           ↓
                              Dual Output Heads
                                ├── Discrete (43 tokens) ← training signal
                                └── Latent (6D)          ← training signal
```

---

## Dataset Extension

### New Phrase Data

Hand-curated phrases organized by complexity:

```python
PHRASE_PHONEMES = {
    # 2-word phrases (6-10 phonemes)
    "hello world": ["HH", "AH", "L", "OW", "W", "ER", "L", "D"],
    "good morning": ["G", "UH", "D", "M", "AO", "R", "N", "IH", "NG"],
    "thank you": ["TH", "AE", "NG", "K", "Y", "UW"],
    "bye bye": ["B", "AY", "B", "AY"],
    "come here": ["K", "AH", "M", "HH", "IH", "R"],
    "go away": ["G", "OW", "AH", "W", "EY"],
    "sit down": ["S", "IH", "T", "D", "AW", "N"],
    "stand up": ["S", "T", "AE", "N", "D", "AH", "P"],
    "look here": ["L", "UH", "K", "HH", "IH", "R"],
    "watch me": ["W", "AA", "CH", "M", "IY"],

    # 3-word phrases (10-15 phonemes)
    "I love you": ["AY", "L", "AH", "V", "Y", "UW"],
    "how are you": ["HH", "AW", "AA", "R", "Y", "UW"],
    "see you later": ["S", "IY", "Y", "UW", "L", "EY", "T", "ER"],
    "nice to meet": ["N", "AY", "S", "T", "UW", "M", "IY", "T"],
    "what is that": ["W", "AH", "T", "IH", "Z", "DH", "AE", "T"],
    "where are you": ["W", "EH", "R", "AA", "R", "Y", "UW"],
    "I want food": ["AY", "W", "AA", "N", "T", "F", "UW", "D"],
    "give me water": ["G", "IH", "V", "M", "IY", "W", "AO", "T", "ER"],
    "open the door": ["OW", "P", "AH", "N", "DH", "AH", "D", "AO", "R"],
    "close your eyes": ["K", "L", "OW", "Z", "Y", "AO", "R", "AY", "Z"],

    # Short sentences (15-22 phonemes)
    "the cat is sleeping": ["DH", "AH", "K", "AE", "T", "IH", "Z", "S", "L", "IY", "P", "IH", "NG"],
    "I see a red ball": ["AY", "S", "IY", "AH", "R", "EH", "D", "B", "AO", "L"],
    "the dog is running": ["DH", "AH", "D", "AO", "G", "IH", "Z", "R", "AH", "N", "IH", "NG"],
    "can you help me": ["K", "AE", "N", "Y", "UW", "HH", "EH", "L", "P", "M", "IY"],
    "I want to go home": ["AY", "W", "AA", "N", "T", "T", "UW", "G", "OW", "HH", "OW", "M"],

    # Longer sentences (22-30 phonemes)
    "the little bird is singing": ["DH", "AH", "L", "IH", "T", "AH", "L", "B", "ER", "D", "IH", "Z", "S", "IH", "NG", "IH", "NG"],
    "I like to eat bananas": ["AY", "L", "AY", "K", "T", "UW", "IY", "T", "B", "AH", "N", "AE", "N", "AH", "Z"],
}
```

Target: 50-100 hand-curated phrases covering:
- Common greetings and farewells
- Simple commands
- Basic questions
- Short declarative sentences
- Child-friendly vocabulary (matching existing word set)

### Dataset Merging

```python
class WordDataset:
    def __init__(self, ..., include_phrases: bool = True):
        self.entries = {}
        self.entries.update(WORD_PHONEMES)  # Existing 51 words

        if include_phrases:
            self.entries.update(PHRASE_PHONEMES)  # New phrases

        # Filter by max_phonemes for curriculum
        self.filtered_entries = {
            k: v for k, v in self.entries.items()
            if len(v) <= self.max_phonemes
        }
```

---

## Acoustic Validation

### Compute Acoustic Match

```python
def compute_acoustic_match(
    model: SpeechSequenceLRN,
    tts: TTSBackend,
    generated_phonemes: List[str],
    target_audio: torch.Tensor,
) -> float:
    """
    Returns acoustic similarity score (0-1) for validation.
    No gradients — purely diagnostic.

    Args:
        model: The speech model (for encoder access)
        tts: TTS backend for synthesis
        generated_phonemes: Model's output, e.g. ["HH", "AH", "L", "OW"]
        target_audio: Original mel spectrogram the model was imitating

    Returns:
        Cosine similarity between produced and target embeddings (0-1)
    """
    # Handle failed generations
    if not generated_phonemes or len(generated_phonemes) == 0:
        return 0.0

    # Synthesize generated sequence
    produced_audio = tts.synthesize_phonemes(generated_phonemes)

    # Convert to mel spectrogram
    produced_mel = audio_to_mel(produced_audio)

    # Encode both through frozen encoder
    with torch.no_grad():
        produced_features = model.encoder(produced_mel).squeeze()
        target_features = model.encoder(target_audio).squeeze()

    # Cosine similarity as interpretable 0-1 score
    similarity = F.cosine_similarity(
        produced_features.unsqueeze(0),
        target_features.unsqueeze(0)
    ).item()

    # Clamp to 0-1 range (cosine can be negative)
    return max(0.0, similarity)
```

### Integration with Trainer

```python
class SequenceTrainer:
    def __init__(
        self,
        model: SpeechSequenceLRN,
        tts: TTSBackend,
        acoustic_check_interval: int = 10,
        ...
    ):
        self.tts = tts
        self.acoustic_check_interval = acoustic_check_interval
        self.acoustic_scores = []

    def train_epoch(self, dataloader, ...):
        self.acoustic_scores = []  # Reset each epoch

        for batch_idx, batch in enumerate(dataloader):
            # Supervised training (unchanged)
            loss = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()

            # Periodic acoustic validation
            if batch_idx % self.acoustic_check_interval == 0:
                with torch.no_grad():
                    generated = self.model.generate(batch.mel)

                    # Only score successful generations
                    if generated.eos_reached and len(generated.phonemes) > 0:
                        score = compute_acoustic_match(
                            self.model,
                            self.tts,
                            generated.phonemes,
                            batch.mel
                        )
                    else:
                        score = 0.0  # Failed generation

                    self.acoustic_scores.append(score)

        return {
            "loss": avg_loss,
            "token_accuracy": accuracy,
            "acoustic_match": np.mean(self.acoustic_scores) if self.acoustic_scores else 0.0,
        }
```

### Divergence Detection

```python
def log_metrics(token_accuracy: float, acoustic_match: float):
    """Log metrics and warn on divergence."""
    print(f"Token accuracy: {token_accuracy:.2%}")
    print(f"Acoustic match: {acoustic_match:.2%}")

    # Detect divergence
    if token_accuracy > 0.90 and acoustic_match < 0.50:
        print("WARNING: High token accuracy but low acoustic match!")
        print("Model may be getting phonemes right but timing/prosody wrong.")

    if acoustic_match > 0.90 and token_accuracy < 0.70:
        print("WARNING: High acoustic match but low token accuracy!")
        print("Unexpected — investigate dataset or evaluation.")
```

---

## Curriculum

### Phase Structure

| Phase | Epochs  | Max Phonemes | Data                          | Token Gate | Acoustic Gate |
|-------|---------|--------------|-------------------------------|------------|---------------|
| 1     | 1-40    | 8            | Single words + 2-word phrases | > 0.85     | > 0.80        |
| 2     | 41-80   | 15           | + 3-word phrases              | > 0.85     | > 0.80        |
| 3     | 81-130  | 22           | + short sentences             | > 0.80     | > 0.75        |
| 4     | 131-200 | 30           | All data                      | —          | —             |

### Configuration

```python
CURRICULUM_PHASES = {
    1: {
        "max_phonemes": 8,
        "token_threshold": 0.85,
        "acoustic_threshold": 0.80,
        "description": "Single words + 2-word phrases",
    },
    2: {
        "max_phonemes": 15,
        "token_threshold": 0.85,
        "acoustic_threshold": 0.80,
        "description": "+ 3-word phrases",
    },
    3: {
        "max_phonemes": 22,
        "token_threshold": 0.80,
        "acoustic_threshold": 0.75,
        "description": "+ short sentences",
    },
    4: {
        "max_phonemes": 30,
        "token_threshold": 0.0,  # No gate
        "acoustic_threshold": 0.0,  # No gate
        "description": "All data (final phase)",
    },
}
```

### Advancement Logic

```python
def should_advance_phase(
    current_phase: int,
    token_accuracy: float,
    acoustic_match: float,
) -> bool:
    """Check if both gates pass for curriculum advancement."""
    if current_phase >= 4:
        return False  # Already at final phase

    phase_config = CURRICULUM_PHASES[current_phase]
    token_gate = token_accuracy > phase_config["token_threshold"]
    acoustic_gate = acoustic_match > phase_config["acoustic_threshold"]

    return token_gate and acoustic_gate
```

---

## File Changes

### Files to Modify

| File | Changes |
|------|---------|
| `primordial/speech/word_dataset.py` | Add `PHRASE_PHONEMES` dict, update `WordDataset` to merge phrases |
| `primordial/speech/training.py` | Add `compute_acoustic_match()`, update `SequenceTrainer` with acoustic validation |
| `primordial/scripts/train_sequence.py` | Update curriculum config, add TTS init, dual gating logic |

### Files Unchanged

- `sequence_decoder.py` — architecture already handles longer sequences
- `tts.py` — already has `synthesize_phonemes()` for lists
- `latent.py` — no changes needed
- `encoders.py` — no changes needed

### New Test Coverage

| File | Tests to Add |
|------|--------------|
| `tests/speech/test_word_dataset.py` | Phrase loading, phoneme count filtering, merge behavior |
| `tests/speech/test_sequence_trainer.py` | Acoustic match computation, dual gating, divergence detection |

---

## Success Criteria

1. **Phase 1 complete:** > 85% token accuracy AND > 80% acoustic match on 2-word phrases
2. **Phase 4 complete:** > 80% token accuracy AND > 75% acoustic match on full dataset
3. **No catastrophic forgetting:** Single-word accuracy remains > 90% throughout
4. **Stable training:** Loss decreases monotonically, no NaN/Inf

---

## Commands

### Training

```bash
# Train with acoustic validation
python -m primordial.scripts.train_sequence \
    --encoder-checkpoint checkpoints/production/curriculum_best.pt \
    --epochs 200 \
    --demo-every 20 \
    --include-phrases

# Resume from checkpoint
python -m primordial.scripts.train_sequence \
    --resume checkpoints/sequence/multi_word_latest.pt \
    --epochs 200
```

### Testing

```bash
# Run all speech tests
pytest tests/speech/ -v

# Run specific new tests
pytest tests/speech/test_word_dataset.py -v -k phrase
pytest tests/speech/test_sequence_trainer.py -v -k acoustic
```

---

## Future Work

1. **Policy gradient training** — If accuracy plateaus, explore REINFORCE-style training using acoustic match as reward signal
2. **Differentiable synthesizer** — When latent synth exists, enable true self-listening with gradient flow
3. **Prosody modeling** — Add pitch contours for questions vs statements
4. **Speaker-invariant validation** — Use Whisper transcription for validation with different voices

---

## References

- Previous session: `docs/SESSION-2025-12-02-sequence-decoder.md`
- Latent synth design: `docs/plans/latent-voice-synth-design.md`
- Existing implementation: `primordial/speech/sequence_decoder.py`
