# Session Summary: Autoregressive Sequence Decoder

**Date:** 2025-12-02
**Goal:** Extend speech production from single phonemes to full word sequences

---

## What We Built

### Autoregressive Sequence Decoder
A transformer-based decoder that generates complete phoneme sequences from audio input. Instead of just recognizing "hello" → HH, the model now produces the full sequence: HH → EH → L → OW.

**Architecture:**
```
Audio → CNN Encoder → Fourier Mixing → Pooled (384)
                                           ↓
                              Memory Projection (384→128)
                                           ↓
                              Transformer Decoder (3 layers)
                                    - 4 attention heads
                                    - 128 hidden dim
                                    - Causal self-attention
                                    - Cross-attention to audio
                                           ↓
                              Dual Output Heads
                                ├── Discrete (43 tokens)
                                └── Latent (6D articulatory)
```

**Key Components:**
- `SequenceDecoder`: 3-layer transformer with dual heads
- `SpeechSequenceLRN`: Full model wrapping encoder + decoder
- `WordDataset`: 51 words with phoneme sequences
- `SequenceTrainer`: Teacher forcing with masked losses

---

## Training Results

### Final Run: 96.9% Accuracy

| Phase | Epochs | Max Phonemes | Words | Final Accuracy |
|-------|--------|--------------|-------|----------------|
| 1 | 1-30 | 3 | 11 | 73% |
| 2 | 31-60 | 5 | 33 | 68% |
| 3 | 61-100 | 7 | 42 | 87% |
| 4 | 101-160 | 12 | 51 | **96.9%** |

### Demo Progression
```
Epoch  20: 'you' → ['IY']                     # Too short, wrong
Epoch  50: 'under' → ['P', 'AH', 'T', 'ER']   # Wrong phonemes
Epoch  80: 'under' → ['B', 'AH', 'N', 'D', 'AH', 'T', 'ER']  # Too long
Epoch 100: 'butterfly' → ['B', 'AH', 'T', 'ER', 'F', 'L', 'AY'] ✓ MATCH
Epoch 120: 'under' → ['AH', 'N', 'D', 'ER']   ✓ MATCH
Epoch 160: 'you' → ['Y', 'UW']                 ✓ MATCH
```

---

## Key Bugs Fixed

### 1. EOS Token Not Being Learned
**Problem:** Model generated infinite sequences (15+ tokens)
**Cause:** `ignore_index=EOS_TOKEN` in cross-entropy loss
**Fix:** Use `ignore_index=-100` for padding only, let EOS be trained normally

### 2. Phase 4 Not Running
**Problem:** Training stopped at epoch 100
**Cause:** Hardcoded `phases = [1, 2, 3]`
**Fix:** Changed to `phases = list(CURRICULUM.keys())`

### 3. Early EOS Bias
**Problem:** Model stopped sequences too early
**Fix:** Added `min_length` and `eos_penalty` parameters to `generate()`

---

## Files Created/Modified

### New Files
- `primordial/speech/sequence_decoder.py` - SequenceDecoder, SpeechSequenceLRN
- `primordial/speech/word_dataset.py` - WORD_PHONEMES (51 words), WordDataset
- `primordial/scripts/train_sequence.py` - Training CLI with 4-phase curriculum
- `tests/speech/test_sequence_decoder.py` - 9 tests
- `tests/speech/test_word_dataset.py` - 4 tests
- `tests/speech/test_sequence_trainer.py` - 3 tests
- `tests/speech/test_integration.py` - 2 integration tests
- `docs/plans/2025-12-02-autoregressive-sequence-decoder.md` - Design doc
- `docs/plans/2025-12-02-sequence-decoder-implementation.md` - Implementation plan

### Modified Files
- `primordial/speech/latent.py` - Added SOS_TOKEN, EOS_TOKEN, TOTAL_VOCAB
- `primordial/speech/training.py` - Added SequenceTrainer
- `primordial/speech/__init__.py` - Exports for new components

---

## Dataset

### Word Distribution by Phoneme Count
```
2 phonemes:  10 words (ba, bee, ma, me, hi, go, no, see, you, we)
3 phonemes:   3 words (over, under → actually 4 phonemes, puppy)
4 phonemes:  12 words (hello, water, mommy, daddy, baby, happy, sorry, open, yellow, ...)
5 phonemes:  10 words (monkey, pizza, chicken, rainbow, purple, orange, rabbit, kitten, ...)
6 phonemes:   8 words (banana, morning, evening, singing, jumping, running, tomorrow, together)
7 phonemes:   5 words (elephant, butterfly, dinosaur, alligator, beautiful, wonderful)
8+ phonemes:  3 words (computer, caterpillar, helicopter, crocodile, strawberry, watermelon, understand)
```

---

## Parameter Count

| Component | Parameters |
|-----------|------------|
| CNN Encoder (existing) | ~800K |
| Memory Projection | 49K |
| Phoneme Embedding | 5.5K |
| Transformer Decoder | 540K |
| Discrete Head | 5.5K |
| Latent Head | 8.6K |
| **Total New** | **~610K** |

---

## Commands

### Train from Scratch
```bash
python -m primordial.scripts.train_sequence --epochs 160 --demo-every 20
```

### Train with Pretrained Encoder (Recommended)
```bash
python -m primordial.scripts.train_sequence \
  --encoder-checkpoint checkpoints/production/curriculum_best.pt \
  --epochs 160 \
  --demo-every 20
```

### Generate from Audio
```python
from primordial.speech import SpeechSequenceLRN, SpeechConfig

config = SpeechConfig(encoder_type='cnn', tts_backend='piper')
model = SpeechSequenceLRN(config)
model.load_state_dict(torch.load('checkpoints/sequence/sequence_best.pt'))

# Generate
phonemes, latents = model.generate(mel_spectrogram, min_length=2)
print(phonemes)  # ['HH', 'EH', 'L', 'OW']
```

---

## Next Steps

1. **Self-listening training** - Agent produces → hears itself → adjusts
2. **Multi-word sequences** - "hello world" as single production
3. **Prosody modeling** - Pitch/duration contours over sequence
4. **Semantic grounding** - Produce word from meaning, not just imitation

---

## Test Results

```
21 tests passing
├── test_latent.py (3 tests)
├── test_sequence_decoder.py (9 tests)
├── test_sequence_trainer.py (3 tests)
├── test_word_dataset.py (4 tests)
└── test_integration.py (2 tests)
```
