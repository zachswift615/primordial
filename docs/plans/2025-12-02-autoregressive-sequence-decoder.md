# Autoregressive Sequence Decoder Design

**Date:** 2025-12-02
**Status:** Approved
**Depends on:** Phase 2 Speech Production (2025-12-02-phase2-speech-production.md)

---

## Overview

Extend the speech production system to generate full phoneme sequences instead of single phonemes. The agent hears "hello" and produces the complete sequence HH→EH→L→OW, not just the first phoneme.

**Core insight:** Use a transformer decoder with dual output heads (discrete tokens + continuous latents) trained via teacher forcing and self-listening.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FULL SEQUENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Audio "hello"                                                   │
│       ↓                                                          │
│  Mel Spectrogram (80×64)                                         │
│       ↓                                                          │
│  CNN Encoder → Fourier Mixing → Pooled (384)   ← EXISTING        │
│       ↓                                                          │
│  Memory Projection (384 → 128)                 ← NEW             │
│       ↓                                                          │
│  ┌─────────────────────────────────────────┐                     │
│  │     Transformer Decoder (3 layers)      │   ← NEW             │
│  │                                         │                     │
│  │  [SOS] → Embed (128) + PosEnc           │                     │
│  │            ↓                            │                     │
│  │  Causal Self-Attention (4 heads)        │                     │
│  │            ↓                            │                     │
│  │  Cross-Attention to Memory (128)        │                     │
│  │            ↓                            │                     │
│  │  FFN (256)                              │                     │
│  │            ↓                            │                     │
│  │  Dual Heads:                            │                     │
│  │    - Discrete: Linear(128→43) → softmax │                     │
│  │    - Latent: Linear(128→6) → tanh       │                     │
│  └─────────────────────────────────────────┘                     │
│       ↓                                                          │
│  [SOS] → [HH] → [EH] → [L] → [OW] → [EOS]                        │
│       ↓                                                          │
│  Piper TTS → Audio "hello"                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Dimensions

| Component | Dimension |
|-----------|-----------|
| Audio pooled (existing) | 384 |
| Memory projection | 384 → 128 |
| Phoneme embedding | 43 × 128 |
| Transformer hidden | 128 |
| Attention heads | 4 (32 dims each) |
| FFN intermediate | 256 |
| Decoder layers | 3 |
| Discrete output | 43 tokens |
| Latent output | 6 dimensions |

### Token Vocabulary

```python
PHONEME_VOCAB_SIZE = 41   # Existing phonemes (IY, IH, ..., SIL, UNK)
SOS_TOKEN = 41            # Start of sequence
EOS_TOKEN = 42            # End of sequence (also used as PAD)
TOTAL_VOCAB = 43
```

---

## Dual-Head Output

The decoder produces two outputs at each position:

1. **Discrete Head** - Softmax over 43 tokens for sequence decisions
2. **Latent Head** - 6D tanh output for articulatory control

**Why both?**
- Discrete provides clean cross-entropy loss for training
- Latent provides smooth motor commands for TTS synthesis
- Disagreement between heads indicates uncertainty

```python
# Training loss
discrete_loss = F.cross_entropy(discrete_logits, target_tokens, ignore_index=EOS_TOKEN)
latent_loss = F.mse_loss(latent_pred[phoneme_mask], target_anchors[phoneme_mask])
total_loss = discrete_loss + 0.5 * latent_loss
```

---

## Training Strategy

### Teacher Forcing

During training, feed ground truth tokens as input (shifted right):

```python
# Word: "hello" → [HH, EH, L, OW]
# Input:  [SOS, HH, EH, L, OW]      ← Teacher forcing
# Target: [HH, EH, L, OW, EOS]      ← What to predict
```

All positions computed in parallel via causal masking.

### Self-Listening Loop

The agent produces a word, hears itself, and learns from the experience:

```
1. HEAR TARGET
   Audio "hello" → Encoder → Pooled (384)

2. GENERATE SEQUENCE (autoregressive)
   [SOS] → Decoder → [HH]
   [SOS, HH] → Decoder → [EH]
   ...until [EOS]

3. SYNTHESIZE
   [HH, EH, L, OW] → Piper TTS → produced_audio

4. SELF-LISTEN
   produced_audio → Encoder → Pooled'
   Pooled' → Decoder → [HH', EH', L', OW']

5. COMPARE & LEARN
   - Sequence match: [HH,EH,L,OW] == [HH',EH',L',OW']?
   - Embedding similarity: cosine(Pooled, Pooled')
```

**Gradient handling:** Use teacher forcing on self-produced sequences to get gradients (TTS is not differentiable).

### Combined Training

```python
def combined_training_step(audio_mel, target_phonemes, self_listen_ratio=0.2):
    if random.random() < self_listen_ratio:
        # Self-listening: generate, synthesize, train on own production
        with torch.no_grad():
            produced_phonemes, _ = generate_sequence(audio_mel, temperature=1.0)
        produced_audio = tts.synthesize_phonemes(produced_phonemes)
        produced_mel = compute_mel_spectrogram(produced_audio)

        loss = train_step(produced_mel, produced_phonemes)
        consistency_loss = 1 - cosine_similarity(encode(produced_mel), encode(audio_mel))
        total_loss = loss + 0.1 * consistency_loss
    else:
        # Supervised: train on ground truth
        total_loss = train_step(audio_mel, target_phonemes)

    return total_loss
```

---

## Curriculum

Progressive training from simple to complex:

| Phase | Epochs | Words | Max Phonemes | Self-Listen | Temperature |
|-------|--------|-------|--------------|-------------|-------------|
| 1 | 1-30 | ba, bee, ma, me, hi, go, no | 3 | 10% | 0.0 (greedy) |
| 2 | 31-60 | hello, water, mommy, daddy, baby | 5 | 20% | 0.5 |
| 3 | 61+ | All available | 10 | 30% | 0.7 |

**Phase advancement:** Move to next phase when accuracy > 90% on current phase.

### Word Dataset

```python
WORD_PHONEMES = {
    # Phase 1: Simple syllables
    'ba':    ['B', 'AA'],
    'bee':   ['B', 'IY'],
    'ma':    ['M', 'AA'],
    'me':    ['M', 'IY'],
    'hi':    ['HH', 'AY'],
    'go':    ['G', 'OW'],
    'no':    ['N', 'OW'],

    # Phase 2: Short words
    'hello': ['HH', 'EH', 'L', 'OW'],
    'water': ['W', 'AO', 'T', 'ER'],
    'mommy': ['M', 'AA', 'M', 'IY'],
    'daddy': ['D', 'AE', 'D', 'IY'],
    'baby':  ['B', 'EY', 'B', 'IY'],

    # Phase 3: Longer words
    'banana':   ['B', 'AH', 'N', 'AE', 'N', 'AH'],
    'computer': ['K', 'AH', 'M', 'P', 'Y', 'UW', 'T', 'ER'],
    'elephant': ['EH', 'L', 'AH', 'F', 'AH', 'N', 'T'],
}
```

---

## File Structure

```
primordial/speech/
├── __init__.py              # Add new exports
├── config.py                # Add sequence decoder config
├── latent.py                # Add SOS_TOKEN, EOS_TOKEN anchors
├── sequence_decoder.py      # NEW: SequenceDecoder, SpeechSequenceLRN
├── training.py              # Add SequenceTrainer
└── (existing files unchanged)

primordial/scripts/
├── train_production_interactive.py  # Existing single-phoneme
└── train_sequence.py                # NEW: Sequence training CLI
```

---

## Parameter Budget

| Component | Parameters |
|-----------|------------|
| Encoder (existing, optional freeze) | ~800K |
| Memory Projection (384→128) | 49K |
| Phoneme Embedding (43×128) | 5.5K |
| Transformer Decoder (3 layers) | 540K |
| Discrete Head (128→43) | 5.5K |
| Latent Head (128→64→6) | 8.6K |
| **Total New** | **~610K** |
| **Total System** | **~1.4M** |

---

## Implementation Notes

### Causal Mask

```python
def _generate_causal_mask(seq_len, device):
    """Generate causal attention mask.

    Returns:
        Upper triangular matrix where True = masked (future tokens)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()
```

### Temperature Sampling

```python
def sample_with_temperature(logits, temperature=1.0):
    if temperature == 0:
        return logits.argmax()
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()
```

### Latent Loss Masking

Only compute latent loss on real phonemes (not SOS/EOS/PAD):

```python
phoneme_mask = (target_tokens < 41)  # Real phonemes only
latent_loss = F.mse_loss(latent_pred[phoneme_mask], target_anchors[phoneme_mask])
```

---

## Success Criteria

1. **Phase 1 Complete:** 90%+ accuracy on 2-phoneme syllables
2. **Phase 2 Complete:** 90%+ accuracy on 4-5 phoneme words
3. **Self-Consistency:** Agent recognizes >95% of its own productions
4. **Word Imitation:** Hear "hello" → produce [HH, EH, L, OW] → sounds like "hello"

---

## Future Extensions

1. **Multi-word sequences:** "hello world" as single production
2. **Prosody modeling:** Pitch/duration contours over sequence
3. **Semantic grounding:** Produce word from meaning (not just audio imitation)
4. **Continuous latent interpolation:** Smooth coarticulation between phonemes

---

## References

- Phase 2 Speech Production: `docs/plans/2025-12-02-phase2-speech-production.md`
- Session summary: `docs/SESSION-2025-12-02.md`
- Latent space: `primordial/speech/latent.py`
