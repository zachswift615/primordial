# LibriSpeech Training Fixes - Mode Collapse Recovery

## Problem Summary

The current training run has mode collapsed. The model outputs nearly identical phoneme sequences regardless of input:

```
Generated: ['AH', 'N', 'D', 'DH', 'AH', 'S', 'T', 'UW', 'M', 'AH', 'N', 'D', 'IH', 'N', 'D']
Generated: ['AH', 'N', 'D', 'DH', 'AH', 'S', 'EH', 'R', 'IY', 'AH', 'N', 'D', 'IH', 'N', 'D']
```

**Evidence:**
- Training accuracy: 81% → 52% (getting worse)
- Validation PER: stuck at 84-85% (no improvement)
- All demo outputs start with "AH N D DH AH..."

## Root Causes

1. **Curriculum too aggressive**: 20% → 80% real data over 50 epochs
2. **Learning rate too high**: 5e-5 destroys learned features during fine-tuning
3. **Utterance length mismatch**: Model trained on 1-4 phoneme words, LibriSpeech has 15+ phoneme sentences

## Required Changes

### 1. Gentler Curriculum (in train_librispeech.py)

Change the real_ratio calculation to:
- Start at 20% real data
- Ramp to maximum 50% (never higher)
- Ramp slower (over 100 epochs instead of 50)

```python
# OLD (too aggressive):
# real_ratio = min(0.8, 0.2 + epoch * 0.012)

# NEW (gentler):
max_real_ratio = 0.5  # Never go above 50% real data
ramp_epochs = 100     # Take 100 epochs to reach max
real_ratio = min(max_real_ratio, 0.2 + epoch * (max_real_ratio - 0.2) / ramp_epochs)
```

### 2. Lower Learning Rate

Change learning rate from 5e-5 to 5e-6 (10x lower):

```python
# OLD:
# learning_rate = 5e-5

# NEW:
learning_rate = 5e-6  # Much gentler for fine-tuning
```

Also consider adding learning rate warmup:

```python
# Optional: warmup for first 5 epochs
if epoch < 5:
    lr = learning_rate * (epoch + 1) / 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### 3. Filter to Short Utterances

In `librispeech_dataset.py`, add filtering by phoneme count:

```python
class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train-clean-100",
        config: SpeechConfig = None,
        max_phonemes: int = 10,      # NEW: filter long utterances
        min_phonemes: int = 2,       # NEW: filter too-short ones
        max_duration: float = 2.0,   # Reduce from 5.0 to 2.0 seconds
        min_duration: float = 0.3,
    ):
        # ... existing init code ...

        # After loading samples, filter by phoneme count
        filtered_samples = []
        for audio_path, text in self.samples:
            phonemes = text_to_phonemes(text)
            if min_phonemes <= len(phonemes) <= max_phonemes:
                filtered_samples.append((audio_path, text, phonemes))

        self.samples = filtered_samples
        print(f"Filtered to {len(self.samples)} utterances with {min_phonemes}-{max_phonemes} phonemes")
```

### 4. Start Fresh from Piper Checkpoint

Don't continue from the collapsed `librispeech_best.pt`. Restart from the original Piper-trained checkpoint:

```bash
python -m primordial.scripts.train_librispeech \
    --checkpoint checkpoints/sequence/sequence_best.pt \  # Original Piper checkpoint
    --output checkpoints/sequence/librispeech_v2_best.pt \
    --epochs 100 \
    --lr 5e-6 \
    --max-real-ratio 0.5
```

## Updated Training Parameters

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| learning_rate | 5e-5 | 5e-6 | Prevent catastrophic forgetting |
| max_real_ratio | 0.8 | 0.5 | Keep enough synthetic data for stability |
| ramp_epochs | 50 | 100 | Slower transition to real data |
| max_phonemes | unlimited | 10 | Match training distribution |
| max_duration | 5.0s | 2.0s | Shorter utterances |

## Success Criteria

After these changes, you should see:
1. **Training accuracy stable** at 70%+ throughout training
2. **Validation PER decreasing** (not stuck at 84%)
3. **Demo outputs varying** - different inputs produce different outputs
4. **No repeated patterns** - outputs shouldn't all start with "AH N D DH AH"

## Files to Modify

1. `primordial/scripts/train_librispeech.py` - learning rate, curriculum
2. `primordial/speech/librispeech_dataset.py` - phoneme count filtering

## Verification

After 10 epochs with new settings, check:
```
- Is training accuracy > 70%?
- Is validation PER < 80%?
- Are demo outputs different from each other?
```

If any answer is no, may need to reduce learning rate further (1e-6) or reduce max_real_ratio to 0.3.

## Command to Run

```bash
# Make sure to start from original Piper checkpoint, not the collapsed one!
python -m primordial.scripts.train_librispeech \
    --data ~/data/LibriSpeech \
    --checkpoint checkpoints/sequence/sequence_best.pt \
    --output checkpoints/sequence/librispeech_v2_best.pt \
    --epochs 100 \
    --lr 5e-6 \
    --max-real-ratio 0.5 \
    --max-phonemes 10 \
    2>&1 | tee training_log_v2.txt
```
