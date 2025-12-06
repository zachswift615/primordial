# LibriSpeech Training V3 - More Data

## Context

Training V2 completed successfully:
- **No mode collapse** (fixed from V1)
- **Val PER: 92.9% → 86.3%** (6.6 point improvement)
- **Training stable** at 70-75% accuracy
- Best individual results ~50% PER on some utterances

The model is learning but needs more data. V2 only had **57 utterances** (heavily filtered for short phoneme sequences).

## What to Change

### 1. Relax Phoneme Filtering

In `primordial/speech/librispeech_dataset.py`, increase `max_phonemes`:

```python
# V2 settings (too restrictive):
# max_phonemes = 10

# V3 settings (more data):
max_phonemes = 15  # Allow slightly longer utterances
min_phonemes = 3   # Keep minimum to avoid too-short ones
max_duration = 3.0 # Allow up to 3 seconds (was 2.0)
```

This should give you **200-400 utterances** instead of 57.

### 2. Training Parameters

Keep most settings the same (they worked!), but run longer:

```python
epochs = 200        # Double the epochs (was 100)
learning_rate = 5e-6  # Keep the same (worked well)
max_real_ratio = 0.6  # Slightly higher ceiling (was 0.5)
ramp_epochs = 150     # Slower ramp to new ceiling
```

### 3. Start from V2 Checkpoint

Continue from the best V2 model (transfer learning):

```bash
python -m primordial.scripts.train_librispeech \
    --data ~/data/LibriSpeech \
    --checkpoint checkpoints/sequence/librispeech_best.pt \
    --output checkpoints/sequence/librispeech_v3_best.pt \
    --epochs 200 \
    --lr 5e-6 \
    --max-real-ratio 0.6 \
    --max-phonemes 15 \
    --max-duration 3.0 \
    2>&1 | tee training_log_v3.txt
```

## Expected Improvements

| Metric | V2 | V3 Target |
|--------|-----|-----------|
| Utterances | 57 | 200-400 |
| Val PER | 86.3% | <70% |
| Epochs | 100 | 200 |
| Max Real Ratio | 50% | 60% |

## Files to Modify

### `primordial/speech/librispeech_dataset.py`

Find the `__init__` method and update defaults:

```python
def __init__(
    self,
    root: Path,
    split: str = "train-clean-100",
    config: SpeechConfig = None,
    max_phonemes: int = 15,      # Changed from 10
    min_phonemes: int = 3,       # Changed from 2
    max_duration: float = 3.0,   # Changed from 2.0
    min_duration: float = 0.3,
):
```

### `primordial/scripts/train_librispeech.py`

Update argument defaults or pass via command line:

```python
parser.add_argument('--epochs', type=int, default=200)  # Was 100
parser.add_argument('--max-real-ratio', type=float, default=0.6)  # Was 0.5
parser.add_argument('--max-phonemes', type=int, default=15)  # Was 10
parser.add_argument('--max-duration', type=float, default=3.0)  # Was 2.0
```

## Verification

After starting, check the log shows more data:

```
# V2 showed:
INFO:primordial.speech.librispeech_dataset:LibriSpeechDataset: 57 utterances from train-clean-100

# V3 should show something like:
INFO:primordial.speech.librispeech_dataset:LibriSpeechDataset: 250 utterances from train-clean-100
```

If you still see ~57 utterances, the filtering changes didn't take effect.

## Command to Run

```bash
# Make sure to start from V2 best checkpoint
python -m primordial.scripts.train_librispeech \
    --data ~/data/LibriSpeech \
    --checkpoint checkpoints/sequence/librispeech_best.pt \
    --output checkpoints/sequence/librispeech_v3_best.pt \
    --epochs 200 \
    --lr 5e-6 \
    --max-real-ratio 0.6 \
    --max-phonemes 15 \
    --max-duration 3.0 \
    2>&1 | tee training_log_v3.txt
```

## Monitoring

After ~50 epochs, check:
1. Is training accuracy still >70%?
2. Is validation PER dropping below 80%?
3. Are demo outputs varied (no mode collapse)?

If PER isn't improving after 100 epochs, the model may need architectural changes (larger encoder, more layers, etc.) rather than just more data.

## Drive Time Estimate

Spring Hill → Nashville is ~45 min. At ~2 min/epoch with more data, you should see:
- ~20-25 epochs completed
- Early indication if the larger dataset is helping

Good luck with the drive! 🚗
