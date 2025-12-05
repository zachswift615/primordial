# LibriSpeech Speaker Diversity Training

## Context

We have a working speech recognition model (`SpeechSequenceLRN`) that:
- Encodes mel spectrograms via CNN encoder
- Decodes to phoneme sequences via transformer decoder
- Achieves high accuracy on synthetic Piper TTS audio

**Problem:** The model is overfitted to Piper's voice. When tested with real human speech (interactive demo), it produces nonsense phoneme sequences. It has no speaker invariance.

**Solution:** Fine-tune on LibriSpeech dataset with diverse speakers to learn speaker-independent acoustic features.

## Files to Download

From https://www.openslr.org/12/ download:

### Recommended: Start Small
```
train-clean-100.tar.gz  (6.3 GB) - 100 hours, ~250 speakers, clean audio
dev-clean.tar.gz        (337 MB) - validation set, clean audio
```

### Later: Scale Up
```
train-clean-360.tar.gz  (23 GB)  - 360 hours, ~900 speakers
train-other-500.tar.gz  (30 GB)  - 500 hours, noisier (good for robustness)
```

### Required Metadata
The transcripts are included in the tar files. Each audio file has a corresponding `.trans.txt` with word-level transcriptions.

Structure after extraction:
```
LibriSpeech/
├── train-clean-100/
│   ├── <speaker_id>/
│   │   ├── <chapter_id>/
│   │   │   ├── <speaker>-<chapter>-<utterance>.flac
│   │   │   └── <speaker>-<chapter>.trans.txt
```

## Existing Codebase

Key files to understand:
- `primordial/speech/sequence_decoder.py` - `SpeechSequenceLRN` model
- `primordial/speech/word_dataset.py` - Current dataset (Piper synthetic only)
- `primordial/speech/training.py` - `SequenceTrainer` class
- `primordial/scripts/train_sequence.py` - Training script
- `primordial/speech/encoders.py` - `compute_mel_spectrogram()` function
- `primordial/speech/config.py` - `SpeechConfig` dataclass

Current training config:
```python
SpeechConfig(
    sample_rate=16000,      # LibriSpeech is 16kHz - perfect match!
    n_mels=80,
    n_fft=400,
    hop_length=160,
    encoder_type='cnn',
)
```

## Training Design

### Phase 1: LibriSpeech Dataset Class

Create `primordial/speech/librispeech_dataset.py`:

```python
"""LibriSpeech dataset for speaker-diverse training."""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
from typing import List, Tuple, Optional
import re

# You'll need a text-to-phoneme converter
# Options:
#   1. g2p_en: pip install g2p_en (simple, good enough)
#   2. phonemizer: pip install phonemizer (more accurate, needs espeak)
#   3. CMU dict lookup (fast but limited vocabulary)

class LibriSpeechDataset(Dataset):
    """Dataset for LibriSpeech audio with phoneme targets."""

    def __init__(
        self,
        root: Path,
        split: str = "train-clean-100",
        config: SpeechConfig,
        max_duration: float = 5.0,  # Skip very long utterances
        min_duration: float = 0.5,  # Skip very short ones
    ):
        self.root = Path(root) / split
        self.config = config
        self.samples = []  # List of (audio_path, transcript)

        # Scan directory structure
        for trans_file in self.root.rglob("*.trans.txt"):
            chapter_dir = trans_file.parent
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        audio_path = chapter_dir / f"{utterance_id}.flac"
                        if audio_path.exists():
                            self.samples.append((audio_path, text))

        # Filter by duration (requires loading - could cache)
        # ... implementation details

    def __getitem__(self, idx) -> Tuple[torch.Tensor, List[int]]:
        audio_path, text = self.samples[idx]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.config.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.config.sample_rate)(waveform)

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(waveform, ...)

        # Convert text to phonemes
        phonemes = text_to_phonemes(text)  # Need to implement
        token_ids = [phoneme_to_index(p) for p in phonemes]

        return mel, token_ids
```

### Phase 2: Text-to-Phoneme Conversion

Option A - Use g2p_en (recommended for simplicity):
```python
from g2p_en import G2p
g2p = G2p()

def text_to_phonemes(text: str) -> List[str]:
    """Convert text to ARPABET phonemes."""
    phonemes = g2p(text)
    # g2p returns list like ['HH', 'AH0', 'L', 'OW1']
    # Strip stress markers (0,1,2) to match our inventory
    return [re.sub(r'[012]', '', p) for p in phonemes if p.strip()]
```

Option B - Use CMU dict (faster, no dependencies):
```python
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
CMU = cmudict.dict()

def text_to_phonemes(text: str) -> List[str]:
    phonemes = []
    for word in text.lower().split():
        word = re.sub(r'[^a-z]', '', word)
        if word in CMU:
            # Take first pronunciation, strip stress
            pron = CMU[word][0]
            phonemes.extend([re.sub(r'[012]', '', p) for p in pron])
    return phonemes
```

### Phase 3: Training Strategy

**Mixed Training (Recommended):**
```python
# Combine synthetic (Piper) and real (LibriSpeech) data
# Start with more synthetic, gradually increase real

class MixedSpeechDataset(Dataset):
    def __init__(self, synthetic_dataset, librispeech_dataset, real_ratio=0.5):
        self.synthetic = synthetic_dataset
        self.real = librispeech_dataset
        self.real_ratio = real_ratio

    def __getitem__(self, idx):
        if random.random() < self.real_ratio:
            return self.real[idx % len(self.real)]
        else:
            return self.synthetic[idx % len(self.synthetic)]
```

**Curriculum:**
1. Epoch 0-5: 20% LibriSpeech, 80% Piper (preserve learned patterns)
2. Epoch 5-15: 50% LibriSpeech, 50% Piper
3. Epoch 15+: 80% LibriSpeech, 20% Piper

**Data Augmentation:**
```python
import torchaudio.transforms as T

def augment_audio(waveform, sample_rate):
    """Apply random augmentations for robustness."""
    # Pitch shift (±2 semitones)
    if random.random() < 0.3:
        shift = random.uniform(-2, 2)
        waveform = T.PitchShift(sample_rate, shift)(waveform)

    # Time stretch (0.9x - 1.1x)
    if random.random() < 0.3:
        rate = random.uniform(0.9, 1.1)
        waveform = T.TimeStretch(rate)(waveform)

    # Add noise
    if random.random() < 0.2:
        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

    return waveform
```

### Phase 4: Evaluation

Create evaluation script to test on held-out speakers:
```python
def evaluate_speaker_invariance(model, dataset):
    """Test recognition accuracy across different speakers."""
    results_by_speaker = defaultdict(list)

    for mel, target_phonemes, speaker_id in dataset:
        predicted, _ = model.generate(mel)
        # Compute phoneme error rate (PER)
        per = phoneme_error_rate(predicted, target_phonemes)
        results_by_speaker[speaker_id].append(per)

    # Report per-speaker and aggregate metrics
    for speaker, pers in results_by_speaker.items():
        print(f"Speaker {speaker}: PER = {np.mean(pers):.2%}")
```

## Checkpoints

Current best model: `checkpoints/sequence/sequence_best.pt`
- Trained on Piper synthetic data only
- Use as initialization for fine-tuning (transfer learning)

New checkpoints should go to: `checkpoints/sequence/librispeech_*.pt`

## Success Criteria

1. **Phoneme Error Rate (PER) < 30%** on dev-clean (real speakers)
2. **Consistent outputs** for same utterance across recordings
3. **Interactive demo works** with user's voice

## Dependencies to Install

```bash
pip install g2p_en          # Text-to-phoneme
pip install torchaudio      # Audio loading (probably already installed)
# OR
pip install nltk            # For CMU dict approach
```

## Suggested File Structure

```
primordial/speech/
├── librispeech_dataset.py  # New: LibriSpeech loader
├── g2p.py                  # New: Text-to-phoneme utilities
├── augmentation.py         # New: Audio augmentation
└── ...existing files...

primordial/scripts/
├── train_librispeech.py    # New: Training script
├── download_librispeech.py # New: Download helper (optional)
└── evaluate_speakers.py    # New: Speaker invariance eval
```

## Quick Start Command

After implementing:
```bash
# Download (or manually download and extract to ~/data/LibriSpeech/)
# wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

# Train
python -m primordial.scripts.train_librispeech \
    --data ~/data/LibriSpeech \
    --checkpoint checkpoints/sequence/sequence_best.pt \
    --output checkpoints/sequence/librispeech_best.pt
```

## Questions for Implementation

1. **g2p_en vs CMU dict?** - g2p_en handles OOV words better, CMU is faster
2. **Fine-tune or train from scratch?** - Fine-tune from Piper checkpoint (transfer learning)
3. **How much data?** - Start with train-clean-100, scale up if needed
4. **Batch handling?** - Variable length sequences need padding/collation
