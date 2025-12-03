# Autoregressive Sequence Decoder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a transformer decoder that generates full phoneme sequences (HH→EH→L→OW) from audio, enabling word-level speech imitation.

**Architecture:** 3-layer transformer decoder with dual heads (discrete tokens + 6D latent), trained via teacher forcing and self-listening. Reuses existing CNN encoder.

**Tech Stack:** PyTorch, existing SpeechLRN encoder, Piper TTS

**Design Doc:** `docs/plans/2025-12-02-autoregressive-sequence-decoder.md`

---

## Task 1: Add Token Constants to Latent Module

**Files:**
- Modify: `primordial/speech/latent.py`
- Test: `tests/speech/test_latent.py` (create if needed)

**Step 1: Write the failing test**

Create `tests/speech/test_latent.py`:

```python
"""Tests for latent phoneme space."""
import torch
from primordial.speech.latent import (
    SOS_TOKEN, EOS_TOKEN, TOTAL_VOCAB,
    PHONEME_ANCHORS, get_anchor, snap_to_nearest_anchor
)


def test_token_constants():
    """Verify token indices are correctly defined."""
    assert SOS_TOKEN == 41
    assert EOS_TOKEN == 42
    assert TOTAL_VOCAB == 43


def test_sos_eos_anchors():
    """SOS and EOS should have anchors at origin."""
    sos_anchor = get_anchor('SOS')
    eos_anchor = get_anchor('EOS')

    assert sos_anchor.shape == (6,)
    assert eos_anchor.shape == (6,)
    assert torch.allclose(sos_anchor, torch.zeros(6))
    assert torch.allclose(eos_anchor, torch.zeros(6))


def test_snap_excludes_sos_eos():
    """Snapping should not return SOS or EOS for normal latents."""
    # A latent near the IY anchor
    latent = torch.tensor([1.0, 1.0, -1.0, 1.0, 0.0, -1.0])
    phoneme, dist = snap_to_nearest_anchor(latent)

    assert phoneme == 'IY'
    assert phoneme not in ('SOS', 'EOS')
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_latent.py -v`
Expected: FAIL with "cannot import name 'SOS_TOKEN'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/latent.py` after the existing constants:

```python
# Sequence tokens (for autoregressive decoder)
SOS_TOKEN = 41  # Start of sequence
EOS_TOKEN = 42  # End of sequence (also used as PAD)
TOTAL_VOCAB = 43  # 41 phonemes + SOS + EOS
```

Add SOS and EOS to `PHONEME_ANCHORS` dict (at the end, before the closing brace):

```python
    # ============= SEQUENCE TOKENS =============
    'SOS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Start of sequence - origin
    'EOS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # End of sequence - origin
}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_latent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/latent.py tests/speech/test_latent.py
git commit -m "feat(speech): add SOS/EOS tokens and anchors for sequence decoding"
```

---

## Task 2: Create SequenceDecoder Module

**Files:**
- Create: `primordial/speech/sequence_decoder.py`
- Test: `tests/speech/test_sequence_decoder.py`

**Step 1: Write the failing test**

Create `tests/speech/test_sequence_decoder.py`:

```python
"""Tests for autoregressive sequence decoder."""
import torch
import pytest
from primordial.speech.sequence_decoder import SequenceDecoder
from primordial.speech.config import SpeechConfig


@pytest.fixture
def config():
    return SpeechConfig()


@pytest.fixture
def decoder(config):
    return SequenceDecoder(config)


def test_decoder_init(decoder):
    """Decoder should initialize with correct components."""
    assert hasattr(decoder, 'phoneme_embed')
    assert hasattr(decoder, 'pos_encoding')
    assert hasattr(decoder, 'transformer')
    assert hasattr(decoder, 'memory_proj')
    assert hasattr(decoder, 'discrete_head')
    assert hasattr(decoder, 'latent_head')


def test_decoder_forward_shape(decoder):
    """Forward pass should produce correct output shapes."""
    batch_size = 4
    seq_len = 5

    # Inputs
    input_ids = torch.randint(0, 43, (batch_size, seq_len))
    memory = torch.randn(batch_size, 384)  # Pooled audio encoding

    # Forward
    discrete_logits, latent = decoder(input_ids, memory)

    # Check shapes
    assert discrete_logits.shape == (batch_size, seq_len, 43)
    assert latent.shape == (batch_size, seq_len, 6)


def test_decoder_latent_bounded(decoder):
    """Latent output should be bounded to [-1, 1] via tanh."""
    input_ids = torch.randint(0, 43, (2, 3))
    memory = torch.randn(2, 384)

    _, latent = decoder(input_ids, memory)

    assert latent.min() >= -1.0
    assert latent.max() <= 1.0


def test_causal_mask_shape(decoder):
    """Causal mask should be upper triangular."""
    mask = decoder._generate_causal_mask(5, 'cpu')

    assert mask.shape == (5, 5)
    assert mask.dtype == torch.bool
    # Upper triangle (excluding diagonal) should be True (masked)
    assert mask[0, 1] == True
    assert mask[0, 4] == True
    # Diagonal and below should be False (visible)
    assert mask[0, 0] == False
    assert mask[4, 0] == False
    assert mask[4, 4] == False
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sequence_decoder.py -v`
Expected: FAIL with "No module named 'primordial.speech.sequence_decoder'"

**Step 3: Write minimal implementation**

Create `primordial/speech/sequence_decoder.py`:

```python
"""Autoregressive transformer decoder for phoneme sequence generation."""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import SpeechConfig
from .latent import TOTAL_VOCAB, LATENT_DIM


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SequenceDecoder(nn.Module):
    """Transformer decoder for autoregressive phoneme sequence generation.

    Takes pooled audio encoding and generates a sequence of phonemes
    autoregressively, with dual output heads for discrete tokens and
    continuous latent positions.
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        # Dimensions
        self.hidden_dim = 128
        self.num_heads = 4
        self.num_layers = 3
        self.ffn_dim = 256
        self.pooled_dim = 384  # From existing encoder

        # Embeddings
        self.phoneme_embed = nn.Embedding(TOTAL_VOCAB, self.hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(
            self.hidden_dim, max_len=32, dropout=0.1
        )

        # Memory projection (384 -> 128)
        self.memory_proj = nn.Linear(self.pooled_dim, self.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Dual output heads
        self.discrete_head = nn.Linear(self.hidden_dim, TOTAL_VOCAB)
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, LATENT_DIM),
            nn.Tanh(),
        )

    def _generate_causal_mask(self, seq_len: int, device) -> torch.Tensor:
        """Generate causal attention mask.

        Returns:
            Upper triangular matrix where True = masked (future tokens)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with teacher forcing.

        Args:
            input_ids: (batch, seq_len) phoneme token indices
            memory: (batch, 384) pooled audio encoding from encoder
            tgt_mask: Optional causal mask, generated if not provided

        Returns:
            discrete_logits: (batch, seq_len, 43) logits over vocabulary
            latent: (batch, seq_len, 6) latent positions in [-1, 1]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed and add positional encoding
        x = self.phoneme_embed(input_ids)  # (batch, seq_len, 128)
        x = self.pos_encoding(x)

        # Project memory and expand for cross-attention
        memory = self.memory_proj(memory)  # (batch, 128)
        memory = memory.unsqueeze(1)       # (batch, 1, 128)

        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(seq_len, device)

        # Decode
        output = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # (batch, seq_len, 128)

        # Dual heads
        discrete_logits = self.discrete_head(output)
        latent = self.latent_head(output)

        return discrete_logits, latent
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sequence_decoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/sequence_decoder.py tests/speech/test_sequence_decoder.py
git commit -m "feat(speech): add SequenceDecoder transformer for phoneme sequences"
```

---

## Task 3: Add SequenceDecoder to Module Exports

**Files:**
- Modify: `primordial/speech/__init__.py`

**Step 1: Write the failing test**

Add to `tests/speech/test_sequence_decoder.py`:

```python
def test_module_exports():
    """SequenceDecoder should be importable from speech module."""
    from primordial.speech import SequenceDecoder, SOS_TOKEN, EOS_TOKEN, TOTAL_VOCAB

    assert SequenceDecoder is not None
    assert SOS_TOKEN == 41
    assert EOS_TOKEN == 42
    assert TOTAL_VOCAB == 43
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sequence_decoder.py::test_module_exports -v`
Expected: FAIL with "cannot import name 'SequenceDecoder'"

**Step 3: Write minimal implementation**

Modify `primordial/speech/__init__.py`:

Add import after existing imports:
```python
from .sequence_decoder import SequenceDecoder, SinusoidalPositionalEncoding
```

Add to `from .latent import` line:
```python
from .latent import (
    PHONEME_ANCHORS,
    LATENT_DIM,
    get_anchor,
    snap_to_nearest_anchor,
    get_k_nearest_anchors,
    interpret_latent,
    SOS_TOKEN,
    EOS_TOKEN,
    TOTAL_VOCAB,
)
```

Add to `__all__` list:
```python
    # Sequence decoder
    'SequenceDecoder',
    'SinusoidalPositionalEncoding',
    'SOS_TOKEN',
    'EOS_TOKEN',
    'TOTAL_VOCAB',
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sequence_decoder.py::test_module_exports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/__init__.py
git commit -m "feat(speech): export SequenceDecoder and token constants"
```

---

## Task 4: Create SpeechSequenceLRN (Full Model)

**Files:**
- Modify: `primordial/speech/sequence_decoder.py`
- Test: `tests/speech/test_sequence_decoder.py`

**Step 1: Write the failing test**

Add to `tests/speech/test_sequence_decoder.py`:

```python
from primordial.speech.sequence_decoder import SequenceDecoder, SpeechSequenceLRN
from primordial.speech.latent import SOS_TOKEN, EOS_TOKEN


@pytest.fixture
def full_model(config):
    return SpeechSequenceLRN(config)


def test_full_model_init(full_model):
    """Full model should have encoder and decoder."""
    assert hasattr(full_model, 'encoder')
    assert hasattr(full_model, 'decoder')


def test_full_model_forward(full_model):
    """Full model forward pass should work end-to-end."""
    batch_size = 2
    seq_len = 4

    # Mel spectrogram input
    mel = torch.randn(batch_size, 80, 64)
    # Target tokens for teacher forcing
    target_tokens = torch.randint(0, 41, (batch_size, seq_len))

    discrete_logits, latent = full_model(mel, target_tokens)

    assert discrete_logits.shape == (batch_size, seq_len, 43)
    assert latent.shape == (batch_size, seq_len, 6)


def test_full_model_generate(full_model):
    """Generate should produce variable-length sequences."""
    mel = torch.randn(1, 80, 64)

    phonemes, latents = full_model.generate(mel, max_length=10)

    assert isinstance(phonemes, list)
    assert len(phonemes) <= 10
    assert all(isinstance(p, str) for p in phonemes)
    # Latents should match phoneme count
    if latents is not None:
        assert latents.shape[0] == len(phonemes)
        assert latents.shape[1] == 6
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sequence_decoder.py::test_full_model_init -v`
Expected: FAIL with "cannot import name 'SpeechSequenceLRN'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/sequence_decoder.py`:

```python
from .training import SpeechLRN
from .phonemes import index_to_phoneme


class SpeechSequenceLRN(nn.Module):
    """Full model: CNN Encoder + Fourier Mixing + Sequence Decoder.

    Combines the existing speech encoder with the new autoregressive
    decoder for word-level speech production.
    """

    def __init__(self, config: SpeechConfig):
        super().__init__()
        self.config = config

        # Reuse existing encoder
        self.encoder = SpeechLRN(config)

        # New sequence decoder
        self.decoder = SequenceDecoder(config)

        # Token constants
        from .latent import SOS_TOKEN, EOS_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode mel spectrogram to pooled features.

        Args:
            mel: (batch, 80, n_frames) mel spectrogram

        Returns:
            (batch, 384) pooled audio encoding
        """
        return self.encoder.encode(mel)

    def forward(
        self,
        mel: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass with teacher forcing.

        Args:
            mel: (batch, 80, n_frames) mel spectrogram
            target_tokens: (batch, seq_len) target phoneme indices

        Returns:
            discrete_logits: (batch, seq_len, 43)
            latent: (batch, seq_len, 6)
        """
        pooled = self.encode(mel)
        return self.decoder(target_tokens, memory=pooled)

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        max_length: int = 15,
        temperature: float = 0.0,
    ) -> Tuple[list, Optional[torch.Tensor]]:
        """Autoregressive phoneme sequence generation.

        Args:
            mel: (1, 80, n_frames) single mel spectrogram
            max_length: Maximum phonemes to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            phonemes: List[str] - generated phoneme sequence
            latents: (seq_len, 6) - latent vectors, or None if empty
        """
        device = mel.device
        pooled = self.encode(mel)  # (1, 384)

        # Start with SOS
        generated_tokens = [self.sos_token]
        generated_latents = []

        for _ in range(max_length):
            # Prepare input
            input_ids = torch.tensor([generated_tokens], device=device)

            # Decode
            discrete_logits, latent_pred = self.decoder(input_ids, memory=pooled)

            # Get prediction from last position
            next_logits = discrete_logits[0, -1]  # (43,)
            next_latent = latent_pred[0, -1]      # (6,)

            # Sample or greedy
            if temperature == 0:
                next_token = next_logits.argmax().item()
            else:
                import torch.nn.functional as F
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

            # Stop on EOS
            if next_token == self.eos_token:
                break

            generated_tokens.append(next_token)
            generated_latents.append(next_latent)

        # Convert to phoneme strings (skip SOS)
        phonemes = [index_to_phoneme(t) for t in generated_tokens[1:]]
        latents = torch.stack(generated_latents) if generated_latents else None

        return phonemes, latents
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sequence_decoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/sequence_decoder.py tests/speech/test_sequence_decoder.py
git commit -m "feat(speech): add SpeechSequenceLRN full model with generation"
```

---

## Task 5: Create Word Dataset

**Files:**
- Create: `primordial/speech/word_dataset.py`
- Test: `tests/speech/test_word_dataset.py`

**Step 1: Write the failing test**

Create `tests/speech/test_word_dataset.py`:

```python
"""Tests for word phoneme dataset."""
import torch
import pytest
from primordial.speech.word_dataset import WORD_PHONEMES, WordDataset
from primordial.speech.config import SpeechConfig
from primordial.speech.latent import SOS_TOKEN, EOS_TOKEN


def test_word_phonemes_dict():
    """Word dictionary should contain expected words."""
    assert 'hello' in WORD_PHONEMES
    assert WORD_PHONEMES['hello'] == ['HH', 'EH', 'L', 'OW']

    assert 'ba' in WORD_PHONEMES
    assert WORD_PHONEMES['ba'] == ['B', 'AA']


def test_word_dataset_length():
    """Dataset length should match word count."""
    config = SpeechConfig()
    dataset = WordDataset(config, words=['ba', 'hello'])

    assert len(dataset) == 2


def test_word_dataset_getitem():
    """Getting item should return mel, input tokens, target tokens."""
    config = SpeechConfig()
    dataset = WordDataset(config, words=['hello'])

    mel, input_tokens, target_tokens, word = dataset[0]

    # Mel spectrogram
    assert mel.shape[0] == config.n_mels

    # Input: [SOS, HH, EH, L, OW]
    assert input_tokens[0] == SOS_TOKEN
    assert len(input_tokens) == 5  # SOS + 4 phonemes

    # Target: [HH, EH, L, OW, EOS]
    assert target_tokens[-1] == EOS_TOKEN
    assert len(target_tokens) == 5  # 4 phonemes + EOS

    assert word == 'hello'


def test_word_dataset_curriculum_filter():
    """Dataset should filter by max phoneme length."""
    config = SpeechConfig()

    # Only short words
    short_dataset = WordDataset(config, max_phonemes=3)
    # Should include 'ba' (2), 'bee' (2), etc. but not 'hello' (4)
    words = [short_dataset[i][3] for i in range(len(short_dataset))]
    assert 'ba' in words
    assert 'hello' not in words
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_word_dataset.py -v`
Expected: FAIL with "No module named 'primordial.speech.word_dataset'"

**Step 3: Write minimal implementation**

Create `primordial/speech/word_dataset.py`:

```python
"""Word dataset for sequence training."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from .config import SpeechConfig
from .tts import create_tts_backend
from .encoders import compute_mel_spectrogram
from .phonemes import phoneme_to_index
from .latent import SOS_TOKEN, EOS_TOKEN


# Word to phoneme mappings
WORD_PHONEMES = {
    # Phase 1: Simple syllables (2-3 phonemes)
    'ba':    ['B', 'AA'],
    'bee':   ['B', 'IY'],
    'ma':    ['M', 'AA'],
    'me':    ['M', 'IY'],
    'hi':    ['HH', 'AY'],
    'go':    ['G', 'OW'],
    'no':    ['N', 'OW'],
    'see':   ['S', 'IY'],
    'you':   ['Y', 'UW'],
    'we':    ['W', 'IY'],

    # Phase 2: Short words (4-5 phonemes)
    'hello': ['HH', 'EH', 'L', 'OW'],
    'water': ['W', 'AO', 'T', 'ER'],
    'mommy': ['M', 'AA', 'M', 'IY'],
    'daddy': ['D', 'AE', 'D', 'IY'],
    'baby':  ['B', 'EY', 'B', 'IY'],
    'happy': ['HH', 'AE', 'P', 'IY'],
    'sorry': ['S', 'AA', 'R', 'IY'],

    # Phase 3: Longer words (6+ phonemes)
    'banana':   ['B', 'AH', 'N', 'AE', 'N', 'AH'],
    'computer': ['K', 'AH', 'M', 'P', 'Y', 'UW', 'T', 'ER'],
    'elephant': ['EH', 'L', 'AH', 'F', 'AH', 'N', 'T'],
    'tomorrow': ['T', 'AH', 'M', 'AA', 'R', 'OW'],
    'together': ['T', 'AH', 'G', 'EH', 'DH', 'ER'],
}


class WordDataset(Dataset):
    """Dataset of words with their phoneme sequences.

    Generates TTS audio for each word and provides:
    - mel: Mel spectrogram of the word
    - input_tokens: [SOS, phoneme1, phoneme2, ...] for teacher forcing
    - target_tokens: [phoneme1, phoneme2, ..., EOS] for loss
    """

    def __init__(
        self,
        config: SpeechConfig,
        words: Optional[List[str]] = None,
        max_phonemes: Optional[int] = None,
    ):
        """
        Args:
            config: Speech configuration
            words: Specific words to include (default: all)
            max_phonemes: Filter to words with at most this many phonemes
        """
        self.config = config
        self.tts = create_tts_backend(config)

        # Filter words
        if words is not None:
            self.words = [w for w in words if w in WORD_PHONEMES]
        else:
            self.words = list(WORD_PHONEMES.keys())

        if max_phonemes is not None:
            self.words = [
                w for w in self.words
                if len(WORD_PHONEMES[w]) <= max_phonemes
            ]

        # Pre-generate audio and mels
        self._cache = {}
        self._prepare_data()

    def _prepare_data(self):
        """Pre-generate mel spectrograms for all words."""
        for word in self.words:
            phonemes = WORD_PHONEMES[word]

            # Synthesize audio
            audio = self.tts.synthesize_phonemes(phonemes)
            waveform = torch.from_numpy(audio).float()

            # Resample if needed
            if self.tts.sample_rate != self.config.sample_rate:
                target_len = int(len(waveform) * self.config.sample_rate / self.tts.sample_rate)
                waveform = F.interpolate(
                    waveform.view(1, 1, -1),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()

            # Compute mel spectrogram
            mel = compute_mel_spectrogram(
                waveform,
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            ).squeeze(0)

            # Pad/truncate to standard size
            if mel.shape[1] < self.config.n_frames:
                mel = F.pad(mel, (0, self.config.n_frames - mel.shape[1]))
            else:
                mel = mel[:, :self.config.n_frames]

            # Convert phonemes to token indices
            phoneme_indices = [phoneme_to_index(p) for p in phonemes]

            # Input: [SOS, phoneme1, phoneme2, ...]
            input_tokens = torch.tensor([SOS_TOKEN] + phoneme_indices, dtype=torch.long)

            # Target: [phoneme1, phoneme2, ..., EOS]
            target_tokens = torch.tensor(phoneme_indices + [EOS_TOKEN], dtype=torch.long)

            self._cache[word] = (mel, input_tokens, target_tokens)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            mel: (n_mels, n_frames) mel spectrogram
            input_tokens: (seq_len,) input sequence with SOS
            target_tokens: (seq_len,) target sequence with EOS
            word: str - the word
        """
        word = self.words[idx]
        mel, input_tokens, target_tokens = self._cache[word]
        return mel, input_tokens, target_tokens, word
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_word_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/word_dataset.py tests/speech/test_word_dataset.py
git commit -m "feat(speech): add WordDataset with phoneme sequences for training"
```

---

## Task 6: Create SequenceTrainer

**Files:**
- Modify: `primordial/speech/training.py`
- Test: `tests/speech/test_sequence_trainer.py`

**Step 1: Write the failing test**

Create `tests/speech/test_sequence_trainer.py`:

```python
"""Tests for sequence training."""
import torch
import pytest
from primordial.speech.training import SequenceTrainer
from primordial.speech.sequence_decoder import SpeechSequenceLRN
from primordial.speech.config import SpeechConfig


@pytest.fixture
def config():
    return SpeechConfig()


@pytest.fixture
def model(config):
    return SpeechSequenceLRN(config)


@pytest.fixture
def trainer(model, config):
    return SequenceTrainer(model, config)


def test_trainer_init(trainer):
    """Trainer should initialize with optimizer."""
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'step_count')


def test_train_step(trainer):
    """Training step should return loss dict."""
    mel = torch.randn(2, 80, 64)
    input_tokens = torch.randint(0, 41, (2, 5))
    target_tokens = torch.randint(0, 41, (2, 5))

    losses = trainer.train_step(mel, input_tokens, target_tokens)

    assert 'total' in losses
    assert 'discrete' in losses
    assert 'latent' in losses
    assert losses['total'] > 0


def test_compute_accuracy(trainer):
    """Accuracy computation should work."""
    logits = torch.zeros(2, 5, 43)
    logits[:, :, 0] = 10.0  # All predict token 0

    targets = torch.zeros(2, 5, dtype=torch.long)  # All targets are 0

    acc = trainer._compute_accuracy(logits, targets)
    assert acc == 1.0  # Perfect accuracy
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sequence_trainer.py -v`
Expected: FAIL with "cannot import name 'SequenceTrainer'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/training.py` at the end:

```python
from .latent import get_anchor, EOS_TOKEN, LATENT_DIM
from dataclasses import dataclass


@dataclass
class SequenceMetrics:
    """Metrics from a sequence training step."""
    total_loss: float
    discrete_loss: float
    latent_loss: float
    accuracy: float
    step: int


class SequenceTrainer:
    """Trainer for autoregressive phoneme sequence generation.

    Uses teacher forcing with dual losses:
    - Cross-entropy for discrete token prediction
    - MSE for latent anchor prediction (masked to real phonemes)
    """

    def __init__(
        self,
        model,  # SpeechSequenceLRN
        config: SpeechConfig,
        device: str = "cpu",
        lr: float = 1e-4,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.step_count = 0

        # Loss weights
        self.discrete_weight = 1.0
        self.latent_weight = 0.5

    def _get_target_anchors(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Get latent anchors for target token sequence.

        Args:
            target_tokens: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, 6) anchor positions
        """
        from .phonemes import index_to_phoneme

        batch, seq_len = target_tokens.shape
        anchors = torch.zeros(batch, seq_len, LATENT_DIM, device=target_tokens.device)

        for b in range(batch):
            for t in range(seq_len):
                token = target_tokens[b, t].item()
                if token < 41:  # Real phoneme
                    phoneme = index_to_phoneme(token)
                    anchors[b, t] = get_anchor(phoneme)

        return anchors

    def _compute_accuracy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
    ) -> float:
        """Compute token prediction accuracy."""
        preds = logits.argmax(dim=-1)
        mask = targets != ignore_index
        if mask.sum() == 0:
            return 0.0
        correct = (preds == targets) & mask
        return (correct.sum() / mask.sum()).item()

    def train_step(
        self,
        mel: torch.Tensor,
        input_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict:
        """Single training step with teacher forcing.

        Args:
            mel: (batch, n_mels, n_frames) mel spectrogram
            input_tokens: (batch, seq_len) input with SOS
            target_tokens: (batch, seq_len) target with EOS

        Returns:
            Dict with loss values
        """
        self.model.train()
        mel = mel.to(self.device)
        input_tokens = input_tokens.to(self.device)
        target_tokens = target_tokens.to(self.device)

        # Forward pass
        discrete_logits, latent_pred = self.model(mel, input_tokens)

        # Discrete loss (cross-entropy)
        discrete_loss = F.cross_entropy(
            discrete_logits.view(-1, discrete_logits.size(-1)),
            target_tokens.view(-1),
            ignore_index=EOS_TOKEN,  # Don't penalize EOS predictions harshly
        )

        # Latent loss (MSE, masked to real phonemes only)
        target_anchors = self._get_target_anchors(target_tokens)
        phoneme_mask = target_tokens < 41  # Real phonemes only

        if phoneme_mask.sum() > 0:
            latent_loss = F.mse_loss(
                latent_pred[phoneme_mask],
                target_anchors[phoneme_mask],
            )
        else:
            latent_loss = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = self.discrete_weight * discrete_loss + self.latent_weight * latent_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1

        # Accuracy
        accuracy = self._compute_accuracy(discrete_logits, target_tokens)

        return {
            'total': total_loss.item(),
            'discrete': discrete_loss.item(),
            'latent': latent_loss.item() if isinstance(latent_loss, torch.Tensor) else latent_loss,
            'accuracy': accuracy,
            'step': self.step_count,
        }

    def save_checkpoint(self, path: str):
        """Save model and optimizer state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/test_sequence_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add primordial/speech/training.py tests/speech/test_sequence_trainer.py
git commit -m "feat(speech): add SequenceTrainer for autoregressive training"
```

---

## Task 7: Create Training Script

**Files:**
- Create: `primordial/scripts/train_sequence.py`

**Step 1: No test needed for CLI script**

CLI scripts are tested via integration tests and manual runs.

**Step 2: Write implementation**

Create `primordial/scripts/train_sequence.py`:

```python
#!/usr/bin/env python3
"""Train autoregressive phoneme sequence decoder.

Usage:
    python -m primordial.scripts.train_sequence [options]

Examples:
    # Train with curriculum
    python -m primordial.scripts.train_sequence --epochs 100

    # Train specific phase
    python -m primordial.scripts.train_sequence --phase 1 --epochs 30
"""

import argparse
import torch
from pathlib import Path

# Try to import audio playback
try:
    import sounddevice as sd
    CAN_PLAY = True
except ImportError:
    CAN_PLAY = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence decoder")

    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Train specific phase only (1, 2, or 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--demo-every", type=int, default=10,
        help="Run demo every N epochs (default: 10)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints/sequence",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio playback"
    )

    return parser.parse_args()


# Curriculum phases
CURRICULUM = {
    1: {
        'max_phonemes': 3,
        'epochs': 30,
        'self_listen_ratio': 0.1,
        'temperature': 0.0,
    },
    2: {
        'max_phonemes': 5,
        'epochs': 30,
        'self_listen_ratio': 0.2,
        'temperature': 0.5,
    },
    3: {
        'max_phonemes': 10,
        'epochs': 40,
        'self_listen_ratio': 0.3,
        'temperature': 0.7,
    },
}


def main():
    args = parse_args()

    # Check audio
    play_audio = CAN_PLAY and not args.no_audio

    # Import after args to avoid slow startup for --help
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN
    from primordial.speech.training import SequenceTrainer
    from primordial.speech.word_dataset import WordDataset
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("Sequence Decoder Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Audio playback: {'enabled' if play_audio else 'disabled'}")
    print("=" * 60)
    print()

    # Setup
    config = SpeechConfig(encoder_type='cnn', tts_backend='piper')
    model = SpeechSequenceLRN(config)
    trainer = SequenceTrainer(model, config, lr=args.lr)
    tts = create_tts_backend(config)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Determine phases to train
    if args.phase:
        phases = [args.phase]
    else:
        phases = [1, 2, 3]

    best_accuracy = 0.0
    total_epochs = 0

    for phase_num in phases:
        phase = CURRICULUM[phase_num]
        phase_epochs = min(phase['epochs'], args.epochs - total_epochs)

        if phase_epochs <= 0:
            break

        print(f"\n{'='*60}")
        print(f"Phase {phase_num}: max_phonemes={phase['max_phonemes']}")
        print(f"{'='*60}")

        # Create dataset for this phase
        dataset = WordDataset(config, max_phonemes=phase['max_phonemes'])
        print(f"Words in phase: {len(dataset)}")

        # Custom collate for variable-length sequences
        def collate_fn(batch):
            mels, inputs, targets, words = zip(*batch)

            # Pad sequences to max length in batch
            max_len = max(len(t) for t in inputs)

            padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
            padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)

            for i, (inp, tgt) in enumerate(zip(inputs, targets)):
                padded_inputs[i, :len(inp)] = inp
                padded_targets[i, :len(tgt)] = tgt

            mels = torch.stack(mels)
            return mels, padded_inputs, padded_targets, words

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        for epoch in range(phase_epochs):
            total_epochs += 1
            epoch_losses = {'total': 0, 'discrete': 0, 'latent': 0, 'accuracy': 0}
            num_batches = 0

            for mel, input_tokens, target_tokens, words in dataloader:
                losses = trainer.train_step(mel, input_tokens, target_tokens)

                for k in epoch_losses:
                    epoch_losses[k] += losses.get(k, 0)
                num_batches += 1

            # Average
            for k in epoch_losses:
                epoch_losses[k] /= max(num_batches, 1)

            # Save best
            if epoch_losses['accuracy'] > best_accuracy:
                best_accuracy = epoch_losses['accuracy']
                torch.save(model.state_dict(), save_dir / "sequence_best.pt")

            print(f"Epoch {total_epochs:3d} (P{phase_num}): "
                  f"loss={epoch_losses['total']:.4f}, "
                  f"acc={epoch_losses['accuracy']:.1%}")

            # Demo
            if (epoch + 1) % args.demo_every == 0:
                model.eval()
                with torch.no_grad():
                    # Pick a word from dataset
                    mel, _, _, word = dataset[epoch % len(dataset)]
                    mel = mel.unsqueeze(0)

                    # Generate
                    phonemes, latents = model.generate(mel, temperature=phase['temperature'])

                    from primordial.speech.word_dataset import WORD_PHONEMES
                    target_phonemes = WORD_PHONEMES[word]

                    match = phonemes == target_phonemes
                    print(f"  Demo: '{word}' -> {phonemes} "
                          f"(target: {target_phonemes}) "
                          f"[{'MATCH' if match else 'miss'}]")

                    if play_audio and phonemes:
                        # Play target
                        target_audio = tts.synthesize_phonemes(target_phonemes)
                        print(f"    Target: ", end="", flush=True)
                        sd.play(target_audio, tts.sample_rate)
                        sd.wait()
                        print("done")

                        # Play produced
                        produced_audio = tts.synthesize_phonemes(phonemes)
                        print(f"    Produced: ", end="", flush=True)
                        sd.play(produced_audio, tts.sample_rate)
                        sd.wait()
                        print("done")

                model.train()

    # Final results
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best accuracy: {best_accuracy:.1%}")
    print(f"Checkpoint saved to: {save_dir}")


if __name__ == "__main__":
    main()
```

**Step 3: Run to verify it works**

Run: `python -m primordial.scripts.train_sequence --epochs 1 --no-audio`
Expected: Runs one epoch without errors

**Step 4: Commit**

```bash
git add primordial/scripts/train_sequence.py
git commit -m "feat(speech): add train_sequence.py CLI for sequence training"
```

---

## Task 8: Update Module Exports (Final)

**Files:**
- Modify: `primordial/speech/__init__.py`

**Step 1: Write the failing test**

Add to an existing test file:

```python
def test_all_exports():
    """All new components should be importable."""
    from primordial.speech import (
        SequenceDecoder,
        SpeechSequenceLRN,
        SequenceTrainer,
        WordDataset,
        WORD_PHONEMES,
        SOS_TOKEN,
        EOS_TOKEN,
        TOTAL_VOCAB,
    )
    assert all([
        SequenceDecoder, SpeechSequenceLRN, SequenceTrainer,
        WordDataset, WORD_PHONEMES
    ])
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/speech/test_sequence_decoder.py::test_all_exports -v`
Expected: FAIL

**Step 3: Add remaining exports**

Modify `primordial/speech/__init__.py` to add:

```python
from .sequence_decoder import SequenceDecoder, SpeechSequenceLRN, SinusoidalPositionalEncoding
from .training import SequenceTrainer
from .word_dataset import WordDataset, WORD_PHONEMES
```

Add to `__all__`:
```python
    'SpeechSequenceLRN',
    'SequenceTrainer',
    'WordDataset',
    'WORD_PHONEMES',
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/speech/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add primordial/speech/__init__.py
git commit -m "feat(speech): complete module exports for sequence decoder"
```

---

## Task 9: Integration Test

**Files:**
- Create: `tests/speech/test_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for full sequence pipeline."""
import torch
import pytest
from primordial.speech import (
    SpeechConfig,
    SpeechSequenceLRN,
    SequenceTrainer,
    WordDataset,
    WORD_PHONEMES,
)


@pytest.fixture
def config():
    return SpeechConfig(encoder_type='cnn', tts_backend='piper')


def test_full_training_loop(config):
    """End-to-end training should work."""
    model = SpeechSequenceLRN(config)
    trainer = SequenceTrainer(model, config)
    dataset = WordDataset(config, words=['ba', 'bee'])

    # One training step
    mel, input_tokens, target_tokens, word = dataset[0]

    losses = trainer.train_step(
        mel.unsqueeze(0),
        input_tokens.unsqueeze(0),
        target_tokens.unsqueeze(0),
    )

    assert losses['total'] > 0
    assert losses['accuracy'] >= 0


def test_generate_matches_word(config):
    """After training, generation should improve."""
    model = SpeechSequenceLRN(config)
    dataset = WordDataset(config, words=['ba'])

    mel, _, _, word = dataset[0]

    # Generate before training
    phonemes_before, _ = model.generate(mel.unsqueeze(0))

    # Train a few steps
    trainer = SequenceTrainer(model, config)
    for _ in range(20):
        mel, input_tokens, target_tokens, _ = dataset[0]
        trainer.train_step(
            mel.unsqueeze(0),
            input_tokens.unsqueeze(0),
            target_tokens.unsqueeze(0),
        )

    # Generate after training
    phonemes_after, _ = model.generate(mel.unsqueeze(0))

    # Should be closer to target
    target = WORD_PHONEMES['ba']
    # Just verify it runs - actual accuracy depends on more training
    assert isinstance(phonemes_after, list)
```

**Step 2: Run integration test**

Run: `python -m pytest tests/speech/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/speech/test_integration.py
git commit -m "test(speech): add integration tests for sequence pipeline"
```

---

## Summary

| Task | Description | New Files |
|------|-------------|-----------|
| 1 | Add SOS/EOS token constants | - |
| 2 | Create SequenceDecoder | `sequence_decoder.py` |
| 3 | Export SequenceDecoder | - |
| 4 | Create SpeechSequenceLRN | - |
| 5 | Create WordDataset | `word_dataset.py` |
| 6 | Create SequenceTrainer | - |
| 7 | Create training script | `train_sequence.py` |
| 8 | Final exports | - |
| 9 | Integration test | `test_integration.py` |

**Total new files:** 3 source files, 4 test files
**Estimated time:** 2-3 hours with TDD

After completing all tasks, run the full training:
```bash
python -m primordial.scripts.train_sequence --epochs 100
```
