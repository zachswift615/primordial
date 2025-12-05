# Multi-Word Sequences with Self-Listening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable multi-word phrase training with acoustic validation for curriculum gating.

**Architecture:** Extend existing SequenceTrainer with `compute_acoustic_match()` function. Update training script to use phrase dataset and dual-gated curriculum advancement.

**Tech Stack:** PyTorch, Piper TTS, existing SpeechLRN/SequenceDecoder

---

## Task 1: Add compute_acoustic_match Function

**Files:**
- Modify: `primordial/speech/training.py` (add function after line 797, before SequenceTrainer class)
- Test: `tests/speech/test_sequence_trainer.py`

**Step 1: Write the failing test**

Add to `tests/speech/test_sequence_trainer.py`:

```python
def test_compute_acoustic_match_identical_audio():
    """Acoustic match of identical audio should be ~1.0."""
    from primordial.speech.training import compute_acoustic_match
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    tts = create_tts_backend(config)

    # Generate audio for same phonemes twice
    phonemes = ['HH', 'EH', 'L', 'OW']
    target_audio = tts.synthesize_phonemes(phonemes)

    score = compute_acoustic_match(model, tts, config, phonemes, target_audio)

    assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    assert score > 0.9, f"Identical audio should match >0.9, got {score}"


def test_compute_acoustic_match_different_phonemes():
    """Different phonemes should have lower acoustic match."""
    from primordial.speech.training import compute_acoustic_match
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    tts = create_tts_backend(config)

    # Generate audio for different phonemes
    target_phonemes = ['HH', 'EH', 'L', 'OW']  # "hello"
    generated_phonemes = ['B', 'AY']  # "bye"
    target_audio = tts.synthesize_phonemes(target_phonemes)

    score = compute_acoustic_match(model, tts, config, generated_phonemes, target_audio)

    assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    assert score < 0.8, f"Different phonemes should match <0.8, got {score}"


def test_compute_acoustic_match_empty_phonemes():
    """Empty phoneme list should return 0.0."""
    from primordial.speech.training import compute_acoustic_match
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    tts = create_tts_backend(config)

    target_audio = tts.synthesize_phonemes(['HH', 'AY'])

    score = compute_acoustic_match(model, tts, config, [], target_audio)

    assert score == 0.0, f"Empty phonemes should return 0.0, got {score}"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/speech/test_sequence_trainer.py::test_compute_acoustic_match_identical_audio -v`

Expected: FAIL with "cannot import name 'compute_acoustic_match'"

**Step 3: Write minimal implementation**

Add to `primordial/speech/training.py` before the `class SequenceTrainer:` line (around line 798):

```python
def compute_acoustic_match(
    model,  # SpeechSequenceLRN or SpeechLRN with encoder
    tts,  # TTSBackend
    config: SpeechConfig,
    generated_phonemes: list,
    target_audio: np.ndarray,
) -> float:
    """Compute acoustic similarity between generated and target audio.

    Returns cosine similarity (0-1) for logging and curriculum gating.
    No gradients - purely diagnostic.

    Args:
        model: Model with encoder attribute
        tts: TTS backend for synthesis
        config: Speech configuration
        generated_phonemes: List of phoneme strings from generation
        target_audio: Target audio waveform (numpy array)

    Returns:
        Similarity score 0.0-1.0 (1.0 = perfect match)
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .encoders import compute_mel_spectrogram

    # Handle empty phonemes
    if not generated_phonemes:
        return 0.0

    # Synthesize generated sequence
    produced_audio = tts.synthesize_phonemes(generated_phonemes)

    # Convert to tensors
    produced_waveform = torch.from_numpy(produced_audio).float()
    target_waveform = torch.from_numpy(target_audio).float()

    # Resample if needed
    if tts.sample_rate != config.sample_rate:
        produced_len = int(len(produced_waveform) * config.sample_rate / tts.sample_rate)
        produced_waveform = F.interpolate(
            produced_waveform.view(1, 1, -1),
            size=produced_len,
            mode='linear',
            align_corners=False
        ).squeeze()

        target_len = int(len(target_waveform) * config.sample_rate / tts.sample_rate)
        target_waveform = F.interpolate(
            target_waveform.view(1, 1, -1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).squeeze()

    # Compute mel spectrograms
    produced_mel = compute_mel_spectrogram(
        produced_waveform,
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    ).squeeze(0)

    target_mel = compute_mel_spectrogram(
        target_waveform,
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    ).squeeze(0)

    # Pad/truncate to standard size
    def normalize_mel(mel, n_frames):
        if mel.shape[1] < n_frames:
            mel = F.pad(mel, (0, n_frames - mel.shape[1]))
        else:
            mel = mel[:, :n_frames]
        return mel

    produced_mel = normalize_mel(produced_mel, config.n_frames)
    target_mel = normalize_mel(target_mel, config.n_frames)

    # Encode through model (no gradients)
    with torch.no_grad():
        # Get encoder - handle both SpeechSequenceLRN and SpeechLRN
        encoder = getattr(model, 'encoder', model)

        # Add batch dimension
        produced_mel = produced_mel.unsqueeze(0)
        target_mel = target_mel.unsqueeze(0)

        # Forward through encoder
        produced_features = encoder(produced_mel)
        target_features = encoder(target_mel)

        # Pool if needed (encoder returns (batch, seq_len, hidden_dim))
        if produced_features.dim() == 3:
            produced_features = produced_features.mean(dim=1)
            target_features = target_features.mean(dim=1)

        # Flatten
        produced_features = produced_features.flatten()
        target_features = target_features.flatten()

        # Cosine similarity
        similarity = F.cosine_similarity(
            produced_features.unsqueeze(0),
            target_features.unsqueeze(0)
        ).item()

        # Clamp to 0-1 range (cosine can be negative)
        return max(0.0, similarity)
```

Also add the import at the top of `training.py`:

```python
import numpy as np
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/speech/test_sequence_trainer.py::test_compute_acoustic_match_identical_audio tests/speech/test_sequence_trainer.py::test_compute_acoustic_match_different_phonemes tests/speech/test_sequence_trainer.py::test_compute_acoustic_match_empty_phonemes -v`

Expected: 3 PASSED

**Step 5: Commit**

```bash
git add primordial/speech/training.py tests/speech/test_sequence_trainer.py
git commit -m "feat(speech): add compute_acoustic_match for self-listening validation"
```

---

## Task 2: Add Tests for Phrase Dataset Loading

**Files:**
- Test: `tests/speech/test_word_dataset.py`

**Step 1: Write the failing tests**

Add to `tests/speech/test_word_dataset.py`:

```python
def test_phrase_phonemes_exist():
    """PHRASE_PHONEMES should contain multi-word phrases."""
    from primordial.speech.word_dataset import PHRASE_PHONEMES

    assert len(PHRASE_PHONEMES) > 0, "PHRASE_PHONEMES should not be empty"
    assert "hello world" in PHRASE_PHONEMES, "Should contain 'hello world'"
    assert len(PHRASE_PHONEMES["hello world"]) == 8, "hello world = 8 phonemes"


def test_get_all_entries_includes_phrases():
    """get_all_entries should merge words and phrases."""
    from primordial.speech.word_dataset import (
        get_all_entries, WORD_PHONEMES, PHRASE_PHONEMES
    )

    all_entries = get_all_entries(include_phrases=True)

    assert len(all_entries) == len(WORD_PHONEMES) + len(PHRASE_PHONEMES)
    assert "hello" in all_entries  # word
    assert "hello world" in all_entries  # phrase


def test_get_all_entries_excludes_phrases():
    """get_all_entries(include_phrases=False) should only return words."""
    from primordial.speech.word_dataset import get_all_entries, WORD_PHONEMES

    entries = get_all_entries(include_phrases=False)

    assert len(entries) == len(WORD_PHONEMES)
    assert "hello" in entries
    assert "hello world" not in entries


def test_word_dataset_max_phonemes_filters_phrases():
    """WordDataset with max_phonemes should filter correctly."""
    from primordial.speech.word_dataset import WordDataset, PHRASE_PHONEMES
    from primordial.speech import SpeechConfig

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')

    # max_phonemes=6 should exclude "hello world" (8 phonemes)
    dataset = WordDataset(config, max_phonemes=6, include_phrases=True)

    assert "hello world" not in dataset.words
    assert "bye bye" in dataset.words  # 4 phonemes

    # Verify we got some phrases
    phrase_count = sum(1 for w in dataset.words if " " in w)
    assert phrase_count > 0, "Should include some short phrases"
```

**Step 2: Run tests to verify they pass**

These should already pass since the phrase data was added earlier.

Run: `pytest tests/speech/test_word_dataset.py -v`

Expected: All tests PASS

**Step 3: Commit (if any changes were needed)**

```bash
git add tests/speech/test_word_dataset.py
git commit -m "test(speech): add tests for phrase dataset loading"
```

---

## Task 3: Update Curriculum with Dual Gating

**Files:**
- Modify: `primordial/scripts/train_sequence.py`

**Step 1: Update CURRICULUM constant**

Replace the existing `CURRICULUM` dict (lines 74-100) with:

```python
# Curriculum phases with dual gating (token accuracy + acoustic match)
# Total: 40 + 40 + 50 + 70 = 200 epochs
CURRICULUM = {
    1: {
        'max_phonemes': 8,
        'epochs': 40,
        'include_phrases': True,
        'token_threshold': 0.85,
        'acoustic_threshold': 0.80,
        'temperature': 0.0,
        'description': 'Single words + 2-word phrases',
    },
    2: {
        'max_phonemes': 15,
        'epochs': 40,
        'include_phrases': True,
        'token_threshold': 0.85,
        'acoustic_threshold': 0.80,
        'temperature': 0.5,
        'description': '+ 3-word phrases',
    },
    3: {
        'max_phonemes': 22,
        'epochs': 50,
        'include_phrases': True,
        'token_threshold': 0.80,
        'acoustic_threshold': 0.75,
        'temperature': 0.5,
        'description': '+ short sentences',
    },
    4: {
        'max_phonemes': 30,
        'epochs': 70,
        'include_phrases': True,
        'token_threshold': 0.0,  # No gate
        'acoustic_threshold': 0.0,  # No gate
        'temperature': 0.7,
        'description': 'All data',
    },
}
```

**Step 2: Verify script still runs**

Run: `python -m primordial.scripts.train_sequence --help`

Expected: Help text displays without errors

**Step 3: Commit**

```bash
git add primordial/scripts/train_sequence.py
git commit -m "feat(speech): update curriculum with dual gating thresholds"
```

---

## Task 4: Add Acoustic Match Tracking to Training Loop

**Files:**
- Modify: `primordial/scripts/train_sequence.py`

**Step 1: Add acoustic match CLI argument**

Add after line 68 (after `--no-audio` argument):

```python
    parser.add_argument(
        "--acoustic-check-interval", type=int, default=10,
        help="Compute acoustic match every N batches (default: 10)"
    )
```

**Step 2: Import compute_acoustic_match**

Update the import block inside `main()` (around line 110-114) to:

```python
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN
    from primordial.speech.training import SequenceTrainer, compute_acoustic_match
    from primordial.speech.word_dataset import WordDataset, WORD_PHONEMES, get_all_entries
    from torch.utils.data import DataLoader
    import numpy as np
```

**Step 3: Update dataset creation to include phrases**

Replace line 200:
```python
        dataset = WordDataset(config, max_phonemes=phase['max_phonemes'])
```

With:
```python
        dataset = WordDataset(
            config,
            max_phonemes=phase['max_phonemes'],
            include_phrases=phase.get('include_phrases', False),
        )
```

**Step 4: Add acoustic match tracking to training loop**

Replace the training loop (lines 231-254) with:

```python
        for epoch in range(phase_epochs):
            total_epochs += 1
            epoch_losses = {'total': 0, 'discrete': 0, 'latent': 0, 'accuracy': 0}
            acoustic_scores = []
            num_batches = 0

            for batch_idx, (mel, input_tokens, target_tokens, words) in enumerate(dataloader):
                losses = trainer.train_step(mel, input_tokens, target_tokens)

                for k in epoch_losses:
                    epoch_losses[k] += losses.get(k, 0)
                num_batches += 1

                # Periodic acoustic match check
                if batch_idx % args.acoustic_check_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        # Check first sample in batch
                        sample_mel = mel[0:1]
                        generated, _ = model.generate(sample_mel, temperature=phase['temperature'])

                        # Get target audio for this word
                        word = words[0]
                        all_entries = get_all_entries(include_phrases=True)
                        target_phonemes = all_entries.get(word, [])
                        target_audio = tts.synthesize_phonemes(target_phonemes)

                        # Compute acoustic match
                        if generated and len(generated) > 0:
                            score = compute_acoustic_match(
                                model, tts, config, generated, target_audio
                            )
                        else:
                            score = 0.0  # Failed generation

                        acoustic_scores.append(score)
                    model.train()

            # Average
            for k in epoch_losses:
                epoch_losses[k] /= max(num_batches, 1)

            avg_acoustic = np.mean(acoustic_scores) if acoustic_scores else 0.0

            # Divergence warning
            if epoch_losses['accuracy'] > 0.9 and avg_acoustic < 0.5:
                print(f"  WARNING: High token accuracy ({epoch_losses['accuracy']:.1%}) "
                      f"but low acoustic match ({avg_acoustic:.2f})")

            # Save best (based on combined score)
            combined_score = epoch_losses['accuracy'] * 0.7 + avg_acoustic * 0.3
            if combined_score > best_accuracy:
                best_accuracy = combined_score
                torch.save(model.state_dict(), save_dir / "sequence_best.pt")

            print(f"Epoch {total_epochs:3d} (P{phase_num}): "
                  f"loss={epoch_losses['total']:.4f}, "
                  f"acc={epoch_losses['accuracy']:.1%}, "
                  f"acoustic={avg_acoustic:.2f}")
```

**Step 5: Add phase advancement check**

Add after the demo section (after line 289, inside the epoch loop):

```python
            # Check phase advancement (dual gating)
            token_gate = phase.get('token_threshold', 0.0)
            acoustic_gate = phase.get('acoustic_threshold', 0.0)

            if (token_gate > 0 and acoustic_gate > 0 and
                epoch_losses['accuracy'] >= token_gate and
                avg_acoustic >= acoustic_gate):
                print(f"\n  Phase {phase_num} gates passed! "
                      f"(acc={epoch_losses['accuracy']:.1%} >= {token_gate:.0%}, "
                      f"acoustic={avg_acoustic:.2f} >= {acoustic_gate:.2f})")
                print(f"  Advancing to next phase...\n")
                break  # Exit epoch loop, advance to next phase
```

**Step 6: Verify script runs**

Run: `python -m primordial.scripts.train_sequence --epochs 1 --demo-every 1`

Expected: Runs one epoch with acoustic match logging

**Step 7: Commit**

```bash
git add primordial/scripts/train_sequence.py
git commit -m "feat(speech): add acoustic match tracking and dual curriculum gating"
```

---

## Task 5: Update Demo to Show Phrases

**Files:**
- Modify: `primordial/scripts/train_sequence.py`

**Step 1: Update demo section to handle phrases**

Replace the demo section (lines 257-289) with:

```python
            # Demo
            if (epoch + 1) % args.demo_every == 0:
                model.eval()
                with torch.no_grad():
                    # Pick a word/phrase from dataset
                    sample_idx = epoch % len(dataset)
                    mel, _, _, word = dataset[sample_idx]
                    mel = mel.unsqueeze(0)

                    # Generate
                    phonemes, latents = model.generate(
                        mel,
                        temperature=phase['temperature'],
                        min_length=2,
                    )

                    # Get target phonemes
                    all_entries = get_all_entries(include_phrases=True)
                    target_phonemes = all_entries.get(word, [])

                    match = phonemes == target_phonemes

                    # Compute acoustic match for demo
                    target_audio = tts.synthesize_phonemes(target_phonemes)
                    demo_acoustic = compute_acoustic_match(
                        model, tts, config, phonemes, target_audio
                    ) if phonemes else 0.0

                    print(f"  Demo: '{word}'")
                    print(f"    Generated: {phonemes}")
                    print(f"    Target:    {target_phonemes}")
                    print(f"    Match: {'YES' if match else 'NO'}, "
                          f"Acoustic: {demo_acoustic:.2f}")

                    if play_audio and phonemes:
                        # Play target
                        print(f"    Playing target...", end=" ", flush=True)
                        sd.play(target_audio, tts.sample_rate)
                        sd.wait()
                        print("done")

                        # Play produced
                        produced_audio = tts.synthesize_phonemes(phonemes)
                        print(f"    Playing produced...", end=" ", flush=True)
                        sd.play(produced_audio, tts.sample_rate)
                        sd.wait()
                        print("done")

                model.train()
```

**Step 2: Verify demo works**

Run: `python -m primordial.scripts.train_sequence --epochs 2 --demo-every 1 --no-audio`

Expected: Demo output shows phrases with acoustic scores

**Step 3: Commit**

```bash
git add primordial/scripts/train_sequence.py
git commit -m "feat(speech): update demo to show phrases with acoustic scores"
```

---

## Task 6: Add Integration Test for Phrase Training

**Files:**
- Test: `tests/speech/test_integration.py`

**Step 1: Add phrase training integration test**

Add to `tests/speech/test_integration.py`:

```python
def test_phrase_training_smoke():
    """Smoke test: phrase training runs without errors."""
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN
    from primordial.speech.training import SequenceTrainer, compute_acoustic_match
    from primordial.speech.word_dataset import WordDataset
    from torch.utils.data import DataLoader

    config = SpeechConfig(encoder_type='cnn', tts_backend='dummy')
    model = SpeechSequenceLRN(config)
    trainer = SequenceTrainer(model, config, lr=1e-3)
    tts = create_tts_backend(config)

    # Create dataset with phrases
    dataset = WordDataset(config, max_phonemes=10, include_phrases=True)

    # Verify we have phrases
    phrase_count = sum(1 for w in dataset.words if " " in w)
    assert phrase_count > 0, "Dataset should include phrases"

    # Custom collate
    def collate_fn(batch):
        from primordial.speech.latent import EOS_TOKEN
        import torch

        mels, inputs, targets, words = zip(*batch)
        max_len = max(len(t) for t in inputs)

        padded_inputs = torch.full((len(batch), max_len), EOS_TOKEN, dtype=torch.long)
        padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            padded_inputs[i, :len(inp)] = inp
            padded_targets[i, :len(tgt)] = tgt

        mels = torch.stack(mels)
        return mels, padded_inputs, padded_targets, words

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Run one training step
    mel, input_tokens, target_tokens, words = next(iter(dataloader))
    losses = trainer.train_step(mel, input_tokens, target_tokens)

    assert 'total' in losses
    assert 'accuracy' in losses
    assert losses['total'] > 0

    # Test acoustic match on a phrase
    model.eval()
    with torch.no_grad():
        phonemes, _ = model.generate(mel[0:1])

        # Get target audio
        word = words[0]
        target_phonemes = dataset._entries[word]
        target_audio = tts.synthesize_phonemes(target_phonemes)

        score = compute_acoustic_match(model, tts, config, phonemes, target_audio)

        assert 0.0 <= score <= 1.0, f"Acoustic score out of range: {score}"
```

**Step 2: Run test**

Run: `pytest tests/speech/test_integration.py::test_phrase_training_smoke -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/speech/test_integration.py
git commit -m "test(speech): add phrase training integration test"
```

---

## Task 7: Final Verification

**Step 1: Run all speech tests**

Run: `pytest tests/speech/ -v`

Expected: All tests PASS

**Step 2: Run short training session**

Run: `python -m primordial.scripts.train_sequence --epochs 5 --demo-every 2 --no-audio`

Expected: Training runs with:
- Phrases included in dataset
- Acoustic match scores logged
- Demo shows phrases with acoustic scores

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(speech): complete multi-word self-listening implementation"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add compute_acoustic_match function | training.py, test_sequence_trainer.py |
| 2 | Add phrase dataset tests | test_word_dataset.py |
| 3 | Update curriculum with dual gating | train_sequence.py |
| 4 | Add acoustic match tracking | train_sequence.py |
| 5 | Update demo for phrases | train_sequence.py |
| 6 | Add integration test | test_integration.py |
| 7 | Final verification | All files |

**Total estimated time:** 30-45 minutes

**After completion:** Run a full training session with:
```bash
python -m primordial.scripts.train_sequence --epochs 200 --demo-every 20
```
