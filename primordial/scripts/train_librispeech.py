#!/usr/bin/env python3
"""Train speech recognition on LibriSpeech for speaker diversity.

Fine-tunes the existing sequence model (trained on Piper TTS) using real human
speech from LibriSpeech to achieve speaker-invariant phoneme recognition.

Usage:
    python -m primordial.scripts.train_librispeech --data ~/data/LibriSpeech

Options:
    --data PATH             Path to LibriSpeech directory (required)
    --checkpoint PATH       Resume from checkpoint (uses sequence_best.pt by default)
    --epochs N              Total training epochs (default: 100)
    --mixed                 Use mixed synthetic+real training (curriculum)
    --real-ratio RATIO      Initial ratio of real data in mixed mode (default: 0.2)
    --max-phonemes N        Filter to sequences with at most N phonemes
    --augment               Apply audio augmentation
    --no-audio              Disable audio playback during demos
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import audio playback
try:
    import sounddevice as sd
    CAN_PLAY = True
except ImportError:
    CAN_PLAY = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on LibriSpeech for speaker-diverse recognition"
    )

    # Required
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to LibriSpeech directory"
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint to fine-tune from (default: checkpoints/sequence/sequence_best.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./checkpoints/sequence",
        help="Directory to save checkpoints"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Learning rate (very low to prevent catastrophic forgetting)"
    )

    # Curriculum
    parser.add_argument(
        "--mixed", action="store_true",
        help="Use mixed synthetic+real training with curriculum"
    )
    parser.add_argument(
        "--real-ratio", type=float, default=0.2,
        help="Initial ratio of real data in mixed mode"
    )
    parser.add_argument(
        "--real-ratio-final", type=float, default=0.6,
        help="Final ratio of real data after curriculum"
    )
    parser.add_argument(
        "--curriculum-epochs", type=int, default=150,
        help="Epochs to ramp from initial to final ratio (slower = more stable)"
    )

    # Data
    parser.add_argument(
        "--split", type=str, default="train-clean-100",
        help="LibriSpeech split to use"
    )
    parser.add_argument(
        "--val-split", type=str, default=None,
        help="Validation split (e.g., dev-clean)"
    )
    parser.add_argument(
        "--max-phonemes", type=int, default=None,
        help="Filter to utterances with at most N phonemes (use --truncate-phonemes instead)"
    )
    parser.add_argument(
        "--truncate-phonemes", type=int, default=20,
        help="Truncate phoneme sequences to first N phonemes (keeps all utterances)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=3.0,
        help="Maximum utterance duration in seconds"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply audio augmentation"
    )

    # Demo
    parser.add_argument(
        "--demo-every", type=int, default=5,
        help="Run demo every N epochs"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio playback during demos"
    )
    parser.add_argument(
        "--eval-speakers", type=int, default=5,
        help="Number of speakers to evaluate per demo"
    )

    return parser.parse_args()


def compute_phoneme_error_rate(predicted: list, target: list) -> float:
    """Compute Phoneme Error Rate (PER) using edit distance.

    PER = (substitutions + insertions + deletions) / len(target)
    """
    if len(target) == 0:
        return 1.0 if len(predicted) > 0 else 0.0

    # Dynamic programming for edit distance
    m, n = len(predicted), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n] / len(target)


def main():
    args = parse_args()

    play_audio = CAN_PLAY and not args.no_audio

    # Import after args
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN
    from primordial.speech.training import SequenceTrainer
    from primordial.speech.librispeech_dataset import (
        LibriSpeechDataset,
        MixedSpeechDataset,
        collate_variable_length,
    )
    from primordial.speech.word_dataset import WordDataset
    from primordial.speech.g2p import text_to_phonemes
    from primordial.speech.phonemes import index_to_phoneme
    from primordial.speech.latent import EOS_TOKEN, SOS_TOKEN
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("LibriSpeech Fine-Tuning")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Split: {args.split}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Mixed training: {args.mixed}")
    if args.mixed:
        print(f"  Real ratio: {args.real_ratio} -> {args.real_ratio_final}")
    print(f"Augmentation: {args.augment}")
    print("=" * 60)
    print()

    # Config
    piper_voice = Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"
    if not piper_voice.exists():
        logger.warning(f"Piper voice not found at {piper_voice}")
        piper_voice = ""

    config = SpeechConfig(
        encoder_type='cnn',
        tts_backend='piper',
        tts_model_path=str(piper_voice),
    )

    # Model
    model = SpeechSequenceLRN(config)

    # Load checkpoint
    checkpoint_path = args.checkpoint or Path(args.output_dir) / "sequence_best.pt"
    if Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        logger.info("  Checkpoint loaded successfully")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("  Training from scratch (not recommended)")

    # Trainer with lower LR for fine-tuning
    trainer = SequenceTrainer(model, config, lr=args.lr)

    # TTS for demos
    tts = create_tts_backend(config) if piper_voice else None

    # Create datasets
    logger.info("Loading LibriSpeech dataset...")
    librispeech_dataset = LibriSpeechDataset(
        root=args.data,
        split=args.split,
        config=config,
        max_phonemes=args.max_phonemes,
        truncate_phonemes=args.truncate_phonemes,
        max_duration=args.max_duration,
        augment=args.augment,
    )
    logger.info(f"  Loaded {len(librispeech_dataset)} utterances")
    logger.info(f"  Speakers: {len(librispeech_dataset.get_speaker_ids())}")

    # Mixed training setup
    if args.mixed:
        logger.info("Creating synthetic dataset for mixed training...")
        synthetic_dataset = WordDataset(
            config,
            include_phrases=True,
            max_phonemes=args.max_phonemes,
        )
        logger.info(f"  Synthetic: {len(synthetic_dataset)} words/phrases")

        train_dataset = MixedSpeechDataset(
            synthetic_dataset=synthetic_dataset,
            real_dataset=librispeech_dataset,
            real_ratio=args.real_ratio,
        )
    else:
        train_dataset = librispeech_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        num_workers=0,
        pin_memory=True,
    )

    # Validation loader
    val_loader = None
    if args.val_split:
        logger.info(f"Loading validation split: {args.val_split}")
        val_dataset = LibriSpeechDataset(
            root=args.data,
            split=args.val_split,
            config=config,
            max_phonemes=args.max_phonemes,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_variable_length,
        )
        logger.info(f"  Validation: {len(val_dataset)} utterances")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_per = float('inf')

    for epoch in range(args.epochs):
        # Update curriculum ratio
        if args.mixed and epoch < args.curriculum_epochs:
            progress = epoch / args.curriculum_epochs
            current_ratio = args.real_ratio + progress * (args.real_ratio_final - args.real_ratio)
            train_dataset.set_real_ratio(current_ratio)
            ratio_str = f" (real={current_ratio:.0%})"
        else:
            ratio_str = ""

        # Training
        model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0

        for batch_idx, (mels, input_tokens, target_tokens, identifiers) in enumerate(train_loader):
            losses = trainer.train_step(mels, input_tokens, target_tokens)

            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1

            # Progress
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: "
                      f"loss={losses['total']:.4f}, acc={losses['accuracy']:.1%}")

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)

        print(f"Epoch {epoch+1:3d}/{args.epochs}{ratio_str}: "
              f"loss={epoch_losses['total']:.4f}, "
              f"acc={epoch_losses['accuracy']:.1%}")

        # Validation
        if val_loader and (epoch + 1) % args.demo_every == 0:
            model.eval()
            val_pers = []

            with torch.no_grad():
                for mels, input_tokens, target_tokens, identifiers in val_loader:
                    # Generate for first sample in batch
                    generated, _ = model.generate(mels[0:1])

                    # Get target phonemes (convert indices to phonemes)
                    target_indices = target_tokens[0].tolist()
                    target_phonemes = []
                    for idx in target_indices:
                        if idx == EOS_TOKEN or idx == -100:
                            break
                        if 0 <= idx < 41:
                            target_phonemes.append(index_to_phoneme(idx))

                    per = compute_phoneme_error_rate(generated, target_phonemes)
                    val_pers.append(per)

                    if len(val_pers) >= 100:  # Sample 100 for speed
                        break

            avg_per = np.mean(val_pers)
            print(f"  Validation PER: {avg_per:.1%}")

            if avg_per < best_per:
                best_per = avg_per
                torch.save(model.state_dict(), output_dir / "librispeech_best.pt")
                print(f"  New best! Saved to librispeech_best.pt")

        # Demo
        if (epoch + 1) % args.demo_every == 0:
            model.eval()
            print(f"\n  Demo (epoch {epoch+1}):")

            with torch.no_grad():
                # Test on a few samples
                demo_samples = min(3, len(librispeech_dataset))
                for i in range(demo_samples):
                    idx = (epoch * demo_samples + i) % len(librispeech_dataset)
                    mel, input_tokens, target_tokens, speaker_id = librispeech_dataset[idx]

                    # Generate
                    generated, _ = model.generate(mel.unsqueeze(0))

                    # Get target phonemes
                    target_indices = target_tokens.tolist()
                    target_phonemes = []
                    for idx in target_indices:
                        if idx == EOS_TOKEN:
                            break
                        if 0 <= idx < 41:
                            target_phonemes.append(index_to_phoneme(idx))

                    per = compute_phoneme_error_rate(generated, target_phonemes)

                    print(f"    Speaker {speaker_id}:")
                    print(f"      Generated: {generated[:15]}{'...' if len(generated) > 15 else ''}")
                    print(f"      Target:    {target_phonemes[:15]}{'...' if len(target_phonemes) > 15 else ''}")
                    print(f"      PER: {per:.1%}")

                    # Play audio if available
                    if play_audio and tts and generated:
                        try:
                            audio = tts.synthesize_phonemes(generated)
                            print(f"      Playing generated...", end=" ", flush=True)
                            sd.play(audio, tts.sample_rate)
                            sd.wait()
                            print("done")
                        except Exception as e:
                            print(f"      Audio error: {e}")

            print()

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(str(output_dir / f"librispeech_epoch{epoch+1}.pt"))

    # Final results
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation PER: {best_per:.1%}")
    print(f"Checkpoints saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Test with interactive demo")
    print("  2. If PER > 30%, try more epochs or data augmentation")
    print("  3. Download dev-clean for proper validation")


if __name__ == "__main__":
    main()
