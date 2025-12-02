#!/usr/bin/env python3
"""Run speech learning experiments.

Usage:
    python -m primordial.scripts.run_speech [options]

Examples:
    # Train phoneme classification (Phase 1)
    python -m primordial.scripts.run_speech --phase classification

    # Test with synthetic data (no real audio needed)
    python -m primordial.scripts.run_speech --synthetic

    # Use specific Piper voice
    python -m primordial.scripts.run_speech --voice /path/to/voice.onnx
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Default Piper voice path
DEFAULT_VOICE = Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Train LRN for speech learning")

    parser.add_argument(
        "--phase",
        choices=["classification", "production", "sequences", "words"],
        default="classification",
        help="Training phase (default: classification)"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=str(DEFAULT_VOICE),
        help=f"Path to Piper voice .onnx file (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (DummyTTS) instead of real TTS"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device (default: auto)"
    )
    parser.add_argument(
        "--encoder",
        choices=["linear", "cnn"],
        default="linear",
        help="Audio encoder type: linear (simple) or cnn (richer features)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/speech",
        help="Directory to save checkpoints"
    )

    return parser.parse_args()


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    device = get_device(args.device)

    print("=" * 60)
    print("Primordial LRN - Speech Learning")
    print("=" * 60)
    print(f"Phase: {args.phase}")
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder}")
    print(f"Epochs: {args.epochs}")
    print(f"Synthetic data: {args.synthetic}")
    if not args.synthetic:
        print(f"Voice: {args.voice}")
    print("=" * 60)

    # Import speech components
    from primordial.speech import (
        SpeechConfig, SpeechLRN, PhonemeTrainer,
        SyntheticPhonemeDataset, NUM_PHONEMES
    )

    # Import production trainer if needed
    if args.phase == "production":
        from primordial.speech.training import ProductionTrainer

    # Create config
    config = SpeechConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        encoder_type=args.encoder,
        tts_backend="piper" if not args.synthetic else "dummy",
        tts_model_path=args.voice if not args.synthetic else "",
    )

    # Create model
    print("\nInitializing model...")
    model = SpeechLRN(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Create trainer based on phase
    if args.phase == "production":
        trainer = ProductionTrainer(model, config, device=device)
        print("Using ProductionTrainer for production phase")
    else:
        trainer = PhonemeTrainer(model, config, device=device)
        print("Using PhonemeTrainer for classification phase")

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Create dataset
    print("\nCreating dataset...")
    if args.synthetic:
        dataset = SyntheticPhonemeDataset(config, samples_per_phoneme=20)
        print(f"Synthetic dataset (DummyTTS): {len(dataset)} samples ({NUM_PHONEMES} phonemes × 20 each)")
    else:
        # Use real Piper TTS for dataset generation
        print(f"Generating dataset with Piper TTS: {args.voice}")
        dataset = SyntheticPhonemeDataset(config, samples_per_phoneme=20)
        print(f"Piper TTS dataset: {len(dataset)} samples ({NUM_PHONEMES} phonemes × 20 each)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for now
    )

    # Training loop
    print(f"\nStarting training ({args.epochs} epochs)...")
    print("-" * 60)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "production":
        # Production phase training loop
        best_match_rate = 0.0

        try:
            for epoch in range(args.epochs):
                metrics = trainer.train_production_epoch(dataloader, epoch)

                print(f"Epoch {epoch+1:3d}: "
                      f"latent_loss={metrics['latent_loss']:.4f}, "
                      f"match_rate={metrics['match_rate']:.1%}, "
                      f"embed_loss={metrics['embed_loss']:.4f}")

                # Save best model based on match rate
                if metrics['match_rate'] > best_match_rate:
                    best_match_rate = metrics['match_rate']
                    trainer.save_checkpoint(save_dir / "speech_best.pt")

                # Periodic save
                if (epoch + 1) % 10 == 0:
                    trainer.save_checkpoint(save_dir / f"speech_epoch{epoch+1}.pt")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted.")

        # Save final
        trainer.save_checkpoint(save_dir / "speech_final.pt")

        print("-" * 60)
        print(f"Training complete!")
        print(f"Best match rate: {best_match_rate:.1%}")
        print(f"Checkpoints saved to: {save_dir}")

    else:
        # Classification phase training loop
        best_acc = 0.0

        try:
            for epoch in range(args.epochs):
                metrics = trainer.train_epoch(dataloader, verbose=False)

                print(f"Epoch {epoch+1:3d}: "
                      f"loss={metrics['total_loss']:.4f}, "
                      f"acc={metrics['phoneme_accuracy']:.1%}")

                # Save best model
                if metrics['phoneme_accuracy'] > best_acc:
                    best_acc = metrics['phoneme_accuracy']
                    trainer.save_checkpoint(save_dir / "speech_best.pt")

                # Periodic save
                if (epoch + 1) % 10 == 0:
                    trainer.save_checkpoint(save_dir / f"speech_epoch{epoch+1}.pt")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted.")

        # Save final
        trainer.save_checkpoint(save_dir / "speech_final.pt")

        print("-" * 60)
        print(f"Training complete!")
        print(f"Best accuracy: {best_acc:.1%}")
        print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
