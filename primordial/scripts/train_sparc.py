#!/usr/bin/env python3
"""Train model to predict SPARC articulatory features.

Phase 2 of SPARC integration: supervised training on pre-computed targets.

Usage:
    python -m primordial.scripts.train_sparc \
        --data data/sparc_features/train-clean-100.h5 \
        --val-data data/sparc_features/dev-clean.h5 \
        --epochs 50 \
        --batch-size 32
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from primordial.speech import SpeechConfig, SpeechLRN, SPARCTrainer
from primordial.speech.sparc_dataset import SPARCDataset


def main():
    parser = argparse.ArgumentParser(description="Train SPARC articulatory model")
    parser.add_argument("--data", type=str, required=True, help="Training HDF5 file")
    parser.add_argument("--val-data", type=str, default=None, help="Validation HDF5 file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/sparc", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--encoder", type=str, default="cnn", choices=["linear", "cnn"])
    parser.add_argument("--duration", type=float, default=2.0, help="Audio duration")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Config
    config = SpeechConfig(
        encoder_type=args.encoder,
        audio_duration=args.duration,
    )

    # Data
    train_dataset = SPARCDataset(args.data, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data:
        val_dataset = SPARCDataset(args.val_data, config)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )

    # Model
    model = SpeechLRN(config, output_head='articulatory')
    trainer = SPARCTrainer(model, config, lr=args.lr, device=device)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed from epoch {start_epoch}")

    print(f"Training on {len(train_dataset)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)

        log = f"Epoch {epoch+1}/{args.epochs}"
        log += f" | train_loss={train_losses['total']:.4f}"
        log += f" | ema={train_losses['ema']:.4f}"
        log += f" | pitch={train_losses['pitch']:.4f}"

        # Validate
        if val_loader:
            val_losses = {}
            for mel, ema, pitch, loudness in val_loader:
                target = {'ema': ema, 'pitch': pitch, 'loudness': loudness}
                batch_losses = trainer.validation_step(mel, target)
                for k, v in batch_losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v

            n_val = len(val_loader)
            val_losses = {k: v / n_val for k, v in val_losses.items()}
            log += f" | val_loss={val_losses['total']:.4f}"

            # Save best
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                trainer.save_checkpoint(output_dir / "best.pt", epoch)

        print(log)

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(output_dir / f"epoch_{epoch+1}.pt", epoch)

    # Save final
    trainer.save_checkpoint(output_dir / "final.pt", args.epochs - 1)
    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
