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
        "--encoder-checkpoint", type=str, default=None,
        help="Load pretrained encoder weights (e.g., from production training)"
    )
    parser.add_argument(
        "--freeze-encoder", action="store_true",
        help="Freeze encoder weights, only train decoder"
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
    from primordial.speech.word_dataset import WordDataset, WORD_PHONEMES
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

    # Setup - find Piper voice model
    piper_voice = Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"
    if not piper_voice.exists():
        print(f"Warning: Piper voice not found at {piper_voice}")
        print("Training will use DummyTTS (random noise). Results will be poor.")
        piper_voice = ""

    config = SpeechConfig(
        encoder_type='cnn',
        tts_backend='piper',
        tts_model_path=str(piper_voice),
    )
    model = SpeechSequenceLRN(config)
    trainer = SequenceTrainer(model, config, lr=args.lr)
    tts = create_tts_backend(config)

    # Load pretrained encoder if provided
    if args.encoder_checkpoint:
        print(f"Loading pretrained encoder: {args.encoder_checkpoint}")
        checkpoint = torch.load(args.encoder_checkpoint, map_location='cpu')

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Extract encoder weights (the SpeechLRN inside SpeechSequenceLRN)
        # The production checkpoint has keys like 'audio_encoder.*', 'mixing_layers.*', etc.
        # SpeechSequenceLRN.encoder is a SpeechLRN, so we load directly
        encoder_state = {}
        for k, v in state_dict.items():
            # Skip production-specific heads
            if k.startswith('production_head.') or k.startswith('speech_head.'):
                continue
            encoder_state[k] = v

        model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"  Loaded encoder weights ({len(encoder_state)} tensors)")

        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("  Encoder frozen - only training decoder")

    # Load full checkpoint if provided (for resuming training)
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
