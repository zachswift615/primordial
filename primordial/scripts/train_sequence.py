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
    parser.add_argument(
        "--acoustic-check-interval", type=int, default=10,
        help="Compute acoustic match every N batches (default: 10)"
    )

    return parser.parse_args()


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


def main():
    args = parse_args()

    # Check audio
    play_audio = CAN_PLAY and not args.no_audio

    # Import after args to avoid slow startup for --help
    from primordial.speech import SpeechConfig, create_tts_backend
    from primordial.speech.sequence_decoder import SpeechSequenceLRN
    from primordial.speech.training import SequenceTrainer, compute_acoustic_match
    from primordial.speech.word_dataset import WordDataset, WORD_PHONEMES, get_all_entries
    from torch.utils.data import DataLoader
    import numpy as np

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
        phases = list(CURRICULUM.keys())  # All phases (1, 2, 3, 4)

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
        dataset = WordDataset(
            config,
            max_phonemes=phase['max_phonemes'],
            include_phrases=phase.get('include_phrases', False),
        )
        print(f"Words in phase: {len(dataset)}")

        # Custom collate for variable-length sequences
        def collate_fn(batch):
            from primordial.speech.latent import EOS_TOKEN, SOS_TOKEN

            mels, inputs, targets, words = zip(*batch)

            # Pad sequences to max length in batch
            max_len = max(len(t) for t in inputs)

            # Pad inputs with EOS (after the real sequence)
            # Pad targets with -100 (ignore_index for cross-entropy)
            padded_inputs = torch.full((len(batch), max_len), EOS_TOKEN, dtype=torch.long)
            padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)

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

    # Final results
    print(f"\n{'='*60}")
    print("Training Complete")
    print(f"{'='*60}")
    print(f"Best accuracy: {best_accuracy:.1%}")
    print(f"Checkpoint saved to: {save_dir}")


if __name__ == "__main__":
    main()
