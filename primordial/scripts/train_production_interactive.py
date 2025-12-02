#!/usr/bin/env python3
"""Interactive speech production training with audio playback.

This script trains the production head while playing audio so you can
hear the agent learning to speak.

Usage:
    python -m primordial.scripts.train_production_interactive [options]

Examples:
    # Train with audio playback every 10 epochs
    python -m primordial.scripts.train_production_interactive --epochs 50 --play-every 10

    # Train specific phonemes
    python -m primordial.scripts.train_production_interactive --phonemes IY,AA,B,M,S

    # Silent mode (no audio)
    python -m primordial.scripts.train_production_interactive --no-audio
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Try to import audio playback
try:
    import sounddevice as sd
    CAN_PLAY = True
except ImportError:
    CAN_PLAY = False


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive production training")

    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--phonemes", type=str, default="IY,AA,EH,UW,AH,B,D,M,S,L",
        help="Comma-separated phonemes to train on"
    )
    parser.add_argument(
        "--play-every", type=int, default=10,
        help="Play audio demo every N epochs (default: 10)"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio playback"
    )
    parser.add_argument(
        "--voice", type=str,
        default=str(Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"),
        help="Path to Piper voice model"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./checkpoints/production",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint to resume training from"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check audio capability
    play_audio = CAN_PLAY and not args.no_audio
    if not CAN_PLAY and not args.no_audio:
        print("Warning: sounddevice not installed. Run: pip install sounddevice")
        print("Continuing without audio playback...")

    # Import after args to avoid slow startup for --help
    from primordial.speech import (
        SpeechConfig, SpeechLRN,
        snap_to_nearest_anchor, get_anchor,
        create_tts_backend, compute_mel_spectrogram
    )

    # Parse phonemes
    train_phonemes = [p.strip().upper() for p in args.phonemes.split(",")]

    print("=" * 60)
    print("Interactive Production Training")
    print("=" * 60)
    print(f"Phonemes: {', '.join(train_phonemes)}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Audio playback: {'enabled' if play_audio else 'disabled'}")
    print("=" * 60)
    print()

    # Create config and model
    config = SpeechConfig(
        encoder_type='cnn',
        tts_backend='piper',
        tts_model_path=args.voice,
    )

    print("Loading model...")
    model = SpeechLRN(config)
    tts = create_tts_backend(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        print("Checkpoint loaded - resuming training")

    # Pre-compute target mels
    print("Pre-computing target audio...")
    target_mels = {}
    for phoneme in train_phonemes:
        audio = tts.synthesize_phonemes([phoneme])
        waveform = torch.from_numpy(audio).float()

        # Resample if needed
        if tts.sample_rate != config.sample_rate:
            target_len = int(len(waveform) * config.sample_rate / tts.sample_rate)
            waveform = F.interpolate(
                waveform.view(1, 1, -1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).squeeze()

        mel = compute_mel_spectrogram(
            waveform, config.sample_rate,
            config.n_mels, config.n_fft, config.hop_length
        ).squeeze(0)

        # Pad/truncate to standard size
        if mel.shape[1] < config.n_frames:
            mel = F.pad(mel, (0, config.n_frames - mel.shape[1]))
        else:
            mel = mel[:, :config.n_frames]

        target_mels[phoneme] = mel

    print()
    print("Starting training...")
    print("-" * 60)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_match_rate = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        matches = 0

        # Shuffle phonemes each epoch
        shuffled = train_phonemes.copy()
        np.random.shuffle(shuffled)

        for target in shuffled:
            target_anchor = get_anchor(target)
            target_mel = target_mels[target].unsqueeze(0)

            # Forward pass
            latent, dur, pitch = model.forward_production(target_mel)
            latent = latent.squeeze(0)

            # Loss: MSE to target anchor
            loss = F.mse_loss(latent, target_anchor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Check match
            with torch.no_grad():
                produced, _ = snap_to_nearest_anchor(latent)
            if produced == target:
                matches += 1

        avg_loss = epoch_loss / len(train_phonemes)
        match_rate = matches / len(train_phonemes)

        # Save best
        if match_rate > best_match_rate:
            best_match_rate = match_rate
            torch.save(model.state_dict(), save_dir / "production_best.pt")

        # Print and optionally play audio
        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, match_rate={match_rate:5.0%}", end="")

        if (epoch + 1) % args.play_every == 0:
            # Pick a random phoneme for demo
            demo_target = train_phonemes[(epoch // args.play_every) % len(train_phonemes)]

            with torch.no_grad():
                latent, _, _ = model.forward_production(target_mels[demo_target].unsqueeze(0))
                produced, dist = snap_to_nearest_anchor(latent.squeeze())

            print(f"  | Demo: {demo_target} -> {produced} (d={dist:.2f})", end="")

            if play_audio:
                print()
                # Play target
                target_audio = tts.synthesize_phonemes([demo_target])
                print(f"         Target ({demo_target}): ", end="", flush=True)
                sd.play(target_audio, tts.sample_rate)
                sd.wait()
                print("done")

                # Play produced
                produced_audio = tts.synthesize_phonemes([produced])
                print(f"         Produced ({produced}): ", end="", flush=True)
                sd.play(produced_audio, tts.sample_rate)
                sd.wait()
                print("done")
            else:
                print()
        else:
            print()

    # Final test
    print()
    print("-" * 60)
    print("Final Results:")
    print("-" * 60)

    final_matches = 0
    for target in train_phonemes:
        with torch.no_grad():
            latent, _, _ = model.forward_production(target_mels[target].unsqueeze(0))
            produced, dist = snap_to_nearest_anchor(latent.squeeze())

        match_str = "MATCH" if produced == target else "miss"
        print(f"  {target:3s} -> {produced:3s} (d={dist:.2f}) [{match_str}]")

        if produced == target:
            final_matches += 1

        if play_audio:
            # Play both
            target_audio = tts.synthesize_phonemes([target])
            produced_audio = tts.synthesize_phonemes([produced])

            sd.play(target_audio, tts.sample_rate)
            sd.wait()
            sd.play(produced_audio, tts.sample_rate)
            sd.wait()

    final_rate = final_matches / len(train_phonemes)
    print()
    print(f"Final match rate: {final_rate:.0%} ({final_matches}/{len(train_phonemes)})")
    print(f"Best match rate: {best_match_rate:.0%}")
    print(f"Checkpoint saved to: {save_dir}")


if __name__ == "__main__":
    main()
