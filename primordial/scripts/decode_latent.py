#!/usr/bin/env python3
"""Interpret latent vectors from the speech production model.

This script decodes 6D latent vectors into phoneme interpretations and
articulatory feature descriptions.

Usage:
    python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5"
    python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5" --play
    python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5" --k 5

Examples:
    # Decode a latent vector
    python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5"

    # Show top 5 nearest phonemes
    python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5" --k 5

    # Decode and play synthesized audio
    python -m primordial.scripts.decode_latent --latent "0.3,-0.7,0.1,0.8,0.2,-0.5" --play
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode latent vectors from speech production model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--latent",
        type=str,
        required=True,
        help="Comma-separated 6D latent vector (e.g., '0.3,-0.7,0.1,0.8,0.2,-0.5')"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest anchors to show (default: 3)"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play synthesized audio using Piper TTS"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=str(Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"),
        help="Path to Piper voice model (only used with --play)"
    )

    return parser.parse_args()


def parse_latent(latent_str: str) -> torch.Tensor:
    """Parse comma-separated string into 6D tensor.

    Args:
        latent_str: Comma-separated float values

    Returns:
        (6,) tensor

    Raises:
        ValueError: If input is not exactly 6 values
    """
    try:
        values = [float(x.strip()) for x in latent_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid latent format: {e}")

    if len(values) != 6:
        raise ValueError(f"Expected 6 values, got {len(values)}")

    return torch.tensor(values, dtype=torch.float32)


def format_latent_vector(latent: torch.Tensor) -> str:
    """Format latent tensor as string."""
    values = latent.tolist()
    return "[" + ", ".join(f"{v:5.2f}" for v in values) + "]"


def print_interpretation(latent: torch.Tensor, k: int = 3):
    """Print formatted interpretation of latent vector.

    Args:
        latent: (6,) latent tensor
        k: Number of nearest anchors to show
    """
    from primordial.speech.latent import (
        snap_to_nearest_anchor,
        get_k_nearest_anchors,
        interpret_latent
    )

    # Print latent vector
    print("\nLatent:", format_latent_vector(latent))
    print()

    # Get nearest anchors
    nearest_anchors = get_k_nearest_anchors(latent, k=k)

    print(f"Nearest anchors (top {k}):")
    for i, (phoneme, distance) in enumerate(nearest_anchors, 1):
        print(f"  {i}. {phoneme:4s} (distance: {distance:.2f})")
    print()

    # Get feature interpretation
    features = interpret_latent(latent)

    print("Feature interpretation:")
    print(f"  Front-Back: {latent[0]:5.2f} ({features['front_back']})")
    print(f"  High-Low:   {latent[1]:5.2f} ({features['high_low']})")
    print(f"  Rounded:    {latent[2]:5.2f} ({features['rounded']})")
    print(f"  Voiced:     {latent[3]:5.2f} ({features['voiced']})")
    print(f"  Manner:     {latent[4]:5.2f} ({features['manner']})")
    print(f"  Type:       {latent[5]:5.2f} ({features['type']})")
    print()


def play_audio(latent: torch.Tensor, voice_path: str):
    """Synthesize and play audio for nearest phoneme.

    Args:
        latent: (6,) latent tensor
        voice_path: Path to Piper voice model
    """
    from primordial.speech.latent import snap_to_nearest_anchor
    from primordial.speech.tts import PiperTTS
    from primordial.speech import SpeechConfig

    # Find nearest phoneme
    nearest_phoneme, distance = snap_to_nearest_anchor(latent)

    print(f"Synthesizing '{nearest_phoneme}' (distance: {distance:.2f})...")

    # Check if voice model exists
    voice_path_obj = Path(voice_path)
    if not voice_path_obj.exists():
        print(f"Error: Voice model not found at {voice_path}")
        print("Please install a Piper voice or specify path with --voice")
        return

    try:
        # Create TTS backend
        config = SpeechConfig(tts_backend="piper", tts_model_path=voice_path)
        tts = PiperTTS(voice_path, config)

        # Synthesize phoneme (repeat 3 times for clarity)
        phonemes = [nearest_phoneme] * 3
        audio = tts.synthesize_phonemes(phonemes)

        # Play audio
        print(f"Playing audio ({len(audio)} samples at {tts.sample_rate} Hz)...")

        try:
            # Try using sounddevice for playback
            import sounddevice as sd
            sd.play(audio, tts.sample_rate)
            sd.wait()
            print("Playback complete.")
        except ImportError:
            # Fallback: save to temporary file
            print("sounddevice not available, saving to temp file...")
            import tempfile
            import subprocess

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            # Write WAV file
            from scipy.io import wavfile
            wavfile.write(temp_path, tts.sample_rate, (audio * 32767).astype(np.int16))
            print(f"Saved to: {temp_path}")

            # Try to play with system player
            try:
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['afplay', temp_path], check=True)
                elif sys.platform == 'linux':
                    subprocess.run(['aplay', temp_path], check=True)
                elif sys.platform == 'win32':
                    subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{temp_path}").PlaySync()'], check=True)
                print("Playback complete.")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"Could not auto-play. Please play manually: {temp_path}")

    except Exception as e:
        print(f"Error during synthesis/playback: {e}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_args()

    # Parse latent vector
    try:
        latent = parse_latent(args.latent)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print header
    print("=" * 60)
    print("Latent Vector Decoder - Speech Production Model")
    print("=" * 60)

    # Print interpretation
    try:
        print_interpretation(latent, k=args.k)
    except Exception as e:
        print(f"Error during interpretation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Play audio if requested
    if args.play:
        try:
            play_audio(latent, args.voice)
        except Exception as e:
            print(f"Error during audio playback: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
