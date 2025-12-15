#!/usr/bin/env python3
"""Preprocess LibriSpeech with SPARC encoder for training.

Creates HDF5 files with:
- mel: Input mel spectrograms
- ema: Target EMA articulator positions (12D)
- pitch: Target F0 (1D)
- loudness: Target energy (1D)

Usage:
    python -m primordial.scripts.preprocess_sparc \
        --librispeech /path/to/LibriSpeech \
        --output data/sparc_features \
        --splits train-clean-100 dev-clean \
        --duration 2.0
"""
import argparse
from pathlib import Path
import numpy as np

from primordial.speech import SpeechConfig, SPARCWrapper
from primordial.speech.sparc_dataset import preprocess_to_hdf5


def find_audio_files(librispeech_root: Path, split: str) -> list:
    """Find all audio files in a LibriSpeech split.

    Returns:
        List of (audio_path, speaker_id) tuples
    """
    split_dir = librispeech_root / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    audio_files = []
    for trans_file in split_dir.rglob("*.trans.txt"):
        chapter_dir = trans_file.parent
        speaker_id = chapter_dir.parent.name

        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) >= 1:
                    utterance_id = parts[0]
                    audio_path = chapter_dir / f"{utterance_id}.flac"
                    if audio_path.exists():
                        audio_files.append((str(audio_path), speaker_id))

    return audio_files


def filter_by_duration(
    audio_files: list,
    min_duration: float,
    max_duration: float,
    sample_rate: int = 16000,
) -> list:
    """Filter audio files by duration."""
    import soundfile as sf

    filtered = []
    for audio_path, speaker_id in audio_files:
        info = sf.info(audio_path)
        duration = info.duration
        if min_duration <= duration <= max_duration:
            filtered.append((audio_path, speaker_id))

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Preprocess LibriSpeech with SPARC")
    parser.add_argument("--librispeech", type=str, required=True, help="LibriSpeech root")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train-clean-100"], help="Splits to process")
    parser.add_argument("--duration", type=float, default=2.0, help="Audio duration in seconds")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum duration")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--mock", action="store_true", help="Use mock SPARC (for testing)")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split")
    args = parser.parse_args()

    librispeech_root = Path(args.librispeech)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SpeechConfig(audio_duration=args.duration)
    sparc = SPARCWrapper(config, mock=args.mock)

    for split in args.splits:
        print(f"\nProcessing {split}...")

        # Find audio files
        audio_files = find_audio_files(librispeech_root, split)
        print(f"  Found {len(audio_files)} utterances")

        # Filter by duration
        audio_files = filter_by_duration(
            audio_files,
            args.min_duration,
            args.duration,
            config.sample_rate,
        )
        print(f"  {len(audio_files)} after duration filter")

        # Limit samples if requested
        if args.max_samples:
            audio_files = audio_files[:args.max_samples]
            print(f"  Limited to {len(audio_files)} samples")

        # Preprocess
        output_path = output_dir / f"{split}.h5"
        preprocess_to_hdf5(
            audio_files,
            str(output_path),
            config,
            sparc,
            batch_size=args.batch_size,
        )
        print(f"  Saved to {output_path}")

        # Print storage info
        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
