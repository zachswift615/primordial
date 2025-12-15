#!/usr/bin/env python3
"""Validate SPARC roundtrip quality before training.

Tests:
1. Audio -> SPARC encode -> decode -> compare to original
2. Measures reconstruction quality (mel MSE, PESQ if available)
3. Saves sample audio for manual listening

Usage:
    python -m primordial.scripts.validate_sparc --audio sample.wav
    python -m primordial.scripts.validate_sparc --librispeech /path/to/data --n-samples 10
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import torchaudio

from primordial.speech import SpeechConfig, SPARCWrapper, VoiceIdentity
from primordial.speech.encoders import compute_mel_spectrogram


def compute_mel_mse(audio1: torch.Tensor, audio2: torch.Tensor, config: SpeechConfig) -> float:
    """Compute MSE between mel spectrograms of two audio signals."""
    mel1 = compute_mel_spectrogram(audio1.unsqueeze(0), config)
    mel2 = compute_mel_spectrogram(audio2.unsqueeze(0), config)
    return torch.nn.functional.mse_loss(mel1, mel2).item()


def validate_single(
    audio_path: str,
    config: SpeechConfig,
    sparc: SPARCWrapper,
    voice: VoiceIdentity,
    output_dir: Optional[Path] = None,
) -> dict:
    """Validate SPARC roundtrip on a single audio file."""
    # Load audio
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    else:
        audio = audio.squeeze(0)

    # Resample if needed
    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        audio = resampler(audio)

    # Truncate/pad to expected duration
    expected_len = int(config.sample_rate * config.audio_duration)
    if len(audio) > expected_len:
        audio = audio[:expected_len]
    elif len(audio) < expected_len:
        audio = torch.nn.functional.pad(audio, (0, expected_len - len(audio)))

    # Encode
    ema, pitch, loudness = sparc.encode(audio.unsqueeze(0))

    # Decode
    spk_emb = voice.get_embedding(batch_size=1, device=audio.device)
    reconstructed = sparc.decode(ema, pitch, loudness, spk_emb)

    # Compute metrics
    mel_mse = compute_mel_mse(audio, reconstructed.squeeze(0), config)

    # Save outputs if requested
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_path).stem

        torchaudio.save(
            str(output_dir / f"{stem}_original.wav"),
            audio.unsqueeze(0),
            config.sample_rate,
        )
        torchaudio.save(
            str(output_dir / f"{stem}_reconstructed.wav"),
            reconstructed.detach().cpu(),
            config.sample_rate,
        )

        # Save features for inspection
        np.savez(
            output_dir / f"{stem}_features.npz",
            ema=ema.detach().numpy(),
            pitch=pitch.detach().numpy(),
            loudness=loudness.detach().numpy(),
        )

    return {
        'mel_mse': mel_mse,
        'ema_mean': ema.abs().mean().item(),
        'ema_std': ema.std().item(),
        'pitch_mean': pitch.mean().item(),
        'pitch_std': pitch.std().item(),
        'loudness_mean': loudness.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate SPARC roundtrip quality")
    parser.add_argument("--audio", type=str, help="Single audio file to test")
    parser.add_argument("--librispeech", type=str, help="LibriSpeech root for batch testing")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples for batch")
    parser.add_argument("--output-dir", type=str, default="sparc_validation", help="Output directory")
    parser.add_argument("--mock", action="store_true", help="Use mock SPARC (for testing)")
    args = parser.parse_args()

    config = SpeechConfig()
    sparc = SPARCWrapper(config, mock=args.mock)
    voice = VoiceIdentity()
    output_dir = Path(args.output_dir)

    if args.audio:
        results = validate_single(args.audio, config, sparc, voice, output_dir)
        print(f"\nValidation results for {args.audio}:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")

    elif args.librispeech:
        from primordial.speech import LibriSpeechDataset

        # Find audio files
        dataset = LibriSpeechDataset(
            args.librispeech,
            split="dev-clean",
            config=config,
            max_duration=config.audio_duration,
        )

        all_results = []
        for i in range(min(args.n_samples, len(dataset))):
            # Get audio path from dataset
            audio_path = dataset.samples[i]['audio_path']
            results = validate_single(
                audio_path, config, sparc, voice,
                output_dir / f"sample_{i}"
            )
            all_results.append(results)
            print(f"Sample {i}: mel_mse={results['mel_mse']:.4f}")

        # Aggregate
        print("\nAggregate results:")
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            print(f"  {key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
