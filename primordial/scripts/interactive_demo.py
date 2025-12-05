"""
Interactive speech demo: Listen → Recognize → Speak back

Usage:
    python -m primordial.scripts.interactive_demo
    python -m primordial.scripts.interactive_demo --duration 3  # longer recording
    python -m primordial.scripts.interactive_demo --voice /path/to/voice.onnx
"""

import argparse
import torch
import sounddevice as sd
import numpy as np
from pathlib import Path

from primordial.speech import SpeechConfig, create_tts_backend
from primordial.speech.sequence_decoder import SpeechSequenceLRN
from primordial.speech.encoders import compute_mel_spectrogram

# Defaults
DEFAULT_SAMPLE_RATE = 16000  # Model trained on 16kHz
DEFAULT_RECORD_SECONDS = 2
DEFAULT_CHECKPOINT = "checkpoints/sequence/sequence_best.pt"
DEFAULT_VOICE = Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"


def record_audio(duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Record from microphone.

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Audio waveform as numpy array (samples,)
    """
    print(f"🎤 Listening for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.

    Args:
        audio: Raw audio waveform

    Returns:
        Normalized audio
    """
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def main():
    parser = argparse.ArgumentParser(description="Interactive speech recognition demo")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                       help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--voice", type=str, default=str(DEFAULT_VOICE),
                       help=f"Path to Piper voice .onnx file (default: {DEFAULT_VOICE})")
    parser.add_argument("--duration", type=float, default=DEFAULT_RECORD_SECONDS,
                       help=f"Recording duration in seconds (default: {DEFAULT_RECORD_SECONDS})")
    parser.add_argument("--no-tts", action="store_true",
                       help="Disable TTS playback (just show recognized phonemes)")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available audio devices and exit")
    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return

    # Load model
    print("Loading model...")
    config = SpeechConfig(encoder_type='cnn', tts_backend='piper', tts_model_path=args.voice)
    model = SpeechSequenceLRN(config)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Train the model first with: python -m primordial.scripts.train_sequence")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    print("Model loaded!")

    # Load TTS
    if not args.no_tts:
        voice_path = Path(args.voice)
        if not voice_path.exists():
            print(f"Warning: Piper voice not found at {voice_path}")
            print("TTS will use DummyTTS (simple tones)")
        tts = create_tts_backend(config)
    else:
        tts = None

    print("=" * 50)
    print("Interactive Speech Demo")
    print("Speak a word or phrase, model will repeat it back")
    print("Press Ctrl+C to exit")
    print("=" * 50)

    try:
        while True:
            input("\nPress Enter to record...")

            # Record
            audio = record_audio(args.duration, config.sample_rate)
            audio = normalize_audio(audio)

            # Check if audio is silent
            if np.abs(audio).max() < 0.01:
                print("⚠️  No audio detected - try speaking louder or check microphone")
                continue

            # Convert to tensor
            waveform = torch.from_numpy(audio).float()

            # Compute mel spectrogram
            mel = compute_mel_spectrogram(
                waveform,
                sample_rate=config.sample_rate,
                n_mels=config.n_mels,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
            )

            # Pad/truncate to expected size
            mel = mel.squeeze(0)  # Remove batch dim from compute_mel_spectrogram
            target_frames = config.n_frames

            if mel.shape[1] < target_frames:
                # Pad with zeros (silence)
                padding = target_frames - mel.shape[1]
                mel = torch.nn.functional.pad(mel, (0, padding))
            else:
                # Truncate
                mel = mel[:, :target_frames]

            mel = mel.unsqueeze(0)  # Add batch dim: (1, n_mels, n_frames)

            # Generate phonemes
            with torch.no_grad():
                phonemes, latents = model.generate(mel)

            if phonemes:
                print(f"📝 Heard: {phonemes}")
                print(f"   ({len(phonemes)} phonemes)")

                # Speak back
                if tts is not None:
                    print("🔊 Speaking...")
                    try:
                        audio_out = tts.synthesize_phonemes(phonemes)
                        sd.play(audio_out, tts.sample_rate)
                        sd.wait()
                    except Exception as e:
                        print(f"⚠️  TTS error: {e}")
            else:
                print("📝 Heard: (nothing recognized)")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
