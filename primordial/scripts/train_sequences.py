#!/usr/bin/env python3
"""Train on phoneme sequences (syllables/words).

Takes the single-phoneme production model and teaches it sequences.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

try:
    import sounddevice as sd
    CAN_PLAY = True
except ImportError:
    CAN_PLAY = False

# Simple syllables and words as phoneme sequences
SEQUENCES = {
    # CVC syllables
    'ba': ['B', 'AA'],
    'bee': ['B', 'IY'],
    'boo': ['B', 'UW'],
    'da': ['D', 'AA'],
    'dee': ['D', 'IY'],
    'ma': ['M', 'AA'],
    'me': ['M', 'IY'],
    'no': ['N', 'OW'],
    'pa': ['P', 'AA'],
    'see': ['S', 'IY'],
    'so': ['S', 'OW'],
    'ta': ['T', 'AA'],
    'too': ['T', 'UW'],

    # Simple words
    'hi': ['HH', 'AY'],
    'hey': ['HH', 'EY'],
    'go': ['G', 'OW'],
    'up': ['AH', 'P'],
    'yes': ['Y', 'EH', 'S'],
    'no': ['N', 'OW'],
    'mom': ['M', 'AA', 'M'],
    'dad': ['D', 'AE', 'D'],
    'dog': ['D', 'AO', 'G'],
    'cat': ['K', 'AE', 'T'],
    'food': ['F', 'UW', 'D'],
    'water': ['W', 'AO', 'T', 'ER'],
    'hello': ['HH', 'EH', 'L', 'OW'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/production/production_best.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--play-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--voice", type=str,
                       default=str(Path.home() / ".claude-tts/voices/en_US-lessac-medium/en_US-lessac-medium.onnx"))
    args = parser.parse_args()

    from primordial.speech import (
        SpeechConfig, SpeechLRN, snap_to_nearest_anchor, get_anchor,
        create_tts_backend, compute_mel_spectrogram
    )

    print("=" * 60)
    print("Sequence Production Training")
    print("=" * 60)
    print(f"Sequences: {len(SEQUENCES)} words/syllables")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 60)

    config = SpeechConfig(encoder_type='cnn', tts_backend='piper', tts_model_path=args.voice)
    model = SpeechLRN(config)
    tts = create_tts_backend(config)

    # Load pretrained single-phoneme model
    if Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Pre-compute sequence audio
    print("\nPre-computing sequence audio...")
    seq_data = {}
    for word, phonemes in SEQUENCES.items():
        audio = tts.synthesize_phonemes(phonemes)
        waveform = torch.from_numpy(audio).float()
        if tts.sample_rate != config.sample_rate:
            target_len = int(len(waveform) * config.sample_rate / tts.sample_rate)
            waveform = F.interpolate(waveform.view(1,1,-1), size=target_len, mode='linear', align_corners=False).squeeze()

        mel = compute_mel_spectrogram(waveform, config.sample_rate, config.n_mels, config.n_fft, config.hop_length).squeeze(0)
        # Pad/truncate - sequences are longer, use 2x frames
        target_frames = config.n_frames * 2
        if mel.shape[1] < target_frames:
            mel = F.pad(mel, (0, target_frames - mel.shape[1]))
        else:
            mel = mel[:, :target_frames]

        # Get target anchors for each phoneme in sequence
        anchors = torch.stack([get_anchor(p) for p in phonemes])
        seq_data[word] = {'mel': mel, 'phonemes': phonemes, 'anchors': anchors, 'audio': audio}

    print(f"\nStarting training...")
    print("-" * 60)

    words = list(SEQUENCES.keys())

    for epoch in range(args.epochs):
        epoch_loss = 0
        seq_matches = 0
        total_phonemes = 0

        np.random.shuffle(words)

        for word in words:
            data = seq_data[word]
            mel = data['mel'].unsqueeze(0)
            anchors = data['anchors']
            phonemes = data['phonemes']

            # Get model's sequence prediction
            # For now: predict first phoneme from full sequence mel
            # TODO: Implement autoregressive sequence production
            latent, _, _ = model.forward_production(mel)
            latent = latent.squeeze(0)

            # Loss: match first phoneme anchor
            loss = F.mse_loss(latent, anchors[0])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Check match
            with torch.no_grad():
                produced, _ = snap_to_nearest_anchor(latent)
            if produced == phonemes[0]:
                seq_matches += 1
            total_phonemes += 1

        avg_loss = epoch_loss / len(words)
        match_rate = seq_matches / total_phonemes

        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, first_phoneme_match={match_rate:5.0%}")

        if (epoch + 1) % args.play_every == 0 and CAN_PLAY:
            # Demo a word
            demo_word = words[epoch % len(words)]
            data = seq_data[demo_word]
            print(f"  Demo: '{demo_word}' = {data['phonemes']}")
            print(f"  Playing...")
            sd.play(data['audio'], tts.sample_rate)
            sd.wait()

    print("-" * 60)
    print("Final test:")
    for word in list(SEQUENCES.keys())[:10]:
        data = seq_data[word]
        with torch.no_grad():
            latent, _, _ = model.forward_production(data['mel'].unsqueeze(0))
            produced, d = snap_to_nearest_anchor(latent.squeeze())
        target = data['phonemes'][0]
        print(f"  '{word}' ({data['phonemes']}) -> first={produced} ({'MATCH' if produced == target else 'miss'})")

    print("\nCheckpoint saved!")
    torch.save(model.state_dict(), "checkpoints/production/sequences_best.pt")


if __name__ == "__main__":
    main()
