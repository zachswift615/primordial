"""Speech learning module for Primordial LRN.

This module enables LRN agents to learn speech through sensorimotor
imitation - hearing audio and learning to reproduce it via phoneme
production, similar to how infants acquire speech.

Components:
- config: SpeechConfig for audio/model settings
- phonemes: ARPABET phoneme inventory (41 phonemes)
- encoders: MelSpectrogramEncoder for audio input
- heads: SpeechHead for phoneme/duration/pitch output
- tts: Text-to-speech synthesis from phonemes (Piper/Sherpa-ONNX)
- training: SpeechLRN model and PhonemeTrainer

Example:
    from primordial.speech import SpeechConfig, SpeechLRN, PhonemeTrainer

    config = SpeechConfig()
    model = SpeechLRN(config)
    trainer = PhonemeTrainer(model, config)
"""

from .config import SpeechConfig
from .phonemes import (
    PHONEME_INVENTORY,
    NUM_PHONEMES,
    phoneme_to_index,
    index_to_phoneme,
    VOWELS,
    CONSONANTS,
)
from .encoders import MelSpectrogramEncoder, CNNMelEncoder, compute_mel_spectrogram
from .heads import SpeechHead, SpeechSequenceHead, AudioReconstructionHead, ProductionHead
from .tts import TTSBackend, PiperTTS, DummyTTS, create_tts_backend
from .training import SpeechLRN, PhonemeTrainer, ProductionTrainer, PhonemeDataset, SyntheticPhonemeDataset
from .sequence_decoder import SequenceDecoder, SinusoidalPositionalEncoding
from .latent import (
    PHONEME_ANCHORS,
    LATENT_DIM,
    get_anchor,
    snap_to_nearest_anchor,
    get_k_nearest_anchors,
    interpret_latent,
    SOS_TOKEN,
    EOS_TOKEN,
    TOTAL_VOCAB,
)

__all__ = [
    # Config
    'SpeechConfig',

    # Phonemes
    'PHONEME_INVENTORY',
    'NUM_PHONEMES',
    'VOWELS',
    'CONSONANTS',
    'phoneme_to_index',
    'index_to_phoneme',

    # Encoders
    'MelSpectrogramEncoder',
    'CNNMelEncoder',
    'compute_mel_spectrogram',

    # Heads
    'SpeechHead',
    'SpeechSequenceHead',
    'AudioReconstructionHead',
    'ProductionHead',

    # TTS
    'TTSBackend',
    'PiperTTS',
    'DummyTTS',
    'create_tts_backend',

    # Training
    'SpeechLRN',
    'PhonemeTrainer',
    'ProductionTrainer',
    'PhonemeDataset',
    'SyntheticPhonemeDataset',

    # Latent space
    'PHONEME_ANCHORS',
    'LATENT_DIM',
    'get_anchor',
    'snap_to_nearest_anchor',
    'get_k_nearest_anchors',
    'interpret_latent',

    # Sequence decoder
    'SequenceDecoder',
    'SinusoidalPositionalEncoding',
    'SOS_TOKEN',
    'EOS_TOKEN',
    'TOTAL_VOCAB',
]
