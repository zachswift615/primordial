"""Word dataset for sequence training."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Optional, Tuple

from .config import SpeechConfig
from .tts import create_tts_backend
from .encoders import compute_mel_spectrogram
from .phonemes import phoneme_to_index
from .latent import SOS_TOKEN, EOS_TOKEN


# Word to phoneme mappings
WORD_PHONEMES = {
    # Phase 1: Simple syllables (2-3 phonemes)
    'ba':    ['B', 'AA'],
    'bee':   ['B', 'IY'],
    'ma':    ['M', 'AA'],
    'me':    ['M', 'IY'],
    'hi':    ['HH', 'AY'],
    'go':    ['G', 'OW'],
    'no':    ['N', 'OW'],
    'see':   ['S', 'IY'],
    'you':   ['Y', 'UW'],
    'we':    ['W', 'IY'],

    # Phase 2: Short words (4-5 phonemes)
    'hello': ['HH', 'EH', 'L', 'OW'],
    'water': ['W', 'AO', 'T', 'ER'],
    'mommy': ['M', 'AA', 'M', 'IY'],
    'daddy': ['D', 'AE', 'D', 'IY'],
    'baby':  ['B', 'EY', 'B', 'IY'],
    'happy': ['HH', 'AE', 'P', 'IY'],
    'sorry': ['S', 'AA', 'R', 'IY'],
    'open':  ['OW', 'P', 'AH', 'N'],
    'over':  ['OW', 'V', 'ER'],
    'under': ['AH', 'N', 'D', 'ER'],

    # Phase 3: Medium words (5-7 phonemes)
    'banana':   ['B', 'AH', 'N', 'AE', 'N', 'AH'],
    'monkey':   ['M', 'AH', 'NG', 'K', 'IY'],
    'pizza':    ['P', 'IY', 'T', 'S', 'AH'],
    'chicken':  ['CH', 'IH', 'K', 'AH', 'N'],
    'rainbow':  ['R', 'EY', 'N', 'B', 'OW'],
    'purple':   ['P', 'ER', 'P', 'AH', 'L'],
    'orange':   ['AO', 'R', 'AH', 'N', 'JH'],
    'yellow':   ['Y', 'EH', 'L', 'OW'],
    'rabbit':   ['R', 'AE', 'B', 'AH', 'T'],
    'kitten':   ['K', 'IH', 'T', 'AH', 'N'],
    'puppy':    ['P', 'AH', 'P', 'IY'],
    'morning':  ['M', 'AO', 'R', 'N', 'IH', 'NG'],
    'evening':  ['IY', 'V', 'N', 'IH', 'NG'],
    'singing':  ['S', 'IH', 'NG', 'IH', 'NG'],
    'jumping':  ['JH', 'AH', 'M', 'P', 'IH', 'NG'],
    'running':  ['R', 'AH', 'N', 'IH', 'NG'],

    # Phase 4: Longer words (7+ phonemes)
    'computer':   ['K', 'AH', 'M', 'P', 'Y', 'UW', 'T', 'ER'],
    'elephant':   ['EH', 'L', 'AH', 'F', 'AH', 'N', 'T'],
    'tomorrow':   ['T', 'AH', 'M', 'AA', 'R', 'OW'],
    'together':   ['T', 'AH', 'G', 'EH', 'DH', 'ER'],
    'beautiful':  ['B', 'Y', 'UW', 'T', 'AH', 'F', 'AH', 'L'],
    'wonderful':  ['W', 'AH', 'N', 'D', 'ER', 'F', 'AH', 'L'],
    'butterfly':  ['B', 'AH', 'T', 'ER', 'F', 'L', 'AY'],
    'caterpillar': ['K', 'AE', 'T', 'ER', 'P', 'IH', 'L', 'ER'],
    'dinosaur':   ['D', 'AY', 'N', 'AH', 'S', 'AO', 'R'],
    'helicopter': ['HH', 'EH', 'L', 'AH', 'K', 'AA', 'P', 'T', 'ER'],
    'alligator':  ['AE', 'L', 'AH', 'G', 'EY', 'T', 'ER'],
    'crocodile':  ['K', 'R', 'AA', 'K', 'AH', 'D', 'AY', 'L'],
    'strawberry': ['S', 'T', 'R', 'AO', 'B', 'EH', 'R', 'IY'],
    'watermelon': ['W', 'AO', 'T', 'ER', 'M', 'EH', 'L', 'AH', 'N'],
    'understand': ['AH', 'N', 'D', 'ER', 'S', 'T', 'AE', 'N', 'D'],
}


class WordDataset(Dataset):
    """Dataset of words with their phoneme sequences.

    Generates TTS audio for each word and provides:
    - mel: Mel spectrogram of the word
    - input_tokens: [SOS, phoneme1, phoneme2, ...] for teacher forcing
    - target_tokens: [phoneme1, phoneme2, ..., EOS] for loss
    """

    def __init__(
        self,
        config: SpeechConfig,
        words: Optional[List[str]] = None,
        max_phonemes: Optional[int] = None,
    ):
        """
        Args:
            config: Speech configuration
            words: Specific words to include (default: all)
            max_phonemes: Filter to words with at most this many phonemes
        """
        self.config = config
        self.tts = create_tts_backend(config)

        # Filter words
        if words is not None:
            self.words = [w for w in words if w in WORD_PHONEMES]
        else:
            self.words = list(WORD_PHONEMES.keys())

        if max_phonemes is not None:
            self.words = [
                w for w in self.words
                if len(WORD_PHONEMES[w]) <= max_phonemes
            ]

        # Pre-generate audio and mels
        self._cache = {}
        self._prepare_data()

    def _prepare_data(self):
        """Pre-generate mel spectrograms for all words."""
        for word in self.words:
            phonemes = WORD_PHONEMES[word]

            # Synthesize audio
            audio = self.tts.synthesize_phonemes(phonemes)
            waveform = torch.from_numpy(audio).float()

            # Resample if needed
            if self.tts.sample_rate != self.config.sample_rate:
                target_len = int(len(waveform) * self.config.sample_rate / self.tts.sample_rate)
                waveform = F.interpolate(
                    waveform.view(1, 1, -1),
                    size=target_len,
                    mode='linear',
                    align_corners=False
                ).squeeze()

            # Compute mel spectrogram
            mel = compute_mel_spectrogram(
                waveform,
                sample_rate=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
            ).squeeze(0)

            # Pad/truncate to standard size
            if mel.shape[1] < self.config.n_frames:
                mel = F.pad(mel, (0, self.config.n_frames - mel.shape[1]))
            else:
                mel = mel[:, :self.config.n_frames]

            # Convert phonemes to token indices
            phoneme_indices = [phoneme_to_index(p) for p in phonemes]

            # Input: [SOS, phoneme1, phoneme2, ...]
            input_tokens = torch.tensor([SOS_TOKEN] + phoneme_indices, dtype=torch.long)

            # Target: [phoneme1, phoneme2, ..., EOS]
            target_tokens = torch.tensor(phoneme_indices + [EOS_TOKEN], dtype=torch.long)

            self._cache[word] = (mel, input_tokens, target_tokens)

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            mel: (n_mels, n_frames) mel spectrogram
            input_tokens: (seq_len,) input sequence with SOS
            target_tokens: (seq_len,) target sequence with EOS
            word: str - the word
        """
        word = self.words[idx]
        mel, input_tokens, target_tokens = self._cache[word]
        return mel, input_tokens, target_tokens, word
