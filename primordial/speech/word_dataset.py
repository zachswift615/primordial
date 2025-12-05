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


# Phrase to phoneme mappings (multi-word)
PHRASE_PHONEMES = {
    # 2-word phrases (6-10 phonemes)
    'hello world': ['HH', 'AH', 'L', 'OW', 'W', 'ER', 'L', 'D'],
    'good morning': ['G', 'UH', 'D', 'M', 'AO', 'R', 'N', 'IH', 'NG'],
    'thank you': ['TH', 'AE', 'NG', 'K', 'Y', 'UW'],
    'bye bye': ['B', 'AY', 'B', 'AY'],
    'come here': ['K', 'AH', 'M', 'HH', 'IH', 'R'],
    'go away': ['G', 'OW', 'AH', 'W', 'EY'],
    'sit down': ['S', 'IH', 'T', 'D', 'AW', 'N'],
    'stand up': ['S', 'T', 'AE', 'N', 'D', 'AH', 'P'],
    'look here': ['L', 'UH', 'K', 'HH', 'IH', 'R'],
    'watch me': ['W', 'AA', 'CH', 'M', 'IY'],
    'help me': ['HH', 'EH', 'L', 'P', 'M', 'IY'],
    'hold on': ['HH', 'OW', 'L', 'D', 'AA', 'N'],
    'wake up': ['W', 'EY', 'K', 'AH', 'P'],
    'slow down': ['S', 'L', 'OW', 'D', 'AW', 'N'],
    'hurry up': ['HH', 'ER', 'IY', 'AH', 'P'],
    'good night': ['G', 'UH', 'D', 'N', 'AY', 'T'],
    'good job': ['G', 'UH', 'D', 'JH', 'AA', 'B'],
    'my turn': ['M', 'AY', 'T', 'ER', 'N'],
    'your turn': ['Y', 'AO', 'R', 'T', 'ER', 'N'],
    'all done': ['AO', 'L', 'D', 'AH', 'N'],

    # 3-word phrases (10-15 phonemes)
    'I love you': ['AY', 'L', 'AH', 'V', 'Y', 'UW'],
    'how are you': ['HH', 'AW', 'AA', 'R', 'Y', 'UW'],
    'see you later': ['S', 'IY', 'Y', 'UW', 'L', 'EY', 'T', 'ER'],
    'nice to meet': ['N', 'AY', 'S', 'T', 'UW', 'M', 'IY', 'T'],
    'what is that': ['W', 'AH', 'T', 'IH', 'Z', 'DH', 'AE', 'T'],
    'where are you': ['W', 'EH', 'R', 'AA', 'R', 'Y', 'UW'],
    'I want food': ['AY', 'W', 'AA', 'N', 'T', 'F', 'UW', 'D'],
    'give me water': ['G', 'IH', 'V', 'M', 'IY', 'W', 'AO', 'T', 'ER'],
    'open the door': ['OW', 'P', 'AH', 'N', 'DH', 'AH', 'D', 'AO', 'R'],
    'close your eyes': ['K', 'L', 'OW', 'Z', 'Y', 'AO', 'R', 'AY', 'Z'],
    'I am happy': ['AY', 'AE', 'M', 'HH', 'AE', 'P', 'IY'],
    'I am hungry': ['AY', 'AE', 'M', 'HH', 'AH', 'NG', 'G', 'R', 'IY'],
    'I am tired': ['AY', 'AE', 'M', 'T', 'AY', 'ER', 'D'],
    'please help me': ['P', 'L', 'IY', 'Z', 'HH', 'EH', 'L', 'P', 'M', 'IY'],
    'let me try': ['L', 'EH', 'T', 'M', 'IY', 'T', 'R', 'AY'],
    'I can do': ['AY', 'K', 'AE', 'N', 'D', 'UW'],
    'yes I can': ['Y', 'EH', 'S', 'AY', 'K', 'AE', 'N'],
    'no thank you': ['N', 'OW', 'TH', 'AE', 'NG', 'K', 'Y', 'UW'],
    'wait for me': ['W', 'EY', 'T', 'F', 'AO', 'R', 'M', 'IY'],
    'come with me': ['K', 'AH', 'M', 'W', 'IH', 'DH', 'M', 'IY'],

    # Short sentences (15-22 phonemes)
    'the cat is sleeping': ['DH', 'AH', 'K', 'AE', 'T', 'IH', 'Z', 'S', 'L', 'IY', 'P', 'IH', 'NG'],
    'I see a ball': ['AY', 'S', 'IY', 'AH', 'B', 'AO', 'L'],
    'the dog is running': ['DH', 'AH', 'D', 'AO', 'G', 'IH', 'Z', 'R', 'AH', 'N', 'IH', 'NG'],
    'can you help me': ['K', 'AE', 'N', 'Y', 'UW', 'HH', 'EH', 'L', 'P', 'M', 'IY'],
    'I want to go home': ['AY', 'W', 'AA', 'N', 'T', 'T', 'UW', 'G', 'OW', 'HH', 'OW', 'M'],
    'where is my mommy': ['W', 'EH', 'R', 'IH', 'Z', 'M', 'AY', 'M', 'AA', 'M', 'IY'],
    'I like this book': ['AY', 'L', 'AY', 'K', 'DH', 'IH', 'S', 'B', 'UH', 'K'],
    'what do you want': ['W', 'AH', 'T', 'D', 'UW', 'Y', 'UW', 'W', 'AA', 'N', 'T'],
    'I need a hug': ['AY', 'N', 'IY', 'D', 'AH', 'HH', 'AH', 'G'],
    'that is so funny': ['DH', 'AE', 'T', 'IH', 'Z', 'S', 'OW', 'F', 'AH', 'N', 'IY'],

    # Longer sentences (22-30 phonemes)
    'the little bird is singing': ['DH', 'AH', 'L', 'IH', 'T', 'AH', 'L', 'B', 'ER', 'D', 'IH', 'Z', 'S', 'IH', 'NG', 'IH', 'NG'],
    'I like to eat bananas': ['AY', 'L', 'AY', 'K', 'T', 'UW', 'IY', 'T', 'B', 'AH', 'N', 'AE', 'N', 'AH', 'Z'],
    'the sun is very bright': ['DH', 'AH', 'S', 'AH', 'N', 'IH', 'Z', 'V', 'EH', 'R', 'IY', 'B', 'R', 'AY', 'T'],
    'I want to play outside': ['AY', 'W', 'AA', 'N', 'T', 'T', 'UW', 'P', 'L', 'EY', 'AW', 'T', 'S', 'AY', 'D'],
    'can we go to the park': ['K', 'AE', 'N', 'W', 'IY', 'G', 'OW', 'T', 'UW', 'DH', 'AH', 'P', 'AA', 'R', 'K'],
}


def get_all_entries(include_phrases: bool = True) -> dict:
    """Get combined word and phrase phoneme mappings.

    Args:
        include_phrases: Whether to include multi-word phrases

    Returns:
        Dict mapping words/phrases to phoneme lists
    """
    entries = dict(WORD_PHONEMES)
    if include_phrases:
        entries.update(PHRASE_PHONEMES)
    return entries


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
        include_phrases: bool = False,
    ):
        """
        Args:
            config: Speech configuration
            words: Specific words to include (default: all)
            max_phonemes: Filter to words with at most this many phonemes
            include_phrases: Include multi-word phrases in dataset
        """
        self.config = config
        self.tts = create_tts_backend(config)
        self.include_phrases = include_phrases

        # Get all entries (words + optionally phrases)
        all_entries = get_all_entries(include_phrases=include_phrases)

        # Filter words
        if words is not None:
            self.words = [w for w in words if w in all_entries]
        else:
            self.words = list(all_entries.keys())

        if max_phonemes is not None:
            self.words = [
                w for w in self.words
                if len(all_entries[w]) <= max_phonemes
            ]

        # Store reference to entries for phoneme lookup
        self._entries = all_entries

        # Pre-generate audio and mels
        self._cache = {}
        self._prepare_data()

    def _prepare_data(self):
        """Pre-generate mel spectrograms for all words."""
        for word in self.words:
            phonemes = self._entries[word]

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
