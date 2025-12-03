"""Latent phoneme space for speech production.

Defines a 6-dimensional articulatory feature space where phonemes are positioned
based on their physical production characteristics. This enables smooth
interpolation between sounds and linguistically meaningful navigation.

Dimensions:
    0: Front-Back     - Vowel frontness / Consonant place of articulation
    1: High-Low       - Vowel height (tongue position)
    2: Rounded        - Lip rounding
    3: Voiced         - Voicing (vocal cord vibration)
    4: Manner         - Manner of articulation (stop/fricative/nasal/approximant)
    5: Vowel-Cons     - Vowel vs consonant distinction
"""
import torch
from typing import Dict, Tuple, List

# Latent space dimensionality
LATENT_DIM = 6

# Feature indices for readability
FRONT_BACK = 0
HIGH_LOW = 1
ROUNDED = 2
VOICED = 3
MANNER = 4
VOWEL_CONS = 5

# Sequence tokens (for autoregressive decoder)
SOS_TOKEN = 41  # Start of sequence
EOS_TOKEN = 42  # End of sequence (also used as PAD)
TOTAL_VOCAB = 43  # 41 phonemes + SOS + EOS

# Phoneme anchor positions in the 6D latent space
# Format: [front-back, high-low, rounded, voiced, manner, vowel-cons]
# Range: -1.0 to 1.0 for each dimension

PHONEME_ANCHORS: Dict[str, List[float]] = {
    # ============= VOWELS (vowel-cons = -1.0) =============
    # Vowels are all voiced (dim 3 = 1.0), manner neutral (dim 4 = 0.0)

    # Front vowels (front-back = 1.0)
    'IY': [ 1.0,  1.0, -1.0,  1.0,  0.0, -1.0],  # beat - front, high, unrounded
    'IH': [ 0.8,  0.7, -1.0,  1.0,  0.0, -1.0],  # bit - front, high-mid, unrounded
    'EY': [ 0.9,  0.4, -1.0,  1.0,  0.0, -1.0],  # bait - front, mid, unrounded (diphthong)
    'EH': [ 0.8,  0.2, -1.0,  1.0,  0.0, -1.0],  # bet - front, mid-low, unrounded
    'AE': [ 0.7, -0.5, -1.0,  1.0,  0.0, -1.0],  # bat - front, low, unrounded

    # Central vowels (front-back = 0.0)
    'AH': [ 0.0,  0.0, -1.0,  1.0,  0.0, -1.0],  # but - central, mid, unrounded
    'ER': [ 0.0,  0.3, -1.0,  1.0,  0.0, -1.0],  # bird - central, mid, r-colored

    # Back vowels (front-back = -1.0)
    'UW': [-1.0,  1.0,  1.0,  1.0,  0.0, -1.0],  # boot - back, high, rounded
    'UH': [-0.8,  0.6,  0.8,  1.0,  0.0, -1.0],  # book - back, high-mid, rounded
    'OW': [-0.9,  0.3,  1.0,  1.0,  0.0, -1.0],  # boat - back, mid, rounded (diphthong)
    'AO': [-0.8, -0.2,  0.8,  1.0,  0.0, -1.0],  # bought - back, low-mid, rounded
    'AA': [-0.6, -0.8, -0.5,  1.0,  0.0, -1.0],  # bot - back, low, unrounded

    # Diphthongs (positioned at their starting point with slight modification)
    'AY': [ 0.3, -0.6, -1.0,  1.0,  0.0, -1.0],  # bite - starts low-central
    'AW': [-0.3, -0.6,  0.3,  1.0,  0.0, -1.0],  # bout - starts low-back
    'OY': [-0.5,  0.0,  0.8,  1.0,  0.0, -1.0],  # boy - starts mid-back rounded

    # ============= CONSONANTS (vowel-cons = 1.0) =============

    # --- STOPS (manner = -1.0) ---
    # Bilabial (front-back = -1.0, lips)
    'P':  [-1.0,  0.0,  0.0, -1.0, -1.0,  1.0],  # unvoiced bilabial stop
    'B':  [-1.0,  0.0,  0.0,  1.0, -1.0,  1.0],  # voiced bilabial stop

    # Alveolar (front-back = 0.0, tongue tip)
    'T':  [ 0.0,  0.0,  0.0, -1.0, -1.0,  1.0],  # unvoiced alveolar stop
    'D':  [ 0.0,  0.0,  0.0,  1.0, -1.0,  1.0],  # voiced alveolar stop

    # Velar (front-back = 1.0, back of tongue)
    'K':  [ 1.0,  0.0,  0.0, -1.0, -1.0,  1.0],  # unvoiced velar stop
    'G':  [ 1.0,  0.0,  0.0,  1.0, -1.0,  1.0],  # voiced velar stop

    # --- FRICATIVES (manner = 0.5) ---
    # Labiodental (front-back = -0.8)
    'F':  [-0.8,  0.0,  0.0, -1.0,  0.5,  1.0],  # unvoiced labiodental fricative
    'V':  [-0.8,  0.0,  0.0,  1.0,  0.5,  1.0],  # voiced labiodental fricative

    # Dental (front-back = -0.4)
    'TH': [-0.4,  0.0,  0.0, -1.0,  0.5,  1.0],  # unvoiced dental fricative (think)
    'DH': [-0.4,  0.0,  0.0,  1.0,  0.5,  1.0],  # voiced dental fricative (the)

    # Alveolar (front-back = 0.0)
    'S':  [ 0.0,  0.0,  0.0, -1.0,  0.5,  1.0],  # unvoiced alveolar fricative
    'Z':  [ 0.0,  0.0,  0.0,  1.0,  0.5,  1.0],  # voiced alveolar fricative

    # Postalveolar (front-back = 0.3)
    'SH': [ 0.3,  0.0,  0.0, -1.0,  0.5,  1.0],  # unvoiced postalveolar fricative
    'ZH': [ 0.3,  0.0,  0.0,  1.0,  0.5,  1.0],  # voiced postalveolar fricative (measure)

    # Glottal (front-back = 1.0)
    'HH': [ 1.0,  0.0,  0.0, -1.0,  0.5,  1.0],  # glottal fricative

    # --- AFFRICATES (manner = -0.5, between stop and fricative) ---
    'CH': [ 0.3,  0.0,  0.0, -1.0, -0.5,  1.0],  # unvoiced postalveolar affricate
    'JH': [ 0.3,  0.0,  0.0,  1.0, -0.5,  1.0],  # voiced postalveolar affricate

    # --- NASALS (manner = 0.8, nasal airflow) ---
    'M':  [-1.0,  0.0,  0.0,  1.0,  0.8,  1.0],  # bilabial nasal
    'N':  [ 0.0,  0.0,  0.0,  1.0,  0.8,  1.0],  # alveolar nasal
    'NG': [ 1.0,  0.0,  0.0,  1.0,  0.8,  1.0],  # velar nasal

    # --- APPROXIMANTS/LIQUIDS (manner = 1.0, vowel-like consonants) ---
    'L':  [ 0.0,  0.0,  0.0,  1.0,  1.0,  1.0],  # alveolar lateral approximant
    'R':  [ 0.2,  0.0,  0.0,  1.0,  1.0,  1.0],  # alveolar approximant
    'W':  [-1.0,  0.5,  1.0,  1.0,  1.0,  1.0],  # labial-velar approximant (rounded)
    'Y':  [ 0.8,  0.8, -1.0,  1.0,  1.0,  1.0],  # palatal approximant

    # --- SPECIAL ---
    'SIL': [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # silence - origin
    'UNK': [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],  # unknown - origin

    # ============= SEQUENCE TOKENS =============
    'SOS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Start of sequence - origin
    'EOS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # End of sequence - origin
}

# Pre-compute tensor anchors for efficient distance calculations
_ANCHOR_TENSOR: torch.Tensor = None
_ANCHOR_PHONEMES: List[str] = None


def _ensure_anchor_tensor(device: str = "cpu") -> Tuple[torch.Tensor, List[str]]:
    """Lazily create anchor tensor on first use."""
    global _ANCHOR_TENSOR, _ANCHOR_PHONEMES
    if _ANCHOR_TENSOR is None:
        _ANCHOR_PHONEMES = list(PHONEME_ANCHORS.keys())
        _ANCHOR_TENSOR = torch.tensor(
            [PHONEME_ANCHORS[p] for p in _ANCHOR_PHONEMES],
            dtype=torch.float32
        )
    return _ANCHOR_TENSOR.to(device), _ANCHOR_PHONEMES


def get_anchor(phoneme: str) -> torch.Tensor:
    """Get the anchor position for a phoneme.

    Args:
        phoneme: ARPABET phoneme string (e.g., 'IY', 'B', 'SIL')

    Returns:
        (6,) tensor of anchor coordinates
    """
    if phoneme not in PHONEME_ANCHORS:
        phoneme = 'UNK'
    return torch.tensor(PHONEME_ANCHORS[phoneme], dtype=torch.float32)


def snap_to_nearest_anchor(latent: torch.Tensor) -> Tuple[str, float]:
    """Find the nearest phoneme anchor to a latent position.

    Args:
        latent: (6,) or (batch, 6) latent position

    Returns:
        If input is (6,): (phoneme, distance)
        If input is (batch, 6): (list of phonemes, tensor of distances)
    """
    anchors, phonemes = _ensure_anchor_tensor(latent.device)

    if latent.dim() == 1:
        # Single latent
        distances = torch.norm(anchors - latent.unsqueeze(0), dim=1)
        min_idx = distances.argmin().item()
        return phonemes[min_idx], distances[min_idx].item()
    else:
        # Batch of latents
        # latent: (batch, 6), anchors: (41, 6)
        # Compute pairwise distances: (batch, 41)
        distances = torch.cdist(latent, anchors)
        min_indices = distances.argmin(dim=1)
        min_distances = distances.gather(1, min_indices.unsqueeze(1)).squeeze(1)
        nearest_phonemes = [phonemes[i] for i in min_indices.tolist()]
        return nearest_phonemes, min_distances


def get_anchor_distance(latent: torch.Tensor, phoneme: str) -> float:
    """Get distance from latent position to a specific phoneme anchor.

    Args:
        latent: (6,) latent position
        phoneme: Target phoneme

    Returns:
        Euclidean distance
    """
    anchor = get_anchor(phoneme).to(latent.device)
    return torch.norm(latent - anchor).item()


def get_k_nearest_anchors(latent: torch.Tensor, k: int = 3) -> List[Tuple[str, float]]:
    """Get the k nearest phoneme anchors to a latent position.

    Args:
        latent: (6,) latent position
        k: Number of nearest neighbors

    Returns:
        List of (phoneme, distance) tuples, sorted by distance
    """
    anchors, phonemes = _ensure_anchor_tensor(latent.device)
    distances = torch.norm(anchors - latent.unsqueeze(0), dim=1)
    topk = distances.topk(k, largest=False)

    return [(phonemes[i], d.item()) for i, d in zip(topk.indices.tolist(), topk.values)]


def interpret_latent(latent: torch.Tensor) -> Dict[str, str]:
    """Interpret a latent vector in terms of articulatory features.

    Args:
        latent: (6,) latent position

    Returns:
        Dict mapping feature names to human-readable descriptions
    """
    l = latent.tolist()

    def describe_range(val: float, low: str, mid: str, high: str) -> str:
        if val < -0.5:
            return low
        elif val > 0.5:
            return high
        else:
            return mid

    return {
        'front_back': describe_range(l[FRONT_BACK], 'back/labial', 'central/alveolar', 'front/velar'),
        'high_low': describe_range(l[HIGH_LOW], 'low', 'mid', 'high'),
        'rounded': describe_range(l[ROUNDED], 'unrounded', 'neutral', 'rounded'),
        'voiced': describe_range(l[VOICED], 'unvoiced', 'neutral', 'voiced'),
        'manner': describe_range(l[MANNER], 'stop', 'fricative', 'nasal/approximant'),
        'type': describe_range(l[VOWEL_CONS], 'vowel', 'ambiguous', 'consonant'),
    }


def get_all_anchors_tensor(device: str = "cpu") -> Tuple[torch.Tensor, List[str]]:
    """Get all anchor positions as a tensor.

    Args:
        device: Target device

    Returns:
        (anchors tensor (41, 6), list of phoneme names)
    """
    return _ensure_anchor_tensor(device)
