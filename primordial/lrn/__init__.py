"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .mixing import FourierMixingLayer
from .heads import SensoryHead, RewardHead
from .prototype import FourierPrototype

__all__ = [
    "PrototypeConfig",
    "FourierMixingLayer",
    "SensoryHead",
    "RewardHead",
    "FourierPrototype",
]
