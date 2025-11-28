"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .lrn_config import LRNConfig
from .mixing import FourierMixingLayer
from .heads import SensoryHead, RewardHead
from .prototype import FourierPrototype

__all__ = [
    "PrototypeConfig",
    "LRNConfig",
    "FourierMixingLayer",
    "SensoryHead",
    "RewardHead",
    "FourierPrototype",
]
