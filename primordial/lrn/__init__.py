"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .lrn_config import LRNConfig
from .mixing import FourierMixingLayer
from .heads import SensoryHead, RewardHead
from .prototype import FourierPrototype
from .utils import init_spectral_filter, complex_to_real, real_to_complex

__all__ = [
    "PrototypeConfig",
    "LRNConfig",
    "FourierMixingLayer",
    "SensoryHead",
    "RewardHead",
    "FourierPrototype",
    "init_spectral_filter",
    "complex_to_real",
    "real_to_complex",
]
