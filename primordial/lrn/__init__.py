"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .mixing import FourierMixingLayer

__all__ = ["PrototypeConfig", "FourierMixingLayer"]
