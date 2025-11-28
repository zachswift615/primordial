"""Living Resonance Network - Fourier-based neural architecture."""
from .config import PrototypeConfig
from .lrn_config import LRNConfig
from .mixing import FourierMixingLayer
from .lrn_mixing import LRNFourierMixingLayer
from .heads import SensoryHead, RewardHead
from .lrn_heads import PredictionHead, LRNRewardHead, ActionHead
from .prototype import FourierPrototype
from .utils import init_spectral_filter, complex_to_real, real_to_complex
from .encoders import (
    WaveletEncoder,
    VisionEncoder,
    AudioEncoder,
    ProprioEncoder,
    TouchEncoder,
)
from .genome import GenomeModulator
from .architecture import LivingResonanceNetwork
from .learning import (
    RewardHistoryBuffer,
    GradientClipper,
    ExponentialMovingAverage,
    GradientMonitor,
    OnlineLRScheduler,
)

__all__ = [
    "PrototypeConfig",
    "LRNConfig",
    "FourierMixingLayer",
    "LRNFourierMixingLayer",
    "SensoryHead",
    "RewardHead",
    "PredictionHead",
    "LRNRewardHead",
    "ActionHead",
    "FourierPrototype",
    "init_spectral_filter",
    "complex_to_real",
    "real_to_complex",
    "WaveletEncoder",
    "VisionEncoder",
    "AudioEncoder",
    "ProprioEncoder",
    "TouchEncoder",
    "GenomeModulator",
    "LivingResonanceNetwork",
    "RewardHistoryBuffer",
    "GradientClipper",
    "ExponentialMovingAverage",
    "GradientMonitor",
    "OnlineLRScheduler",
]
