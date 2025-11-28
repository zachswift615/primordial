"""Human teaching interface for Primordial agents."""

from primordial.interface.app import TeachingApp
from primordial.interface.config import UIConfig
from primordial.interface.teaching_signals import (
    TeachingSignal, TeachingSignalType, TeachingSignalQueue
)

__version__ = "0.1.0"
__all__ = [
    "TeachingApp",
    "UIConfig",
    "TeachingSignal",
    "TeachingSignalType",
    "TeachingSignalQueue"
]
