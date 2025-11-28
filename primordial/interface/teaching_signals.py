"""Teaching signal system for human-to-agent communication."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np


class TeachingSignalType(Enum):
    """Types of teaching signals."""
    REWARD = "reward"
    PUNISH = "punish"
    POINT = "point"
    DEMONSTRATE = "demonstrate"
    VOICE = "voice"


@dataclass
class TeachingSignal:
    """A single teaching signal from human to agent."""
    signal_type: TeachingSignalType
    timestamp: float
    intensity: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def reward(cls, timestamp: float, intensity: float = 1.0) -> "TeachingSignal":
        """Create a reward signal."""
        return cls(
            signal_type=TeachingSignalType.REWARD,
            timestamp=timestamp,
            intensity=intensity
        )

    @classmethod
    def punish(cls, timestamp: float, intensity: float = 1.0) -> "TeachingSignal":
        """Create a punishment signal."""
        return cls(
            signal_type=TeachingSignalType.PUNISH,
            timestamp=timestamp,
            intensity=intensity
        )

    @classmethod
    def point(cls, timestamp: float, x: int, y: int) -> "TeachingSignal":
        """Create a pointing signal."""
        return cls(
            signal_type=TeachingSignalType.POINT,
            timestamp=timestamp,
            data={"x": x, "y": y}
        )

    @classmethod
    def demonstrate(cls, timestamp: float, action: str) -> "TeachingSignal":
        """Create a demonstration signal."""
        return cls(
            signal_type=TeachingSignalType.DEMONSTRATE,
            timestamp=timestamp,
            data={"action": action}
        )

    @classmethod
    def voice(cls, timestamp: float, audio: np.ndarray) -> "TeachingSignal":
        """Create a voice signal."""
        return cls(
            signal_type=TeachingSignalType.VOICE,
            timestamp=timestamp,
            data={"audio": audio}
        )


class TeachingSignalQueue:
    """Thread-safe queue for teaching signals."""

    def __init__(self, max_size: int = 1000):
        self._queue: deque = deque(maxlen=max_size)
        self._max_size = max_size

    def enqueue(self, signal: TeachingSignal) -> None:
        """Add a signal to the queue."""
        self._queue.append(signal)

    def dequeue(self) -> Optional[TeachingSignal]:
        """Remove and return the oldest signal."""
        if self._queue:
            return self._queue.popleft()
        return None

    def size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def get_recent(self, since_timestamp: float) -> List[TeachingSignal]:
        """Get all signals since a given timestamp."""
        return [s for s in self._queue if s.timestamp > since_timestamp]

    def clear(self) -> None:
        """Clear all signals."""
        self._queue.clear()
