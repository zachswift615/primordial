import pytest
import numpy as np
from primordial.interface.teaching_signals import (
    TeachingSignal, TeachingSignalType, TeachingSignalQueue
)


def test_reward_signal_creation():
    signal = TeachingSignal.reward(timestamp=1.5, intensity=0.8)
    assert signal.signal_type == TeachingSignalType.REWARD
    assert signal.timestamp == 1.5
    assert signal.intensity == 0.8


def test_punishment_signal_creation():
    signal = TeachingSignal.punish(timestamp=2.0, intensity=1.0)
    assert signal.signal_type == TeachingSignalType.PUNISH
    assert signal.timestamp == 2.0
    assert signal.intensity == 1.0


def test_pointing_signal_with_coordinates():
    signal = TeachingSignal.point(timestamp=3.0, x=100, y=200)
    assert signal.signal_type == TeachingSignalType.POINT
    assert signal.data["x"] == 100
    assert signal.data["y"] == 200


def test_demonstration_signal_with_action():
    signal = TeachingSignal.demonstrate(timestamp=4.0, action="move_forward")
    assert signal.signal_type == TeachingSignalType.DEMONSTRATE
    assert signal.data["action"] == "move_forward"


def test_voice_signal_with_audio():
    audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    signal = TeachingSignal.voice(timestamp=5.0, audio=audio_data)
    assert signal.signal_type == TeachingSignalType.VOICE
    np.testing.assert_array_equal(signal.data["audio"], audio_data)


def test_signal_queue_enqueue_dequeue():
    queue = TeachingSignalQueue(max_size=100)
    signal = TeachingSignal.reward(timestamp=1.0)

    queue.enqueue(signal)
    assert queue.size() == 1

    retrieved = queue.dequeue()
    assert retrieved == signal
    assert queue.size() == 0


def test_signal_queue_get_recent_signals():
    queue = TeachingSignalQueue(max_size=100)

    queue.enqueue(TeachingSignal.reward(timestamp=1.0))
    queue.enqueue(TeachingSignal.punish(timestamp=2.0))
    queue.enqueue(TeachingSignal.point(timestamp=3.0, x=0, y=0))

    recent = queue.get_recent(since_timestamp=1.5)
    assert len(recent) == 2
    assert recent[0].timestamp == 2.0
    assert recent[1].timestamp == 3.0
