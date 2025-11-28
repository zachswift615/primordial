import pytest
import numpy as np
from unittest.mock import Mock, patch
from primordial.interface.audio_capture import AudioCapture


def test_audio_capture_initialization():
    capture = AudioCapture(sample_rate=44100, channels=1, buffer_size=1024)
    assert capture.sample_rate == 44100
    assert capture.channels == 1
    assert not capture.is_recording()


def test_audio_capture_start_stop():
    capture = AudioCapture(sample_rate=44100, channels=1, buffer_size=1024)

    capture.start()
    assert capture.is_recording()

    capture.stop()
    assert not capture.is_recording()


@patch('sounddevice.InputStream')
def test_audio_capture_collects_data(mock_stream):
    """Test that audio data is collected in buffer."""
    capture = AudioCapture(sample_rate=44100, channels=1, buffer_size=1024)

    # Simulate audio callback
    test_audio = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    capture._audio_callback(test_audio, frames=3, time_info=None, status=None)

    buffer = capture.get_buffer()
    assert len(buffer) == 3
    assert buffer[0] == pytest.approx(0.1)


def test_get_recent_audio():
    capture = AudioCapture(sample_rate=44100, channels=1, buffer_size=1024)

    # Add some test data
    test_data = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    capture._audio_callback(test_data, frames=3, time_info=None, status=None)

    recent = capture.get_recent(num_samples=2)
    assert len(recent) == 2
    np.testing.assert_array_almost_equal(recent, np.array([0.2, 0.3]))


def test_buffer_wraps_when_full():
    """Test circular buffer behavior."""
    capture = AudioCapture(sample_rate=44100, channels=1, buffer_size=5)

    # Fill buffer beyond capacity
    for i in range(10):
        data = np.array([[float(i)]], dtype=np.float32)
        capture._audio_callback(data, frames=1, time_info=None, status=None)

    buffer = capture.get_buffer()
    # Should only have last 5 samples
    assert len(buffer) <= 5
