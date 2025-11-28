"""Background audio capture from microphone."""

import threading
from typing import Optional
import numpy as np
import sounddevice as sd
from collections import deque


class AudioCapture:
    """Captures audio from microphone in background thread."""

    def __init__(self, sample_rate: int = 44100, channels: int = 1,
                 buffer_size: int = 44100):
        """
        Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1=mono, 2=stereo)
            buffer_size: Number of samples to keep in circular buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size

        # Circular buffer for audio samples
        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # Stream state
        self._stream: Optional[sd.InputStream] = None
        self._recording = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio callback status: {status}")

        # Convert to 1D array and store in buffer
        audio_data = indata[:, 0] if self.channels == 1 else indata

        with self._lock:
            for sample in audio_data:
                self._buffer.append(float(sample))

    def start(self) -> None:
        """Start recording audio."""
        if self._recording:
            return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=1024
        )
        self._stream.start()
        self._recording = True

    def stop(self) -> None:
        """Stop recording audio."""
        if not self._recording:
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._recording = False

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_buffer(self) -> np.ndarray:
        """Get the entire audio buffer."""
        with self._lock:
            return np.array(list(self._buffer), dtype=np.float32)

    def get_recent(self, num_samples: int) -> np.ndarray:
        """Get the most recent N samples."""
        with self._lock:
            buffer_list = list(self._buffer)
            recent = buffer_list[-num_samples:] if len(buffer_list) >= num_samples else buffer_list
            return np.array(recent, dtype=np.float32)

    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        with self._lock:
            self._buffer.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
