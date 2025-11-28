# Human Teaching Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a real-time interactive teaching interface where humans train AI agents through reward/punishment, pointing, demonstration, and voice input, displayed via Pygame with multiple view panels.

**Architecture:** Event-driven Pygame application with modular input handlers (keyboard, mouse, controller, microphone), rendering engine with multiple viewport panels, and teaching signal system that converts human inputs into agent learning signals. Audio capture runs in background thread, inputs processed at 60 FPS with low-latency response.

**Tech Stack:** Pygame (rendering/input), sounddevice (microphone), pygame-joystick (controller), numpy (audio/graphs), dataclasses (state management)

---

## UI Layout Specification

```
┌─────────────────────────────────────────────────────────────────────┐
│  Primordial Teaching Interface                    [FPS: 60] [●REC]  │
├──────────────────────────────┬──────────────────────────────────────┤
│                              │  Agent POV (320x240)                 │
│  World View (640x480)        │  ┌────────────────────────────────┐  │
│  ┌────────────────────────┐  │  │                                │  │
│  │                        │  │  │   [First-person view of what   │  │
│  │  [Top-down zoomable    │  │  │    agent currently sees]       │  │
│  │   world with agent,    │  │  │                                │  │
│  │   entities, pointing   │  │  │                                │  │
│  │   cursor]              │  │  └────────────────────────────────┘  │
│  │                        │  │                                      │
│  │                        │  │  Status Panel (320x120)             │
│  │                        │  │  ┌────────────────────────────────┐  │
│  │                        │  │  │ Energy: ████████░░ 80%         │  │
│  │                        │  │  │ Health: ██████████ 100%        │  │
│  └────────────────────────┘  │  │ Age: 42s  Survival: 1m 23s     │  │
│                              │  │ Mode: OBSERVE  Signal: NONE    │  │
├──────────────────────────────┤  └────────────────────────────────┘  │
│  Audio Waveform (640x80)     │                                      │
│  ┌────────────────────────┐  │  Learning Metrics (320x120)         │
│  │ ─╱╲─╱╲──╱╲─╱╲─╱╲──╱╲─ │  │  ┌────────────────────────────────┐  │
│  └────────────────────────┘  │  │ Loss: 0.0234 ↓                 │  │
├──────────────────────────────┤  │ Rewards: 12  Punishments: 3    │  │
│  Controls & Help (640x120)   │  │ Demonstrations: 5              │  │
│  SPACE=Reward X=Punish       │  │ Voice Samples: 47              │  │
│  C=Control  ARROWS=Move      │  └────────────────────────────────┘  │
│  CLICK=Point SCROLL=Zoom     │                                      │
└──────────────────────────────┴──────────────────────────────────────┘
Total Window: 960x720 pixels
```

## File Structure

```
primordial/
├── interface/
│   ├── __init__.py
│   ├── app.py                    # Main application loop
│   ├── renderer.py                # Rendering engine with viewports
│   ├── input_handler.py           # Unified input processing
│   ├── teaching_signals.py        # Convert inputs to learning signals
│   ├── demo_mode.py               # Human demonstration controller
│   ├── audio_capture.py           # Microphone background thread
│   ├── ui_panels.py               # Individual panel rendering
│   └── config.py                  # UI constants and settings
├── core/
│   ├── agent.py                   # Agent state (from previous plans)
│   ├── world.py                   # World state (from previous plans)
│   └── sensors.py                 # Agent sensors (from previous plans)
└── tests/
    └── interface/
        ├── test_input_handler.py
        ├── test_teaching_signals.py
        ├── test_demo_mode.py
        ├── test_audio_capture.py
        └── test_renderer.py
```

---

## Task 1: Configuration and Constants

**Files:**
- Create: `primordial/interface/__init__.py`
- Create: `primordial/interface/config.py`
- Test: `tests/interface/test_config.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_config.py`

```python
import pytest
from primordial.interface.config import UIConfig, Colors, Layout, KeyBindings


def test_ui_config_creates_valid_dimensions():
    config = UIConfig()
    assert config.window_width == 960
    assert config.window_height == 720
    assert config.fps == 60


def test_layout_viewport_dimensions():
    layout = Layout()
    # World view
    assert layout.world_view_rect == (0, 40, 640, 480)
    # Agent POV
    assert layout.agent_pov_rect == (640, 40, 320, 240)
    # Status panel
    assert layout.status_rect == (640, 280, 320, 120)


def test_keybindings_are_defined():
    kb = KeyBindings()
    assert kb.REWARD is not None
    assert kb.PUNISH is not None
    assert kb.CONTROL is not None
    assert kb.POINT is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.config'"

**Step 3: Create package init**

Create: `primordial/interface/__init__.py`

```python
"""Human teaching interface for Primordial agents."""

__version__ = "0.1.0"
```

**Step 4: Write minimal implementation**

Create: `primordial/interface/config.py`

```python
"""Configuration constants for the teaching interface."""

from dataclasses import dataclass
from typing import Tuple
import pygame


@dataclass
class Colors:
    """Color palette for UI."""
    BACKGROUND = (20, 20, 30)
    PANEL_BG = (30, 30, 40)
    TEXT = (200, 200, 200)
    TEXT_BRIGHT = (255, 255, 255)
    AGENT = (100, 200, 255)
    ENTITY = (255, 200, 100)
    REWARD = (100, 255, 100)
    PUNISH = (255, 100, 100)
    POINT_CURSOR = (255, 255, 0)
    WAVEFORM = (0, 255, 150)
    GRID = (50, 50, 60)


@dataclass
class Layout:
    """UI layout dimensions (x, y, width, height)."""
    # Header bar
    header_rect: Tuple[int, int, int, int] = (0, 0, 960, 40)

    # Main panels
    world_view_rect: Tuple[int, int, int, int] = (0, 40, 640, 480)
    agent_pov_rect: Tuple[int, int, int, int] = (640, 40, 320, 240)
    status_rect: Tuple[int, int, int, int] = (640, 280, 320, 120)
    metrics_rect: Tuple[int, int, int, int] = (640, 400, 320, 120)

    # Bottom panels
    waveform_rect: Tuple[int, int, int, int] = (0, 520, 640, 80)
    controls_rect: Tuple[int, int, int, int] = (0, 600, 640, 120)


@dataclass
class KeyBindings:
    """Keyboard input mappings."""
    REWARD = pygame.K_SPACE
    PUNISH = pygame.K_x
    CONTROL = pygame.K_c
    MOVE_UP = pygame.K_UP
    MOVE_DOWN = pygame.K_DOWN
    MOVE_LEFT = pygame.K_LEFT
    MOVE_RIGHT = pygame.K_RIGHT
    SAVE = pygame.K_s
    LOAD = pygame.K_l
    QUIT = pygame.K_ESCAPE


@dataclass
class ControllerBindings:
    """Game controller button mappings."""
    REWARD = 0      # A button
    PUNISH = 1      # B button
    CONTROL = 2     # X button
    MENU = 3        # Y button


@dataclass
class UIConfig:
    """Main UI configuration."""
    window_width: int = 960
    window_height: int = 720
    fps: int = 60
    font_size: int = 16
    font_size_small: int = 12
    font_size_large: int = 20

    # Zoom settings
    zoom_min: float = 0.5
    zoom_max: float = 4.0
    zoom_step: float = 0.1

    # Audio settings
    audio_sample_rate: int = 44100
    audio_channels: int = 1
    audio_buffer_size: int = 1024
    waveform_history: int = 2048  # Samples to display

    colors: Colors = None
    layout: Layout = None
    keys: KeyBindings = None
    controller: ControllerBindings = None

    def __post_init__(self):
        if self.colors is None:
            self.colors = Colors()
        if self.layout is None:
            self.layout = Layout()
        if self.keys is None:
            self.keys = KeyBindings()
        if self.controller is None:
            self.controller = ControllerBindings()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/interface/test_config.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add primordial/interface/ tests/interface/test_config.py
git commit -m "feat: add UI configuration and layout constants"
```

---

## Task 2: Teaching Signal System

**Files:**
- Create: `primordial/interface/teaching_signals.py`
- Test: `tests/interface/test_teaching_signals.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_teaching_signals.py`

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_teaching_signals.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.teaching_signals'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/teaching_signals.py`

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_teaching_signals.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add primordial/interface/teaching_signals.py tests/interface/test_teaching_signals.py
git commit -m "feat: add teaching signal system for human inputs"
```

---

## Task 3: Audio Capture System

**Files:**
- Create: `primordial/interface/audio_capture.py`
- Test: `tests/interface/test_audio_capture.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_audio_capture.py`

```python
import pytest
import numpy as np
import time
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
    assert buffer[0] == 0.1


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_audio_capture.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.audio_capture'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/audio_capture.py`

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_audio_capture.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add primordial/interface/audio_capture.py tests/interface/test_audio_capture.py
git commit -m "feat: add background microphone audio capture"
```

---

## Task 4: Input Handler

**Files:**
- Create: `primordial/interface/input_handler.py`
- Test: `tests/interface/test_input_handler.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_input_handler.py`

```python
import pytest
import pygame
from unittest.mock import Mock
from primordial.interface.input_handler import InputHandler, InputState
from primordial.interface.config import UIConfig
from primordial.interface.teaching_signals import TeachingSignalType


def test_input_state_initialization():
    state = InputState()
    assert state.mouse_pos == (0, 0)
    assert not state.control_mode
    assert state.zoom_level == 1.0


def test_input_handler_processes_keyboard():
    config = UIConfig()
    handler = InputHandler(config)

    # Mock keyboard event
    event = Mock()
    event.type = pygame.KEYDOWN
    event.key = pygame.K_SPACE

    signals = handler.process_event(event, timestamp=1.0)
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.REWARD


def test_input_handler_punish_key():
    config = UIConfig()
    handler = InputHandler(config)

    event = Mock()
    event.type = pygame.KEYDOWN
    event.key = pygame.K_x

    signals = handler.process_event(event, timestamp=2.0)
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.PUNISH


def test_input_handler_control_mode_toggle():
    config = UIConfig()
    handler = InputHandler(config)

    event = Mock()
    event.type = pygame.KEYDOWN
    event.key = pygame.K_c

    handler.process_event(event, timestamp=1.0)
    assert handler.state.control_mode

    handler.process_event(event, timestamp=2.0)
    assert not handler.state.control_mode


def test_input_handler_mouse_click_pointing():
    config = UIConfig()
    handler = InputHandler(config)

    event = Mock()
    event.type = pygame.MOUSEBUTTONDOWN
    event.button = 1  # Left click
    event.pos = (100, 200)

    signals = handler.process_event(event, timestamp=3.0)
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.POINT
    assert signals[0].data["x"] == 100
    assert signals[0].data["y"] == 200


def test_input_handler_mouse_scroll_zoom():
    config = UIConfig()
    handler = InputHandler(config)

    # Scroll up
    event = Mock()
    event.type = pygame.MOUSEWHEEL
    event.y = 1

    handler.process_event(event, timestamp=1.0)
    assert handler.state.zoom_level > 1.0

    # Scroll down
    event.y = -1
    handler.process_event(event, timestamp=2.0)
    assert handler.state.zoom_level == 1.0  # Back to default
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_input_handler.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.input_handler'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/input_handler.py`

```python
"""Unified input handling for keyboard, mouse, and controller."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import pygame
from primordial.interface.config import UIConfig
from primordial.interface.teaching_signals import TeachingSignal


@dataclass
class InputState:
    """Current state of input devices."""
    mouse_pos: Tuple[int, int] = (0, 0)
    control_mode: bool = False
    zoom_level: float = 1.0
    movement_vector: Tuple[float, float] = (0.0, 0.0)
    controller: Optional[pygame.joystick.Joystick] = None


class InputHandler:
    """Processes input from keyboard, mouse, and game controller."""

    def __init__(self, config: UIConfig):
        self.config = config
        self.state = InputState()

        # Initialize controller if available
        self._init_controller()

    def _init_controller(self) -> None:
        """Initialize game controller if connected."""
        pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            self.state.controller = pygame.joystick.Joystick(0)
            self.state.controller.init()
            print(f"Controller connected: {self.state.controller.get_name()}")

    def process_event(self, event: pygame.event.Event,
                     timestamp: float) -> List[TeachingSignal]:
        """
        Process a single input event and generate teaching signals.

        Args:
            event: Pygame event
            timestamp: Current timestamp in seconds

        Returns:
            List of teaching signals generated from this event
        """
        signals = []

        if event.type == pygame.KEYDOWN:
            signals.extend(self._handle_keydown(event, timestamp))
        elif event.type == pygame.KEYUP:
            signals.extend(self._handle_keyup(event, timestamp))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            signals.extend(self._handle_mouse_button(event, timestamp))
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event)
        elif event.type == pygame.MOUSEWHEEL:
            self._handle_mouse_wheel(event)
        elif event.type == pygame.JOYBUTTONDOWN:
            signals.extend(self._handle_joy_button(event, timestamp))
        elif event.type == pygame.JOYAXISMOTION:
            self._handle_joy_axis(event)

        return signals

    def _handle_keydown(self, event: pygame.event.Event,
                       timestamp: float) -> List[TeachingSignal]:
        """Handle keyboard key press."""
        signals = []

        if event.key == self.config.keys.REWARD:
            signals.append(TeachingSignal.reward(timestamp))
        elif event.key == self.config.keys.PUNISH:
            signals.append(TeachingSignal.punish(timestamp))
        elif event.key == self.config.keys.CONTROL:
            self.state.control_mode = not self.state.control_mode

        # Handle movement keys in control mode
        if self.state.control_mode:
            if event.key == self.config.keys.MOVE_UP:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_up"))
            elif event.key == self.config.keys.MOVE_DOWN:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_down"))
            elif event.key == self.config.keys.MOVE_LEFT:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_left"))
            elif event.key == self.config.keys.MOVE_RIGHT:
                signals.append(TeachingSignal.demonstrate(timestamp, "move_right"))

        return signals

    def _handle_keyup(self, event: pygame.event.Event,
                     timestamp: float) -> List[TeachingSignal]:
        """Handle keyboard key release."""
        # Currently no signals on key release
        return []

    def _handle_mouse_button(self, event: pygame.event.Event,
                            timestamp: float) -> List[TeachingSignal]:
        """Handle mouse button click."""
        signals = []

        if event.button == 1:  # Left click
            x, y = event.pos
            signals.append(TeachingSignal.point(timestamp, x, y))

        return signals

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        """Handle mouse movement."""
        self.state.mouse_pos = event.pos

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        """Handle mouse wheel for zooming."""
        if event.y > 0:  # Scroll up
            self.state.zoom_level = min(
                self.config.zoom_max,
                self.state.zoom_level + self.config.zoom_step
            )
        elif event.y < 0:  # Scroll down
            self.state.zoom_level = max(
                self.config.zoom_min,
                self.state.zoom_level - self.config.zoom_step
            )

    def _handle_joy_button(self, event: pygame.event.Event,
                          timestamp: float) -> List[TeachingSignal]:
        """Handle game controller button press."""
        signals = []

        if event.button == self.config.controller.REWARD:
            signals.append(TeachingSignal.reward(timestamp))
        elif event.button == self.config.controller.PUNISH:
            signals.append(TeachingSignal.punish(timestamp))
        elif event.button == self.config.controller.CONTROL:
            self.state.control_mode = not self.state.control_mode

        return signals

    def _handle_joy_axis(self, event: pygame.event.Event) -> None:
        """Handle game controller joystick movement."""
        # Left stick for movement (axis 0=horizontal, 1=vertical)
        if event.axis == 0:  # Horizontal
            x, y = self.state.movement_vector
            self.state.movement_vector = (event.value, y)
        elif event.axis == 1:  # Vertical
            x, y = self.state.movement_vector
            self.state.movement_vector = (x, event.value)

    def get_continuous_movement(self) -> Tuple[float, float]:
        """Get current movement vector from controller or keyboard state."""
        return self.state.movement_vector
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_input_handler.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add primordial/interface/input_handler.py tests/interface/test_input_handler.py
git commit -m "feat: add unified input handler for keyboard/mouse/controller"
```

---

## Task 5: Demonstration Mode Controller

**Files:**
- Create: `primordial/interface/demo_mode.py`
- Test: `tests/interface/test_demo_mode.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_demo_mode.py`

```python
import pytest
from unittest.mock import Mock
from primordial.interface.demo_mode import DemonstrationController, DemoAction


def test_demo_controller_initialization():
    controller = DemonstrationController()
    assert not controller.is_active()
    assert controller.get_current_action() is None


def test_demo_controller_activate_deactivate():
    controller = DemonstrationController()

    controller.activate()
    assert controller.is_active()

    controller.deactivate()
    assert not controller.is_active()


def test_demo_action_from_keyboard():
    action = DemoAction.from_key_input(
        up=True, down=False, left=False, right=False
    )
    assert action.move_direction == (0, -1)
    assert action.action_type == "move"


def test_demo_action_diagonal_movement():
    action = DemoAction.from_key_input(
        up=True, down=False, left=True, right=False
    )
    # Should normalize diagonal movement
    assert action.move_direction[0] < 0  # Left
    assert action.move_direction[1] < 0  # Up


def test_demo_controller_records_action_sequence():
    controller = DemonstrationController()
    controller.activate()

    action1 = DemoAction.from_key_input(up=True, down=False, left=False, right=False)
    action2 = DemoAction.from_key_input(up=False, down=False, left=True, right=False)

    controller.record_action(action1, timestamp=1.0)
    controller.record_action(action2, timestamp=2.0)

    sequence = controller.get_recorded_sequence()
    assert len(sequence) == 2
    assert sequence[0][0] == 1.0  # timestamp
    assert sequence[1][0] == 2.0


def test_demo_controller_apply_to_agent():
    controller = DemonstrationController()
    controller.activate()

    # Mock agent
    agent = Mock()
    agent.position = [0.0, 0.0]

    action = DemoAction.from_key_input(up=True, down=False, left=False, right=False)
    controller.set_current_action(action)

    result = controller.apply_to_agent(agent, dt=0.1)
    assert result is not None
    assert result["action"] == "move"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_demo_mode.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.demo_mode'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/demo_mode.py`

```python
"""Demonstration mode where human directly controls the agent."""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import math


@dataclass
class DemoAction:
    """A single demonstration action."""
    action_type: str  # "move", "interact", "wait"
    move_direction: Tuple[float, float] = (0.0, 0.0)
    intensity: float = 1.0

    @classmethod
    def from_key_input(cls, up: bool, down: bool, left: bool,
                      right: bool) -> "DemoAction":
        """Create action from keyboard directional input."""
        dx = 0.0
        dy = 0.0

        if up:
            dy -= 1.0
        if down:
            dy += 1.0
        if left:
            dx -= 1.0
        if right:
            dx += 1.0

        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            magnitude = math.sqrt(dx * dx + dy * dy)
            dx /= magnitude
            dy /= magnitude

        action_type = "move" if (dx != 0 or dy != 0) else "wait"

        return cls(
            action_type=action_type,
            move_direction=(dx, dy)
        )

    @classmethod
    def from_joystick(cls, axis_x: float, axis_y: float,
                     deadzone: float = 0.1) -> "DemoAction":
        """Create action from joystick analog input."""
        # Apply deadzone
        if abs(axis_x) < deadzone:
            axis_x = 0.0
        if abs(axis_y) < deadzone:
            axis_y = 0.0

        magnitude = math.sqrt(axis_x * axis_x + axis_y * axis_y)
        action_type = "move" if magnitude > deadzone else "wait"

        return cls(
            action_type=action_type,
            move_direction=(axis_x, axis_y),
            intensity=min(magnitude, 1.0)
        )


class DemonstrationController:
    """Controls agent during human demonstration."""

    def __init__(self):
        self._active = False
        self._current_action: Optional[DemoAction] = None
        self._recorded_sequence: List[Tuple[float, DemoAction]] = []

    def activate(self) -> None:
        """Enter demonstration mode."""
        self._active = True
        self._recorded_sequence.clear()

    def deactivate(self) -> None:
        """Exit demonstration mode."""
        self._active = False
        self._current_action = None

    def is_active(self) -> bool:
        """Check if demonstration mode is active."""
        return self._active

    def set_current_action(self, action: DemoAction) -> None:
        """Set the current demonstration action."""
        self._current_action = action

    def get_current_action(self) -> Optional[DemoAction]:
        """Get the current demonstration action."""
        return self._current_action

    def record_action(self, action: DemoAction, timestamp: float) -> None:
        """Record an action in the demonstration sequence."""
        self._recorded_sequence.append((timestamp, action))

    def get_recorded_sequence(self) -> List[Tuple[float, DemoAction]]:
        """Get the full recorded demonstration sequence."""
        return self._recorded_sequence.copy()

    def clear_recording(self) -> None:
        """Clear the recorded sequence."""
        self._recorded_sequence.clear()

    def apply_to_agent(self, agent: Any, dt: float) -> Optional[Dict[str, Any]]:
        """
        Apply current demonstration action to agent.

        Args:
            agent: The agent to control
            dt: Delta time in seconds

        Returns:
            Dictionary describing the applied action, or None
        """
        if not self._active or self._current_action is None:
            return None

        action = self._current_action

        if action.action_type == "move":
            dx, dy = action.move_direction
            speed = 100.0 * action.intensity  # pixels per second

            # Apply movement (assuming agent has position attribute)
            if hasattr(agent, 'position'):
                agent.position[0] += dx * speed * dt
                agent.position[1] += dy * speed * dt

            return {
                "action": "move",
                "direction": (dx, dy),
                "speed": speed
            }

        return None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_demo_mode.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add primordial/interface/demo_mode.py tests/interface/test_demo_mode.py
git commit -m "feat: add demonstration mode for human agent control"
```

---

## Task 6: UI Panels Rendering

**Files:**
- Create: `primordial/interface/ui_panels.py`
- Test: `tests/interface/test_ui_panels.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_ui_panels.py`

```python
import pytest
import pygame
import numpy as np
from unittest.mock import Mock
from primordial.interface.ui_panels import (
    HeaderPanel, WorldViewPanel, AgentPOVPanel,
    StatusPanel, MetricsPanel, WaveformPanel, ControlsPanel
)
from primordial.interface.config import UIConfig


@pytest.fixture
def pygame_surface():
    """Create a test pygame surface."""
    pygame.init()
    return pygame.Surface((960, 720))


def test_header_panel_renders(pygame_surface):
    config = UIConfig()
    panel = HeaderPanel(config)

    # Should not raise exception
    panel.render(pygame_surface, fps=60, recording=True)


def test_status_panel_displays_agent_state(pygame_surface):
    config = UIConfig()
    panel = StatusPanel(config)

    agent_state = {
        "energy": 0.8,
        "health": 1.0,
        "age": 42.5,
        "survival_time": 83.0
    }

    # Should not raise exception
    panel.render(pygame_surface, agent_state=agent_state, mode="OBSERVE")


def test_waveform_panel_renders_audio(pygame_surface):
    config = UIConfig()
    panel = WaveformPanel(config)

    # Create test waveform
    waveform = np.sin(np.linspace(0, 10 * np.pi, 1000))

    # Should not raise exception
    panel.render(pygame_surface, waveform=waveform)


def test_metrics_panel_displays_learning_stats(pygame_surface):
    config = UIConfig()
    panel = MetricsPanel(config)

    metrics = {
        "loss": 0.0234,
        "rewards": 12,
        "punishments": 3,
        "demonstrations": 5,
        "voice_samples": 47
    }

    # Should not raise exception
    panel.render(pygame_surface, metrics=metrics)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_ui_panels.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.ui_panels'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/ui_panels.py`

```python
"""Individual UI panel renderers."""

from typing import Dict, Any, Optional
import pygame
import numpy as np
from primordial.interface.config import UIConfig


class BasePanel:
    """Base class for UI panels."""

    def __init__(self, config: UIConfig):
        self.config = config
        self.font = pygame.font.Font(None, config.font_size)
        self.font_small = pygame.font.Font(None, config.font_size_small)
        self.font_large = pygame.font.Font(None, config.font_size_large)

    def draw_panel_background(self, surface: pygame.Surface, rect: tuple) -> None:
        """Draw panel background."""
        pygame.draw.rect(surface, self.config.colors.PANEL_BG, rect)
        pygame.draw.rect(surface, self.config.colors.GRID, rect, 1)


class HeaderPanel(BasePanel):
    """Top header bar with FPS and recording status."""

    def render(self, surface: pygame.Surface, fps: int, recording: bool) -> None:
        """Render header panel."""
        rect = self.config.layout.header_rect
        self.draw_panel_background(surface, rect)

        # Title
        title_text = self.font_large.render(
            "Primordial Teaching Interface",
            True,
            self.config.colors.TEXT_BRIGHT
        )
        surface.blit(title_text, (10, 10))

        # FPS counter
        fps_text = self.font.render(f"FPS: {fps}", True, self.config.colors.TEXT)
        surface.blit(fps_text, (rect[2] - 150, 10))

        # Recording indicator
        if recording:
            rec_text = self.font.render("[●REC]", True, self.config.colors.REWARD)
            surface.blit(rec_text, (rect[2] - 70, 10))


class WorldViewPanel(BasePanel):
    """Top-down view of the world."""

    def render(self, surface: pygame.Surface, world_state: Dict[str, Any],
               zoom: float = 1.0, offset: tuple = (0, 0)) -> None:
        """Render world view."""
        rect = self.config.layout.world_view_rect
        self.draw_panel_background(surface, rect)

        # Create subsurface for clipping
        viewport = surface.subsurface(rect)

        # Draw grid
        self._draw_grid(viewport, zoom, offset)

        # Draw entities if provided
        if "entities" in world_state:
            self._draw_entities(viewport, world_state["entities"], zoom, offset)

    def _draw_grid(self, surface: pygame.Surface, zoom: float, offset: tuple) -> None:
        """Draw background grid."""
        grid_size = int(50 * zoom)
        w, h = surface.get_size()

        for x in range(0, w, grid_size):
            pygame.draw.line(surface, self.config.colors.GRID, (x, 0), (x, h))
        for y in range(0, h, grid_size):
            pygame.draw.line(surface, self.config.colors.GRID, (0, y), (w, y))

    def _draw_entities(self, surface: pygame.Surface, entities: list,
                      zoom: float, offset: tuple) -> None:
        """Draw world entities."""
        for entity in entities:
            pos = entity.get("position", (0, 0))
            entity_type = entity.get("type", "unknown")

            # Transform position with zoom and offset
            screen_x = int((pos[0] - offset[0]) * zoom)
            screen_y = int((pos[1] - offset[1]) * zoom)

            # Choose color based on type
            color = self.config.colors.AGENT if entity_type == "agent" else self.config.colors.ENTITY

            # Draw circle
            radius = max(3, int(10 * zoom))
            pygame.draw.circle(surface, color, (screen_x, screen_y), radius)


class AgentPOVPanel(BasePanel):
    """First-person view from agent's perspective."""

    def render(self, surface: pygame.Surface, agent_view: Optional[np.ndarray]) -> None:
        """Render agent POV."""
        rect = self.config.layout.agent_pov_rect
        self.draw_panel_background(surface, rect)

        if agent_view is not None:
            # Convert numpy array to pygame surface
            # Assuming agent_view is RGB image array
            view_surface = pygame.surfarray.make_surface(agent_view)
            view_surface = pygame.transform.scale(view_surface, (rect[2], rect[3]))
            surface.blit(view_surface, (rect[0], rect[1]))
        else:
            # Placeholder text
            text = self.font.render("No agent view", True, self.config.colors.TEXT)
            text_rect = text.get_rect(center=(rect[0] + rect[2]//2, rect[1] + rect[3]//2))
            surface.blit(text, text_rect)


class StatusPanel(BasePanel):
    """Agent status information."""

    def render(self, surface: pygame.Surface, agent_state: Dict[str, Any],
               mode: str) -> None:
        """Render status panel."""
        rect = self.config.layout.status_rect
        self.draw_panel_background(surface, rect)

        x, y = rect[0] + 10, rect[1] + 10
        line_height = 25

        # Energy bar
        energy = agent_state.get("energy", 0.0)
        self._draw_progress_bar(surface, "Energy", energy, x, y,
                               self.config.colors.REWARD)

        # Health bar
        health = agent_state.get("health", 0.0)
        self._draw_progress_bar(surface, "Health", health, x, y + line_height,
                               self.config.colors.PUNISH)

        # Age and survival time
        age = agent_state.get("age", 0.0)
        survival = agent_state.get("survival_time", 0.0)

        text = self.font_small.render(
            f"Age: {age:.1f}s  Survival: {survival:.1f}s",
            True,
            self.config.colors.TEXT
        )
        surface.blit(text, (x, y + 2 * line_height))

        # Mode and signal
        text = self.font_small.render(
            f"Mode: {mode}  Signal: NONE",
            True,
            self.config.colors.TEXT
        )
        surface.blit(text, (x, y + 3 * line_height))

    def _draw_progress_bar(self, surface: pygame.Surface, label: str,
                          value: float, x: int, y: int, color: tuple) -> None:
        """Draw a labeled progress bar."""
        bar_width = 200
        bar_height = 15

        # Label
        text = self.font_small.render(f"{label}:", True, self.config.colors.TEXT)
        surface.blit(text, (x, y))

        # Background
        bar_x = x + 70
        pygame.draw.rect(surface, self.config.colors.GRID,
                        (bar_x, y, bar_width, bar_height))

        # Fill
        fill_width = int(bar_width * value)
        pygame.draw.rect(surface, color, (bar_x, y, fill_width, bar_height))

        # Percentage
        pct_text = self.font_small.render(f"{int(value * 100)}%", True,
                                          self.config.colors.TEXT_BRIGHT)
        surface.blit(pct_text, (bar_x + bar_width + 10, y))


class MetricsPanel(BasePanel):
    """Learning metrics display."""

    def render(self, surface: pygame.Surface, metrics: Dict[str, Any]) -> None:
        """Render metrics panel."""
        rect = self.config.layout.metrics_rect
        self.draw_panel_background(surface, rect)

        x, y = rect[0] + 10, rect[1] + 10
        line_height = 22

        # Loss
        loss = metrics.get("loss", 0.0)
        text = self.font.render(f"Loss: {loss:.4f} ↓", True,
                               self.config.colors.TEXT_BRIGHT)
        surface.blit(text, (x, y))

        # Counts
        rewards = metrics.get("rewards", 0)
        punishments = metrics.get("punishments", 0)
        text = self.font_small.render(
            f"Rewards: {rewards}  Punishments: {punishments}",
            True,
            self.config.colors.TEXT
        )
        surface.blit(text, (x, y + line_height))

        demos = metrics.get("demonstrations", 0)
        text = self.font_small.render(f"Demonstrations: {demos}", True,
                                      self.config.colors.TEXT)
        surface.blit(text, (x, y + 2 * line_height))

        voice = metrics.get("voice_samples", 0)
        text = self.font_small.render(f"Voice Samples: {voice}", True,
                                      self.config.colors.TEXT)
        surface.blit(text, (x, y + 3 * line_height))


class WaveformPanel(BasePanel):
    """Audio waveform visualization."""

    def render(self, surface: pygame.Surface, waveform: np.ndarray) -> None:
        """Render waveform panel."""
        rect = self.config.layout.waveform_rect
        self.draw_panel_background(surface, rect)

        if len(waveform) == 0:
            return

        # Downsample waveform to fit panel width
        panel_width = rect[2] - 20
        if len(waveform) > panel_width:
            step = len(waveform) // panel_width
            waveform = waveform[::step][:panel_width]

        # Draw waveform
        center_y = rect[1] + rect[3] // 2
        scale = (rect[3] - 20) / 2

        points = []
        for i, sample in enumerate(waveform):
            x = rect[0] + 10 + i
            y = center_y - int(sample * scale)
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(surface, self.config.colors.WAVEFORM, False, points, 2)

        # Center line
        pygame.draw.line(surface, self.config.colors.GRID,
                        (rect[0], center_y), (rect[0] + rect[2], center_y))


class ControlsPanel(BasePanel):
    """Control hints and help."""

    def render(self, surface: pygame.Surface) -> None:
        """Render controls panel."""
        rect = self.config.layout.controls_rect
        self.draw_panel_background(surface, rect)

        x, y = rect[0] + 10, rect[1] + 10
        line_height = 20

        controls = [
            "Controls & Help",
            "SPACE=Reward  X=Punish  C=Control Mode",
            "ARROWS=Move (in control mode)",
            "CLICK=Point  SCROLL=Zoom",
            "S=Save  L=Load  ESC=Quit"
        ]

        for i, line in enumerate(controls):
            font = self.font if i == 0 else self.font_small
            color = self.config.colors.TEXT_BRIGHT if i == 0 else self.config.colors.TEXT
            text = font.render(line, True, color)
            surface.blit(text, (x, y + i * line_height))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_ui_panels.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add primordial/interface/ui_panels.py tests/interface/test_ui_panels.py
git commit -m "feat: add UI panel renderers for interface"
```

---

## Task 7: Main Renderer

**Files:**
- Create: `primordial/interface/renderer.py`
- Test: `tests/interface/test_renderer.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_renderer.py`

```python
import pytest
import pygame
from unittest.mock import Mock
from primordial.interface.renderer import Renderer
from primordial.interface.config import UIConfig


def test_renderer_initialization():
    config = UIConfig()
    renderer = Renderer(config)
    assert renderer.screen is not None
    assert renderer.clock is not None


def test_renderer_frame_timing():
    config = UIConfig()
    renderer = Renderer(config)

    # First frame
    dt1 = renderer.tick()
    assert dt1 >= 0

    # Second frame should have elapsed time
    dt2 = renderer.tick()
    assert dt2 > 0


def test_renderer_get_fps():
    config = UIConfig()
    renderer = Renderer(config)

    renderer.tick()
    fps = renderer.get_fps()
    assert fps >= 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_renderer.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.renderer'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/renderer.py`

```python
"""Main rendering engine for the teaching interface."""

from typing import Dict, Any, Optional
import pygame
import numpy as np
from primordial.interface.config import UIConfig
from primordial.interface.ui_panels import (
    HeaderPanel, WorldViewPanel, AgentPOVPanel,
    StatusPanel, MetricsPanel, WaveformPanel, ControlsPanel
)


class Renderer:
    """Main renderer managing all UI panels."""

    def __init__(self, config: UIConfig):
        """
        Initialize renderer.

        Args:
            config: UI configuration
        """
        self.config = config

        # Initialize Pygame
        pygame.init()

        # Create window
        self.screen = pygame.display.set_mode(
            (config.window_width, config.window_height)
        )
        pygame.display.set_caption("Primordial Teaching Interface")

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Initialize panels
        self.header = HeaderPanel(config)
        self.world_view = WorldViewPanel(config)
        self.agent_pov = AgentPOVPanel(config)
        self.status = StatusPanel(config)
        self.metrics = MetricsPanel(config)
        self.waveform = WaveformPanel(config)
        self.controls = ControlsPanel(config)

        # Recording state
        self.recording = False

    def tick(self) -> float:
        """
        Advance one frame and control FPS.

        Returns:
            Delta time in seconds since last frame
        """
        dt = self.clock.tick(self.config.fps) / 1000.0  # Convert ms to seconds
        return dt

    def get_fps(self) -> int:
        """Get current FPS."""
        return int(self.clock.get_fps())

    def render_frame(self,
                    world_state: Dict[str, Any],
                    agent_state: Dict[str, Any],
                    agent_view: Optional[np.ndarray],
                    waveform: np.ndarray,
                    metrics: Dict[str, Any],
                    mode: str,
                    zoom: float = 1.0,
                    offset: tuple = (0, 0)) -> None:
        """
        Render complete frame with all panels.

        Args:
            world_state: World state data
            agent_state: Agent status data
            agent_view: Agent's first-person view (RGB image)
            waveform: Audio waveform samples
            metrics: Learning metrics
            mode: Current interaction mode
            zoom: World view zoom level
            offset: World view camera offset
        """
        # Clear screen
        self.screen.fill(self.config.colors.BACKGROUND)

        # Render all panels
        self.header.render(self.screen, self.get_fps(), self.recording)
        self.world_view.render(self.screen, world_state, zoom, offset)
        self.agent_pov.render(self.screen, agent_view)
        self.status.render(self.screen, agent_state, mode)
        self.metrics.render(self.screen, metrics)
        self.waveform.render(self.screen, waveform)
        self.controls.render(self.screen)

        # Update display
        pygame.display.flip()

    def set_recording(self, recording: bool) -> None:
        """Set recording indicator state."""
        self.recording = recording

    def cleanup(self) -> None:
        """Cleanup resources."""
        pygame.quit()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_renderer.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add primordial/interface/renderer.py tests/interface/test_renderer.py
git commit -m "feat: add main rendering engine with panel management"
```

---

## Task 8: Main Application Loop

**Files:**
- Create: `primordial/interface/app.py`
- Test: `tests/interface/test_app.py`

**Step 1: Write the failing test**

Create: `tests/interface/test_app.py`

```python
import pytest
from unittest.mock import Mock, patch
from primordial.interface.app import TeachingApp
from primordial.interface.config import UIConfig


def test_app_initialization():
    config = UIConfig()
    app = TeachingApp(config)

    assert app.running is False
    assert app.renderer is not None
    assert app.input_handler is not None


def test_app_start_stop():
    config = UIConfig()
    app = TeachingApp(config)

    # App starts not running
    assert not app.running

    # Can manually stop
    app.stop()
    assert not app.running


@patch('pygame.event.get')
def test_app_processes_quit_event(mock_event_get):
    config = UIConfig()
    app = TeachingApp(config)

    # Mock quit event
    quit_event = Mock()
    quit_event.type = pygame.QUIT
    mock_event_get.return_value = [quit_event]

    app.running = True
    app._process_events(0.0)

    assert not app.running
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/interface/test_app.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'primordial.interface.app'"

**Step 3: Write minimal implementation**

Create: `primordial/interface/app.py`

```python
"""Main teaching interface application."""

import time
from typing import Optional, Dict, Any
import pygame
import numpy as np

from primordial.interface.config import UIConfig
from primordial.interface.renderer import Renderer
from primordial.interface.input_handler import InputHandler
from primordial.interface.teaching_signals import TeachingSignalQueue
from primordial.interface.demo_mode import DemonstrationController
from primordial.interface.audio_capture import AudioCapture


class TeachingApp:
    """Main application for human teaching interface."""

    def __init__(self, config: UIConfig):
        """
        Initialize teaching application.

        Args:
            config: UI configuration
        """
        self.config = config
        self.running = False

        # Core components
        self.renderer = Renderer(config)
        self.input_handler = InputHandler(config)
        self.signal_queue = TeachingSignalQueue()
        self.demo_controller = DemonstrationController()
        self.audio_capture = AudioCapture(
            sample_rate=config.audio_sample_rate,
            channels=config.audio_channels,
            buffer_size=config.waveform_history
        )

        # State
        self.start_time = 0.0
        self.current_time = 0.0

        # Statistics
        self.stats = {
            "rewards": 0,
            "punishments": 0,
            "demonstrations": 0,
            "voice_samples": 0,
            "loss": 0.0
        }

        # Placeholder states (would be replaced by actual agent/world)
        self.world_state: Dict[str, Any] = {"entities": []}
        self.agent_state: Dict[str, Any] = {
            "energy": 1.0,
            "health": 1.0,
            "age": 0.0,
            "survival_time": 0.0
        }

    def start(self) -> None:
        """Start the application."""
        self.running = True
        self.start_time = time.time()
        self.audio_capture.start()
        self.renderer.set_recording(True)

    def stop(self) -> None:
        """Stop the application."""
        self.running = False
        self.audio_capture.stop()
        self.renderer.set_recording(False)

    def _process_events(self, timestamp: float) -> None:
        """Process input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if event.type == pygame.KEYDOWN and event.key == self.config.keys.QUIT:
                self.running = False
                continue

            # Process event and get teaching signals
            signals = self.input_handler.process_event(event, timestamp)

            # Enqueue signals
            for signal in signals:
                self.signal_queue.enqueue(signal)

                # Update statistics
                if signal.signal_type.value == "reward":
                    self.stats["rewards"] += 1
                elif signal.signal_type.value == "punish":
                    self.stats["punishments"] += 1
                elif signal.signal_type.value == "demonstrate":
                    self.stats["demonstrations"] += 1

    def _update(self, dt: float) -> None:
        """Update application state."""
        self.current_time = time.time() - self.start_time

        # Update agent age
        self.agent_state["age"] = self.current_time
        self.agent_state["survival_time"] = self.current_time

        # Process demonstration mode
        if self.demo_controller.is_active():
            # In demo mode, update agent from human input
            pass  # Would apply to actual agent

    def _render(self) -> None:
        """Render current frame."""
        # Get audio waveform
        waveform = self.audio_capture.get_recent(self.config.waveform_history)

        # Get current mode
        mode = "CONTROL" if self.input_handler.state.control_mode else "OBSERVE"

        # Render frame
        self.renderer.render_frame(
            world_state=self.world_state,
            agent_state=self.agent_state,
            agent_view=None,  # Would come from agent
            waveform=waveform,
            metrics=self.stats,
            mode=mode,
            zoom=self.input_handler.state.zoom_level
        )

    def run(self) -> None:
        """Main application loop."""
        self.start()

        try:
            while self.running:
                # Tick and get delta time
                dt = self.renderer.tick()

                # Process inputs
                self._process_events(self.current_time)

                # Update state
                self._update(dt)

                # Render
                self._render()

        finally:
            self.cleanup()

    def save_state(self, filepath: str) -> None:
        """
        Save agent state to file.

        Args:
            filepath: Path to save file
        """
        import json

        state = {
            "agent_state": self.agent_state,
            "stats": self.stats,
            "timestamp": self.current_time
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """
        Load agent state from file.

        Args:
            filepath: Path to save file
        """
        import json

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.agent_state = state.get("agent_state", self.agent_state)
        self.stats = state.get("stats", self.stats)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.audio_capture.stop()
        self.renderer.cleanup()


def main():
    """Entry point for teaching interface."""
    config = UIConfig()
    app = TeachingApp(config)
    app.run()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/interface/test_app.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add primordial/interface/app.py tests/interface/test_app.py
git commit -m "feat: add main teaching application with event loop"
```

---

## Task 9: Integration Tests

**Files:**
- Create: `tests/interface/test_integration.py`

**Step 1: Write integration test**

Create: `tests/interface/test_integration.py`

```python
import pytest
import pygame
import numpy as np
from primordial.interface.app import TeachingApp
from primordial.interface.config import UIConfig
from primordial.interface.teaching_signals import TeachingSignalType


def test_end_to_end_reward_flow():
    """Test complete flow from keyboard input to teaching signal."""
    config = UIConfig()
    app = TeachingApp(config)

    # Simulate reward key press
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_SPACE})

    signals = app.input_handler.process_event(event, timestamp=1.0)

    # Verify signal was created
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.REWARD

    # Enqueue and verify
    app.signal_queue.enqueue(signals[0])
    assert app.signal_queue.size() == 1


def test_demonstration_mode_workflow():
    """Test entering demo mode and controlling agent."""
    config = UIConfig()
    app = TeachingApp(config)

    # Enter control mode
    app.input_handler.state.control_mode = True
    app.demo_controller.activate()

    assert app.demo_controller.is_active()
    assert app.input_handler.state.control_mode

    # Send movement command
    event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
    signals = app.input_handler.process_event(event, timestamp=1.0)

    # Verify demonstration signal
    assert len(signals) == 1
    assert signals[0].signal_type == TeachingSignalType.DEMONSTRATE
    assert signals[0].data["action"] == "move_up"


def test_audio_capture_integration():
    """Test audio capture integration."""
    config = UIConfig()
    app = TeachingApp(config)

    # Audio capture should be initialized
    assert app.audio_capture is not None

    # Start recording
    app.audio_capture.start()
    assert app.audio_capture.is_recording()

    # Get waveform (might be empty if no audio)
    waveform = app.audio_capture.get_recent(100)
    assert isinstance(waveform, np.ndarray)

    # Stop recording
    app.audio_capture.stop()
    assert not app.audio_capture.is_recording()


def test_save_load_state():
    """Test state persistence."""
    import tempfile
    import os

    config = UIConfig()
    app = TeachingApp(config)

    # Modify state
    app.stats["rewards"] = 42
    app.agent_state["energy"] = 0.5

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        app.save_state(temp_path)

        # Create new app and load
        app2 = TeachingApp(config)
        app2.load_state(temp_path)

        # Verify state restored
        assert app2.stats["rewards"] == 42
        assert app2.agent_state["energy"] == 0.5

    finally:
        os.unlink(temp_path)
```

**Step 2: Run integration tests**

Run: `pytest tests/interface/test_integration.py -v`
Expected: PASS (4 tests)

**Step 3: Commit**

```bash
git add tests/interface/test_integration.py
git commit -m "test: add integration tests for teaching interface"
```

---

## Task 10: Documentation and CLI Entry Point

**Files:**
- Create: `primordial/interface/README.md`
- Create: `primordial/__main__.py`

**Step 1: Write interface documentation**

Create: `primordial/interface/README.md`

```markdown
# Human Teaching Interface

Interactive teaching interface for Primordial AI agents.

## Overview

The teaching interface allows humans to train AI agents through:
- **Reward/Punishment**: Immediate feedback signals
- **Pointing**: Direct attention to world locations
- **Demonstration**: Direct control to show desired behavior
- **Voice**: Microphone input for audio teaching

## Quick Start

```bash
# Run the teaching interface
python -m primordial

# With custom config
python -m primordial --fps 30 --zoom 2.0
```

## Controls

### Keyboard
- `SPACE` - Send reward signal
- `X` - Send punishment signal
- `C` - Toggle control mode (demonstration)
- `ARROWS` - Move agent (in control mode)
- `S` - Save agent state
- `L` - Load agent state
- `ESC` - Quit

### Mouse
- `LEFT CLICK` - Point at location
- `SCROLL` - Zoom world view in/out

### Game Controller (Optional)
- `A Button` - Reward
- `B Button` - Punishment
- `X Button` - Toggle control mode
- `Left Stick` - Move agent (in control mode)

## UI Layout

```
┌─────────────────────────────────────────┐
│  Header (FPS, Recording Status)         │
├────────────────────┬────────────────────┤
│  World View        │  Agent POV         │
│  (Top-down)        │  (First-person)    │
│                    ├────────────────────┤
│                    │  Status Panel      │
│                    │  (Energy, Health)  │
├────────────────────┤                    │
│  Audio Waveform    │  Learning Metrics  │
├────────────────────┴────────────────────┤
│  Controls Help                          │
└─────────────────────────────────────────┘
```

## Architecture

### Components

- **TeachingApp**: Main application loop (60 FPS)
- **Renderer**: Multi-panel rendering engine
- **InputHandler**: Unified keyboard/mouse/controller input
- **TeachingSignalQueue**: Thread-safe signal queue
- **DemonstrationController**: Human agent control
- **AudioCapture**: Background microphone recording

### Teaching Signals

All human inputs are converted to `TeachingSignal` objects:

```python
TeachingSignal(
    signal_type=TeachingSignalType.REWARD,
    timestamp=1.5,
    intensity=1.0,
    data={}
)
```

Signal types:
- `REWARD` - Positive reinforcement
- `PUNISH` - Negative reinforcement
- `POINT` - Spatial attention (x, y coordinates)
- `DEMONSTRATE` - Action demonstration
- `VOICE` - Audio data (numpy array)

## API Example

```python
from primordial.interface import TeachingApp, UIConfig

# Create custom config
config = UIConfig()
config.fps = 30
config.zoom_min = 0.25
config.zoom_max = 8.0

# Create and run app
app = TeachingApp(config)
app.run()
```

## Testing

```bash
# Run all interface tests
pytest tests/interface/ -v

# Run specific test file
pytest tests/interface/test_app.py -v

# Run with coverage
pytest tests/interface/ --cov=primordial.interface
```

## Performance

- Target: 60 FPS
- Audio latency: < 50ms
- Input response: < 16ms (1 frame)
- Memory: Circular buffers prevent unbounded growth

## Dependencies

- pygame >= 2.0.0
- sounddevice >= 0.4.0
- numpy >= 1.20.0
```

**Step 2: Create CLI entry point**

Create: `primordial/__main__.py`

```python
"""CLI entry point for Primordial."""

import argparse
import sys


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Primordial - Human Teaching Interface for AI Agents"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target frames per second (default: 60)"
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Initial zoom level (default: 1.0)"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Window width in pixels (default: 960)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height in pixels (default: 720)"
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio capture"
    )

    args = parser.parse_args()

    # Import here to avoid pygame init on import
    from primordial.interface.config import UIConfig
    from primordial.interface.app import TeachingApp

    # Create config from args
    config = UIConfig()
    config.fps = args.fps
    config.window_width = args.width
    config.window_height = args.height

    # Create and run app
    app = TeachingApp(config)

    # Disable audio if requested
    if args.no_audio:
        app.audio_capture.stop()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        app.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 3: Create package init**

Create: `primordial/__init__.py`

```python
"""Primordial - Human teaching interface for AI agents."""

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
```

**Step 4: Commit**

```bash
git add primordial/interface/README.md primordial/__main__.py primordial/__init__.py
git commit -m "docs: add interface documentation and CLI entry point"
```

---

## API Specification

### Core Classes

#### TeachingApp
Main application orchestrating all components.

```python
class TeachingApp:
    def __init__(self, config: UIConfig)
    def start(self) -> None
    def stop(self) -> None
    def run(self) -> None  # Main loop
    def save_state(self, filepath: str) -> None
    def load_state(self, filepath: str) -> None
    def cleanup(self) -> None
```

#### Renderer
Multi-panel rendering at 60 FPS.

```python
class Renderer:
    def __init__(self, config: UIConfig)
    def tick(self) -> float  # Returns dt
    def get_fps(self) -> int
    def render_frame(
        self,
        world_state: Dict[str, Any],
        agent_state: Dict[str, Any],
        agent_view: Optional[np.ndarray],
        waveform: np.ndarray,
        metrics: Dict[str, Any],
        mode: str,
        zoom: float = 1.0,
        offset: tuple = (0, 0)
    ) -> None
    def set_recording(self, recording: bool) -> None
    def cleanup(self) -> None
```

#### InputHandler
Unified input processing.

```python
class InputHandler:
    def __init__(self, config: UIConfig)
    def process_event(
        self,
        event: pygame.event.Event,
        timestamp: float
    ) -> List[TeachingSignal]
    def get_continuous_movement(self) -> Tuple[float, float]
```

#### TeachingSignalQueue
Thread-safe signal queue.

```python
class TeachingSignalQueue:
    def __init__(self, max_size: int = 1000)
    def enqueue(self, signal: TeachingSignal) -> None
    def dequeue(self) -> Optional[TeachingSignal]
    def size(self) -> int
    def get_recent(self, since_timestamp: float) -> List[TeachingSignal]
    def clear(self) -> None
```

#### DemonstrationController
Human demonstration mode.

```python
class DemonstrationController:
    def activate(self) -> None
    def deactivate(self) -> None
    def is_active(self) -> bool
    def set_current_action(self, action: DemoAction) -> None
    def get_current_action(self) -> Optional[DemoAction]
    def record_action(self, action: DemoAction, timestamp: float) -> None
    def get_recorded_sequence(self) -> List[Tuple[float, DemoAction]]
    def apply_to_agent(self, agent: Any, dt: float) -> Optional[Dict[str, Any]]
```

#### AudioCapture
Background microphone recording.

```python
class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        buffer_size: int = 44100
    )
    def start(self) -> None
    def stop(self) -> None
    def is_recording(self) -> bool
    def get_buffer(self) -> np.ndarray
    def get_recent(self, num_samples: int) -> np.ndarray
    def clear_buffer(self) -> None
```

---

## Testing Strategy

### Unit Tests
Each module has isolated unit tests:
- `test_config.py` - Configuration validation
- `test_teaching_signals.py` - Signal creation and queueing
- `test_audio_capture.py` - Audio buffering (mocked sounddevice)
- `test_input_handler.py` - Input event processing
- `test_demo_mode.py` - Demonstration controller
- `test_ui_panels.py` - Panel rendering (no display)
- `test_renderer.py` - Frame rendering
- `test_app.py` - Application lifecycle

### Integration Tests
End-to-end workflows:
- `test_integration.py` - Complete input-to-signal flows

### Manual Testing
1. Run `python -m primordial`
2. Test each input:
   - Press SPACE (reward counter increments)
   - Press X (punishment counter increments)
   - Click mouse (pointing cursor appears)
   - Scroll wheel (world zooms in/out)
   - Press C, then arrows (agent moves)
   - Speak into mic (waveform displays)
3. Test save/load:
   - Press S to save
   - Press L to load
4. Test controller (if available)

### Performance Testing
Monitor FPS counter:
- Should maintain 60 FPS
- No lag on input (< 1 frame)
- Audio waveform updates smoothly

---

## Implementation Order

1. **Configuration** (Task 1) - Foundation
2. **Teaching Signals** (Task 2) - Core data structures
3. **Audio Capture** (Task 3) - Background service
4. **Input Handler** (Task 4) - Input processing
5. **Demo Mode** (Task 5) - Demonstration controller
6. **UI Panels** (Task 6) - Individual renderers
7. **Main Renderer** (Task 7) - Composite rendering
8. **Main App** (Task 8) - Event loop
9. **Integration Tests** (Task 9) - Validation
10. **Documentation** (Task 10) - Polish

Each task is independent and testable. Follow TDD: write test, see it fail, implement, see it pass, commit.

---

## Future Enhancements

- Multiple agent views simultaneously
- Real-time learning curve graphs
- Replay recorded demonstrations
- Export teaching sessions to video
- Multi-modal attention heatmaps
- Voice command recognition
- Gesture input via webcam

---

## Dependencies

Add to `requirements.txt`:
```
pygame>=2.0.0
sounddevice>=0.4.0
numpy>=1.20.0
```

Or `pyproject.toml`:
```toml
[project]
dependencies = [
    "pygame>=2.0.0",
    "sounddevice>=0.4.0",
    "numpy>=1.20.0",
]
```
