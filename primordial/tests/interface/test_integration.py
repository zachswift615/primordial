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
