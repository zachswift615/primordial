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
