"""Integrated teaching interface with live simulation."""

import time
from typing import Optional, Dict, Any, List
import pygame
import numpy as np

from primordial.interface.config import UIConfig
from primordial.interface.renderer import Renderer
from primordial.interface.input_handler import InputHandler
from primordial.interface.teaching_signals import TeachingSignalQueue, TeachingSignalType
from primordial.interface.demo_mode import DemonstrationController, DemoAction
from primordial.interface.audio_capture import AudioCapture

from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig


class IntegratedTeachingApp:
    """Teaching interface connected to a live simulation."""

    def __init__(self, ui_config: UIConfig, sim_config: Optional[SimulationConfig] = None):
        """Initialize integrated app with simulation."""
        self.ui_config = ui_config
        self.sim_config = sim_config or SimulationConfig(
            world_width=640,
            world_height=480,
            max_agents=1,
            predator_count=2,
            initial_food=30,
            learning_enabled=True,
            render_enabled=False,
        )

        self.running = False

        # Create simulation
        self.simulation = Simulation(self.sim_config)

        # UI components
        self.renderer = Renderer(ui_config)
        self.input_handler = InputHandler(ui_config)
        self.signal_queue = TeachingSignalQueue()
        self.demo_controller = DemonstrationController()
        self.audio_capture = AudioCapture(
            sample_rate=ui_config.audio_sample_rate,
            channels=ui_config.audio_channels,
            buffer_size=ui_config.waveform_history
        )

        # Stats
        self.stats = {
            "rewards": 0,
            "punishments": 0,
            "demonstrations": 0,
            "voice_samples": 0,
            "loss": 0.0
        }

        self.start_time = 0.0

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

    def _get_world_state(self) -> Dict[str, Any]:
        """Extract world state for rendering."""
        entities = []

        # Add agents
        for agent_id, wrapper in self.simulation.agents.items():
            if wrapper.agent.is_alive:
                entities.append({
                    "type": "agent",
                    "position": (wrapper.agent.position.x, wrapper.agent.position.y),
                    "radius": wrapper.agent.radius,
                })

        # Add food
        for food in self.simulation.world.food_items:
            if food.is_active:
                entities.append({
                    "type": "food",
                    "position": (food.position.x, food.position.y),
                    "radius": food.radius,
                })

        # Add predators
        for predator in self.simulation.world.predators:
            if predator.is_active:
                entities.append({
                    "type": "predator",
                    "position": (predator.position.x, predator.position.y),
                    "radius": predator.radius,
                })

        return {"entities": entities}

    def _get_agent_state(self) -> Dict[str, Any]:
        """Get first agent's state for display."""
        for wrapper in self.simulation.agents.values():
            agent = wrapper.agent
            return {
                "energy": agent.energy / agent.genome.max_energy,
                "health": agent.health / agent.genome.max_health,
                "age": agent.age,
                "survival_time": agent.age,
            }
        return {"energy": 0, "health": 0, "age": 0, "survival_time": 0}

    def _process_events(self, timestamp: float) -> None:
        """Process input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if event.type == pygame.KEYDOWN and event.key == self.ui_config.keys.QUIT:
                self.running = False
                continue

            # Process teaching signals
            signals = self.input_handler.process_event(event, timestamp)

            for signal in signals:
                self.signal_queue.enqueue(signal)

                # Inject into simulation
                if signal.signal_type == TeachingSignalType.REWARD:
                    self.stats["rewards"] += 1
                    for wrapper in self.simulation.agents.values():
                        wrapper.events.append('human_reward')

                elif signal.signal_type == TeachingSignalType.PUNISH:
                    self.stats["punishments"] += 1
                    for wrapper in self.simulation.agents.values():
                        wrapper.events.append('human_punish')

                elif signal.signal_type == TeachingSignalType.DEMONSTRATE:
                    self.stats["demonstrations"] += 1

    def _update(self, dt: float) -> None:
        """Update simulation."""
        # Run one simulation tick
        self.simulation.tick()

        # Handle demo mode - human controls agent
        if self.input_handler.state.control_mode:
            keys = pygame.key.get_pressed()
            action = DemoAction.from_key_input(
                up=keys[pygame.K_UP],
                down=keys[pygame.K_DOWN],
                left=keys[pygame.K_LEFT],
                right=keys[pygame.K_RIGHT]
            )

            if action.action_type == "move":
                for wrapper in self.simulation.agents.values():
                    if wrapper.agent.is_alive:
                        # Apply movement directly
                        dx, dy = action.move_direction
                        speed = 100.0 * dt
                        wrapper.agent.position.x += dx * speed
                        wrapper.agent.position.y += dy * speed

    def _render(self) -> None:
        """Render frame."""
        waveform = self.audio_capture.get_recent(self.ui_config.waveform_history)
        mode = "CONTROL" if self.input_handler.state.control_mode else "OBSERVE"

        self.renderer.render_frame(
            world_state=self._get_world_state(),
            agent_state=self._get_agent_state(),
            agent_view=None,  # Would need to render agent's POV
            waveform=waveform,
            metrics=self.stats,
            mode=mode,
            zoom=self.input_handler.state.zoom_level
        )

    def run(self) -> None:
        """Main loop."""
        self.start()

        try:
            while self.running:
                dt = self.renderer.tick()
                current_time = time.time() - self.start_time

                self._process_events(current_time)
                self._update(dt)
                self._render()

                # Check if all agents dead - respawn
                if all(not w.agent.is_alive for w in self.simulation.agents.values()):
                    print(f"All agents died. Resetting simulation...")
                    self.simulation.reset()

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.audio_capture.stop()
        self.renderer.cleanup()


def main():
    """Run integrated teaching interface."""
    ui_config = UIConfig()
    sim_config = SimulationConfig(
        world_width=640,
        world_height=480,
        max_agents=1,
        predator_count=2,
        initial_food=30,
        learning_enabled=True,
    )

    app = IntegratedTeachingApp(ui_config, sim_config)
    app.run()


if __name__ == "__main__":
    main()
