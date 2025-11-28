"""Wrapper combining AgentBody, LRN model, and OnlineLearningLoop."""

from typing import Tuple, Dict, Any, Optional, List
import torch

from primordial.agents.body import AgentBody
from primordial.agents.actions import AgentAction
from primordial.agents.genome import create_default_genome
from primordial.world.geometry import Vec2
from primordial.lrn.lrn_config import LRNConfig
from primordial.lrn.architecture import LivingResonanceNetwork
from primordial.learning.learning_loop import OnlineLearningLoop
from primordial.simulation.config import SimulationConfig


class AgentWrapper:
    """Wraps AgentBody with neural network and learning loop.

    Orchestrates the sense -> think -> act -> learn cycle.
    """

    def __init__(
        self,
        agent_id: str,
        config: SimulationConfig,
        initial_position: Optional[Vec2] = None,
    ):
        """Initialize agent wrapper.

        Args:
            agent_id: Unique agent identifier.
            config: Simulation configuration.
            initial_position: Starting position (random if None).
        """
        self.config = config
        self.agent_id = agent_id

        # Create agent body
        genome = create_default_genome()
        pos = initial_position or Vec2(
            config.world_width / 2,
            config.world_height / 2
        )
        self.agent = AgentBody(
            agent_id=agent_id,
            genome=genome,
            initial_position=pos
        )

        # Create LRN model
        lrn_config = LRNConfig(
            hidden_dim=config.lrn_hidden_dim,
            num_mixing_layers=config.lrn_num_mixing_layers,
        )
        self.model = LivingResonanceNetwork(lrn_config)

        # Create learning loop (if learning enabled)
        self.learning_enabled = config.learning_enabled
        if self.learning_enabled:
            optimizer_config = {
                'type': 'adamw',
                'params': {'lr': config.learning_rate},
                'lr_schedule': {'warmup_steps': 100}
            }
            reward_config = {
                'modulation': {'reward_scale': config.reward_modulation_scale}
            }
            self.learning_loop = OnlineLearningLoop(
                model=self.model,
                optimizer_config=optimizer_config,
                reward_config=reward_config,
            )
        else:
            self.learning_loop = None

        # State tracking
        self.prev_senses = None
        self.prev_agent_state = None
        self.step_count = 0
        self.events: List[str] = []

    def step(self, world) -> Tuple[AgentAction, Dict[str, Any]]:
        """Perform one simulation step.

        Args:
            world: World instance.

        Returns:
            action: Action to apply.
            metrics: Learning metrics (empty if no learning).
        """
        metrics = {}

        # 1. Sense - Get observations
        observations = self.agent.get_observations(world)

        # Convert to model input format
        vision = torch.from_numpy(observations['vision']).float().unsqueeze(0)
        audio = self._expand_audio(observations['audio'])
        # Fix shape mismatch: AgentBody returns (8,) but LRN expects (batch, 7)
        proprio = torch.from_numpy(observations['proprioception'][:7]).float().unsqueeze(0)
        touch = torch.from_numpy(observations['touch']).float().unsqueeze(0)

        # Current state for reward computation
        current_state = {
            'health': self.agent.health,
            'energy': self.agent.energy,
            'is_alive': self.agent.is_alive,
            'is_eating': self.agent.is_eating,
        }

        # 2. Think + Learn
        if self.learning_enabled and self.learning_loop is not None:
            if self.prev_senses is not None:
                # Create combined senses tensor for learning loop
                senses = torch.cat([
                    vision.flatten(1),
                    audio.flatten(1),
                    proprio,
                    touch
                ], dim=1)

                prev_senses = self.prev_senses

                # Collect events
                events = self._collect_events(self.prev_agent_state, current_state)

                # Learning step
                action_tensor, prediction = self.learning_loop.step(
                    senses=senses,
                    prev_senses=prev_senses,
                    agent_state=current_state,
                    prev_agent_state=self.prev_agent_state,
                    events=events
                )

                metrics['step'] = self.step_count
            else:
                # First step - just forward pass
                with torch.no_grad():
                    _, _, action_tensor = self.model(vision, audio, proprio, touch)
        else:
            # No learning - just inference
            with torch.no_grad():
                _, _, action_tensor = self.model(vision, audio, proprio, touch)

        # Store for next step
        self.prev_senses = torch.cat([
            vision.flatten(1),
            audio.flatten(1),
            proprio,
            touch
        ], dim=1)
        self.prev_agent_state = current_state
        self.step_count += 1

        # 3. Act - Convert tensor to AgentAction
        action = self._tensor_to_action(action_tensor)

        return action, metrics

    def _expand_audio(self, audio: Any) -> torch.Tensor:
        """Expand stereo audio to expected shape (batch, 100, 2)."""
        # audio is (2,) stereo - expand to (1, 100, 2)
        audio_tensor = torch.from_numpy(audio).float()
        expanded = audio_tensor.unsqueeze(0).unsqueeze(0).expand(1, 100, 2)
        return expanded

    def _tensor_to_action(self, action_tensor: torch.Tensor) -> AgentAction:
        """Convert model output to AgentAction."""
        a = action_tensor.squeeze(0).detach().cpu().numpy()
        return AgentAction(
            thrust=float(a[0]),
            torque=float(a[1]),
            vocalize=(float(a[2]), float(a[3])),
            eat=float(a[4])
        )

    def _collect_events(
        self,
        prev_state: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> List[str]:
        """Collect events for reward computation."""
        events = []

        if prev_state is None:
            return events

        # Eating event
        if current_state['is_eating'] and not prev_state['is_eating']:
            events.append('eat')

        # Damage event
        if current_state['health'] < prev_state['health']:
            events.append('damage')

        # Death event
        if not current_state['is_alive'] and prev_state['is_alive']:
            events.append('death')

        return events

    def on_death(self) -> Dict[str, Any]:
        """Handle agent death."""
        if self.learning_loop is not None:
            return self.learning_loop.on_death()
        return {'death_count': 0}

    def save_checkpoint(self, path: str) -> None:
        """Save agent state and model."""
        import json

        # Save agent state
        agent_data = self.agent.save()
        with open(f"{path}_agent.json", 'w') as f:
            json.dump(agent_data, f)

        # Save learning state
        if self.learning_loop is not None:
            self.learning_loop.save_checkpoint(f"{path}_learning.pt")

    def load_checkpoint(self, path: str) -> None:
        """Load agent state and model."""
        import json

        # Load agent state
        with open(f"{path}_agent.json", 'r') as f:
            agent_data = json.load(f)
        self.agent = AgentBody.load(agent_data)

        # Load learning state
        if self.learning_loop is not None:
            self.learning_loop.load_checkpoint(f"{path}_learning.pt")
