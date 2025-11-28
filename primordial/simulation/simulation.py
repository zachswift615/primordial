"""Main simulation orchestrator."""

from typing import Dict, Any, List, Optional
import random
import numpy as np

from primordial.world.world import World
from primordial.world.geometry import Vec2
from primordial.world.entities import Food, Predator
from primordial.simulation.config import SimulationConfig
from primordial.simulation.agent_wrapper import AgentWrapper


class Simulation:
    """Main simulation class orchestrating world and agents.

    Coordinates the simulation loop:
    1. Agents sense the world
    2. Agents decide on actions (neural network)
    3. Agents learn from experience
    4. Actions are applied
    5. World physics tick
    6. Metrics are collected
    """

    def __init__(self, config: SimulationConfig):
        """Initialize simulation.

        Args:
            config: Simulation configuration.
        """
        self.config = config

        # Set random seed if provided
        if config.seed is not None:
            random.seed(config.seed)

        # Create world
        self.world = World(
            width=config.world_width,
            height=config.world_height,
            tick_rate=config.tick_rate
        )

        # Spawn initial entities
        self._spawn_initial_entities()

        # Create agents
        self.agents: Dict[str, AgentWrapper] = {}
        for i in range(config.max_agents):
            agent_id = f"agent_{i}"
            pos = Vec2(
                random.uniform(100, config.world_width - 100),
                random.uniform(100, config.world_height - 100)
            )
            wrapper = AgentWrapper(
                agent_id=agent_id,
                config=config,
                initial_position=pos
            )
            self.agents[agent_id] = wrapper
            self.world.add_entity(wrapper.agent)

        # Metrics
        self.step_count = 0
        self.metrics_history: List[Dict[str, Any]] = []

    def _spawn_initial_entities(self) -> None:
        """Spawn initial food and predators."""
        # Spawn food
        for _ in range(self.config.initial_food):
            margin = 50.0
            x = np.random.uniform(margin, self.world.width - margin)
            y = np.random.uniform(margin, self.world.height - margin)

            food = Food(
                entity_id=self.world.next_entity_id,
                position=Vec2(x, y),
                energy_value=50.0,
                sound_intensity=0.1,
            )
            self.world.add_entity(food)

        # Spawn predators
        for _ in range(self.config.predator_count):
            patrol_center = Vec2(
                np.random.uniform(200, self.world.width - 200),
                np.random.uniform(200, self.world.height - 200),
            )
            predator = Predator(
                entity_id=self.world.next_entity_id,
                position=patrol_center,
                patrol_center=patrol_center,
                patrol_radius=150.0,
            )
            self.world.add_entity(predator)

    def tick(self) -> Dict[str, Any]:
        """Perform one simulation tick.

        Returns:
            Metrics from this tick.
        """
        self.step_count += 1
        tick_metrics = {'step': self.step_count}

        # 1. Agent sensing and action selection
        agent_actions = {}
        for agent_id, wrapper in self.agents.items():
            if not wrapper.agent.is_alive:
                continue

            action, learn_metrics = wrapper.step(self.world)
            agent_actions[agent_id] = action

            # Collect learning metrics
            for k, v in learn_metrics.items():
                tick_metrics[f'{agent_id}_{k}'] = v

        # 2. Apply actions
        for agent_id, action in agent_actions.items():
            wrapper = self.agents[agent_id]
            wrapper.agent.apply_action(action, self.world.dt, self.world)

        # 3. World physics tick
        self.world.tick()

        # 4. Check for deaths
        for agent_id, wrapper in self.agents.items():
            if not wrapper.agent.is_alive:
                death_info = wrapper.on_death()
                tick_metrics[f'{agent_id}_death'] = death_info

        # 5. Collect agent metrics
        for agent_id, wrapper in self.agents.items():
            tick_metrics[f'{agent_id}_alive'] = wrapper.agent.is_alive
            tick_metrics[f'{agent_id}_energy'] = wrapper.agent.energy
            tick_metrics[f'{agent_id}_health'] = wrapper.agent.health
            tick_metrics[f'{agent_id}_age'] = wrapper.agent.age

        # Store metrics
        if self.step_count % self.config.metrics_interval == 0:
            self.metrics_history.append(tick_metrics)

        return tick_metrics

    def run(self, steps: int) -> List[Dict[str, Any]]:
        """Run simulation for specified steps.

        Args:
            steps: Number of steps to run.

        Returns:
            List of metrics from each step.
        """
        all_metrics = []
        for _ in range(steps):
            metrics = self.tick()
            all_metrics.append(metrics)
        return all_metrics

    def get_agent_survival_time(self, agent_id: str) -> float:
        """Get survival time for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Survival time in seconds.
        """
        if agent_id in self.agents:
            return self.agents[agent_id].agent.age
        return 0.0

    def reset(self) -> None:
        """Reset simulation to initial state."""
        # Clear world
        self.world = World(
            width=self.config.world_width,
            height=self.config.world_height,
            tick_rate=self.config.tick_rate
        )

        # Respawn entities
        self._spawn_initial_entities()

        # Recreate agents (keeping learned weights)
        for agent_id, wrapper in self.agents.items():
            pos = Vec2(
                random.uniform(100, self.config.world_width - 100),
                random.uniform(100, self.config.world_height - 100)
            )
            # Create new body but keep learning loop
            old_learning = wrapper.learning_loop
            wrapper.agent = wrapper.agent.__class__(
                agent_id=agent_id,
                genome=wrapper.agent.genome,
                initial_position=pos
            )
            wrapper.learning_loop = old_learning
            wrapper.prev_senses = None
            wrapper.prev_modalities = None
            wrapper.prev_agent_state = None
            self.world.add_entity(wrapper.agent)

        self.step_count = 0
        self.metrics_history = []

    def save_state(self, path: str) -> None:
        """Save complete simulation state."""
        import json

        state = {
            'config': self.config.to_dict(),
            'step_count': self.step_count,
        }

        with open(f"{path}_sim.json", 'w') as f:
            json.dump(state, f, indent=2)

        # Save each agent
        for agent_id, wrapper in self.agents.items():
            wrapper.save_checkpoint(f"{path}_{agent_id}")

    def load_state(self, path: str) -> None:
        """Load simulation state."""
        import json

        with open(f"{path}_sim.json", 'r') as f:
            state = json.load(f)

        self.step_count = state['step_count']

        # Load each agent
        for agent_id, wrapper in self.agents.items():
            wrapper.load_checkpoint(f"{path}_{agent_id}")
