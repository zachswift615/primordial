"""Main simulation orchestrator."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from primordial.simulation.agent_wrapper import AgentWrapper
import random
import numpy as np

from primordial.world.world import World
from primordial.world.geometry import Vec2
from primordial.world.entities import Food, Predator, Vegetation, Water
from primordial.simulation.config import SimulationConfig
from primordial.simulation.agent_wrapper import AgentWrapper
from primordial.agents.body import Gender
from primordial.agents.genome import breed


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
        """Spawn initial food, predators, and vegetation."""
        # Spawn vegetation clusters (for hiding)
        for _ in range(self.config.vegetation_clusters):
            cluster_center = Vec2(
                np.random.uniform(100, self.world.width - 100),
                np.random.uniform(100, self.world.height - 100),
            )
            cluster_size = np.random.randint(3, 6)  # 3-5 plants per cluster

            for _ in range(cluster_size):
                offset = Vec2(
                    np.random.uniform(-40, 40),
                    np.random.uniform(-40, 40),
                )
                veg = Vegetation(
                    entity_id=self.world.next_entity_id,
                    position=cluster_center + offset,
                    radius=np.random.uniform(15, 25),
                )
                self.world.add_entity(veg)

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

        # Spawn water bodies
        for _ in range(self.config.water_bodies):
            water = Water(
                entity_id=self.world.next_entity_id,
                position=Vec2(
                    np.random.uniform(100, self.world.width - 100),
                    np.random.uniform(100, self.world.height - 100),
                ),
                radius=np.random.uniform(25, 40),
            )
            self.world.add_entity(water)

    def tick(self, dt: float = None) -> Dict[str, Any]:
        """Perform one simulation tick.

        Args:
            dt: Elapsed time in seconds. If None, uses fixed timestep.

        Returns:
            Metrics from this tick.
        """
        self.step_count += 1
        tick_metrics = {'step': self.step_count}

        # Use provided dt or fall back to world's fixed timestep
        step_dt = dt if dt is not None else self.world.dt

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
            wrapper.agent.apply_action(action, step_dt, self.world)

        # 3. World physics tick
        self.world.tick(step_dt)

        # 4. Check for breeding opportunities
        breeding_results = self._check_breeding()
        if breeding_results:
            tick_metrics['breeding'] = breeding_results

        # 5. Track deaths (no auto-respawn - breeding is the only way to create new agents)
        for agent_id, wrapper in self.agents.items():
            if not wrapper.agent.is_alive and wrapper.agent.is_active:
                # Agent just died this tick
                death_info = wrapper.on_death()
                tick_metrics[f'{agent_id}_death'] = death_info
                wrapper.agent.is_active = False  # Mark as processed

        # 6. Collect agent metrics
        for agent_id, wrapper in self.agents.items():
            tick_metrics[f'{agent_id}_alive'] = wrapper.agent.is_alive
            tick_metrics[f'{agent_id}_energy'] = wrapper.agent.energy
            tick_metrics[f'{agent_id}_health'] = wrapper.agent.health
            tick_metrics[f'{agent_id}_age'] = wrapper.agent.age

        # Store metrics
        if self.step_count % self.config.metrics_interval == 0:
            self.metrics_history.append(tick_metrics)

        return tick_metrics

    def _get_fittest_agent(self, exclude: str = None) -> Optional[AgentWrapper]:
        """Get the fittest living agent (for inheritance when another dies)."""
        candidates = [
            w for aid, w in self.agents.items()
            if w.agent.is_alive and aid != exclude
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda w: w.get_fitness())

    def _check_breeding(self) -> Optional[Dict[str, Any]]:
        """Check for breeding opportunities between nearby agents.

        Breeding requires:
        - Two agents of opposite genders
        - Both ready to breed (mature, high drive, not on cooldown)
        - Close proximity (within breeding distance)
        - Room for more agents (below max_agents)
        - 50% chance of successful offspring

        Returns:
            Dict with breeding info if successful, None otherwise.
        """
        # Check if at max capacity
        alive_count = sum(1 for w in self.agents.values() if w.agent.is_alive)
        if alive_count >= self.config.max_agents:
            return None

        breeding_distance = 20.0  # Must be within this distance
        offspring_chance = 0.50  # 50% chance per breeding encounter

        # Get all agents ready to breed
        ready_agents = [
            w for w in self.agents.values()
            if w.agent.is_alive and w.agent.can_breed()
        ]

        if len(ready_agents) < 2:
            return None

        # Check pairs for breeding opportunities
        for i, agent1_wrapper in enumerate(ready_agents):
            for agent2_wrapper in ready_agents[i + 1:]:
                agent1 = agent1_wrapper.agent
                agent2 = agent2_wrapper.agent

                # Must be opposite genders
                if agent1.gender == agent2.gender:
                    continue

                # Must be close enough
                distance = agent1.position.distance_to(agent2.position)
                if distance > breeding_distance:
                    continue

                # Both agents "attempt" breeding - reset their drives
                agent1.on_breed_success()
                agent2.on_breed_success()

                # Track breeding attempts
                agent1_wrapper.lifetime_stats['times_bred'] += 1
                agent2_wrapper.lifetime_stats['times_bred'] += 1

                # 25% chance of offspring
                if random.random() > offspring_chance:
                    return {'attempted': True, 'success': False,
                            'parent1': agent1.agent_id, 'parent2': agent2.agent_id}

                # Create offspring!
                offspring_genome = breed(agent1.genome, agent2.genome)

                # Track offspring count
                agent1_wrapper.lifetime_stats['offspring_count'] += 1
                agent2_wrapper.lifetime_stats['offspring_count'] += 1

                # Find a dead agent slot to replace, or create new if room
                dead_wrapper = None
                for w in self.agents.values():
                    if not w.agent.is_alive:
                        dead_wrapper = w
                        break

                if dead_wrapper:
                    # Respawn dead agent with new genome
                    new_pos = Vec2(
                        (agent1.position.x + agent2.position.x) / 2 + random.uniform(-20, 20),
                        (agent1.position.y + agent2.position.y) / 2 + random.uniform(-20, 20)
                    )
                    # Clamp to world bounds
                    new_pos = Vec2(
                        max(50, min(self.config.world_width - 50, new_pos.x)),
                        max(50, min(self.config.world_height - 50, new_pos.y))
                    )

                    parent_gen = max(agent1_wrapper.generation, agent2_wrapper.generation)
                    dead_wrapper.respawn_with_genome(offspring_genome, new_pos, parent_gen)
                    self.world.add_entity(dead_wrapper.agent)

                    return {
                        'attempted': True,
                        'success': True,
                        'parent1': agent1.agent_id,
                        'parent2': agent2.agent_id,
                        'offspring': dead_wrapper.agent_id,
                        'generation': dead_wrapper.generation
                    }

        return None

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
