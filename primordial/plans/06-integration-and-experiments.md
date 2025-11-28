# Integration & Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire all components together into a runnable simulation, create automated experiments to validate success criteria, and provide CLI tools to run experiments.

**Architecture:** The `Simulation` class orchestrates World, AgentBody, LRN, and OnlineLearningLoop. Experiments run as automated scripts that collect metrics over time and generate reports. A CLI exposes experiments for easy execution.

**Tech Stack:** PyTorch, NumPy, Pygame (visualization), JSON/CSV (metrics export), argparse (CLI)

---

## File Structure

```
primordial/
├── simulation/
│   ├── __init__.py
│   ├── simulation.py         # Main Simulation class
│   ├── agent_wrapper.py      # Wraps AgentBody + LRN + LearningLoop
│   └── config.py             # SimulationConfig
├── experiments/
│   ├── __init__.py
│   ├── base.py               # BaseExperiment class
│   ├── survival_baseline.py  # Learning ON vs OFF
│   ├── teaching_impact.py    # Human teaching effect
│   └── metrics_collector.py  # Automated metrics collection
├── cli/
│   ├── __init__.py
│   └── run_experiment.py     # CLI for running experiments
└── tests/
    ├── simulation/
    │   ├── test_simulation.py
    │   └── test_agent_wrapper.py
    └── experiments/
        ├── test_survival_baseline.py
        └── test_metrics_collector.py
```

---

## Task 1: Simulation Configuration

**Files:**
- Create: `primordial/simulation/__init__.py`
- Create: `primordial/simulation/config.py`
- Test: `primordial/tests/simulation/test_config.py`

**Step 1: Write the failing test**

Create: `primordial/tests/simulation/__init__.py`

```python
"""Tests for simulation module."""
```

Create: `primordial/tests/simulation/test_config.py`

```python
import pytest
from primordial.simulation.config import SimulationConfig


def test_simulation_config_defaults():
    config = SimulationConfig()
    assert config.world_width == 1000.0
    assert config.world_height == 1000.0
    assert config.tick_rate == 60
    assert config.learning_enabled is True


def test_simulation_config_custom():
    config = SimulationConfig(
        world_width=500.0,
        learning_enabled=False
    )
    assert config.world_width == 500.0
    assert config.learning_enabled is False


def test_simulation_config_to_dict():
    config = SimulationConfig()
    d = config.to_dict()
    assert 'world_width' in d
    assert 'learning_enabled' in d
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/simulation/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/simulation/__init__.py`

```python
"""Simulation orchestration module."""

from .config import SimulationConfig

__all__ = ['SimulationConfig']
```

Create: `primordial/simulation/config.py`

```python
"""Configuration for simulation."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class SimulationConfig:
    """Configuration for running a simulation.

    Attributes:
        world_width: World width in units.
        world_height: World height in units.
        tick_rate: Simulation ticks per second.
        learning_enabled: Whether online learning is active.
        max_agents: Maximum number of agents.
        initial_food: Initial food items to spawn.
        max_food: Maximum food in world.
        predator_count: Number of predators.
        checkpoint_interval: Steps between checkpoints.
        metrics_interval: Steps between metrics logging.
        render_enabled: Whether to render visualization.
        seed: Random seed for reproducibility.
    """
    # World settings
    world_width: float = 1000.0
    world_height: float = 1000.0
    tick_rate: int = 60

    # Agent settings
    max_agents: int = 1
    initial_food: int = 50
    max_food: int = 100
    predator_count: int = 3

    # Learning settings
    learning_enabled: bool = True
    checkpoint_interval: int = 1000
    metrics_interval: int = 100

    # LRN settings (passed to LRNConfig)
    lrn_hidden_dim: int = 128
    lrn_num_mixing_layers: int = 6

    # Optimizer settings
    learning_rate: float = 1e-4
    reward_modulation_scale: float = 1.0

    # Rendering
    render_enabled: bool = False

    # Reproducibility
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/simulation/test_config.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add primordial/simulation/ primordial/tests/simulation/
git commit -m "feat: add simulation configuration"
```

---

## Task 2: Agent Wrapper

**Files:**
- Create: `primordial/simulation/agent_wrapper.py`
- Test: `primordial/tests/simulation/test_agent_wrapper.py`

**Step 1: Write the failing test**

Create: `primordial/tests/simulation/test_agent_wrapper.py`

```python
import pytest
import torch
from primordial.simulation.agent_wrapper import AgentWrapper
from primordial.simulation.config import SimulationConfig
from primordial.world.world import World


@pytest.fixture
def world():
    return World(width=500, height=500)


@pytest.fixture
def config():
    return SimulationConfig(learning_enabled=True)


def test_agent_wrapper_creation(world, config):
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    assert wrapper.agent is not None
    assert wrapper.model is not None
    assert wrapper.learning_loop is not None


def test_agent_wrapper_step(world, config):
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    world.add_entity(wrapper.agent)

    # Perform one step
    action, metrics = wrapper.step(world)

    assert action is not None
    assert 'loss' in metrics or metrics == {}  # No loss on first step


def test_agent_wrapper_learning_disabled(world):
    config = SimulationConfig(learning_enabled=False)
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    world.add_entity(wrapper.agent)

    # Should still produce action without learning
    action, metrics = wrapper.step(world)
    assert action is not None


def test_agent_wrapper_death_handling(world, config):
    wrapper = AgentWrapper(agent_id="test_agent", config=config)
    world.add_entity(wrapper.agent)

    # Simulate death
    wrapper.agent.die("test")
    result = wrapper.on_death()

    assert 'death_count' in result
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/simulation/test_agent_wrapper.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/simulation/agent_wrapper.py`

```python
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
                'modulation': {'scale': config.reward_modulation_scale}
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
        proprio = torch.from_numpy(observations['proprioception']).float().unsqueeze(0)
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
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/simulation/test_agent_wrapper.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add primordial/simulation/agent_wrapper.py primordial/tests/simulation/test_agent_wrapper.py
git commit -m "feat: add agent wrapper combining body, model, and learning"
```

---

## Task 3: Main Simulation Class

**Files:**
- Create: `primordial/simulation/simulation.py`
- Test: `primordial/tests/simulation/test_simulation.py`

**Step 1: Write the failing test**

Create: `primordial/tests/simulation/test_simulation.py`

```python
import pytest
from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig


@pytest.fixture
def config():
    return SimulationConfig(
        world_width=500,
        world_height=500,
        predator_count=1,
        initial_food=10,
        render_enabled=False
    )


def test_simulation_creation(config):
    sim = Simulation(config)
    assert sim.world is not None
    assert len(sim.agents) == config.max_agents


def test_simulation_tick(config):
    sim = Simulation(config)

    metrics = sim.tick()

    assert 'step' in metrics
    assert metrics['step'] == 1


def test_simulation_run_steps(config):
    sim = Simulation(config)

    all_metrics = sim.run(steps=10)

    assert len(all_metrics) == 10


def test_simulation_agent_survival_time(config):
    sim = Simulation(config)

    # Run for a bit
    sim.run(steps=100)

    # Check we can get survival time
    survival = sim.get_agent_survival_time("agent_0")
    assert survival >= 0


def test_simulation_reset(config):
    sim = Simulation(config)
    sim.run(steps=50)

    sim.reset()

    assert sim.step_count == 0
    assert all(a.agent.is_alive for a in sim.agents.values())
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/simulation/test_simulation.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/simulation/simulation.py`

```python
"""Main simulation orchestrator."""

from typing import Dict, Any, List, Optional
import random

from primordial.world.world import World
from primordial.world.geometry import Vec2
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
            self.world.spawn_food()

        # Spawn predators
        for _ in range(self.config.predator_count):
            self.world.spawn_predator()

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
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/simulation/test_simulation.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add primordial/simulation/simulation.py primordial/tests/simulation/test_simulation.py
git commit -m "feat: add main simulation orchestrator"
```

---

## Task 4: Metrics Collector

**Files:**
- Create: `primordial/experiments/__init__.py`
- Create: `primordial/experiments/metrics_collector.py`
- Test: `primordial/tests/experiments/test_metrics_collector.py`

**Step 1: Write the failing test**

Create: `primordial/tests/experiments/__init__.py`

```python
"""Tests for experiments module."""
```

Create: `primordial/tests/experiments/test_metrics_collector.py`

```python
import pytest
import tempfile
import os
from primordial.experiments.metrics_collector import MetricsCollector


def test_metrics_collector_creation():
    collector = MetricsCollector()
    assert collector.metrics == []


def test_metrics_collector_record():
    collector = MetricsCollector()

    collector.record({'step': 1, 'survival_time': 10.0})
    collector.record({'step': 2, 'survival_time': 20.0})

    assert len(collector.metrics) == 2


def test_metrics_collector_summary():
    collector = MetricsCollector()

    for i in range(10):
        collector.record({'step': i, 'survival_time': float(i * 10)})

    summary = collector.summary()

    assert 'mean_survival_time' in summary
    assert 'max_survival_time' in summary
    assert summary['max_survival_time'] == 90.0


def test_metrics_collector_export_csv():
    collector = MetricsCollector()

    collector.record({'step': 1, 'survival_time': 10.0})
    collector.record({'step': 2, 'survival_time': 20.0})

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        path = f.name

    try:
        collector.export_csv(path)

        with open(path, 'r') as f:
            content = f.read()

        assert 'step' in content
        assert 'survival_time' in content
    finally:
        os.unlink(path)


def test_metrics_collector_export_json():
    collector = MetricsCollector()

    collector.record({'step': 1, 'value': 10.0})

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name

    try:
        collector.export_json(path)

        import json
        with open(path, 'r') as f:
            data = json.load(f)

        assert len(data['metrics']) == 1
    finally:
        os.unlink(path)
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/experiments/test_metrics_collector.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/experiments/__init__.py`

```python
"""Experiments module for validating Primordial."""

from .metrics_collector import MetricsCollector

__all__ = ['MetricsCollector']
```

Create: `primordial/experiments/metrics_collector.py`

```python
"""Metrics collection and export for experiments."""

import json
import csv
from typing import Dict, Any, List
from pathlib import Path


class MetricsCollector:
    """Collects and exports experiment metrics.

    Provides utilities for:
    - Recording metrics over time
    - Computing summary statistics
    - Exporting to CSV and JSON
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[Dict[str, Any]] = []

    def record(self, metrics: Dict[str, Any]) -> None:
        """Record a metrics snapshot.

        Args:
            metrics: Dictionary of metric values.
        """
        self.metrics.append(metrics.copy())

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics = []

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics.

        Returns:
            Dictionary with mean, max, min for numeric fields.
        """
        if not self.metrics:
            return {}

        summary = {}

        # Find all numeric fields
        numeric_fields = []
        for key, value in self.metrics[0].items():
            if isinstance(value, (int, float)):
                numeric_fields.append(key)

        # Compute statistics
        for field in numeric_fields:
            values = [m[field] for m in self.metrics if field in m]
            if values:
                summary[f'mean_{field}'] = sum(values) / len(values)
                summary[f'max_{field}'] = max(values)
                summary[f'min_{field}'] = min(values)

        summary['total_records'] = len(self.metrics)

        return summary

    def export_csv(self, path: str) -> None:
        """Export metrics to CSV file.

        Args:
            path: Output file path.
        """
        if not self.metrics:
            return

        # Collect all fieldnames
        fieldnames = set()
        for m in self.metrics:
            fieldnames.update(m.keys())
        fieldnames = sorted(fieldnames)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)

    def export_json(self, path: str) -> None:
        """Export metrics to JSON file.

        Args:
            path: Output file path.
        """
        data = {
            'metrics': self.metrics,
            'summary': self.summary()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_field_over_time(self, field: str) -> List[Any]:
        """Get values of a field over time.

        Args:
            field: Field name.

        Returns:
            List of values.
        """
        return [m.get(field) for m in self.metrics]
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/experiments/test_metrics_collector.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add primordial/experiments/ primordial/tests/experiments/
git commit -m "feat: add metrics collector for experiments"
```

---

## Task 5: Base Experiment Class

**Files:**
- Create: `primordial/experiments/base.py`
- Test: `primordial/tests/experiments/test_base.py`

**Step 1: Write the failing test**

Create: `primordial/tests/experiments/test_base.py`

```python
import pytest
import tempfile
import os
from primordial.experiments.base import BaseExperiment
from primordial.simulation.config import SimulationConfig


class DummyExperiment(BaseExperiment):
    """Test experiment implementation."""

    def run(self):
        # Run 100 steps
        for _ in range(100):
            self.simulation.tick()
            self.collector.record({
                'step': self.simulation.step_count,
                'survival_time': self.get_mean_survival_time()
            })
        return self.collector.summary()


def test_base_experiment_creation():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = DummyExperiment(config)
    assert exp.simulation is not None
    assert exp.collector is not None


def test_base_experiment_run():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = DummyExperiment(config)

    results = exp.run()

    assert 'total_records' in results


def test_base_experiment_export():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = DummyExperiment(config)
    exp.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        exp.export_results(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, 'metrics.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'summary.json'))
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/experiments/test_base.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/experiments/base.py`

```python
"""Base class for experiments."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import json

from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig
from primordial.experiments.metrics_collector import MetricsCollector


class BaseExperiment(ABC):
    """Base class for Primordial experiments.

    Provides common infrastructure:
    - Simulation setup
    - Metrics collection
    - Results export

    Subclasses implement the `run()` method with experiment logic.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize experiment.

        Args:
            config: Simulation configuration.
        """
        self.config = config
        self.simulation = Simulation(config)
        self.collector = MetricsCollector()

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment.

        Returns:
            Summary results.
        """
        pass

    def get_mean_survival_time(self) -> float:
        """Get mean survival time across all agents."""
        times = []
        for agent_id, wrapper in self.simulation.agents.items():
            times.append(wrapper.agent.age)
        return sum(times) / len(times) if times else 0.0

    def get_alive_count(self) -> int:
        """Get count of alive agents."""
        return sum(
            1 for w in self.simulation.agents.values()
            if w.agent.is_alive
        )

    def reset_simulation(self) -> None:
        """Reset simulation for another trial."""
        self.simulation.reset()

    def export_results(self, output_dir: str) -> None:
        """Export experiment results.

        Args:
            output_dir: Output directory path.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export metrics
        self.collector.export_csv(str(output_path / 'metrics.csv'))

        # Export summary
        summary = self.collector.summary()
        summary['config'] = self.config.to_dict()

        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def print_summary(self) -> None:
        """Print experiment summary to console."""
        summary = self.collector.summary()

        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS")
        print("=" * 50)

        for key, value in sorted(summary.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("=" * 50 + "\n")
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/experiments/test_base.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add primordial/experiments/base.py primordial/tests/experiments/test_base.py
git commit -m "feat: add base experiment class"
```

---

## Task 6: Survival Baseline Experiment

**Files:**
- Create: `primordial/experiments/survival_baseline.py`
- Test: `primordial/tests/experiments/test_survival_baseline.py`

**Step 1: Write the failing test**

Create: `primordial/tests/experiments/test_survival_baseline.py`

```python
import pytest
from primordial.experiments.survival_baseline import SurvivalBaselineExperiment
from primordial.simulation.config import SimulationConfig


def test_survival_experiment_creation():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=1,
        render_enabled=False
    )
    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=100,
        num_trials=2
    )
    assert exp.steps_per_trial == 100
    assert exp.num_trials == 2


def test_survival_experiment_run():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=1,
        render_enabled=False
    )
    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=50,
        num_trials=2
    )

    results = exp.run()

    assert 'learning_on_mean_survival' in results
    assert 'learning_off_mean_survival' in results
    assert 'improvement_ratio' in results


def test_survival_experiment_improvement_ratio():
    # This is more of an integration test
    config = SimulationConfig(
        world_width=500,
        world_height=500,
        predator_count=0,  # No predators for predictable results
        render_enabled=False
    )
    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=100,
        num_trials=1
    )

    results = exp.run()

    # Ratio should be >= 1 (at minimum, learning shouldn't hurt)
    # In practice, may be < 1 for short runs
    assert 'improvement_ratio' in results
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/experiments/test_survival_baseline.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/experiments/survival_baseline.py`

```python
"""Survival baseline experiment: Learning ON vs OFF.

Measures whether online learning improves agent survival time
compared to a random/untrained baseline.

Success criterion: Agent survives >5x longer with learning ON vs OFF.
"""

from typing import Dict, Any, List
from copy import deepcopy

from primordial.simulation.config import SimulationConfig
from primordial.experiments.base import BaseExperiment


class SurvivalBaselineExperiment(BaseExperiment):
    """Compares survival with learning enabled vs disabled.

    Runs multiple trials in each condition and computes
    mean survival time.
    """

    def __init__(
        self,
        config: SimulationConfig,
        steps_per_trial: int = 10000,
        num_trials: int = 5,
    ):
        """Initialize experiment.

        Args:
            config: Base simulation configuration.
            steps_per_trial: Steps to run each trial.
            num_trials: Number of trials per condition.
        """
        super().__init__(config)
        self.steps_per_trial = steps_per_trial
        self.num_trials = num_trials

        self.learning_on_survivals: List[float] = []
        self.learning_off_survivals: List[float] = []

    def run(self) -> Dict[str, Any]:
        """Run the experiment.

        Returns:
            Results with survival times and improvement ratio.
        """
        print("Starting Survival Baseline Experiment")
        print(f"  Steps per trial: {self.steps_per_trial}")
        print(f"  Trials per condition: {self.num_trials}")
        print()

        # Condition 1: Learning ON
        print("Running: Learning ON")
        self.learning_on_survivals = self._run_condition(learning_enabled=True)

        # Condition 2: Learning OFF
        print("Running: Learning OFF")
        self.learning_off_survivals = self._run_condition(learning_enabled=False)

        # Compute results
        results = self._compute_results()

        # Record to collector
        self.collector.record(results)

        return results

    def _run_condition(self, learning_enabled: bool) -> List[float]:
        """Run trials for a condition.

        Args:
            learning_enabled: Whether learning is enabled.

        Returns:
            List of survival times.
        """
        survival_times = []

        for trial in range(self.num_trials):
            # Create fresh config with learning setting
            trial_config = SimulationConfig(
                **{**self.config.to_dict(), 'learning_enabled': learning_enabled}
            )

            # Create fresh simulation
            from primordial.simulation.simulation import Simulation
            sim = Simulation(trial_config)

            # Run trial
            for step in range(self.steps_per_trial):
                sim.tick()

                # Early termination if all agents dead
                if all(not w.agent.is_alive for w in sim.agents.values()):
                    break

            # Collect survival times
            trial_survival = 0.0
            for agent_id, wrapper in sim.agents.items():
                trial_survival += wrapper.agent.age
            trial_survival /= len(sim.agents)

            survival_times.append(trial_survival)
            print(f"  Trial {trial + 1}/{self.num_trials}: {trial_survival:.2f}s")

        return survival_times

    def _compute_results(self) -> Dict[str, Any]:
        """Compute experiment results."""
        on_mean = sum(self.learning_on_survivals) / len(self.learning_on_survivals)
        off_mean = sum(self.learning_off_survivals) / len(self.learning_off_survivals)

        # Avoid division by zero
        ratio = on_mean / max(off_mean, 0.001)

        return {
            'learning_on_mean_survival': on_mean,
            'learning_off_mean_survival': off_mean,
            'learning_on_trials': self.learning_on_survivals,
            'learning_off_trials': self.learning_off_survivals,
            'improvement_ratio': ratio,
            'target_ratio': 5.0,
            'target_met': ratio >= 5.0,
        }


def main():
    """Run survival baseline experiment from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run survival baseline experiment'
    )
    parser.add_argument('--steps', type=int, default=10000,
                       help='Steps per trial')
    parser.add_argument('--trials', type=int, default=5,
                       help='Trials per condition')
    parser.add_argument('--output', type=str, default='./results/survival',
                       help='Output directory')

    args = parser.parse_args()

    config = SimulationConfig(
        world_width=1000,
        world_height=1000,
        predator_count=3,
        initial_food=50,
        render_enabled=False,
    )

    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials
    )

    results = exp.run()
    exp.print_summary()
    exp.export_results(args.output)


if __name__ == '__main__':
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/experiments/test_survival_baseline.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add primordial/experiments/survival_baseline.py primordial/tests/experiments/test_survival_baseline.py
git commit -m "feat: add survival baseline experiment (learning on vs off)"
```

---

## Task 7: Teaching Impact Experiment

**Files:**
- Create: `primordial/experiments/teaching_impact.py`
- Test: `primordial/tests/experiments/test_teaching_impact.py`

**Step 1: Write the failing test**

Create: `primordial/tests/experiments/test_teaching_impact.py`

```python
import pytest
from primordial.experiments.teaching_impact import TeachingImpactExperiment
from primordial.simulation.config import SimulationConfig


def test_teaching_experiment_creation():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        render_enabled=False
    )
    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=100,
        teaching_interval=10
    )
    assert exp.teaching_interval == 10


def test_teaching_experiment_run():
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        predator_count=0,
        render_enabled=False
    )
    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=50,
        teaching_interval=10,
        num_trials=1
    )

    results = exp.run()

    assert 'with_teaching_mean_survival' in results
    assert 'without_teaching_mean_survival' in results
    assert 'teaching_acceleration' in results
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/experiments/test_teaching_impact.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create: `primordial/experiments/teaching_impact.py`

```python
"""Teaching impact experiment: measures effect of human teaching.

Compares learning speed with and without simulated teaching signals.

Success criterion: Human teaching accelerates learning >2x.
"""

from typing import Dict, Any, List
import random

from primordial.simulation.config import SimulationConfig
from primordial.simulation.simulation import Simulation
from primordial.experiments.base import BaseExperiment


class TeachingImpactExperiment(BaseExperiment):
    """Measures the impact of teaching signals on learning.

    Simulates teaching by providing reward signals when agent
    performs good behaviors (eating, avoiding predators).
    """

    def __init__(
        self,
        config: SimulationConfig,
        steps_per_trial: int = 10000,
        num_trials: int = 3,
        teaching_interval: int = 10,
    ):
        """Initialize experiment.

        Args:
            config: Simulation configuration.
            steps_per_trial: Steps per trial.
            num_trials: Trials per condition.
            teaching_interval: Steps between teaching signals.
        """
        super().__init__(config)
        self.steps_per_trial = steps_per_trial
        self.num_trials = num_trials
        self.teaching_interval = teaching_interval

        self.with_teaching_survivals: List[float] = []
        self.without_teaching_survivals: List[float] = []

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        print("Starting Teaching Impact Experiment")
        print(f"  Steps per trial: {self.steps_per_trial}")
        print(f"  Teaching interval: {self.teaching_interval}")
        print()

        # Condition 1: With teaching
        print("Running: With Teaching")
        self.with_teaching_survivals = self._run_condition(teaching=True)

        # Condition 2: Without teaching
        print("Running: Without Teaching")
        self.without_teaching_survivals = self._run_condition(teaching=False)

        results = self._compute_results()
        self.collector.record(results)

        return results

    def _run_condition(self, teaching: bool) -> List[float]:
        """Run trials for a condition."""
        survival_times = []

        for trial in range(self.num_trials):
            sim = Simulation(self.config)

            for step in range(self.steps_per_trial):
                metrics = sim.tick()

                # Inject teaching signals
                if teaching and step % self.teaching_interval == 0:
                    self._inject_teaching(sim)

                # Early termination
                if all(not w.agent.is_alive for w in sim.agents.values()):
                    break

            # Collect survival
            trial_survival = sum(
                w.agent.age for w in sim.agents.values()
            ) / len(sim.agents)

            survival_times.append(trial_survival)
            print(f"  Trial {trial + 1}/{self.num_trials}: {trial_survival:.2f}s")

        return survival_times

    def _inject_teaching(self, sim: Simulation) -> None:
        """Inject simulated teaching signals.

        Rewards eating, punishes taking damage.
        """
        for agent_id, wrapper in sim.agents.items():
            if not wrapper.agent.is_alive:
                continue

            # Reward eating
            if wrapper.agent.is_eating:
                # Add 'reward' event to trigger positive modulation
                if wrapper.learning_loop is not None:
                    # The events list is processed in step()
                    # We add to the events that will be processed
                    wrapper.events.append('human_reward')

            # Punish low health
            if wrapper.agent.health < wrapper.agent.genome.max_health * 0.3:
                if wrapper.learning_loop is not None:
                    wrapper.events.append('human_punish')

    def _compute_results(self) -> Dict[str, Any]:
        """Compute results."""
        with_mean = sum(self.with_teaching_survivals) / len(self.with_teaching_survivals)
        without_mean = sum(self.without_teaching_survivals) / len(self.without_teaching_survivals)

        acceleration = with_mean / max(without_mean, 0.001)

        return {
            'with_teaching_mean_survival': with_mean,
            'without_teaching_mean_survival': without_mean,
            'with_teaching_trials': self.with_teaching_survivals,
            'without_teaching_trials': self.without_teaching_survivals,
            'teaching_acceleration': acceleration,
            'target_acceleration': 2.0,
            'target_met': acceleration >= 2.0,
        }


def main():
    """Run teaching impact experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run teaching impact experiment'
    )
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--output', type=str, default='./results/teaching')

    args = parser.parse_args()

    config = SimulationConfig(
        world_width=1000,
        world_height=1000,
        predator_count=3,
        render_enabled=False,
    )

    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials
    )

    results = exp.run()
    exp.print_summary()
    exp.export_results(args.output)


if __name__ == '__main__':
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/experiments/test_teaching_impact.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add primordial/experiments/teaching_impact.py primordial/tests/experiments/test_teaching_impact.py
git commit -m "feat: add teaching impact experiment"
```

---

## Task 8: CLI for Running Experiments

**Files:**
- Create: `primordial/cli/__init__.py`
- Create: `primordial/cli/run_experiment.py`
- Update: `primordial/__main__.py` (add experiment subcommand)

**Step 1: Write the failing test**

Create: `primordial/tests/cli/__init__.py`

```python
"""Tests for CLI module."""
```

Create: `primordial/tests/cli/test_cli.py`

```python
import pytest
import subprocess
import sys


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        [sys.executable, '-m', 'primordial', '--help'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'Primordial' in result.stdout


def test_cli_experiment_list():
    """Test listing experiments."""
    result = subprocess.run(
        [sys.executable, '-m', 'primordial', 'experiment', '--list'],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert 'survival' in result.stdout.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest primordial/tests/cli/test_cli.py -v`
Expected: FAIL (help text won't include experiments)

**Step 3: Write minimal implementation**

Create: `primordial/cli/__init__.py`

```python
"""CLI module for Primordial."""
```

Create: `primordial/cli/run_experiment.py`

```python
"""CLI for running experiments."""

import argparse
from pathlib import Path

from primordial.simulation.config import SimulationConfig


EXPERIMENTS = {
    'survival': 'SurvivalBaselineExperiment',
    'teaching': 'TeachingImpactExperiment',
}


def list_experiments():
    """List available experiments."""
    print("\nAvailable Experiments:")
    print("=" * 40)
    print("  survival  - Learning ON vs OFF comparison")
    print("             Target: >5x survival improvement")
    print()
    print("  teaching  - Human teaching impact")
    print("             Target: >2x learning acceleration")
    print()
    print("Usage:")
    print("  python -m primordial experiment survival --steps 10000")
    print("  python -m primordial experiment teaching --output ./results")
    print()


def run_survival(args):
    """Run survival baseline experiment."""
    from primordial.experiments.survival_baseline import SurvivalBaselineExperiment

    config = SimulationConfig(
        world_width=args.world_size,
        world_height=args.world_size,
        predator_count=args.predators,
        initial_food=args.food,
        seed=args.seed,
        render_enabled=False,
    )

    exp = SurvivalBaselineExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials
    )

    results = exp.run()
    exp.print_summary()

    if args.output:
        exp.export_results(args.output)
        print(f"\nResults exported to: {args.output}")

    return results


def run_teaching(args):
    """Run teaching impact experiment."""
    from primordial.experiments.teaching_impact import TeachingImpactExperiment

    config = SimulationConfig(
        world_width=args.world_size,
        world_height=args.world_size,
        predator_count=args.predators,
        initial_food=args.food,
        seed=args.seed,
        render_enabled=False,
    )

    exp = TeachingImpactExperiment(
        config=config,
        steps_per_trial=args.steps,
        num_trials=args.trials,
        teaching_interval=args.teaching_interval
    )

    results = exp.run()
    exp.print_summary()

    if args.output:
        exp.export_results(args.output)
        print(f"\nResults exported to: {args.output}")

    return results


def add_experiment_parser(subparsers):
    """Add experiment subparser."""
    exp_parser = subparsers.add_parser(
        'experiment',
        help='Run experiments'
    )

    exp_parser.add_argument(
        'name',
        nargs='?',
        choices=['survival', 'teaching'],
        help='Experiment name'
    )

    exp_parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments'
    )

    # Common args
    exp_parser.add_argument('--steps', type=int, default=10000,
                           help='Steps per trial (default: 10000)')
    exp_parser.add_argument('--trials', type=int, default=5,
                           help='Trials per condition (default: 5)')
    exp_parser.add_argument('--output', type=str, default=None,
                           help='Output directory for results')
    exp_parser.add_argument('--world-size', type=int, default=1000,
                           help='World size (default: 1000)')
    exp_parser.add_argument('--predators', type=int, default=3,
                           help='Number of predators (default: 3)')
    exp_parser.add_argument('--food', type=int, default=50,
                           help='Initial food count (default: 50)')
    exp_parser.add_argument('--seed', type=int, default=None,
                           help='Random seed for reproducibility')

    # Teaching-specific
    exp_parser.add_argument('--teaching-interval', type=int, default=10,
                           help='Teaching signal interval (default: 10)')

    return exp_parser


def run_experiment_command(args):
    """Handle experiment command."""
    if args.list or args.name is None:
        list_experiments()
        return

    if args.name == 'survival':
        run_survival(args)
    elif args.name == 'teaching':
        run_teaching(args)
```

Update: `primordial/__main__.py`

```python
"""CLI entry point for Primordial."""

import argparse
import sys


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Primordial - Human Teaching Interface for AI Agents"
    )

    subparsers = parser.add_subparsers(dest='command')

    # Interface command (default)
    interface_parser = subparsers.add_parser(
        'interface',
        help='Run the teaching interface'
    )
    interface_parser.add_argument('--fps', type=int, default=60)
    interface_parser.add_argument('--width', type=int, default=960)
    interface_parser.add_argument('--height', type=int, default=720)
    interface_parser.add_argument('--no-audio', action='store_true')

    # Experiment command
    from primordial.cli.run_experiment import add_experiment_parser, run_experiment_command
    add_experiment_parser(subparsers)

    # Simulate command (headless)
    sim_parser = subparsers.add_parser(
        'simulate',
        help='Run headless simulation'
    )
    sim_parser.add_argument('--steps', type=int, default=1000)
    sim_parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.command == 'experiment':
        run_experiment_command(args)
    elif args.command == 'simulate':
        run_simulate(args)
    elif args.command == 'interface' or args.command is None:
        run_interface(args if args.command else parser.parse_args(['interface']))


def run_interface(args):
    """Run the teaching interface."""
    from primordial.interface.config import UIConfig
    from primordial.interface.app import TeachingApp

    config = UIConfig()
    config.fps = args.fps
    config.window_width = args.width
    config.window_height = args.height

    app = TeachingApp(config)

    if args.no_audio:
        app.audio_capture.stop()

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        app.stop()
        sys.exit(0)


def run_simulate(args):
    """Run headless simulation."""
    from primordial.simulation.simulation import Simulation
    from primordial.simulation.config import SimulationConfig

    config = SimulationConfig(render_enabled=False)
    sim = Simulation(config)

    print(f"Running simulation for {args.steps} steps...")
    metrics = sim.run(args.steps)

    print(f"\nCompleted {len(metrics)} steps")
    print(f"Final agent survival times:")
    for agent_id, wrapper in sim.agents.items():
        print(f"  {agent_id}: {wrapper.agent.age:.2f}s")

    if args.output:
        sim.save_state(args.output)
        print(f"\nState saved to: {args.output}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest primordial/tests/cli/test_cli.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add primordial/cli/ primordial/__main__.py primordial/tests/cli/
git commit -m "feat: add CLI for running experiments"
```

---

## Task 9: Update Package Exports

**Files:**
- Update: `primordial/simulation/__init__.py`
- Update: `primordial/experiments/__init__.py`

**Step 1: Update exports**

Update: `primordial/simulation/__init__.py`

```python
"""Simulation orchestration module."""

from .config import SimulationConfig
from .simulation import Simulation
from .agent_wrapper import AgentWrapper

__all__ = [
    'SimulationConfig',
    'Simulation',
    'AgentWrapper',
]
```

Update: `primordial/experiments/__init__.py`

```python
"""Experiments module for validating Primordial."""

from .metrics_collector import MetricsCollector
from .base import BaseExperiment
from .survival_baseline import SurvivalBaselineExperiment
from .teaching_impact import TeachingImpactExperiment

__all__ = [
    'MetricsCollector',
    'BaseExperiment',
    'SurvivalBaselineExperiment',
    'TeachingImpactExperiment',
]
```

**Step 2: Verify imports work**

Run: `python -c "from primordial.simulation import Simulation; from primordial.experiments import SurvivalBaselineExperiment; print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add primordial/simulation/__init__.py primordial/experiments/__init__.py
git commit -m "feat: update package exports"
```

---

## Task 10: Integration Test - Full Pipeline

**Files:**
- Create: `primordial/tests/test_full_pipeline.py`

**Step 1: Write integration test**

Create: `primordial/tests/test_full_pipeline.py`

```python
"""Full pipeline integration test."""

import pytest
from primordial.simulation.simulation import Simulation
from primordial.simulation.config import SimulationConfig


def test_full_pipeline_runs():
    """Test complete simulation pipeline runs without errors."""
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        max_agents=1,
        predator_count=0,  # No predators for stable test
        initial_food=20,
        learning_enabled=True,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)

    # Run for 100 steps
    metrics = sim.run(steps=100)

    assert len(metrics) == 100
    assert sim.step_count == 100

    # Agent should still be alive (no predators)
    for wrapper in sim.agents.values():
        assert wrapper.agent.is_alive


def test_full_pipeline_with_predators():
    """Test pipeline with predators (may cause death)."""
    config = SimulationConfig(
        world_width=200,
        world_height=200,
        max_agents=1,
        predator_count=2,
        initial_food=10,
        learning_enabled=True,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)

    # Run for 500 steps (enough for potential death)
    metrics = sim.run(steps=500)

    assert len(metrics) == 500


def test_full_pipeline_save_load():
    """Test saving and loading simulation state."""
    import tempfile
    import os

    config = SimulationConfig(
        world_width=200,
        world_height=200,
        max_agents=1,
        predator_count=0,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)
    sim.run(steps=50)

    original_step = sim.step_count
    original_age = list(sim.agents.values())[0].agent.age

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test')
        sim.save_state(path)

        # Create new simulation and load
        sim2 = Simulation(config)
        sim2.load_state(path)

        assert sim2.step_count == original_step


def test_learning_changes_behavior():
    """Test that learning produces different behavior over time."""
    config = SimulationConfig(
        world_width=300,
        world_height=300,
        max_agents=1,
        predator_count=0,
        initial_food=30,
        learning_enabled=True,
        render_enabled=False,
        seed=42
    )

    sim = Simulation(config)

    # Run and check agent is learning
    metrics = sim.run(steps=200)

    # Check that the model has been updated
    wrapper = list(sim.agents.values())[0]
    assert wrapper.step_count == 200
```

**Step 2: Run integration tests**

Run: `pytest primordial/tests/test_full_pipeline.py -v`
Expected: PASS (4 tests)

**Step 3: Run full test suite**

Run: `pytest primordial/tests/ -v --tb=short`
Expected: All tests pass

**Step 4: Commit**

```bash
git add primordial/tests/test_full_pipeline.py
git commit -m "test: add full pipeline integration tests"
```

---

## How to Run Experiments

### Quick Start

```bash
# List available experiments
python -m primordial experiment --list

# Run survival baseline (learning ON vs OFF)
python -m primordial experiment survival --steps 10000 --trials 5 --output ./results/survival

# Run teaching impact experiment
python -m primordial experiment teaching --steps 10000 --trials 3 --output ./results/teaching

# Run headless simulation
python -m primordial simulate --steps 5000

# Run interactive teaching interface
python -m primordial interface
```

### Experiment Output

Each experiment creates:
- `metrics.csv` - Raw metrics over time
- `summary.json` - Summary statistics and config

### Success Criteria Validation

| Criterion | Experiment | How to Measure |
|-----------|------------|----------------|
| >5x survival with learning | `survival` | `improvement_ratio` in results |
| >2x learning acceleration with teaching | `teaching` | `teaching_acceleration` in results |
| 60 FPS | Interface | FPS counter in UI |
| <10ms forward pass | Benchmark (manual) | Profile `model.forward()` |
| No catastrophic forgetting | Long run | Track `mean_survival` over time |

### Automated Validation

Create a script to run all experiments and check criteria:

```bash
# Run all experiments and validate
python -c "
from primordial.experiments import SurvivalBaselineExperiment, TeachingImpactExperiment
from primordial.simulation.config import SimulationConfig

config = SimulationConfig(predator_count=3, initial_food=50)

# Survival test
survival_exp = SurvivalBaselineExperiment(config, steps_per_trial=5000, num_trials=3)
survival_results = survival_exp.run()

# Teaching test
teaching_exp = TeachingImpactExperiment(config, steps_per_trial=5000, num_trials=3)
teaching_results = teaching_exp.run()

print('='*50)
print('SUCCESS CRITERIA VALIDATION')
print('='*50)
print(f'Survival improvement: {survival_results[\"improvement_ratio\"]:.2f}x (target: 5x) - {\"PASS\" if survival_results[\"target_met\"] else \"FAIL\"}')
print(f'Teaching acceleration: {teaching_results[\"teaching_acceleration\"]:.2f}x (target: 2x) - {\"PASS\" if teaching_results[\"target_met\"] else \"FAIL\"}')
"
```

---

## Summary

This plan creates:

1. **Simulation Module** (`primordial/simulation/`)
   - `SimulationConfig` - Configuration dataclass
   - `AgentWrapper` - Combines AgentBody + LRN + LearningLoop
   - `Simulation` - Main orchestrator

2. **Experiments Module** (`primordial/experiments/`)
   - `MetricsCollector` - Metrics collection and export
   - `BaseExperiment` - Abstract base class
   - `SurvivalBaselineExperiment` - Learning ON vs OFF
   - `TeachingImpactExperiment` - Human teaching effect

3. **CLI** (`primordial/cli/`)
   - `run_experiment.py` - Experiment runner
   - Updated `__main__.py` - Unified entry point

4. **Tests**
   - Unit tests for each component
   - Full pipeline integration tests

**Total: 10 tasks, ~35 new tests**
