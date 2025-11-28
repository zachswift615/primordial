"""Tests for AgentBody."""

import pytest
import math
import numpy as np
import torch

from primordial.agents import AgentBody, AgentGenome, AgentAction
from primordial.world import World
from primordial.world.geometry import Vec2
from primordial.world.entities import Food, EntityType


class TestAgentBodyCreation:
    """Tests for AgentBody initialization."""

    def test_default_creation(self):
        """Test agent creation with defaults."""
        agent = AgentBody(agent_id="test_agent")

        assert agent.agent_id == "test_agent"
        assert agent.is_alive
        assert agent.is_active
        assert agent.energy == 100.0
        assert agent.health == 100.0
        assert agent.age == 0.0

    def test_custom_position(self):
        """Test agent creation with custom position."""
        agent = AgentBody(
            agent_id="test",
            initial_position=Vec2(100, 200),
        )

        assert agent.position.x == 100
        assert agent.position.y == 200

    def test_custom_angle(self):
        """Test agent creation with custom facing angle."""
        agent = AgentBody(
            agent_id="test",
            initial_angle=math.pi / 2,
        )

        assert agent.angle == pytest.approx(math.pi / 2)
        assert agent.facing.x == pytest.approx(0.0, abs=1e-6)
        assert agent.facing.y == pytest.approx(1.0)

    def test_custom_genome(self):
        """Test agent creation with custom genome."""
        genome = AgentGenome(max_speed=200.0, radius=10.0)
        agent = AgentBody(agent_id="test", genome=genome)

        assert agent.genome.max_speed == 200.0
        assert agent.radius == 10.0

    def test_entity_type(self):
        """Test agent has correct entity type."""
        agent = AgentBody(agent_id="test")

        assert agent.entity_type == EntityType.AGENT

    def test_sensors_initialized(self):
        """Test all sensors are initialized."""
        agent = AgentBody(agent_id="test")

        assert agent.vision is not None
        assert agent.audio is not None
        assert agent.proprioception is not None
        assert agent.touch is not None


class TestAgentBodyPhysics:
    """Tests for agent physics and movement."""

    @pytest.fixture
    def agent(self):
        """Create agent at origin facing right."""
        return AgentBody(
            agent_id="test",
            initial_position=Vec2(250, 250),
            initial_angle=0.0,
        )

    @pytest.fixture
    def world(self):
        """Create empty world."""
        return World(width=500, height=500)

    def test_thrust_applies_force(self, agent, world):
        """Test that thrust accelerates agent forward."""
        action = AgentAction(
            thrust=1.0,
            torque=0.0,
            vocalize=np.zeros(2),
            eat=0.0,
        )

        initial_velocity = Vec2(agent.velocity.x, agent.velocity.y)
        agent.apply_action(action, dt=1/60, world=world)

        # Should have positive x velocity (facing right)
        assert agent.velocity.x > initial_velocity.x

    def test_negative_thrust(self, agent, world):
        """Test that negative thrust moves agent backward."""
        action = AgentAction(
            thrust=-1.0,
            torque=0.0,
            vocalize=np.zeros(2),
            eat=0.0,
        )

        agent.apply_action(action, dt=1/60, world=world)

        # Should have negative x velocity (reverse)
        assert agent.velocity.x < 0

    def test_torque_rotates_agent(self, agent, world):
        """Test that torque changes agent angle."""
        action = AgentAction(
            thrust=0.0,
            torque=1.0,  # Counter-clockwise
            vocalize=np.zeros(2),
            eat=0.0,
        )

        initial_angle = agent.angle
        agent.apply_action(action, dt=1/60, world=world)

        # Angle should increase (CCW rotation)
        assert agent.angle > initial_angle

    def test_velocity_limit(self, agent, world):
        """Test that velocity is clamped to max_speed."""
        # Give very high thrust for many frames
        action = AgentAction(
            thrust=1.0,
            torque=0.0,
            vocalize=np.zeros(2),
            eat=0.0,
        )

        for _ in range(1000):
            agent.apply_action(action, dt=1/60, world=world)

        speed = agent.velocity.magnitude()
        assert speed <= agent.genome.max_speed

    def test_angular_velocity_limit(self, agent, world):
        """Test that angular velocity is clamped."""
        action = AgentAction(
            thrust=0.0,
            torque=1.0,
            vocalize=np.zeros(2),
            eat=0.0,
        )

        for _ in range(1000):
            agent.apply_action(action, dt=1/60, world=world)

        assert abs(agent.angular_velocity) <= agent.genome.max_angular_speed

    def test_angle_wraps(self, agent, world):
        """Test that angle wraps around 2*pi."""
        action = AgentAction(
            thrust=0.0,
            torque=1.0,
            vocalize=np.zeros(2),
            eat=0.0,
        )

        # Rotate many times
        for _ in range(10000):
            agent.apply_action(action, dt=1/60, world=world)

        assert 0 <= agent.angle < 2 * math.pi


class TestAgentBodySurvival:
    """Tests for survival mechanics."""

    @pytest.fixture
    def agent(self):
        """Create agent with full energy and health."""
        return AgentBody(
            agent_id="test",
            initial_position=Vec2(250, 250),
        )

    @pytest.fixture
    def world(self):
        """Create empty world."""
        return World(width=500, height=500)

    def test_energy_depletes_over_time(self, agent, world):
        """Test that energy depletes even when idle."""
        initial_energy = agent.energy
        action = AgentAction.zero()

        agent.apply_action(action, dt=1.0, world=world)

        assert agent.energy < initial_energy

    def test_movement_costs_extra_energy(self, agent, world):
        """Test that movement costs more energy than idle."""
        # Clone agent for comparison
        idle_agent = AgentBody(agent_id="idle", initial_position=Vec2(100, 100))
        moving_agent = AgentBody(agent_id="moving", initial_position=Vec2(200, 200))

        idle_action = AgentAction.zero()
        moving_action = AgentAction(
            thrust=1.0,
            torque=0.0,
            vocalize=np.zeros(2),
            eat=0.0,
        )

        idle_agent.apply_action(idle_action, dt=1.0, world=world)
        moving_agent.apply_action(moving_action, dt=1.0, world=world)

        assert moving_agent.energy < idle_agent.energy

    def test_healing_when_energy_high(self, agent, world):
        """Test that agent heals when energy is above 50%."""
        agent.health = 50.0  # Half health
        agent.energy = 80.0  # High energy

        action = AgentAction.zero()
        agent.apply_action(action, dt=1.0, world=world)

        assert agent.health > 50.0

    def test_no_healing_when_energy_low(self, agent, world):
        """Test that agent doesn't heal when energy is low."""
        agent.health = 50.0
        agent.energy = 30.0  # Below 50%

        action = AgentAction.zero()
        agent.apply_action(action, dt=1.0, world=world)

        # Health shouldn't increase (might decrease from starvation)
        assert agent.health <= 50.0

    def test_starvation_damage(self, agent, world):
        """Test that zero energy causes health damage."""
        agent.energy = 0.0
        agent.health = 100.0

        action = AgentAction.zero()
        agent.apply_action(action, dt=1.0, world=world)

        assert agent.health < 100.0

    def test_death_from_starvation(self, agent, world):
        """Test that agent dies when health reaches zero."""
        agent.energy = 0.0
        agent.health = 1.0

        action = AgentAction.zero()

        # Run until dead
        for _ in range(100):
            if not agent.is_alive:
                break
            agent.apply_action(action, dt=0.1, world=world)

        assert not agent.is_alive
        assert agent.death_cause == "health_depleted"

    def test_take_damage(self, agent, world):
        """Test damage application."""
        initial_health = agent.health
        agent.take_damage(20.0)

        assert agent.health == initial_health - 20.0

    def test_damage_resistance(self, world):
        """Test that damage resistance reduces damage."""
        genome = AgentGenome(damage_resistance=2.0)  # Double resistance
        agent = AgentBody(agent_id="tough", genome=genome)

        initial_health = agent.health
        agent.take_damage(20.0)

        # Should only take 10 damage (20 / 2.0)
        assert agent.health == initial_health - 10.0

    def test_death_from_damage(self, agent, world):
        """Test death from taking too much damage."""
        agent.take_damage(150.0)

        assert not agent.is_alive
        assert agent.death_cause == "damage"


class TestAgentBodyEating:
    """Tests for eating mechanics."""

    @pytest.fixture
    def agent(self):
        """Create agent with depleted energy."""
        agent = AgentBody(
            agent_id="test",
            initial_position=Vec2(250, 250),
        )
        agent.energy = 50.0  # Half energy
        return agent

    @pytest.fixture
    def world_with_food(self):
        """Create world with food near agent."""
        world = World(width=500, height=500)
        food = Food(
            entity_id=0,
            position=Vec2(255, 250),  # Very close to agent
            energy_value=100.0,
        )
        world.add_entity(food)
        world._rebuild_spatial_grid()
        return world

    def test_eating_gains_energy(self, agent, world_with_food):
        """Test that eating food increases energy."""
        initial_energy = agent.energy
        action = AgentAction(
            thrust=0.0,
            torque=0.0,
            vocalize=np.zeros(2),
            eat=1.0,
        )

        # Need to be close enough to eat
        agent.position = Vec2(250, 250)
        agent.apply_action(action, dt=0.1, world=world_with_food)

        assert agent.energy > initial_energy
        assert agent.is_eating

    def test_eating_requires_proximity(self, agent):
        """Test that eating doesn't work when food is far."""
        world = World(width=500, height=500)
        food = Food(
            entity_id=0,
            position=Vec2(400, 250),  # Far from agent
            energy_value=100.0,
        )
        world.add_entity(food)
        world._rebuild_spatial_grid()

        initial_energy = agent.energy
        action = AgentAction(
            thrust=0.0,
            torque=0.0,
            vocalize=np.zeros(2),
            eat=1.0,
        )

        agent.apply_action(action, dt=0.1, world=world)

        assert agent.energy <= initial_energy  # Only metabolic loss
        assert not agent.is_eating


class TestAgentBodyObservations:
    """Tests for observation methods."""

    @pytest.fixture
    def agent(self):
        """Create agent."""
        return AgentBody(
            agent_id="test",
            initial_position=Vec2(250, 250),
        )

    @pytest.fixture
    def world(self):
        """Create world."""
        return World(width=500, height=500)

    def test_get_observations_returns_dict(self, agent, world):
        """Test that get_observations returns expected dictionary."""
        obs = agent.get_observations(world)

        assert isinstance(obs, dict)
        assert "vision" in obs
        assert "audio" in obs
        assert "proprioception" in obs
        assert "touch" in obs

    def test_observation_shapes(self, agent, world):
        """Test observation array shapes."""
        obs = agent.get_observations(world)

        assert obs["vision"].shape == (32, 4)
        assert obs["audio"].shape == (2,)
        assert obs["proprioception"].shape == (8,)
        assert obs["touch"].shape == (8,)

    def test_observation_tensor_shape(self, agent, world):
        """Test flattened observation tensor shape."""
        tensor = agent.get_observation_tensor(world)

        assert tensor.shape == (AgentBody.OBSERVATION_DIM,)
        assert tensor.dtype == torch.float32

    def test_observation_tensor_layout(self, agent, world):
        """Test observation tensor has correct layout."""
        tensor = agent.get_observation_tensor(world)

        # Vision: 0:128
        # Audio: 128:130
        # Proprioception: 130:138
        # Touch: 138:146
        assert len(tensor) == 146


class TestAgentBodySerialization:
    """Tests for save/load methods."""

    def test_save(self):
        """Test agent serialization."""
        agent = AgentBody(
            agent_id="test",
            initial_position=Vec2(100, 200),
            initial_angle=1.5,
        )
        agent.energy = 75.0
        agent.health = 80.0
        agent.age = 10.0

        data = agent.save()

        assert data["agent_id"] == "test"
        assert data["position"] == (100, 200)
        assert data["angle"] == 1.5
        assert data["energy"] == 75.0
        assert data["health"] == 80.0
        assert data["age"] == 10.0

    def test_load(self):
        """Test agent deserialization."""
        original = AgentBody(
            agent_id="test",
            initial_position=Vec2(100, 200),
            initial_angle=1.5,
        )
        original.energy = 75.0
        original.health = 80.0
        original.velocity = Vec2(10, 20)

        data = original.save()
        restored = AgentBody.load(data)

        assert restored.agent_id == original.agent_id
        assert restored.position.x == original.position.x
        assert restored.position.y == original.position.y
        assert restored.angle == original.angle
        assert restored.energy == original.energy
        assert restored.health == original.health
        assert restored.velocity.x == original.velocity.x
        assert restored.velocity.y == original.velocity.y

    def test_roundtrip(self):
        """Test save/load roundtrip preserves state."""
        genome = AgentGenome(max_speed=180.0, radius=12.0)
        original = AgentBody(
            agent_id="roundtrip_test",
            genome=genome,
            initial_position=Vec2(300, 400),
            initial_angle=2.0,
        )
        original.energy = 60.0
        original.health = 70.0
        original.age = 25.0
        original.velocity = Vec2(15, -10)
        original.angular_velocity = 0.5

        data = original.save()
        restored = AgentBody.load(data)

        assert restored.genome.max_speed == genome.max_speed
        assert restored.genome.radius == genome.radius
        assert restored.angular_velocity == original.angular_velocity

    def test_dead_agent_roundtrip(self):
        """Test that dead agent state is preserved."""
        agent = AgentBody(agent_id="dead_test")
        agent.die("testing")

        data = agent.save()
        restored = AgentBody.load(data)

        assert not restored.is_alive
        assert not restored.is_active
        assert restored.death_cause == "testing"
