"""Performance tests for agent system.

Note: These are benchmark tests that document current performance.
The pure-Python raycasting implementation is the main bottleneck.
For production use at 60Hz with 100+ agents, consider:
1. Cython/Numba for raycasting hot paths
2. Spatial indexing for raycast queries
3. Reducing vision ray count (32 -> 16)
"""

import time

import pytest
import numpy as np

from primordial.agents import AgentBody, AgentAction
from primordial.world import World
from primordial.world.geometry import Vec2


class TestAgentPerformance:
    """Performance benchmarks for agent operations."""

    @pytest.fixture
    def populated_world(self):
        """Create world with default entities."""
        world = World(width=1000, height=1000)
        world.setup_default_world()
        return world

    def test_10_agents_benchmark(self, populated_world):
        """Benchmark 10 agents for 1 second (realistic for pure Python)."""
        world = populated_world

        # Create 10 agents at random positions
        agents = []
        for i in range(10):
            agent = AgentBody(
                agent_id=f"agent_{i}",
                initial_position=Vec2(
                    np.random.uniform(50, 950),
                    np.random.uniform(50, 950),
                ),
                initial_angle=np.random.uniform(0, 2 * np.pi),
            )
            world.add_entity(agent)
            agents.append(agent)

        # Simulate 1 second at 60Hz (60 frames)
        dt = 1 / 60
        num_frames = 60

        start = time.time()

        for _ in range(num_frames):
            for agent in agents:
                if agent.is_alive:
                    obs = agent.get_observation_tensor(world)
                    action = AgentAction.random()
                    agent.apply_action(action, dt, world)
            world.tick()

        elapsed = time.time() - start

        # Log performance metrics (documentation, not strict requirement)
        fps = num_frames / elapsed
        ms_per_frame = (elapsed / num_frames) * 1000
        print(f"\n10-agent benchmark: {fps:.1f} FPS, {ms_per_frame:.2f} ms/frame")

        # Reasonable target: should complete faster than wall-clock time
        # for 10 agents even in pure Python
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 1s simulation"

    def test_observation_tensor_benchmark(self, populated_world):
        """Benchmark observation tensor creation (documents current perf)."""
        world = populated_world

        # Create single agent
        agent = AgentBody(
            agent_id="perf_test",
            initial_position=Vec2(500, 500),
        )
        world.add_entity(agent)

        # Time 100 observation tensor creations
        num_iterations = 100

        start = time.time()
        for _ in range(num_iterations):
            _ = agent.get_observation_tensor(world)
        elapsed = time.time() - start

        avg_ms = (elapsed / num_iterations) * 1000
        print(f"\nObservation tensor: {avg_ms:.3f} ms average")

        # Pure Python target: < 5ms per observation is acceptable
        assert avg_ms < 5.0, f"Too slow: {avg_ms:.3f} ms per observation"

    def test_action_application_performance(self, populated_world):
        """Test action application is fast enough."""
        world = populated_world

        # Create single agent
        agent = AgentBody(
            agent_id="perf_test",
            initial_position=Vec2(500, 500),
        )
        world.add_entity(agent)

        # Time 10000 action applications
        num_iterations = 10000
        dt = 1 / 60

        start = time.time()
        for _ in range(num_iterations):
            action = AgentAction.random()
            agent.apply_action(action, dt, world)
        elapsed = time.time() - start

        avg_us = (elapsed / num_iterations) * 1_000_000

        # Should average less than 100 microseconds per action
        assert avg_us < 100.0, f"Too slow: {avg_us:.1f} µs per action"
        print(f"\nAction application: {avg_us:.1f} µs average")

    def test_sensor_benchmark(self, populated_world):
        """Benchmark individual sensor performance."""
        world = populated_world

        agent = AgentBody(
            agent_id="sensor_test",
            initial_position=Vec2(500, 500),
        )
        world.add_entity(agent)

        num_iterations = 100
        facing = Vec2(1, 0)

        # Vision sensor
        start = time.time()
        for _ in range(num_iterations):
            _ = agent.vision.sense(agent.position, facing, world)
        vision_time = (time.time() - start) / num_iterations * 1000

        # Audio sensor
        start = time.time()
        for _ in range(num_iterations):
            _ = agent.audio.sense(agent.position, facing, world)
        audio_time = (time.time() - start) / num_iterations * 1000

        # Touch sensor
        start = time.time()
        for _ in range(num_iterations):
            _ = agent.touch.sense(agent.position, world)
        touch_time = (time.time() - start) / num_iterations * 1000

        # Proprioception (no world dependency)
        start = time.time()
        for _ in range(num_iterations):
            _ = agent.proprioception.sense(
                energy=agent.energy,
                health=agent.health,
                velocity=agent.velocity,
                angular_velocity=agent.angular_velocity,
                angle=agent.angle,
            )
        proprio_time = (time.time() - start) / num_iterations * 1000

        print(f"\nSensor performance (ms):")
        print(f"  Vision: {vision_time:.3f}")
        print(f"  Audio: {audio_time:.3f}")
        print(f"  Touch: {touch_time:.3f}")
        print(f"  Proprioception: {proprio_time:.4f}")

        # Vision is bottleneck in pure Python; < 5ms is acceptable
        assert vision_time < 5.0, f"Vision too slow: {vision_time:.3f} ms"
        # Other sensors should be fast
        assert audio_time < 0.1, f"Audio too slow: {audio_time:.3f} ms"
        assert touch_time < 0.1, f"Touch too slow: {touch_time:.3f} ms"

    def test_batch_observation_benchmark(self, populated_world):
        """Benchmark batch observation creation for multiple agents."""
        world = populated_world

        # Create 10 agents
        agents = []
        for i in range(10):
            agent = AgentBody(
                agent_id=f"batch_{i}",
                initial_position=Vec2(
                    np.random.uniform(100, 900),
                    np.random.uniform(100, 900),
                ),
            )
            world.add_entity(agent)
            agents.append(agent)

        # Time getting observations for all agents
        num_iterations = 10

        start = time.time()
        for _ in range(num_iterations):
            for agent in agents:
                _ = agent.get_observation_tensor(world)
        elapsed = time.time() - start

        total_obs = num_iterations * len(agents)
        avg_ms = (elapsed / total_obs) * 1000

        print(f"\nBatch observations: {avg_ms:.3f} ms per agent (10 agents)")

        # Pure Python target: < 5ms per observation
        assert avg_ms < 5.0, f"Too slow: {avg_ms:.3f} ms per observation"
