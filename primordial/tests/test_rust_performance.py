"""Performance comparison tests for Rust vs Python."""
import time
import pytest
import numpy as np

from primordial.world import World
from primordial.world.geometry import Vec2
from primordial.world import helpers

try:
    from primordial.world.helpers_rust import get_vision_input_fast, RUST_AVAILABLE
except ImportError:
    RUST_AVAILABLE = False


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")
class TestRustPerformance:
    """Performance benchmarks comparing Rust vs Python."""

    @pytest.fixture
    def populated_world(self):
        """Create world with default entities for testing."""
        world = World(width=1000, height=1000)
        world.setup_default_world()
        return world

    def test_rust_faster_than_python(self, populated_world):
        """Rust should be at least 5x faster than Python."""
        world = populated_world
        position = Vec2(500, 500)
        facing = Vec2(1, 0)

        num_iterations = 50

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(num_iterations):
            helpers.get_vision_input(world, position, facing)
        python_time = (time.perf_counter() - start) / num_iterations

        # Benchmark Rust
        start = time.perf_counter()
        for _ in range(num_iterations):
            get_vision_input_fast(world, position, facing)
        rust_time = (time.perf_counter() - start) / num_iterations

        speedup = python_time / rust_time
        print(f"\nPython: {python_time*1000:.3f}ms, Rust: {rust_time*1000:.3f}ms")
        print(f"Speedup: {speedup:.1f}x")

        assert speedup > 5.0, f"Rust only {speedup:.1f}x faster, expected >5x"

    def test_rust_python_equivalence(self, populated_world):
        """Rust and Python produce equivalent results."""
        world = populated_world
        position = Vec2(500, 500)
        facing = Vec2(1, 0)

        python_result = helpers.get_vision_input(world, position, facing)
        python_result[:, 0] = 1.0 - python_result[:, 0]  # Match Rust normalization

        rust_result = get_vision_input_fast(world, position, facing)

        # Allow small floating point differences
        np.testing.assert_allclose(
            rust_result[:, :2],  # distance and type
            python_result[:, :2],
            rtol=0.01,
            atol=0.01
        )
