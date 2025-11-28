"""Tests for Rust raycast implementation."""
import pytest
import numpy as np
import math

# Skip all tests if Rust extension not available
try:
    from primordial._rust import raycast_vision
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not built")


class TestRaycastVision:
    """Tests for raycast_vision function."""

    def test_no_entities_returns_zeros(self):
        """Empty entity list returns max distance (0.0 normalized)."""
        origin = np.array([0.0, 0.0], dtype=np.float32)
        directions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        result = raycast_vision(
            origin, directions, 100.0,
            np.zeros((0, 2), dtype=np.float32),  # No entities
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.uint8),
            np.zeros(0, dtype=np.int32),
        )

        assert result.shape == (2, 4)
        assert np.allclose(result[:, 0], 0.0)  # Far = 0.0

    def test_hit_entity_in_front(self):
        """Ray hits entity directly ahead."""
        origin = np.array([0.0, 0.0], dtype=np.float32)
        directions = np.array([[1.0, 0.0]], dtype=np.float32)  # Facing right

        # Entity at (50, 0) with radius 10
        positions = np.array([[50.0, 0.0]], dtype=np.float32)
        radii = np.array([10.0], dtype=np.float32)
        types = np.array([1], dtype=np.uint8)  # Food
        ids = np.array([42], dtype=np.int32)

        result = raycast_vision(
            origin, directions, 100.0,
            positions, radii, types, ids
        )

        # Hit at distance 40 (50 - 10 radius), normalized: 1 - 40/100 = 0.6
        assert result[0, 0] == pytest.approx(0.6, rel=0.01)
        assert result[0, 1] == 1.0  # Food type

    def test_ignore_entity_id(self):
        """Ignored entity is not hit."""
        origin = np.array([0.0, 0.0], dtype=np.float32)
        directions = np.array([[1.0, 0.0]], dtype=np.float32)

        positions = np.array([[30.0, 0.0]], dtype=np.float32)
        radii = np.array([10.0], dtype=np.float32)
        types = np.array([1], dtype=np.uint8)
        ids = np.array([42], dtype=np.int32)

        # Without ignore - should hit
        result = raycast_vision(
            origin, directions, 100.0,
            positions, radii, types, ids
        )
        assert result[0, 0] > 0.5  # Close hit

        # With ignore - should miss
        result = raycast_vision(
            origin, directions, 100.0,
            positions, radii, types, ids,
            ignore_entity_id=42
        )
        assert result[0, 0] == pytest.approx(0.0)  # No hit

    def test_closest_entity_wins(self):
        """Nearest entity is reported when multiple in path."""
        origin = np.array([0.0, 0.0], dtype=np.float32)
        directions = np.array([[1.0, 0.0]], dtype=np.float32)

        # Two entities: food at 30, predator at 60
        positions = np.array([[30.0, 0.0], [60.0, 0.0]], dtype=np.float32)
        radii = np.array([5.0, 5.0], dtype=np.float32)
        types = np.array([1, 2], dtype=np.uint8)  # Food=1, Predator=2
        ids = np.array([1, 2], dtype=np.int32)

        result = raycast_vision(
            origin, directions, 100.0,
            positions, radii, types, ids
        )

        # Should hit food (closer), not predator
        assert result[0, 1] == 1.0  # Food type
