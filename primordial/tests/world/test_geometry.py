"""Tests for geometry primitives (Vec2, Circle, AABB)."""

import math

import numpy as np
import pytest

from primordial.world.geometry import AABB, Circle, Vec2


class TestVec2:
    """Tests for Vec2 class."""

    def test_creation(self):
        """Test Vec2 creation."""
        v = Vec2(3.0, 4.0)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_addition(self):
        """Test vector addition."""
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)
        result = v1 + v2
        assert result.x == 4.0
        assert result.y == 6.0

    def test_subtraction(self):
        """Test vector subtraction."""
        v1 = Vec2(5.0, 7.0)
        v2 = Vec2(2.0, 3.0)
        result = v1 - v2
        assert result.x == 3.0
        assert result.y == 4.0

    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vec2(2.0, 3.0)
        result = v * 2.0
        assert result.x == 4.0
        assert result.y == 6.0

    def test_scalar_rmul(self):
        """Test reverse scalar multiplication."""
        v = Vec2(2.0, 3.0)
        result = 2.0 * v
        assert result.x == 4.0
        assert result.y == 6.0

    def test_division(self):
        """Test scalar division."""
        v = Vec2(6.0, 8.0)
        result = v / 2.0
        assert result.x == 3.0
        assert result.y == 4.0

    def test_negation(self):
        """Test vector negation."""
        v = Vec2(3.0, -4.0)
        result = -v
        assert result.x == -3.0
        assert result.y == 4.0

    def test_equality(self):
        """Test vector equality."""
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(1.0, 2.0)
        v3 = Vec2(1.0, 3.0)
        assert v1 == v2
        assert v1 != v3

    def test_iadd(self):
        """Test in-place addition."""
        v = Vec2(1.0, 2.0)
        v += Vec2(3.0, 4.0)
        assert v.x == 4.0
        assert v.y == 6.0

    def test_isub(self):
        """Test in-place subtraction."""
        v = Vec2(5.0, 7.0)
        v -= Vec2(2.0, 3.0)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_imul(self):
        """Test in-place multiplication."""
        v = Vec2(2.0, 3.0)
        v *= 2.0
        assert v.x == 4.0
        assert v.y == 6.0

    def test_magnitude(self):
        """Test vector magnitude (3-4-5 triangle)."""
        v = Vec2(3.0, 4.0)
        assert v.magnitude() == 5.0

    def test_magnitude_squared(self):
        """Test squared magnitude."""
        v = Vec2(3.0, 4.0)
        assert v.magnitude_squared() == 25.0

    def test_normalized(self):
        """Test vector normalization."""
        v = Vec2(3.0, 4.0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 1e-6
        assert abs(n.x - 0.6) < 1e-6
        assert abs(n.y - 0.8) < 1e-6

    def test_normalized_zero_vector(self):
        """Test normalizing zero vector returns zero vector."""
        v = Vec2(0.0, 0.0)
        n = v.normalized()
        assert n.x == 0.0
        assert n.y == 0.0

    def test_distance_to(self):
        """Test distance between vectors."""
        v1 = Vec2(0.0, 0.0)
        v2 = Vec2(3.0, 4.0)
        assert v1.distance_to(v2) == 5.0

    def test_distance_squared_to(self):
        """Test squared distance."""
        v1 = Vec2(0.0, 0.0)
        v2 = Vec2(3.0, 4.0)
        assert v1.distance_squared_to(v2) == 25.0

    def test_dot_product(self):
        """Test dot product."""
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)
        assert v1.dot(v2) == 11.0  # 1*3 + 2*4

    def test_dot_perpendicular(self):
        """Test dot product of perpendicular vectors is zero."""
        v1 = Vec2(1.0, 0.0)
        v2 = Vec2(0.0, 1.0)
        assert v1.dot(v2) == 0.0

    def test_cross_product(self):
        """Test 2D cross product."""
        v1 = Vec2(1.0, 0.0)
        v2 = Vec2(0.0, 1.0)
        assert v1.cross(v2) == 1.0

    def test_perpendicular(self):
        """Test perpendicular vector."""
        v = Vec2(1.0, 0.0)
        p = v.perpendicular()
        assert p.x == 0.0
        assert p.y == 1.0
        assert v.dot(p) == 0.0  # Should be perpendicular

    def test_angle(self):
        """Test angle calculation."""
        v = Vec2(1.0, 0.0)
        assert abs(v.angle()) < 1e-6

        v = Vec2(0.0, 1.0)
        assert abs(v.angle() - math.pi / 2) < 1e-6

        v = Vec2(-1.0, 0.0)
        assert abs(v.angle() - math.pi) < 1e-6

    def test_rotate(self):
        """Test vector rotation."""
        v = Vec2(1.0, 0.0)
        rotated = v.rotate(math.pi / 2)  # 90 degrees
        assert abs(rotated.x) < 1e-6
        assert abs(rotated.y - 1.0) < 1e-6

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        v = Vec2(3.0, 4.0)
        arr = v.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr[0] == 3.0
        assert arr[1] == 4.0

    def test_from_numpy(self):
        """Test creation from numpy array."""
        arr = np.array([3.0, 4.0])
        v = Vec2.from_numpy(arr)
        assert v.x == 3.0
        assert v.y == 4.0

    def test_from_angle(self):
        """Test creation from angle."""
        v = Vec2.from_angle(0.0, 5.0)
        assert abs(v.x - 5.0) < 1e-6
        assert abs(v.y) < 1e-6

        v = Vec2.from_angle(math.pi / 2, 5.0)
        assert abs(v.x) < 1e-6
        assert abs(v.y - 5.0) < 1e-6

    def test_copy(self):
        """Test vector copy."""
        v1 = Vec2(3.0, 4.0)
        v2 = v1.copy()
        assert v1 == v2
        assert v1 is not v2


class TestCircle:
    """Tests for Circle class."""

    def test_creation(self):
        """Test Circle creation."""
        c = Circle(Vec2(5.0, 5.0), 10.0)
        assert c.center.x == 5.0
        assert c.center.y == 5.0
        assert c.radius == 10.0

    def test_contains_point_inside(self):
        """Test point inside circle."""
        c = Circle(Vec2(0.0, 0.0), 10.0)
        assert c.contains_point(Vec2(0.0, 0.0))
        assert c.contains_point(Vec2(5.0, 0.0))
        assert c.contains_point(Vec2(0.0, 5.0))

    def test_contains_point_outside(self):
        """Test point outside circle."""
        c = Circle(Vec2(0.0, 0.0), 10.0)
        assert not c.contains_point(Vec2(15.0, 0.0))
        assert not c.contains_point(Vec2(8.0, 8.0))  # ~11.3 distance

    def test_contains_point_on_boundary(self):
        """Test point on circle boundary."""
        c = Circle(Vec2(0.0, 0.0), 10.0)
        assert c.contains_point(Vec2(10.0, 0.0))
        assert c.contains_point(Vec2(0.0, 10.0))

    def test_intersects_overlapping(self):
        """Test intersecting circles."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(15.0, 0.0), 10.0)
        assert c1.intersects(c2)

    def test_intersects_non_overlapping(self):
        """Test non-intersecting circles."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(25.0, 0.0), 10.0)
        assert not c1.intersects(c2)

    def test_intersects_touching(self):
        """Test circles just touching."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(20.0, 0.0), 10.0)
        assert c1.intersects(c2)

    def test_intersects_contained(self):
        """Test one circle contained in another."""
        c1 = Circle(Vec2(0.0, 0.0), 20.0)
        c2 = Circle(Vec2(5.0, 0.0), 5.0)
        assert c1.intersects(c2)

    def test_distance_to_non_overlapping(self):
        """Test distance between non-overlapping circles."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(25.0, 0.0), 10.0)
        assert c1.distance_to(c2) == 5.0  # 25 - 10 - 10

    def test_distance_to_overlapping(self):
        """Test distance between overlapping circles (negative)."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(15.0, 0.0), 10.0)
        assert c1.distance_to(c2) == -5.0  # 15 - 10 - 10

    def test_overlap_depth_overlapping(self):
        """Test overlap depth for overlapping circles."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(15.0, 0.0), 10.0)
        assert c1.overlap_depth(c2) == 5.0

    def test_overlap_depth_non_overlapping(self):
        """Test overlap depth for non-overlapping circles."""
        c1 = Circle(Vec2(0.0, 0.0), 10.0)
        c2 = Circle(Vec2(25.0, 0.0), 10.0)
        assert c1.overlap_depth(c2) == 0.0


class TestAABB:
    """Tests for AABB class."""

    def test_creation(self):
        """Test AABB creation."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        assert box.min_x == 0.0
        assert box.min_y == 0.0
        assert box.max_x == 100.0
        assert box.max_y == 100.0

    def test_width_height(self):
        """Test width and height properties."""
        box = AABB(10.0, 20.0, 50.0, 80.0)
        assert box.width == 40.0
        assert box.height == 60.0

    def test_center(self):
        """Test center property."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        center = box.center
        assert center.x == 50.0
        assert center.y == 50.0

    def test_contains_point_inside(self):
        """Test point inside AABB."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        assert box.contains_point(Vec2(50.0, 50.0))
        assert box.contains_point(Vec2(1.0, 1.0))

    def test_contains_point_outside(self):
        """Test point outside AABB."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        assert not box.contains_point(Vec2(150.0, 50.0))
        assert not box.contains_point(Vec2(-10.0, 50.0))

    def test_contains_point_on_boundary(self):
        """Test point on AABB boundary."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        assert box.contains_point(Vec2(0.0, 50.0))
        assert box.contains_point(Vec2(100.0, 50.0))

    def test_intersects_overlapping(self):
        """Test intersecting AABBs."""
        box1 = AABB(0.0, 0.0, 100.0, 100.0)
        box2 = AABB(50.0, 50.0, 150.0, 150.0)
        assert box1.intersects(box2)
        assert box2.intersects(box1)

    def test_intersects_non_overlapping(self):
        """Test non-intersecting AABBs."""
        box1 = AABB(0.0, 0.0, 100.0, 100.0)
        box2 = AABB(150.0, 150.0, 200.0, 200.0)
        assert not box1.intersects(box2)

    def test_intersects_touching(self):
        """Test AABBs just touching."""
        box1 = AABB(0.0, 0.0, 100.0, 100.0)
        box2 = AABB(100.0, 0.0, 200.0, 100.0)
        assert box1.intersects(box2)

    def test_intersects_contained(self):
        """Test one AABB contained in another."""
        box1 = AABB(0.0, 0.0, 100.0, 100.0)
        box2 = AABB(25.0, 25.0, 75.0, 75.0)
        assert box1.intersects(box2)

    def test_contains_circle_inside(self):
        """Test circle fully inside AABB."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        circle = Circle(Vec2(50.0, 50.0), 10.0)
        assert box.contains_circle(circle)

    def test_contains_circle_partial(self):
        """Test circle partially outside AABB."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        circle = Circle(Vec2(95.0, 50.0), 10.0)
        assert not box.contains_circle(circle)

    def test_intersects_circle_inside(self):
        """Test circle inside AABB."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        circle = Circle(Vec2(50.0, 50.0), 10.0)
        assert box.intersects_circle(circle)

    def test_intersects_circle_overlapping(self):
        """Test circle overlapping AABB edge."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        circle = Circle(Vec2(105.0, 50.0), 10.0)
        assert box.intersects_circle(circle)

    def test_intersects_circle_outside(self):
        """Test circle outside AABB."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        circle = Circle(Vec2(150.0, 50.0), 10.0)
        assert not box.intersects_circle(circle)

    def test_intersects_circle_corner(self):
        """Test circle near AABB corner."""
        box = AABB(0.0, 0.0, 100.0, 100.0)
        # Circle at corner, just touching
        circle = Circle(Vec2(110.0, 110.0), 15.0)  # ~14.14 to corner
        assert box.intersects_circle(circle)

    def test_from_circle(self):
        """Test creating AABB from circle."""
        circle = Circle(Vec2(50.0, 50.0), 10.0)
        box = AABB.from_circle(circle)
        assert box.min_x == 40.0
        assert box.min_y == 40.0
        assert box.max_x == 60.0
        assert box.max_y == 60.0

    def test_expand(self):
        """Test AABB expansion."""
        box = AABB(10.0, 10.0, 20.0, 20.0)
        expanded = box.expand(5.0)
        assert expanded.min_x == 5.0
        assert expanded.min_y == 5.0
        assert expanded.max_x == 25.0
        assert expanded.max_y == 25.0
