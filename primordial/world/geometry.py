"""Geometry primitives for the world system.

Provides Vec2, Circle, and AABB classes for positions, collision shapes,
and spatial partitioning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Vec2:
    """2D vector for positions, velocities, etc."""

    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __iadd__(self, other: Vec2) -> Vec2:
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other: Vec2) -> Vec2:
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, scalar: float) -> Vec2:
        self.x *= scalar
        self.y *= scalar
        return self

    def magnitude(self) -> float:
        """Return the length of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y)

    def magnitude_squared(self) -> float:
        """Return the squared length (faster, avoids sqrt)."""
        return self.x * self.x + self.y * self.y

    def normalized(self) -> Vec2:
        """Return a unit vector in the same direction."""
        mag = self.magnitude()
        if mag == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def distance_to(self, other: Vec2) -> float:
        """Return distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_squared_to(self, other: Vec2) -> float:
        """Return squared distance (faster, avoids sqrt)."""
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def dot(self, other: Vec2) -> float:
        """Return dot product with another vector."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vec2) -> float:
        """Return 2D cross product (z-component of 3D cross)."""
        return self.x * other.y - self.y * other.x

    def perpendicular(self) -> Vec2:
        """Return perpendicular vector (rotated 90 degrees counter-clockwise)."""
        return Vec2(-self.y, self.x)

    def angle(self) -> float:
        """Return angle in radians from positive x-axis."""
        return math.atan2(self.y, self.x)

    def rotate(self, angle: float) -> Vec2:
        """Return vector rotated by angle (radians)."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y], dtype=np.float32)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> Vec2:
        """Create Vec2 from numpy array."""
        return Vec2(float(arr[0]), float(arr[1]))

    @staticmethod
    def from_angle(angle: float, magnitude: float = 1.0) -> Vec2:
        """Create Vec2 from angle (radians) and optional magnitude."""
        return Vec2(math.cos(angle) * magnitude, math.sin(angle) * magnitude)

    def copy(self) -> Vec2:
        """Return a copy of this vector."""
        return Vec2(self.x, self.y)


@dataclass
class Circle:
    """Circular collision shape."""

    center: Vec2
    radius: float

    def contains_point(self, point: Vec2) -> bool:
        """Check if point is inside circle."""
        return self.center.distance_squared_to(point) <= self.radius * self.radius

    def intersects(self, other: Circle) -> bool:
        """Check if this circle intersects another circle."""
        min_dist = self.radius + other.radius
        return self.center.distance_squared_to(other.center) <= min_dist * min_dist

    def distance_to(self, other: Circle) -> float:
        """Return distance between circle surfaces (negative if overlapping)."""
        center_dist = self.center.distance_to(other.center)
        return center_dist - self.radius - other.radius

    def overlap_depth(self, other: Circle) -> float:
        """Return overlap depth (positive if overlapping, else 0)."""
        distance = self.distance_to(other)
        return max(0.0, -distance)


@dataclass
class AABB:
    """Axis-aligned bounding box for spatial partitioning."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.max_y - self.min_y

    @property
    def center(self) -> Vec2:
        """Center point of the bounding box."""
        return Vec2(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
        )

    def contains_point(self, point: Vec2) -> bool:
        """Check if point is inside bounding box."""
        return (
            self.min_x <= point.x <= self.max_x
            and self.min_y <= point.y <= self.max_y
        )

    def intersects(self, other: AABB) -> bool:
        """Check if this AABB intersects another AABB."""
        return (
            self.min_x <= other.max_x
            and self.max_x >= other.min_x
            and self.min_y <= other.max_y
            and self.max_y >= other.min_y
        )

    def contains_circle(self, circle: Circle) -> bool:
        """Check if this AABB fully contains a circle."""
        return (
            self.min_x <= circle.center.x - circle.radius
            and self.max_x >= circle.center.x + circle.radius
            and self.min_y <= circle.center.y - circle.radius
            and self.max_y >= circle.center.y + circle.radius
        )

    def intersects_circle(self, circle: Circle) -> bool:
        """Check if this AABB intersects a circle."""
        # Find closest point on AABB to circle center
        closest_x = max(self.min_x, min(circle.center.x, self.max_x))
        closest_y = max(self.min_y, min(circle.center.y, self.max_y))
        closest = Vec2(closest_x, closest_y)

        # Check if closest point is within circle radius
        return circle.center.distance_squared_to(closest) <= circle.radius * circle.radius

    @staticmethod
    def from_circle(circle: Circle) -> AABB:
        """Create AABB that bounds a circle."""
        return AABB(
            circle.center.x - circle.radius,
            circle.center.y - circle.radius,
            circle.center.x + circle.radius,
            circle.center.y + circle.radius,
        )

    def expand(self, margin: float) -> AABB:
        """Return expanded AABB with margin on all sides."""
        return AABB(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin,
        )
