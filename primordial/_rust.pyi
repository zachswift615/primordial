"""Type stubs for Rust extension module."""
import numpy as np
from numpy.typing import NDArray

__version__: str

class Vec2:
    x: float
    y: float
    def __init__(self, x: float, y: float) -> None: ...
    def magnitude(self) -> float: ...
    def normalized(self) -> Vec2: ...
    def dot(self, other: Vec2) -> float: ...

def raycast_vision(
    origin: NDArray[np.float32],
    ray_directions: NDArray[np.float32],
    max_distance: float,
    entity_positions: NDArray[np.float32],
    entity_radii: NDArray[np.float32],
    entity_types: NDArray[np.uint8],
    entity_ids: NDArray[np.int32],
    ignore_entity_id: int | None = None,
) -> NDArray[np.float32]: ...

def batch_raycast_vision(
    agent_positions: NDArray[np.float32],
    agent_angles: NDArray[np.float32],
    agent_ids: NDArray[np.int32],
    num_rays: int,
    fov: float,
    max_distance: float,
    entity_positions: NDArray[np.float32],
    entity_radii: NDArray[np.float32],
    entity_types: NDArray[np.uint8],
    entity_ids: NDArray[np.int32],
) -> NDArray[np.float32]: ...
