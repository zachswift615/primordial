# Agent Body & Sensory System Implementation Plan

## Overview

This document specifies the implementation of the agent's physical embodiment and sensory systems for the Primordial simulation. Each agent is a physical entity with continuous sensory streams, internal state, and survival needs. The design prioritizes:

- **Continuous sensory data**: All sensors output float arrays, not discrete tokens
- **Performance**: Vision raycasting and sensor updates must run at 60Hz for all agents
- **Evolvability**: Genome system allows heritable hyperparameters for future evolution
- **Neural network integration**: All sensory outputs are shaped for direct PyTorch consumption

The agent body acts as the interface between the physical simulation world and the neural network-based Living Resonance Network that controls it.

## File Structure

```
primordial/
├── agents/
│   ├── __init__.py
│   ├── body.py              # AgentBody class - physical state and physics
│   ├── sensors.py           # All sensory system implementations
│   ├── genome.py            # Genome definition and mutation
│   └── actions.py           # Action space and action application
├── physics/
│   ├── __init__.py
│   ├── raycasting.py        # Efficient raycasting for vision and touch
│   └── collision.py         # Collision detection utilities
└── tests/
    ├── test_body.py
    ├── test_sensors.py
    ├── test_genome.py
    └── test_actions.py
```

## Data Structures

### AgentState

```python
@dataclass
class AgentState:
    """Complete physical and internal state of an agent."""

    # Identity
    agent_id: str
    genome: AgentGenome

    # Physics (world coordinates)
    position: np.ndarray  # (x, y) float32
    velocity: np.ndarray  # (vx, vy) float32
    angle: float          # radians, 0 = right, increases CCW
    angular_velocity: float  # radians/sec

    # Survival metrics
    energy: float         # 0.0 to 100.0, depletes over time
    health: float         # 0.0 to 100.0, reduced by damage
    age: float           # seconds since birth

    # Physical properties (from genome)
    radius: float        # collision radius
    mass: float          # affects acceleration

    # Status
    is_alive: bool
    is_eating: bool      # currently consuming food
    last_damage_time: float  # for damage cooldown/invincibility
```

### AgentAction

```python
@dataclass
class AgentAction:
    """Continuous action outputs from neural network."""

    thrust: float        # -1.0 to 1.0, forward/backward force
    torque: float        # -1.0 to 1.0, rotation force
    vocalize: np.ndarray # (2,) float32, [frequency, amplitude] both 0-1
    eat: float           # 0.0 to 1.0, eating effort (only works when touching food)

    @classmethod
    def from_tensor(cls, action_tensor: torch.Tensor) -> 'AgentAction':
        """Create action from neural network output tensor (5,)."""
        assert action_tensor.shape == (5,)
        arr = action_tensor.cpu().numpy()
        return cls(
            thrust=np.clip(arr[0], -1.0, 1.0),
            torque=np.clip(arr[1], -1.0, 1.0),
            vocalize=np.clip(arr[2:4], 0.0, 1.0),
            eat=np.clip(arr[4], 0.0, 1.0)
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for logging/analysis."""
        return torch.tensor([
            self.thrust,
            self.torque,
            self.vocalize[0],
            self.vocalize[1],
            self.eat
        ], dtype=torch.float32)
```

### AgentGenome

```python
@dataclass
class AgentGenome:
    """Heritable hyperparameters that define agent capabilities."""

    # Physical traits
    max_speed: float = 150.0        # units/sec
    max_angular_speed: float = 3.0  # radians/sec
    thrust_force: float = 500.0     # force units
    torque_force: float = 1000.0    # torque units
    radius: float = 8.0             # collision radius
    mass: float = 1.0               # physics mass

    # Sensory capabilities
    vision_range: float = 200.0     # max ray distance
    vision_fov: float = 120.0       # degrees
    vision_rays: int = 32           # number of vision rays
    audio_range: float = 300.0      # max hearing distance
    touch_range: float = 15.0       # touch sensor reach beyond radius

    # Metabolic parameters
    base_energy_cost: float = 0.5   # energy/sec while idle
    movement_energy_mult: float = 2.0   # multiplier for thrust/torque
    vocalize_energy_mult: float = 1.5   # multiplier for sound production
    eating_efficiency: float = 0.8  # energy gained per food unit

    # Health parameters
    max_health: float = 100.0
    max_energy: float = 100.0
    damage_resistance: float = 1.0  # multiplier on incoming damage
    healing_rate: float = 0.1       # health/sec when energy > 50

    # Neural architecture hints (for future LRN construction)
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 0.001

    # Mutation parameters (for breeding system)
    mutation_rate: float = 0.1      # probability per gene
    mutation_scale: float = 0.1     # std dev of gaussian mutation

    def mutate(self) -> 'AgentGenome':
        """Create mutated copy for offspring."""
        import copy
        child = copy.deepcopy(self)

        # Mutate physical traits
        for field in ['max_speed', 'max_angular_speed', 'thrust_force',
                      'torque_force', 'radius', 'mass']:
            if random.random() < self.mutation_rate:
                current = getattr(child, field)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, field, max(0.1, current + delta))

        # Mutate sensory traits
        for field in ['vision_range', 'vision_fov', 'audio_range', 'touch_range']:
            if random.random() < self.mutation_rate:
                current = getattr(child, field)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, field, max(10.0, current + delta))

        # Mutate metabolic traits
        for field in ['base_energy_cost', 'movement_energy_mult',
                      'eating_efficiency', 'healing_rate']:
            if random.random() < self.mutation_rate:
                current = getattr(child, field)
                delta = random.gauss(0, current * self.mutation_scale)
                setattr(child, field, max(0.01, current + delta))

        return child

    def to_dict(self) -> dict:
        """Serialize for saving/loading."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentGenome':
        """Deserialize from saved data."""
        return cls(**data)
```

## Sensory Systems

### Vision System

Vision uses raycasting with 32 rays spread across a 120-degree FOV. Each ray returns distance and color information.

**Implementation (sensors.py):**

```python
class VisionSensor:
    """Ray-based vision system."""

    def __init__(self, genome: AgentGenome):
        self.num_rays = genome.vision_rays
        self.max_range = genome.vision_range
        self.fov = np.radians(genome.vision_fov)

        # Pre-compute ray angles relative to agent heading
        self.ray_angles = np.linspace(
            -self.fov / 2,
            self.fov / 2,
            self.num_rays
        )

    def sense(self, agent_state: AgentState, world: 'World') -> np.ndarray:
        """
        Cast vision rays and return sensory data.

        Returns:
            np.ndarray: shape (num_rays, 4) float32
                        Each row: [distance, r, g, b]
                        distance: 0.0 (max_range) to 1.0 (touching)
                        rgb: 0.0 to 1.0
        """
        output = np.zeros((self.num_rays, 4), dtype=np.float32)

        for i, rel_angle in enumerate(self.ray_angles):
            # Calculate absolute ray angle
            ray_angle = agent_state.angle + rel_angle

            # Ray direction vector
            direction = np.array([
                np.cos(ray_angle),
                np.sin(ray_angle)
            ])

            # Cast ray and get hit info
            hit_distance, hit_color = world.raycast(
                origin=agent_state.position,
                direction=direction,
                max_distance=self.max_range
            )

            # Normalize distance (far=0, near=1)
            normalized_dist = 1.0 - (hit_distance / self.max_range)

            output[i] = [normalized_dist, hit_color[0], hit_color[1], hit_color[2]]

        return output
```

**Raycasting Implementation (physics/raycasting.py):**

```python
def raycast(world: 'World', origin: np.ndarray, direction: np.ndarray,
            max_distance: float) -> Tuple[float, np.ndarray]:
    """
    Cast a ray and return distance to first hit and color.

    Checks:
    - World boundaries
    - Food items
    - Other agents
    - Obstacles (if present)

    Returns:
        distance: float, actual distance to hit
        color: np.ndarray (3,) RGB values 0-1
    """
    closest_distance = max_distance
    closest_color = np.array([0.0, 0.0, 0.0])  # Black = nothing

    # Check world boundaries
    boundary_dist = _raycast_boundary(world.bounds, origin, direction)
    if boundary_dist < closest_distance:
        closest_distance = boundary_dist
        closest_color = np.array([0.3, 0.3, 0.3])  # Gray walls

    # Check food items
    for food in world.food_items:
        dist = _raycast_circle(origin, direction, food.position, food.radius)
        if dist < closest_distance:
            closest_distance = dist
            closest_color = np.array([0.0, 1.0, 0.0])  # Green food

    # Check other agents
    for agent in world.agents:
        if np.array_equal(agent.state.position, origin):
            continue  # Skip self
        dist = _raycast_circle(origin, direction, agent.state.position,
                              agent.state.radius)
        if dist < closest_distance:
            closest_distance = dist
            closest_color = np.array([1.0, 0.0, 0.0])  # Red agents

    return closest_distance, closest_color


def _raycast_circle(origin: np.ndarray, direction: np.ndarray,
                    center: np.ndarray, radius: float) -> float:
    """Ray-circle intersection, returns distance or inf."""
    # Vector from origin to circle center
    oc = origin - center

    # Quadratic formula coefficients
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return float('inf')  # No intersection

    # Return nearest intersection
    t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    return t if t > 0 else float('inf')
```

### Audio System

Agents hear sounds from vocalizations of nearby agents. Stereo hearing provides directional information.

```python
class AudioSensor:
    """Stereo hearing system."""

    def __init__(self, genome: AgentGenome):
        self.max_range = genome.audio_range
        self.ear_separation = genome.radius  # Distance between "ears"

    def sense(self, agent_state: AgentState, world: 'World') -> np.ndarray:
        """
        Mix sounds from all vocalizing agents.

        Returns:
            np.ndarray: shape (2,) float32 [left_ear, right_ear]
                        Values 0.0 to 1.0 (clamped sum of all sounds)
        """
        left_ear = 0.0
        right_ear = 0.0

        # Agent's perpendicular vector (left is +90 degrees from heading)
        left_vec = np.array([
            -np.sin(agent_state.angle),
            np.cos(agent_state.angle)
        ])

        for other_agent in world.agents:
            if other_agent.state.agent_id == agent_state.agent_id:
                continue

            # Get vocalization from other agent's last action
            if not hasattr(other_agent, 'last_action'):
                continue

            freq, amp = other_agent.last_action.vocalize
            if amp < 0.01:  # Not vocalizing
                continue

            # Distance attenuation
            delta = other_agent.state.position - agent_state.position
            distance = np.linalg.norm(delta)

            if distance > self.max_range:
                continue

            # Amplitude falls off with distance
            attenuation = 1.0 - (distance / self.max_range)
            effective_amp = amp * attenuation

            # Stereo positioning based on direction
            direction = delta / (distance + 1e-6)
            left_amount = np.dot(direction, left_vec)  # -1 to 1

            # Split amplitude between ears
            left_contribution = effective_amp * (0.5 + 0.5 * left_amount)
            right_contribution = effective_amp * (0.5 - 0.5 * left_amount)

            left_ear += left_contribution
            right_ear += right_contribution

        # Clamp to valid range
        return np.clip([left_ear, right_ear], 0.0, 1.0).astype(np.float32)
```

### Proprioception System

Internal state awareness - the agent's "sense" of its own body state.

```python
class ProprioceptionSensor:
    """Internal state awareness."""

    def sense(self, agent_state: AgentState) -> np.ndarray:
        """
        Return normalized internal state information.

        Returns:
            np.ndarray: shape (8,) float32
                [energy, health, speed, angular_velocity,
                 velocity_x, velocity_y, angle_sin, angle_cos]
        """
        # Normalize energy and health to 0-1
        energy_norm = agent_state.energy / agent_state.genome.max_energy
        health_norm = agent_state.health / agent_state.genome.max_health

        # Normalize speed to 0-1 based on genome max
        speed = np.linalg.norm(agent_state.velocity)
        speed_norm = speed / agent_state.genome.max_speed

        # Normalize angular velocity
        ang_vel_norm = agent_state.angular_velocity / agent_state.genome.max_angular_speed

        # Velocity components (normalized by max_speed)
        vel_x = agent_state.velocity[0] / agent_state.genome.max_speed
        vel_y = agent_state.velocity[1] / agent_state.genome.max_speed

        # Angle as sin/cos (handles wraparound naturally)
        angle_sin = np.sin(agent_state.angle)
        angle_cos = np.cos(agent_state.angle)

        return np.array([
            energy_norm,
            health_norm,
            speed_norm,
            ang_vel_norm,
            vel_x,
            vel_y,
            angle_sin,
            angle_cos
        ], dtype=np.float32)
```

### Touch System

8 directional contact sensors detect nearby objects.

```python
class TouchSensor:
    """8-directional contact sensors."""

    def __init__(self, genome: AgentGenome):
        self.num_sensors = 8
        self.touch_range = genome.touch_range + genome.radius

        # Sensor angles (relative to agent heading)
        self.sensor_angles = np.linspace(0, 2 * np.pi, self.num_sensors, endpoint=False)

    def sense(self, agent_state: AgentState, world: 'World') -> np.ndarray:
        """
        Detect contact in 8 directions.

        Returns:
            np.ndarray: shape (8, 2) float32
                Each row: [distance, object_type]
                distance: 0.0 (far) to 1.0 (touching)
                object_type: 0.0=nothing, 0.33=wall, 0.66=food, 1.0=agent
        """
        output = np.zeros((self.num_sensors, 2), dtype=np.float32)

        for i, rel_angle in enumerate(self.sensor_angles):
            # Absolute sensor angle
            sensor_angle = agent_state.angle + rel_angle

            # Sensor position (just beyond agent radius)
            sensor_dir = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
            sensor_pos = agent_state.position + sensor_dir * agent_state.radius

            # Check for nearest object
            closest_dist = self.touch_range
            object_type = 0.0

            # Check walls
            wall_dist = world.distance_to_boundary(sensor_pos)
            if wall_dist < closest_dist:
                closest_dist = wall_dist
                object_type = 0.33

            # Check food
            for food in world.food_items:
                dist = np.linalg.norm(food.position - sensor_pos) - food.radius
                if dist < closest_dist:
                    closest_dist = dist
                    object_type = 0.66

            # Check other agents
            for other in world.agents:
                if other.state.agent_id == agent_state.agent_id:
                    continue
                dist = np.linalg.norm(other.state.position - sensor_pos) - other.state.radius
                if dist < closest_dist:
                    closest_dist = dist
                    object_type = 1.0

            # Normalize distance (far=0, near=1)
            distance_norm = 1.0 - (closest_dist / self.touch_range)
            distance_norm = np.clip(distance_norm, 0.0, 1.0)

            output[i] = [distance_norm, object_type]

        return output
```

## Action Application

Actions are applied during the physics update step to modify agent state.

```python
class AgentBody:
    """Physical body of an agent."""

    def apply_action(self, action: AgentAction, dt: float, world: 'World'):
        """Apply actions and update physics."""

        # 1. Apply thrust (forward/backward movement)
        if abs(action.thrust) > 0.001:
            thrust_force = action.thrust * self.state.genome.thrust_force
            direction = np.array([
                np.cos(self.state.angle),
                np.sin(self.state.angle)
            ])
            acceleration = (direction * thrust_force) / self.state.mass
            self.state.velocity += acceleration * dt

        # 2. Apply torque (rotation)
        if abs(action.torque) > 0.001:
            torque = action.torque * self.state.genome.torque_force
            angular_accel = torque / (self.state.mass * self.state.radius ** 2)
            self.state.angular_velocity += angular_accel * dt

        # 3. Limit velocities to genome maximums
        speed = np.linalg.norm(self.state.velocity)
        if speed > self.state.genome.max_speed:
            self.state.velocity *= self.state.genome.max_speed / speed

        self.state.angular_velocity = np.clip(
            self.state.angular_velocity,
            -self.state.genome.max_angular_speed,
            self.state.genome.max_angular_speed
        )

        # 4. Update position and angle
        self.state.position += self.state.velocity * dt
        self.state.angle += self.state.angular_velocity * dt
        self.state.angle = self.state.angle % (2 * np.pi)  # Wrap angle

        # 5. Apply drag
        drag_coefficient = 0.95  # Per-frame drag multiplier
        self.state.velocity *= drag_coefficient ** dt
        self.state.angular_velocity *= drag_coefficient ** dt

        # 6. Handle eating
        if action.eat > 0.1:
            self._try_eat(action.eat, world)

        # 7. Store action for vocalization (used by other agents' audio sensors)
        self.last_action = action

    def _try_eat(self, effort: float, world: 'World'):
        """Attempt to consume nearby food."""
        eating_radius = self.state.radius + 5.0  # Must be very close

        for food in world.food_items:
            distance = np.linalg.norm(food.position - self.state.position)

            if distance < eating_radius:
                # Consume food proportional to effort
                consumed = food.consume(effort * 10.0)  # Max 10 units/sec

                # Gain energy based on efficiency
                energy_gain = consumed * self.state.genome.eating_efficiency
                self.state.energy = min(
                    self.state.genome.max_energy,
                    self.state.energy + energy_gain
                )

                self.state.is_eating = True
                return

        self.state.is_eating = False
```

## Energy & Health Mechanics

### Energy Depletion

Energy depletes continuously based on activity level.

```python
def update_energy(self, action: AgentAction, dt: float):
    """Update energy based on activity."""
    genome = self.state.genome

    # Base metabolic cost
    energy_cost = genome.base_energy_cost * dt

    # Movement costs
    thrust_cost = abs(action.thrust) * genome.movement_energy_mult * dt
    torque_cost = abs(action.torque) * genome.movement_energy_mult * dt

    # Vocalization cost
    vocalize_cost = action.vocalize[1] * genome.vocalize_energy_mult * dt

    # Total depletion
    total_cost = energy_cost + thrust_cost + torque_cost + vocalize_cost
    self.state.energy = max(0.0, self.state.energy - total_cost)

    # Death from starvation
    if self.state.energy <= 0:
        self.die(cause="starvation")
```

### Health Mechanics

```python
def update_health(self, dt: float):
    """Update health (healing or damage from starvation)."""
    genome = self.state.genome

    # Healing when energy is high
    if self.state.energy > genome.max_energy * 0.5:
        if self.state.health < genome.max_health:
            self.state.health = min(
                genome.max_health,
                self.state.health + genome.healing_rate * dt
            )

    # Damage from starvation
    elif self.state.energy <= 0:
        starvation_damage = 5.0 * dt  # 5 health/sec when starving
        self.take_damage(starvation_damage)

def take_damage(self, amount: float):
    """Apply damage with resistance."""
    actual_damage = amount / self.state.genome.damage_resistance
    self.state.health = max(0.0, self.state.health - actual_damage)
    self.state.last_damage_time = self.state.age

    if self.state.health <= 0:
        self.die(cause="damage")

def die(self, cause: str):
    """Mark agent as dead."""
    self.state.is_alive = False
    print(f"Agent {self.state.agent_id} died from {cause} at age {self.state.age:.1f}s")
```

## Genome System

The genome system is designed for future breeding and evolution.

### Inheritance

```python
def breed(parent1: AgentGenome, parent2: AgentGenome) -> AgentGenome:
    """Create offspring genome from two parents."""
    import copy

    # Start with copy of parent1
    child = copy.deepcopy(parent1)

    # Randomly inherit each trait from either parent
    for field in child.__dataclass_fields__:
        if random.random() < 0.5:
            setattr(child, field, getattr(parent2, field))

    # Apply mutation
    child = child.mutate()

    return child
```

### Default Genome Factory

```python
def create_default_genome() -> AgentGenome:
    """Create a baseline genome for initial population."""
    return AgentGenome(
        max_speed=150.0,
        max_angular_speed=3.0,
        thrust_force=500.0,
        torque_force=1000.0,
        radius=8.0,
        mass=1.0,
        vision_range=200.0,
        vision_fov=120.0,
        vision_rays=32,
        audio_range=300.0,
        touch_range=15.0,
        base_energy_cost=0.5,
        movement_energy_mult=2.0,
        vocalize_energy_mult=1.5,
        eating_efficiency=0.8,
        max_health=100.0,
        max_energy=100.0,
        damage_resistance=1.0,
        healing_rate=0.1,
        hidden_dim=128,
        num_layers=3,
        learning_rate=0.001,
        mutation_rate=0.1,
        mutation_scale=0.1
    )
```

## API Specification

### AgentBody Public Interface

```python
class AgentBody:
    """Main agent body class."""

    def __init__(self, agent_id: str, genome: AgentGenome,
                 initial_position: np.ndarray, initial_angle: float):
        """Initialize agent body with genome and starting state."""
        pass

    def update(self, action: AgentAction, dt: float, world: 'World'):
        """
        Complete update cycle: apply action, update physics, update health/energy.

        Args:
            action: Action outputs from neural network
            dt: Time step in seconds (1/60 for 60Hz)
            world: World reference for physics queries
        """
        pass

    def get_observations(self, world: 'World') -> dict:
        """
        Collect all sensory observations.

        Returns:
            dict with keys: 'vision', 'audio', 'proprioception', 'touch'
            Each value is a numpy array ready for neural network input
        """
        pass

    def get_observation_tensor(self, world: 'World') -> torch.Tensor:
        """
        Get all observations as a single flattened tensor.

        Returns:
            torch.Tensor: shape (observation_dim,) float32
        """
        pass

    def take_damage(self, amount: float):
        """Apply damage to agent."""
        pass

    def die(self, cause: str):
        """Mark agent as dead."""
        pass

    @property
    def is_alive(self) -> bool:
        """Check if agent is alive."""
        return self.state.is_alive

    def save(self) -> dict:
        """Serialize agent state for checkpointing."""
        pass

    @classmethod
    def load(cls, data: dict) -> 'AgentBody':
        """Deserialize agent from saved state."""
        pass
```

### World Interface Requirements

The agent body requires the following methods from the World class:

```python
class World:
    """World interface required by agent body."""

    def raycast(self, origin: np.ndarray, direction: np.ndarray,
                max_distance: float) -> Tuple[float, np.ndarray]:
        """Cast ray, return (distance, color)."""
        pass

    def distance_to_boundary(self, position: np.ndarray) -> float:
        """Get distance from position to nearest boundary."""
        pass

    @property
    def agents(self) -> List['AgentBody']:
        """All agents in the world."""
        pass

    @property
    def food_items(self) -> List['Food']:
        """All food items in the world."""
        pass

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """World boundaries (min_x, min_y, max_x, max_y)."""
        pass
```

## Tensor Output Format

All sensory data is structured for direct consumption by PyTorch neural networks.

### Individual Sensor Shapes

```python
# Vision: 32 rays × 4 values (distance, r, g, b)
vision_shape = (32, 4)  # 128 values

# Audio: stereo (left, right)
audio_shape = (2,)  # 2 values

# Proprioception: 8 internal state values
proprioception_shape = (8,)  # 8 values

# Touch: 8 sensors × 2 values (distance, type)
touch_shape = (8, 2)  # 16 values

# Total observation dimension: 128 + 2 + 8 + 16 = 154
```

### Flattened Observation Tensor

```python
def get_observation_tensor(self, world: 'World') -> torch.Tensor:
    """
    Returns flattened observation vector for neural network.

    Shape: (154,) float32

    Layout:
        [0:128]   - Vision (32 rays × 4)
        [128:130] - Audio (2)
        [130:138] - Proprioception (8)
        [138:154] - Touch (8 × 2)
    """
    obs = self.get_observations(world)

    vision_flat = obs['vision'].flatten()        # (128,)
    audio_flat = obs['audio']                    # (2,)
    proprio_flat = obs['proprioception']         # (8,)
    touch_flat = obs['touch'].flatten()          # (16,)

    combined = np.concatenate([
        vision_flat,
        audio_flat,
        proprio_flat,
        touch_flat
    ])

    return torch.from_numpy(combined).float()
```

### Action Tensor Format

```python
# Action output from neural network
action_shape = (5,)  # [thrust, torque, vocalize_freq, vocalize_amp, eat]

# Network outputs should use:
# - tanh activation for thrust and torque (range: -1 to 1)
# - sigmoid activation for vocalize_freq, vocalize_amp, eat (range: 0 to 1)
```

### Batched Operations

For processing multiple agents in parallel:

```python
def get_batch_observations(agents: List[AgentBody], world: 'World') -> torch.Tensor:
    """
    Get observations for multiple agents.

    Args:
        agents: List of N agents
        world: World reference

    Returns:
        torch.Tensor: shape (N, 154) float32
    """
    batch = []
    for agent in agents:
        if agent.is_alive:
            obs = agent.get_observation_tensor(world)
            batch.append(obs)

    return torch.stack(batch) if batch else torch.zeros((0, 154))
```

## Testing Strategy

### Unit Tests

**test_genome.py:**
- Test genome creation and default values
- Test mutation produces valid genomes
- Test serialization/deserialization
- Test breeding produces valid offspring
- Test mutation statistics (mean, variance)

**test_sensors.py:**
- Vision:
  - Ray correctly detects objects at known positions
  - Distance normalization is accurate
  - FOV limits work correctly
  - Color detection returns expected values
- Audio:
  - Sound attenuation with distance
  - Stereo positioning (left/right balance)
  - Multiple sound sources mix correctly
- Proprioception:
  - All values normalized to expected ranges
  - Angle sin/cos encoding
- Touch:
  - 8 sensors detect in correct directions
  - Object type classification
  - Distance normalization

**test_actions.py:**
- Thrust applies correct force
- Torque creates rotation
- Velocity limits enforced
- Eating only works when touching food
- Energy consumption scales with action magnitude

**test_body.py:**
- Agent initialization with genome
- Complete update cycle runs without errors
- Death conditions (starvation, damage)
- Healing when energy is high
- Position/velocity updates are correct

### Integration Tests

**test_agent_lifecycle.py:**
```python
def test_agent_survives_with_food():
    """Agent with food should maintain energy and survive."""
    # Create agent, world with nearby food
    # Run 1000 steps with eating action
    # Assert agent is still alive and energy > 0

def test_agent_starves_without_food():
    """Agent without food should eventually die."""
    # Create agent in empty world
    # Run until agent dies
    # Assert death cause is starvation

def test_agent_moves_correctly():
    """Thrust and torque should move agent predictably."""
    # Apply thrust action for 60 frames
    # Check position changed in expected direction
    # Apply torque action for 60 frames
    # Check angle changed correctly
```

### Performance Tests

**test_performance.py:**
```python
def test_100_agents_60hz():
    """Verify 100 agents can update at 60Hz."""
    world = create_test_world()
    agents = [create_test_agent() for _ in range(100)]

    start = time.time()
    for _ in range(600):  # 10 seconds at 60Hz
        for agent in agents:
            obs = agent.get_observation_tensor(world)
            action = create_random_action()
            agent.update(action, 1/60, world)

    elapsed = time.time() - start
    assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s for 10s simulation"

def test_vision_raycast_performance():
    """Vision raycasting should be fast enough."""
    # Create world with 50 food items and 50 agents
    # Time 1000 vision updates for single agent
    # Assert average time < 1ms per update
```

### Visual Tests (for manual verification)

Create a simple renderer to visually verify:
- Vision rays point in correct directions
- Agent rotates and moves correctly
- Touch sensors are positioned correctly
- Eating animation works

## Implementation Order

### Phase 1: Core Data Structures (Day 1)
1. Create `agents/genome.py` with `AgentGenome` dataclass
2. Create `agents/body.py` with `AgentState` and `AgentAction` dataclasses
3. Write unit tests for genome creation and mutation
4. Test serialization/deserialization

### Phase 2: Physics Foundation (Day 1-2)
1. Create `physics/raycasting.py` with circle and boundary intersection functions
2. Implement basic `AgentBody` class with physics update (no sensors yet)
3. Write tests for position/velocity updates
4. Test velocity limits and drag

### Phase 3: Sensory Systems (Day 2-3)
1. Implement `VisionSensor` with raycasting integration
2. Implement `ProprioceptionSensor` (simplest)
3. Implement `TouchSensor`
4. Implement `AudioSensor`
5. Unit test each sensor independently

### Phase 4: Action Application (Day 3)
1. Implement `apply_action` method in `AgentBody`
2. Implement eating mechanics
3. Test thrust, torque, eating actions
4. Verify action clamping and limits

### Phase 5: Survival Mechanics (Day 4)
1. Implement energy depletion system
2. Implement health mechanics (damage, healing)
3. Implement death conditions
4. Test lifecycle: spawn -> survive -> starve -> die

### Phase 6: Integration & Tensor Interface (Day 4-5)
1. Implement `get_observations()` method
2. Implement `get_observation_tensor()` method
3. Implement `AgentAction.from_tensor()` for neural network integration
4. Write integration tests with mock World
5. Test batched operations

### Phase 7: Performance Optimization (Day 5)
1. Profile vision raycasting
2. Optimize sensor updates (possibly cache world queries)
3. Run performance tests with 100+ agents
4. Optimize if needed to achieve 60Hz target

### Phase 8: Documentation & Polish (Day 6)
1. Add docstrings to all public methods
2. Create example usage scripts
3. Write integration guide for LRN system
4. Final testing pass

## Dependencies

### Required Libraries
```
numpy>=1.24.0
torch>=2.0.0
dataclasses (Python 3.7+)
typing
```

### Optional (for testing/visualization)
```
pytest>=7.0.0
pygame>=2.5.0  # For visual tests
matplotlib>=3.7.0  # For plotting sensor data
```

## Performance Targets

- **Sensor Update Rate**: 60 Hz for 100 agents (target: < 16ms per frame)
- **Vision Raycasting**: < 0.5ms per agent (32 rays)
- **Total Agent Update**: < 1ms per agent per frame
- **Memory**: < 10 KB per agent state

## Future Extensions

These are not part of the initial implementation but should be considered in the design:

1. **Additional Senses**:
   - Smell (gradient field for food detection)
   - Taste (chemical analysis when eating)
   - Pain sensors (localized damage detection)

2. **Enhanced Vision**:
   - Color-coded vision for different object types
   - Motion detection (optical flow)
   - Peripheral vision with lower resolution

3. **Communication**:
   - Pheromone trails (persistent world state)
   - Visual signals (body color changes)
   - Gesture recognition

4. **Advanced Genome**:
   - Sexual dimorphism (male/female traits)
   - Age-dependent traits
   - Epigenetic modifications

5. **Social Behaviors**:
   - Collision damage between agents
   - Cooperative eating
   - Territory marking

## Notes

- All angles use radians, 0 = pointing right, increases counter-clockwise
- All coordinates use standard Cartesian (x right, y up)
- Energy and health use 0-100 scale for intuitive understanding
- All sensor outputs are normalized 0-1 for neural network stability
- Genome parameters are tuned for ~60 second survival without food
- Touch sensors extend slightly beyond collision radius to give advance warning
- Vision rays are evenly distributed across FOV (not log-scale)
- Audio uses simple distance attenuation (could be enhanced with occlusion)
- Eating requires sustained action over time (prevents accidental feeding)

## Questions for Review

1. Should vision include depth-from-stereo (two slightly offset ray fans)?
2. Should angular velocity damping be separate from linear drag?
3. Should there be a minimum energy threshold for actions (can't move when starving)?
4. Should damage have directional information (which touch sensor was hit)?
5. Should genome include color/appearance for agent identification?
6. Should there be a "birth energy" cost for spawning new agents?

---

**Total Lines**: ~750
**Estimated Implementation Time**: 5-6 days for single developer
**Critical Path**: Raycasting performance optimization
