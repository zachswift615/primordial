"""Behavior-based training system for agents.

Detects qualifying behaviors and applies stat growth based on actions.
Stats grow from behavior - "use it, build it" philosophy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Callable
import math

from primordial.agents.genome import TRAINING_CONFIG, TRAINABLE_STATS

if TYPE_CHECKING:
    from primordial.agents.body import AgentBody
    from primordial.world.world import World


@dataclass
class TrainingEvent:
    """Record of a training event that occurred."""
    stat_name: str
    behavior: str
    gain: float
    timestamp: float  # Agent age when it occurred


@dataclass
class BehaviorTracker:
    """Tracks behavior over time for training detection.

    Monitors continuous behaviors (like sustained running) and
    event-based behaviors (like narrow escapes).
    """

    # Sustained running tracking
    ticks_at_high_speed: int = 0
    HIGH_SPEED_THRESHOLD: float = 0.50  # % of max speed to count as "running"
    HIGH_SPEED_TICKS_REQUIRED: int = 60  # ~1 second at 60fps of sustained running

    # Escape tracking
    was_in_danger: bool = False
    danger_entry_time: float = 0.0
    closest_predator_distance: float = float('inf')
    DANGER_ZONE_MULTIPLIER: float = 1.5  # Predator attack range * this = danger zone
    NARROW_ESCAPE_THRESHOLD: float = 1.2  # How close is "narrow"

    # Vision training tracking
    last_food_spot_distance: float = 0.0

    # Recent training events (for UI feedback)
    recent_events: List[TrainingEvent] = field(default_factory=list)
    MAX_RECENT_EVENTS: int = 10

    def record_event(self, stat_name: str, behavior: str, gain: float, agent_age: float) -> None:
        """Record a training event for UI feedback."""
        event = TrainingEvent(
            stat_name=stat_name,
            behavior=behavior,
            gain=gain,
            timestamp=agent_age,
        )
        self.recent_events.append(event)
        # Keep only recent events
        if len(self.recent_events) > self.MAX_RECENT_EVENTS:
            self.recent_events.pop(0)


class AgentTrainer:
    """Monitors agent behavior and applies training gains.

    Attach to an AgentWrapper to enable behavior-based stat growth.
    """

    def __init__(self, print_gains: bool = True):
        """Initialize the trainer.

        Args:
            print_gains: If True, print stat gains to console.
        """
        self.trackers: Dict[str, BehaviorTracker] = {}
        self.print_gains = print_gains

    def get_tracker(self, agent_id: str) -> BehaviorTracker:
        """Get or create a behavior tracker for an agent."""
        if agent_id not in self.trackers:
            self.trackers[agent_id] = BehaviorTracker()
        return self.trackers[agent_id]

    def update(self, agent: AgentBody, world: World, dt: float) -> List[TrainingEvent]:
        """Update training for an agent based on current behavior.

        Should be called every tick.

        Args:
            agent: The agent to check.
            world: World instance for context.
            dt: Time step.

        Returns:
            List of training events that occurred this tick.
        """
        if not agent.is_alive:
            return []

        tracker = self.get_tracker(agent.agent_id)
        events = []

        # Check all behavior types
        events.extend(self._check_speed_training(agent, tracker, dt))
        events.extend(self._check_escape_training(agent, world, tracker, dt))
        events.extend(self._check_vision_training(agent, world, tracker, dt))

        # Apply decay to active gains
        agent.genome.decay_active_gains(dt)

        return events

    def _check_speed_training(
        self,
        agent: AgentBody,
        tracker: BehaviorTracker,
        dt: float
    ) -> List[TrainingEvent]:
        """Check for sustained high-speed running."""
        events = []

        current_speed = agent.velocity.magnitude()
        effective_max_speed = agent.genome.get_effective_stat('max_speed')
        threshold = effective_max_speed * tracker.HIGH_SPEED_THRESHOLD

        if current_speed >= threshold:
            tracker.ticks_at_high_speed += 1

            if tracker.ticks_at_high_speed >= tracker.HIGH_SPEED_TICKS_REQUIRED:
                # Grant speed training!
                gain = agent.genome.apply_training_gain(
                    'max_speed',
                    TRAINING_CONFIG['base_gain']
                )

                if gain > 0:
                    event = TrainingEvent(
                        stat_name='max_speed',
                        behavior='sustained_running',
                        gain=gain,
                        timestamp=agent.age,
                    )
                    events.append(event)
                    tracker.record_event('max_speed', 'sustained_running', gain, agent.age)

                    if self.print_gains:
                        total = agent.genome.get_effective_stat('max_speed')
                        print(f"🏃 {agent.agent_id} trained SPEED +{gain:.3f} (total: {total:.1f})")

                tracker.ticks_at_high_speed = 0  # Reset counter
        else:
            tracker.ticks_at_high_speed = 0  # Reset if slowed down

        return events

    def _check_escape_training(
        self,
        agent: AgentBody,
        world: World,
        tracker: BehaviorTracker,
        dt: float
    ) -> List[TrainingEvent]:
        """Check for narrow escapes from predators."""
        events = []

        # Find closest predator
        closest_distance = float('inf')
        closest_predator = None
        predator_attack_range = 30.0  # Default, will be overridden if predator found

        for predator in world.predators:
            if not predator.is_active:
                continue
            dist = agent.position.distance_to(predator.position)
            if dist < closest_distance:
                closest_distance = dist
                closest_predator = predator
                predator_attack_range = getattr(predator, 'attack_range', 30.0)

        danger_zone = predator_attack_range * tracker.DANGER_ZONE_MULTIPLIER
        currently_in_danger = closest_distance < danger_zone

        # Track closest approach while in danger
        if currently_in_danger and closest_distance < tracker.closest_predator_distance:
            tracker.closest_predator_distance = closest_distance

        # Check for escape event: was in danger, now safe, didn't take damage recently
        if tracker.was_in_danger and not currently_in_danger:
            # Escaped! Check if it was a narrow escape
            time_in_danger = agent.age - tracker.danger_entry_time
            escape_margin = tracker.closest_predator_distance / predator_attack_range

            # Only count if we were actually close and didn't just get hit
            recent_damage_threshold = 0.5  # seconds
            took_recent_damage = (agent.age - agent.last_damage_time) < recent_damage_threshold if agent.last_damage_time > 0 else False

            if not took_recent_damage and tracker.closest_predator_distance < danger_zone:
                # Determine gain multiplier based on how narrow the escape was
                if escape_margin < tracker.NARROW_ESCAPE_THRESHOLD:
                    multiplier = 2.0  # Close call bonus!
                else:
                    multiplier = 1.0

                gain = agent.genome.apply_training_gain(
                    'max_angular_speed',  # Reaction time / agility
                    TRAINING_CONFIG['base_gain'] * multiplier
                )

                if gain > 0:
                    event = TrainingEvent(
                        stat_name='max_angular_speed',
                        behavior='narrow_escape' if multiplier > 1 else 'escape',
                        gain=gain,
                        timestamp=agent.age,
                    )
                    events.append(event)
                    tracker.record_event('max_angular_speed', event.behavior, gain, agent.age)

                    if self.print_gains:
                        total = agent.genome.get_effective_stat('max_angular_speed')
                        escape_type = "NARROW ESCAPE" if multiplier > 1 else "escape"
                        print(f"⚡ {agent.agent_id} trained REACTION ({escape_type}) +{gain:.3f} (total: {total:.2f})")

            # Reset tracking
            tracker.closest_predator_distance = float('inf')

        # Update danger state
        if currently_in_danger and not tracker.was_in_danger:
            # Just entered danger
            tracker.danger_entry_time = agent.age
            tracker.closest_predator_distance = closest_distance

        tracker.was_in_danger = currently_in_danger

        return events

    def _check_vision_training(
        self,
        agent: AgentBody,
        world: World,
        tracker: BehaviorTracker,
        dt: float
    ) -> List[TrainingEvent]:
        """Check for long-distance food spotting.

        This is a bit tricky since we need to know when the agent
        'spots' food. For now, we'll check if the agent is moving
        toward distant food.
        """
        events = []

        # Find nearest food
        nearest_food = None
        nearest_distance = float('inf')

        for food in world.food_items:
            if not food.is_active:
                continue
            dist = agent.position.distance_to(food.position)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_food = food

        if nearest_food is None:
            return events

        effective_vision = agent.genome.get_effective_stat('vision_range')

        # Check if food is at edge of vision (80%+ of range) and agent is moving toward it
        if nearest_distance > effective_vision * 0.80 and nearest_distance < effective_vision:
            # Check if agent is moving toward the food
            to_food = nearest_food.position - agent.position
            if agent.velocity.magnitude() > 0.1:
                velocity_dir = agent.velocity.normalized()
                food_dir = to_food.normalized()
                dot = velocity_dir.dot(food_dir)

                if dot > 0.7:  # Moving generally toward distant food
                    # Only trigger occasionally, not every tick
                    if tracker.last_food_spot_distance == 0 or abs(nearest_distance - tracker.last_food_spot_distance) > 20:
                        gain = agent.genome.apply_training_gain(
                            'vision_range',
                            TRAINING_CONFIG['base_gain'] * 0.5  # Smaller gains for vision
                        )

                        if gain > 0:
                            event = TrainingEvent(
                                stat_name='vision_range',
                                behavior='distant_spotting',
                                gain=gain,
                                timestamp=agent.age,
                            )
                            events.append(event)
                            tracker.record_event('vision_range', 'distant_spotting', gain, agent.age)

                            if self.print_gains:
                                total = agent.genome.get_effective_stat('vision_range')
                                print(f"👁️ {agent.agent_id} trained VISION +{gain:.3f} (total: {total:.1f})")

                        tracker.last_food_spot_distance = nearest_distance

        return events

    def reset_tracker(self, agent_id: str) -> None:
        """Reset tracking for an agent (e.g., on respawn)."""
        if agent_id in self.trackers:
            self.trackers[agent_id] = BehaviorTracker()

    def get_recent_events(self, agent_id: str) -> List[TrainingEvent]:
        """Get recent training events for an agent."""
        if agent_id in self.trackers:
            return self.trackers[agent_id].recent_events
        return []


# Global trainer instance (can be accessed from simulation)
_global_trainer: Optional[AgentTrainer] = None


def get_trainer() -> AgentTrainer:
    """Get the global trainer instance."""
    global _global_trainer
    if _global_trainer is None:
        _global_trainer = AgentTrainer()
    return _global_trainer


def reset_trainer() -> None:
    """Reset the global trainer (e.g., for testing)."""
    global _global_trainer
    _global_trainer = None
