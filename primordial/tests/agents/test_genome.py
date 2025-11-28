"""Tests for AgentGenome."""

import pytest
import random

from primordial.agents.genome import AgentGenome, create_default_genome, breed


class TestAgentGenome:
    """Tests for AgentGenome dataclass."""

    def test_default_creation(self):
        """Test genome creation with default values."""
        genome = AgentGenome()

        assert genome.max_speed == 150.0
        assert genome.max_angular_speed == 3.0
        assert genome.thrust_force == 500.0
        assert genome.radius == 8.0
        assert genome.vision_range == 200.0
        assert genome.vision_rays == 32

    def test_custom_values(self):
        """Test genome creation with custom values."""
        genome = AgentGenome(
            max_speed=200.0,
            radius=12.0,
            vision_fov=180.0,
        )

        assert genome.max_speed == 200.0
        assert genome.radius == 12.0
        assert genome.vision_fov == 180.0
        # Defaults should still apply
        assert genome.thrust_force == 500.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        genome = AgentGenome(max_speed=100.0)
        data = genome.to_dict()

        assert isinstance(data, dict)
        assert data["max_speed"] == 100.0
        assert "radius" in data
        assert "vision_rays" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        original = AgentGenome(max_speed=100.0, radius=15.0)
        data = original.to_dict()
        restored = AgentGenome.from_dict(data)

        assert restored.max_speed == original.max_speed
        assert restored.radius == original.radius
        assert restored.vision_rays == original.vision_rays

    def test_from_dict_ignores_unknown_fields(self):
        """Test that unknown fields are ignored during deserialization."""
        data = {"max_speed": 100.0, "unknown_field": 999}
        genome = AgentGenome.from_dict(data)

        assert genome.max_speed == 100.0
        assert not hasattr(genome, "unknown_field")


class TestGenomeMutation:
    """Tests for genome mutation."""

    def test_mutate_returns_new_genome(self):
        """Test that mutation returns a new genome instance."""
        original = AgentGenome()
        mutated = original.mutate()

        assert mutated is not original
        assert isinstance(mutated, AgentGenome)

    def test_mutate_preserves_approximate_values(self):
        """Test that mutation produces values close to original (statistically)."""
        random.seed(42)
        original = AgentGenome()

        # Collect many mutations and check means are close
        mutated_speeds = []
        for _ in range(100):
            mutated = original.mutate()
            mutated_speeds.append(mutated.max_speed)

        mean_speed = sum(mutated_speeds) / len(mutated_speeds)
        # Should be within ~20% of original (statistically)
        assert 100 < mean_speed < 200

    def test_mutate_enforces_minimums(self):
        """Test that mutation enforces minimum values."""
        # Create genome with very small values
        genome = AgentGenome(
            max_speed=1.0,
            radius=0.5,
            base_energy_cost=0.02,
        )

        # Mutate many times - should never go below minimums
        for _ in range(50):
            mutated = genome.mutate()
            assert mutated.max_speed >= 0.1
            assert mutated.radius >= 0.1
            assert mutated.base_energy_cost >= 0.01

    def test_mutation_rate_affects_frequency(self):
        """Test that higher mutation rate causes more changes."""
        random.seed(42)

        low_rate = AgentGenome(mutation_rate=0.01)
        high_rate = AgentGenome(mutation_rate=0.99)

        # Count changes with low rate
        low_changes = 0
        for _ in range(100):
            mutated = low_rate.mutate()
            if mutated.max_speed != low_rate.max_speed:
                low_changes += 1

        # Count changes with high rate
        high_changes = 0
        for _ in range(100):
            mutated = high_rate.mutate()
            if mutated.max_speed != high_rate.max_speed:
                high_changes += 1

        # High rate should cause more changes
        assert high_changes > low_changes


class TestCreateDefaultGenome:
    """Tests for create_default_genome factory."""

    def test_returns_genome(self):
        """Test factory returns AgentGenome."""
        genome = create_default_genome()
        assert isinstance(genome, AgentGenome)

    def test_has_reasonable_defaults(self):
        """Test default genome has reasonable survival values."""
        genome = create_default_genome()

        # Should have enough energy for ~60 seconds
        assert genome.max_energy >= 50.0
        assert genome.base_energy_cost <= 1.0

        # Should have reasonable vision
        assert genome.vision_range >= 100.0
        assert genome.vision_rays >= 8


class TestBreed:
    """Tests for breeding function."""

    def test_breed_returns_new_genome(self):
        """Test breeding returns a new genome."""
        parent1 = AgentGenome(max_speed=100.0)
        parent2 = AgentGenome(max_speed=200.0)

        child = breed(parent1, parent2)

        assert child is not parent1
        assert child is not parent2
        assert isinstance(child, AgentGenome)

    def test_breed_mixes_traits(self):
        """Test that child inherits traits from both parents."""
        random.seed(42)

        parent1 = AgentGenome(max_speed=100.0, radius=5.0)
        parent2 = AgentGenome(max_speed=200.0, radius=15.0)

        # Breed many times and track distributions
        child_speeds = []
        child_radii = []
        for _ in range(100):
            child = breed(parent1, parent2)
            child_speeds.append(child.max_speed)
            child_radii.append(child.radius)

        # Children should have variety (not all from one parent)
        # Due to mutation, values will vary but should be in reasonable range
        assert min(child_speeds) < 150  # Some closer to parent1
        assert max(child_speeds) > 150  # Some closer to parent2

    def test_breed_applies_mutation(self):
        """Test that breeding applies mutation to child."""
        # Set same values for both parents
        parent1 = AgentGenome(max_speed=100.0, mutation_rate=1.0, mutation_scale=0.5)
        parent2 = AgentGenome(max_speed=100.0, mutation_rate=1.0, mutation_scale=0.5)

        # Child should be mutated (not exactly 100.0)
        random.seed(42)
        child = breed(parent1, parent2)

        # With 100% mutation rate, value should change
        assert child.max_speed != 100.0
