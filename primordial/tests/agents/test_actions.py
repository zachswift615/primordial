"""Tests for AgentAction."""

import pytest
import numpy as np
import torch

from primordial.agents.actions import AgentAction


class TestAgentAction:
    """Tests for AgentAction dataclass."""

    def test_creation(self):
        """Test action creation with valid values."""
        action = AgentAction(
            thrust=0.5,
            torque=-0.3,
            vocalize=np.array([0.2, 0.8]),
            eat=0.0,
        )

        assert action.thrust == 0.5
        assert action.torque == -0.3
        assert np.allclose(action.vocalize, [0.2, 0.8])
        assert action.eat == 0.0

    def test_vocalize_converted_to_array(self):
        """Test that vocalize is converted to numpy array."""
        action = AgentAction(
            thrust=0.0,
            torque=0.0,
            vocalize=[0.1, 0.2],  # List instead of array
            eat=0.0,
        )

        assert isinstance(action.vocalize, np.ndarray)
        assert action.vocalize.dtype == np.float32


class TestAgentActionFromTensor:
    """Tests for from_tensor class method."""

    def test_from_tensor_basic(self):
        """Test conversion from tensor to action."""
        tensor = torch.tensor([0.5, -0.3, 0.2, 0.8, 0.1])
        action = AgentAction.from_tensor(tensor)

        assert action.thrust == pytest.approx(0.5)
        assert action.torque == pytest.approx(-0.3)
        assert action.vocalize[0] == pytest.approx(0.2)
        assert action.vocalize[1] == pytest.approx(0.8)
        assert action.eat == pytest.approx(0.1)

    def test_from_tensor_clamps_thrust(self):
        """Test that thrust is clamped to [-1, 1]."""
        tensor = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
        action = AgentAction.from_tensor(tensor)
        assert action.thrust == 1.0

        tensor = torch.tensor([-2.0, 0.0, 0.0, 0.0, 0.0])
        action = AgentAction.from_tensor(tensor)
        assert action.thrust == -1.0

    def test_from_tensor_clamps_torque(self):
        """Test that torque is clamped to [-1, 1]."""
        tensor = torch.tensor([0.0, 1.5, 0.0, 0.0, 0.0])
        action = AgentAction.from_tensor(tensor)
        assert action.torque == 1.0

    def test_from_tensor_clamps_vocalize(self):
        """Test that vocalize is clamped to [0, 1]."""
        tensor = torch.tensor([0.0, 0.0, -0.5, 1.5, 0.0])
        action = AgentAction.from_tensor(tensor)
        assert action.vocalize[0] == 0.0
        assert action.vocalize[1] == 1.0

    def test_from_tensor_clamps_eat(self):
        """Test that eat is clamped to [0, 1]."""
        tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
        action = AgentAction.from_tensor(tensor)
        assert action.eat == 1.0

    def test_from_tensor_wrong_shape_raises(self):
        """Test that wrong tensor shape raises assertion error."""
        tensor = torch.tensor([0.0, 0.0, 0.0])  # Wrong shape

        with pytest.raises(AssertionError):
            AgentAction.from_tensor(tensor)

    def test_from_tensor_handles_gradients(self):
        """Test that from_tensor detaches gradients properly."""
        tensor = torch.tensor([0.5, 0.3, 0.2, 0.1, 0.0], requires_grad=True)
        action = AgentAction.from_tensor(tensor)

        # Should work without errors (detached from graph)
        assert action.thrust == pytest.approx(0.5)


class TestAgentActionToTensor:
    """Tests for to_tensor method."""

    def test_to_tensor_basic(self):
        """Test conversion from action to tensor."""
        action = AgentAction(
            thrust=0.5,
            torque=-0.3,
            vocalize=np.array([0.2, 0.8]),
            eat=0.1,
        )
        tensor = action.to_tensor()

        assert tensor.shape == (5,)
        assert tensor[0].item() == pytest.approx(0.5)
        assert tensor[1].item() == pytest.approx(-0.3)
        assert tensor[2].item() == pytest.approx(0.2)
        assert tensor[3].item() == pytest.approx(0.8)
        assert tensor[4].item() == pytest.approx(0.1)

    def test_to_tensor_dtype(self):
        """Test that to_tensor returns float32."""
        action = AgentAction.zero()
        tensor = action.to_tensor()

        assert tensor.dtype == torch.float32

    def test_roundtrip(self):
        """Test tensor -> action -> tensor roundtrip."""
        original = torch.tensor([0.3, -0.7, 0.5, 0.5, 0.9])
        action = AgentAction.from_tensor(original)
        recovered = action.to_tensor()

        assert torch.allclose(original, recovered)


class TestAgentActionFactories:
    """Tests for zero and random factory methods."""

    def test_zero_action(self):
        """Test zero/idle action creation."""
        action = AgentAction.zero()

        assert action.thrust == 0.0
        assert action.torque == 0.0
        assert np.allclose(action.vocalize, [0.0, 0.0])
        assert action.eat == 0.0

    def test_random_action(self):
        """Test random action creation."""
        np.random.seed(42)
        action = AgentAction.random()

        # Should be within valid ranges
        assert -1.0 <= action.thrust <= 1.0
        assert -1.0 <= action.torque <= 1.0
        assert 0.0 <= action.vocalize[0] <= 1.0
        assert 0.0 <= action.vocalize[1] <= 1.0
        assert 0.0 <= action.eat <= 1.0

    def test_random_produces_variety(self):
        """Test that random produces different values."""
        np.random.seed(42)
        actions = [AgentAction.random() for _ in range(10)]

        thrusts = [a.thrust for a in actions]
        # Should have some variety
        assert len(set(thrusts)) > 1
