import pytest
from unittest.mock import Mock
from primordial.interface.demo_mode import DemonstrationController, DemoAction


def test_demo_controller_initialization():
    controller = DemonstrationController()
    assert not controller.is_active()
    assert controller.get_current_action() is None


def test_demo_controller_activate_deactivate():
    controller = DemonstrationController()

    controller.activate()
    assert controller.is_active()

    controller.deactivate()
    assert not controller.is_active()


def test_demo_action_from_keyboard():
    action = DemoAction.from_key_input(
        up=True, down=False, left=False, right=False
    )
    assert action.move_direction == (0, -1)
    assert action.action_type == "move"


def test_demo_action_diagonal_movement():
    action = DemoAction.from_key_input(
        up=True, down=False, left=True, right=False
    )
    # Should normalize diagonal movement
    assert action.move_direction[0] < 0  # Left
    assert action.move_direction[1] < 0  # Up


def test_demo_controller_records_action_sequence():
    controller = DemonstrationController()
    controller.activate()

    action1 = DemoAction.from_key_input(up=True, down=False, left=False, right=False)
    action2 = DemoAction.from_key_input(up=False, down=False, left=True, right=False)

    controller.record_action(action1, timestamp=1.0)
    controller.record_action(action2, timestamp=2.0)

    sequence = controller.get_recorded_sequence()
    assert len(sequence) == 2
    assert sequence[0][0] == 1.0  # timestamp
    assert sequence[1][0] == 2.0


def test_demo_controller_apply_to_agent():
    controller = DemonstrationController()
    controller.activate()

    # Mock agent
    agent = Mock()
    agent.position = [0.0, 0.0]

    action = DemoAction.from_key_input(up=True, down=False, left=False, right=False)
    controller.set_current_action(action)

    result = controller.apply_to_agent(agent, dt=0.1)
    assert result is not None
    assert result["action"] == "move"
