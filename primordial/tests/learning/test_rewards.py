"""
Tests for reward system.
"""

import pytest
import torch
from primordial.learning.rewards import (
    RewardHistoryBuffer,
    SurvivalRewards,
    HumanTeaching,
    RewardCombiner,
)


class MockAgentState:
    """Mock AgentState for testing rewards"""

    def __init__(self, health=100.0, energy=100.0, max_health=100.0, max_energy=100.0):
        self.health = health
        self.energy = energy
        self.max_health = max_health
        self.max_energy = max_energy


def test_survival_rewards_eating():
    """Test reward for eating food"""
    prev_state = MockAgentState(health=50, energy=50)
    curr_state = MockAgentState(health=50, energy=70)
    events = ['ate_food']

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    assert reward == SurvivalRewards.EATING_FOOD


def test_survival_rewards_damage():
    """Test negative reward for taking damage"""
    prev_state = MockAgentState(health=80, energy=50)
    curr_state = MockAgentState(health=60, energy=50)
    events = ['took_damage']

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    assert reward == SurvivalRewards.TAKING_DAMAGE


def test_survival_rewards_death():
    """Test large negative reward for death"""
    prev_state = MockAgentState(health=10, energy=50)
    curr_state = MockAgentState(health=0, energy=50)
    events = ['died']

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    # Death event (-10.0) + LOW_HEALTH penalty (-0.05) since health is 0 < 50%
    assert reward == SurvivalRewards.DEATH + SurvivalRewards.LOW_HEALTH


def test_survival_rewards_starving():
    """Test continuous negative reward when starving"""
    prev_state = MockAgentState(health=50, energy=50)
    curr_state = MockAgentState(health=50, energy=20)  # 20% of max
    events = []

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    assert reward == SurvivalRewards.STARVING


def test_survival_rewards_healthy():
    """Test continuous positive reward when healthy"""
    prev_state = MockAgentState(health=80, energy=80)
    curr_state = MockAgentState(health=85, energy=85)
    events = []

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    assert reward == SurvivalRewards.HEALTHY


def test_survival_rewards_combined():
    """Test multiple rewards can combine"""
    prev_state = MockAgentState(health=80, energy=50)
    curr_state = MockAgentState(health=60, energy=20)  # Took damage and low energy
    events = ['took_damage']

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    # Should get both damage penalty and starving penalty
    expected = SurvivalRewards.TAKING_DAMAGE + SurvivalRewards.STARVING
    assert reward == expected


def test_human_teaching_window():
    """Test human teaching signal spreads over window"""
    teaching = HumanTeaching(window_size=10)
    teaching.add_teaching_signal(HumanTeaching.REWARD_BUTTON)

    # Should distribute reward evenly over 10 steps
    total = 0.0
    for _ in range(10):
        reward = teaching.get_current_reward()
        total += reward

    # Total should equal original signal
    assert abs(total - HumanTeaching.REWARD_BUTTON) < 1e-6

    # After window expires, should be zero
    reward = teaching.get_current_reward()
    assert reward == 0.0


def test_human_teaching_multiple_signals():
    """Test multiple teaching signals accumulate"""
    teaching = HumanTeaching(window_size=5)
    teaching.add_teaching_signal(HumanTeaching.REWARD_BUTTON)
    teaching.add_teaching_signal(HumanTeaching.REWARD_BUTTON)

    # Should get 2 rewards at once
    reward = teaching.get_current_reward()
    expected = 2 * (HumanTeaching.REWARD_BUTTON / 5)
    assert abs(reward - expected) < 1e-6


def test_human_teaching_punishment():
    """Test punishment signal"""
    teaching = HumanTeaching(window_size=10)
    teaching.add_teaching_signal(HumanTeaching.PUNISH_BUTTON)

    total = 0.0
    for _ in range(10):
        reward = teaching.get_current_reward()
        total += reward

    # Total should equal punishment signal (negative)
    assert abs(total - HumanTeaching.PUNISH_BUTTON) < 1e-6


def test_reward_combiner():
    """Test combining survival and teaching rewards"""
    combiner = RewardCombiner(survival_weight=1.0, teaching_weight=1.5)
    combiner.human_teaching.add_teaching_signal(HumanTeaching.REWARD_BUTTON)

    prev_state = MockAgentState(health=50, energy=50)
    curr_state = MockAgentState(health=50, energy=70)
    events = ['ate_food']

    total, survival, teaching = combiner.compute_total_reward(
        prev_state, curr_state, events
    )

    # Survival reward should be eating food
    assert survival == SurvivalRewards.EATING_FOOD

    # Teaching should be non-zero (reward button distributed over window)
    assert teaching > 0

    # Total should be weighted combination
    expected_total = 1.0 * survival + 1.5 * teaching
    assert abs(total - expected_total) < 1e-6


def test_reward_combiner_weights():
    """Test different weight combinations"""
    combiner = RewardCombiner(survival_weight=2.0, teaching_weight=0.5)
    combiner.human_teaching.add_teaching_signal(HumanTeaching.REWARD_BUTTON)

    prev_state = MockAgentState(health=50, energy=50)
    curr_state = MockAgentState(health=50, energy=70)
    events = ['ate_food']

    total, survival, teaching = combiner.compute_total_reward(
        prev_state, curr_state, events
    )

    expected_total = 2.0 * survival + 0.5 * teaching
    assert abs(total - expected_total) < 1e-6


def test_reward_history_buffer_basic():
    """Test basic reward history buffer operations"""
    buffer = RewardHistoryBuffer(horizon=5, max_pending=100, max_stale_steps=50)

    # Record a prediction at step 0
    predictions = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    buffer.record_prediction(step=0, reward_preds=predictions)

    # Record actual rewards for steps 1-5
    for i in range(1, 6):
        buffer.record_actual_reward(step=i, reward=float(i))

    # After 5 steps, we should have a ready pair
    # Wait 5 steps
    for _ in range(5):
        ready_pairs = buffer.get_ready_pairs()

    # Should have one pair now
    assert len(ready_pairs) == 1
    pred, actual = ready_pairs[0]

    assert torch.equal(pred, predictions)
    assert torch.equal(actual, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_reward_history_buffer_ready_pairs():
    """Test getting ready prediction/actual pairs"""
    buffer = RewardHistoryBuffer(horizon=3, max_pending=100, max_stale_steps=50)

    # Record predictions at steps 0, 1, 2
    buffer.record_prediction(0, torch.tensor([1.0, 2.0, 3.0]))
    buffer.record_prediction(1, torch.tensor([4.0, 5.0, 6.0]))
    buffer.record_prediction(2, torch.tensor([7.0, 8.0, 9.0]))

    # Record actual rewards
    for i in range(1, 10):
        buffer.record_actual_reward(i, float(i) * 0.5)

    # Collect ready pairs over multiple steps
    all_ready_pairs = []
    for _ in range(3):
        ready_pairs = buffer.get_ready_pairs()
        all_ready_pairs.extend(ready_pairs)

    # All predictions should be ready now
    assert len(all_ready_pairs) == 3


def test_reward_history_buffer_cleanup():
    """Test that old entries are cleaned up"""
    buffer = RewardHistoryBuffer(horizon=5, max_pending=10, max_stale_steps=10)

    # Add many predictions and rewards
    for step in range(100):
        buffer.record_prediction(step, torch.randn(5))
        buffer.record_actual_reward(step, float(step))

    # History should be bounded
    assert len(buffer.reward_history) < 100
    # Should keep only recent ones
    assert len(buffer.pending_predictions) <= 10


def test_reward_history_buffer_on_death():
    """Test death clears buffer state"""
    buffer = RewardHistoryBuffer(horizon=5)

    # Add some data
    buffer.record_prediction(0, torch.randn(5))
    buffer.record_actual_reward(1, 1.0)

    assert len(buffer.pending_predictions) > 0
    assert len(buffer.reward_history) > 0

    # Death should clear everything
    buffer.on_death()

    assert len(buffer.pending_predictions) == 0
    assert len(buffer.reward_history) == 0


def test_reward_history_buffer_stale_predictions():
    """Test that stale predictions are discarded"""
    buffer = RewardHistoryBuffer(horizon=3, max_pending=100, max_stale_steps=5)

    # Record a prediction
    buffer.record_prediction(0, torch.tensor([1.0, 2.0, 3.0]))

    # Wait too long (more than max_stale_steps)
    for _ in range(10):
        buffer.get_ready_pairs()

    # Prediction should be discarded, no pairs returned
    ready_pairs = buffer.get_ready_pairs()
    assert len(ready_pairs) == 0
