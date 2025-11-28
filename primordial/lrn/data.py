"""Synthetic data generation for multi-task prototype validation."""
import torch
from typing import Dict


def generate_multitask_batch(
    batch_size: int = 8,
    seq_len: int = 64,
    reward_horizon: int = 5,
    freq_range: tuple = (0.5, 2.0),
    noise_std: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Generate batch of noisy sine waves with synthetic reward signals.

    This creates a multi-task learning scenario:
    1. Sensory prediction: Predict next timestep in sequence
    2. Reward prediction: Predict upcoming "rewards" based on signal amplitude

    The reward signal is derived from the sine wave amplitude - high amplitude
    regions get positive reward (simulating "food found"), low amplitude gets
    negative reward (simulating "danger"). This creates a learnable correlation
    between sensory patterns and reward outcomes.

    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        reward_horizon: Number of future reward steps to predict
        freq_range: (min, max) frequency range
        noise_std: Standard deviation of noise

    Returns:
        Dict with:
            'input': (batch, seq_len, 1) - input sequence
            'sensory_target': (batch, seq_len, 1) - next-step prediction target
            'reward_target': (batch, reward_horizon) - future reward targets
    """
    # Random frequencies for each sequence
    freqs = torch.rand(batch_size, 1) * (freq_range[1] - freq_range[0]) + freq_range[0]

    # Random phases
    phases = torch.rand(batch_size, 1) * 2 * 3.14159

    # Time steps (extra for target and reward horizon)
    total_len = seq_len + reward_horizon + 1
    t = torch.linspace(0, 4 * 3.14159, total_len).unsqueeze(0)  # (1, total_len)

    # Generate sine waves: (batch, total_len)
    signal = torch.sin(freqs * t + phases)

    # Add noise to signal
    noisy_signal = signal + torch.randn_like(signal) * noise_std

    # Split into input and sensory target
    input_seq = noisy_signal[:, :seq_len].unsqueeze(-1)  # (batch, seq_len, 1)
    sensory_target = noisy_signal[:, 1:seq_len+1].unsqueeze(-1)  # (batch, seq_len, 1)

    # Generate reward targets based on amplitude of FUTURE signal
    # Reward = amplitude of signal at t+1, t+2, ..., t+horizon
    # This simulates: "if I'm at a peak, good things happen; if at a trough, bad things"
    reward_target = torch.zeros(batch_size, reward_horizon)
    for h in range(reward_horizon):
        # Future signal value (clean, not noisy) determines reward
        future_val = signal[:, seq_len + h]
        # Map sine [-1, 1] to reward [-1, 1] with some scaling
        reward_target[:, h] = future_val

    return {
        "input": input_seq,
        "sensory_target": sensory_target,
        "reward_target": reward_target,
    }


# Backward compatibility alias
def generate_sine_batch(
    batch_size: int = 8,
    seq_len: int = 64,
    freq_range: tuple = (0.5, 2.0),
    noise_std: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Legacy function - use generate_multitask_batch instead."""
    batch = generate_multitask_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        reward_horizon=5,
        freq_range=freq_range,
        noise_std=noise_std,
    )
    return {
        "input": batch["input"],
        "target": batch["sensory_target"],
    }
