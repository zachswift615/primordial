"""Multi-task training loop for Fourier prototype validation."""
import torch
import torch.nn as nn
from typing import Dict, List

from .config import PrototypeConfig
from .prototype import FourierPrototype
from .data import generate_multitask_batch


def train_prototype(
    num_steps: int = 1000,
    batch_size: int = 8,
    seq_len: int = 64,
    hidden_dim: int = 32,
    num_layers: int = 2,
    reward_horizon: int = 5,
    reward_loss_weight: float = 1.0,
    learning_rate: float = 1e-3,
    verbose: bool = True,
    log_every: int = 100,
) -> Dict:
    """
    Train the Fourier prototype on multi-task prediction.

    Multi-task learning:
    1. Sensory prediction: Predict next timestep in sequence
    2. Reward prediction: Predict upcoming reward signals

    Args:
        num_steps: Number of training steps
        batch_size: Batch size (use 1 for online learning test)
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of Fourier mixing layers
        reward_horizon: Number of future reward steps to predict
        reward_loss_weight: Weight for reward loss vs sensory loss
        learning_rate: Learning rate
        verbose: Print progress
        log_every: Print every N steps

    Returns:
        Dict with training results
    """
    # Create config and model
    config = PrototypeConfig(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_mixing_layers=num_layers,
        reward_horizon=reward_horizon,
        reward_loss_weight=reward_loss_weight,
    )
    model = FourierPrototype(config, input_dim=1)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history
    total_losses: List[float] = []
    sensory_losses: List[float] = []
    reward_losses: List[float] = []

    for step in range(num_steps):
        # Generate multi-task batch
        batch = generate_multitask_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            reward_horizon=reward_horizon,
        )

        # Forward pass (returns both predictions)
        sensory_pred, reward_pred = model(batch["input"])

        # Compute losses
        sensory_loss = criterion(sensory_pred, batch["sensory_target"])
        reward_loss = criterion(reward_pred, batch["reward_target"])

        # Combined loss (multi-task)
        total_loss = sensory_loss + config.reward_loss_weight * reward_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Record losses
        total_losses.append(total_loss.item())
        sensory_losses.append(sensory_loss.item())
        reward_losses.append(reward_loss.item())

        # Log progress
        if verbose and (step + 1) % log_every == 0:
            avg_total = sum(total_losses[-log_every:]) / log_every
            avg_sensory = sum(sensory_losses[-log_every:]) / log_every
            avg_reward = sum(reward_losses[-log_every:]) / log_every
            print(
                f"Step {step + 1}/{num_steps} | "
                f"Total: {avg_total:.4f} | "
                f"Sensory: {avg_sensory:.4f} | "
                f"Reward: {avg_reward:.4f}"
            )

    return {
        "model": model,
        "config": config,
        "total_losses": total_losses,
        "sensory_losses": sensory_losses,
        "reward_losses": reward_losses,
        "final_total_loss": total_losses[-1],
        "final_sensory_loss": sensory_losses[-1],
        "final_reward_loss": reward_losses[-1],
    }


def evaluate_prototype(model: FourierPrototype, num_batches: int = 10) -> Dict:
    """
    Evaluate prototype model on both tasks.

    Returns:
        Dict with evaluation metrics for sensory and reward prediction
    """
    model.eval()
    config = model.config
    criterion = nn.MSELoss()

    sensory_total = 0.0
    reward_total = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            batch = generate_multitask_batch(
                batch_size=8,
                seq_len=config.seq_len,
                reward_horizon=config.reward_horizon,
            )
            sensory_pred, reward_pred = model(batch["input"])

            sensory_total += criterion(sensory_pred, batch["sensory_target"]).item()
            reward_total += criterion(reward_pred, batch["reward_target"]).item()

    sensory_avg = sensory_total / num_batches
    reward_avg = reward_total / num_batches

    # Compare to random baseline
    sensory_baseline = 0.0
    reward_baseline = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            batch = generate_multitask_batch(
                batch_size=8,
                seq_len=config.seq_len,
                reward_horizon=config.reward_horizon,
            )
            sensory_baseline += criterion(
                torch.randn_like(batch["sensory_target"]),
                batch["sensory_target"]
            ).item()
            reward_baseline += criterion(
                torch.randn_like(batch["reward_target"]),
                batch["reward_target"]
            ).item()

    sensory_baseline /= num_batches
    reward_baseline /= num_batches

    return {
        "sensory_loss": sensory_avg,
        "reward_loss": reward_avg,
        "sensory_baseline": sensory_baseline,
        "reward_baseline": reward_baseline,
        "sensory_improvement": sensory_baseline / sensory_avg if sensory_avg > 0 else float("inf"),
        "reward_improvement": reward_baseline / reward_avg if reward_avg > 0 else float("inf"),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("FOURIER MIXING PROTOTYPE - MULTI-TASK VALIDATION")
    print("=" * 60)

    # Train
    print("\nTraining with multi-task learning...")
    result = train_prototype(
        num_steps=1000,
        batch_size=8,
        learning_rate=1e-3,
        verbose=True,
        log_every=100,
    )

    # Evaluate
    print("\nEvaluating...")
    eval_result = evaluate_prototype(result["model"])

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final sensory loss:  {result['final_sensory_loss']:.4f}")
    print(f"Final reward loss:   {result['final_reward_loss']:.4f}")
    print(f"Final total loss:    {result['final_total_loss']:.4f}")
    print()
    print(f"Sensory baseline:    {eval_result['sensory_baseline']:.4f}")
    print(f"Reward baseline:     {eval_result['reward_baseline']:.4f}")
    print()
    print(f"Sensory improvement: {eval_result['sensory_improvement']:.1f}x better than random")
    print(f"Reward improvement:  {eval_result['reward_improvement']:.1f}x better than random")

    # Success criteria
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    success = True

    # Check 1: Sensory loss decreased
    if result["sensory_losses"][-1] < result["sensory_losses"][0]:
        print("[PASS] Sensory loss decreased during training")
    else:
        print("[FAIL] Sensory loss did not decrease")
        success = False

    # Check 2: Reward loss decreased
    if result["reward_losses"][-1] < result["reward_losses"][0]:
        print("[PASS] Reward loss decreased during training")
    else:
        print("[FAIL] Reward loss did not decrease")
        success = False

    # Check 3: Better than random on sensory
    if eval_result["sensory_improvement"] > 1.5:
        print(f"[PASS] Sensory prediction {eval_result['sensory_improvement']:.1f}x better than random")
    else:
        print("[FAIL] Sensory prediction not better than random")
        success = False

    # Check 4: Better than random on reward
    if eval_result["reward_improvement"] > 1.2:
        print(f"[PASS] Reward prediction {eval_result['reward_improvement']:.1f}x better than random")
    else:
        print("[FAIL] Reward prediction not better than random")
        success = False

    # Check 5: No NaN
    all_losses = result["total_losses"] + result["sensory_losses"] + result["reward_losses"]
    if not any(torch.isnan(torch.tensor(l)) for l in all_losses):
        print("[PASS] No NaN losses")
    else:
        print("[FAIL] NaN losses detected")
        success = False

    # Check 6: Online learning test
    print("\nTesting online learning (batch_size=1)...")
    online_result = train_prototype(
        num_steps=100,
        batch_size=1,
        learning_rate=1e-3,
        verbose=False,
    )
    online_nan = any(torch.isnan(torch.tensor(l)) for l in online_result["total_losses"])
    if not online_nan:
        print("[PASS] Online learning stable (no NaN with batch_size=1)")
    else:
        print("[FAIL] Online learning unstable")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("MULTI-TASK PROTOTYPE VALIDATION: SUCCESS")
        print("Fourier mixing learns BOTH sensory and reward prediction!")
        print("Ready to build full LRN architecture.")
    else:
        print("MULTI-TASK PROTOTYPE VALIDATION: FAILED")
        print("Investigate before proceeding with full implementation.")
    print("=" * 60)
