"""Standalone demo of Living Resonance Network (LRN).

This script demonstrates:
1. Creating LRNConfig and LivingResonanceNetwork
2. Forward pass with synthetic multi-modal inputs
3. Computing loss with sensory + reward prediction
4. Single training step
5. Parameter count and tensor shapes
"""
import torch
import torch.optim as optim

from .lrn_config import LRNConfig
from .architecture import LivingResonanceNetwork


def generate_synthetic_inputs(batch_size: int, config: LRNConfig):
    """Generate synthetic multi-modal sensory inputs."""
    return {
        'vision': torch.randn(batch_size, *config.vision_shape),
        'audio': torch.randn(batch_size, *config.audio_shape),
        'proprio': torch.randn(batch_size, config.proprio_dim),
        'touch': torch.randn(batch_size, config.touch_dim),
        'genome': torch.randn(batch_size, config.genome_dim) if config.use_genome_modulation else None,
    }


def generate_targets(batch_size: int, config: LRNConfig):
    """Generate synthetic targets for next sensory state and future rewards."""
    return {
        'next_sensory': {
            'vision': torch.randn(batch_size, *config.vision_shape),
            'audio': torch.randn(batch_size, *config.audio_shape),
            'proprio': torch.randn(batch_size, config.proprio_dim),
            'touch': torch.randn(batch_size, config.touch_dim),
        },
        'rewards': torch.randn(batch_size, config.reward_horizon),  # Future rewards
    }


def main():
    print("=" * 70)
    print("LIVING RESONANCE NETWORK (LRN) - DEMO")
    print("=" * 70)

    # 1. Create configuration
    print("\n1. Creating LRNConfig...")
    config = LRNConfig(
        vision_shape=(32, 4),
        audio_shape=(100, 2),
        proprio_dim=7,
        touch_dim=8,
        hidden_dim=128,
        num_mixing_layers=6,
        action_dim=5,
        reward_horizon=5,
        use_genome_modulation=True,
    )
    print(f"   Total sequence length: {config.total_seq_len}")
    print(f"   Total sensory dimension: {config.total_sensory_dim}")
    print(f"   Frequency bins: {config.freq_bins}")

    # 2. Create model
    print("\n2. Creating LivingResonanceNetwork...")
    model = LivingResonanceNetwork(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # 3. Forward pass
    print("\n3. Running forward pass...")
    batch_size = 4
    inputs = generate_synthetic_inputs(batch_size, config)

    with torch.no_grad():
        predictions, reward_preds, actions = model(
            vision=inputs['vision'],
            audio=inputs['audio'],
            proprio=inputs['proprio'],
            touch=inputs['touch'],
            genome=inputs['genome'],
        )

    print(f"   Input shapes:")
    print(f"     Vision:  {inputs['vision'].shape}")
    print(f"     Audio:   {inputs['audio'].shape}")
    print(f"     Proprio: {inputs['proprio'].shape}")
    print(f"     Touch:   {inputs['touch'].shape}")
    if inputs['genome'] is not None:
        print(f"     Genome:  {inputs['genome'].shape}")

    print(f"\n   Output shapes:")
    print(f"     Predictions (next sensory): {predictions.shape}")
    print(f"     Reward predictions:         {reward_preds.shape}")
    print(f"     Actions:                    {actions.shape}")

    # 4. Compute loss
    print("\n4. Computing loss...")
    targets = generate_targets(batch_size, config)

    losses = model.compute_loss(
        predictions=predictions,
        reward_preds=reward_preds,
        next_sensory=targets['next_sensory'],
        actions=actions,
        actual_rewards=targets['rewards'],
    )

    print(f"   Loss components:")
    print(f"     Total loss:    {losses['total'].item():.4f}")
    print(f"     Sensory loss:  {losses['sensory'].item():.4f}")
    print(f"     Reward loss:   {losses['reward'].item():.4f}")
    print(f"     Vision loss:   {losses['vision'].item():.4f}")
    print(f"     Audio loss:    {losses['audio'].item():.4f}")
    print(f"     Proprio loss:  {losses['proprio'].item():.4f}")
    print(f"     Touch loss:    {losses['touch'].item():.4f}")

    # 5. Training step
    print("\n5. Performing single training step...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Forward pass (in training mode)
    model.train()
    predictions, reward_preds, actions = model(
        vision=inputs['vision'],
        audio=inputs['audio'],
        proprio=inputs['proprio'],
        touch=inputs['touch'],
        genome=inputs['genome'],
    )

    # Compute loss
    losses = model.compute_loss(
        predictions=predictions,
        reward_preds=reward_preds,
        next_sensory=targets['next_sensory'],
        actions=actions,
        actual_rewards=targets['rewards'],
    )

    # Backward pass
    optimizer.zero_grad()
    losses['total'].backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    max_grad_norm = max(grad_norms) if grad_norms else 0.0

    print(f"   Gradient statistics:")
    print(f"     Average gradient norm: {avg_grad_norm:.6f}")
    print(f"     Max gradient norm:     {max_grad_norm:.6f}")

    # Update weights
    optimizer.step()
    print(f"   Training step complete!")

    # 6. Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"Model:       LivingResonanceNetwork")
    print(f"Parameters:  {total_params:,}")
    print(f"Architecture: {config.num_mixing_layers} Fourier mixing layers")
    print(f"Multi-modal: vision + audio + proprioception + touch")
    print(f"Multi-task:  sensory prediction + reward prediction + action")
    print("=" * 70)


if __name__ == "__main__":
    main()
