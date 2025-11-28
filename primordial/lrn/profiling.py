"""Performance benchmarking for Living Resonance Network (LRN).

This script measures:
1. Forward pass timing
2. Backward pass timing
3. Memory usage
4. Tests with batch_size=1 (online learning) and batch_size=8
"""
import time
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
        'rewards': torch.randn(batch_size, config.reward_horizon),
    }


def benchmark_forward(model, inputs, num_iterations=100):
    """Benchmark forward pass timing."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(**inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def benchmark_backward(model, inputs, targets, num_iterations=100):
    """Benchmark forward + backward pass timing."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        predictions, reward_preds, actions = model(**inputs)
        losses = model.compute_loss(
            predictions=predictions,
            reward_preds=reward_preds,
            next_sensory=targets['next_sensory'],
            actions=actions,
            actual_rewards=targets['rewards'],
        )
        losses['total'].backward()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        optimizer.zero_grad()

        start = time.perf_counter()
        predictions, reward_preds, actions = model(**inputs)
        losses = model.compute_loss(
            predictions=predictions,
            reward_preds=reward_preds,
            next_sensory=targets['next_sensory'],
            actions=actions,
            actual_rewards=targets['rewards'],
        )
        losses['total'].backward()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        # CPU memory is harder to track, return 0
        return 0.0


def profile_batch_size(batch_size: int, config: LRNConfig):
    """Profile model with specific batch size."""
    print(f"\n{'=' * 70}")
    print(f"BATCH SIZE: {batch_size}")
    print('=' * 70)

    # Create model
    model = LivingResonanceNetwork(config)
    total_params = sum(p.numel() for p in model.parameters())

    # Generate data
    inputs = generate_synthetic_inputs(batch_size, config)
    targets = generate_targets(batch_size, config)

    # Memory before
    mem_before = get_memory_usage()

    # Forward pass benchmark
    print(f"\nForward pass timing ({batch_size} samples):")
    forward_stats = benchmark_forward(model, inputs, num_iterations=100)
    print(f"  Mean:  {forward_stats['mean_ms']:.3f} ms")
    print(f"  Min:   {forward_stats['min_ms']:.3f} ms")
    print(f"  Max:   {forward_stats['max_ms']:.3f} ms")
    print(f"  Std:   {forward_stats['std_ms']:.3f} ms")

    # Backward pass benchmark
    print(f"\nBackward pass timing ({batch_size} samples):")
    backward_stats = benchmark_backward(model, inputs, targets, num_iterations=100)
    print(f"  Mean:  {backward_stats['mean_ms']:.3f} ms")
    print(f"  Min:   {backward_stats['min_ms']:.3f} ms")
    print(f"  Max:   {backward_stats['max_ms']:.3f} ms")
    print(f"  Std:   {backward_stats['std_ms']:.3f} ms")

    # Memory after
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    print(f"\nMemory usage:")
    print(f"  Model parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    if torch.cuda.is_available():
        print(f"  GPU memory used: {mem_used:.2f} MB")
    else:
        print(f"  CPU mode (memory tracking unavailable)")

    # Performance assessment
    print(f"\nPerformance assessment:")
    if batch_size == 1:
        target_ms = 20.0  # Online learning target
        if forward_stats['mean_ms'] < target_ms:
            print(f"  [PASS] Online learning: {forward_stats['mean_ms']:.3f} ms < {target_ms} ms")
        else:
            print(f"  [WARN] Online learning: {forward_stats['mean_ms']:.3f} ms > {target_ms} ms")
    else:
        throughput = batch_size / (forward_stats['mean_ms'] / 1000)
        print(f"  Throughput: {throughput:.1f} samples/sec")

    return {
        'batch_size': batch_size,
        'forward': forward_stats,
        'backward': backward_stats,
        'params': total_params,
    }


def main():
    print("=" * 70)
    print("LIVING RESONANCE NETWORK (LRN) - PERFORMANCE PROFILING")
    print("=" * 70)

    # Configuration
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

    print(f"\nModel configuration:")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Mixing layers: {config.num_mixing_layers}")
    print(f"  Total sequence length: {config.total_seq_len}")
    print(f"  Total sensory dimension: {config.total_sensory_dim}")
    print(f"  Reward horizon: {config.reward_horizon}")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Profile different batch sizes
    results = []

    # Batch size 1 - online learning
    results.append(profile_batch_size(1, config))

    # Batch size 8 - batched learning
    results.append(profile_batch_size(8, config))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)
    print(f"\n{'Batch':>8} | {'Forward (ms)':>14} | {'Backward (ms)':>14} | {'Throughput':>12}")
    print('-' * 70)
    for r in results:
        batch = r['batch_size']
        fwd = r['forward']['mean_ms']
        bwd = r['backward']['mean_ms']
        throughput = batch / (fwd / 1000)
        print(f"{batch:>8} | {fwd:>14.3f} | {bwd:>14.3f} | {throughput:>10.1f} s/s")

    print('=' * 70)
    print("\nKey takeaways:")
    print("  - Online learning (batch=1) shows per-sample latency")
    print("  - Batched learning (batch=8) shows throughput capacity")
    print("  - Backward pass includes forward + gradient computation")
    print("  - LRN is designed for real-time embodied AI agents")
    print('=' * 70)


if __name__ == "__main__":
    main()
