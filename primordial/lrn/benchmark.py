"""Performance benchmarking for Fourier prototype."""
import time
import torch

from .config import PrototypeConfig
from .prototype import FourierPrototype


def benchmark_forward_pass(
    num_iterations: int = 100,
    batch_size: int = 1,
    seq_len: int = 64,
    hidden_dim: int = 32,
) -> dict:
    """
    Benchmark forward pass timing.

    Returns timing statistics.
    """
    config = PrototypeConfig(seq_len=seq_len, hidden_dim=hidden_dim)
    model = FourierPrototype(config, input_dim=1)
    model.eval()

    # Warmup
    x = torch.randn(batch_size, seq_len, 1)
    for _ in range(10):
        _ = model(x)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            x = torch.randn(batch_size, seq_len, 1)
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Standard config
    result = benchmark_forward_pass(
        num_iterations=100,
        batch_size=1,
        seq_len=64,
        hidden_dim=32,
    )

    print(f"\nForward pass timing (batch=1, seq=64, hidden=32):")
    print(f"  Mean:  {result['mean_ms']:.3f} ms")
    print(f"  Min:   {result['min_ms']:.3f} ms")
    print(f"  Max:   {result['max_ms']:.3f} ms")
    print(f"  Std:   {result['std_ms']:.3f} ms")

    # Check against target
    if result["mean_ms"] < 5.0:
        print(f"\n[PASS] Forward pass < 5ms target")
    else:
        print(f"\n[WARN] Forward pass > 5ms target")

    # Larger config (closer to full LRN)
    result_large = benchmark_forward_pass(
        num_iterations=100,
        batch_size=1,
        seq_len=164,  # Full LRN seq_len
        hidden_dim=128,  # Full LRN hidden_dim
    )

    print(f"\nForward pass timing (batch=1, seq=164, hidden=128):")
    print(f"  Mean:  {result_large['mean_ms']:.3f} ms")
    print(f"  Min:   {result_large['min_ms']:.3f} ms")
    print(f"  Max:   {result_large['max_ms']:.3f} ms")

    if result_large["mean_ms"] < 10.0:
        print(f"\n[PASS] Large config forward pass < 10ms target")
    else:
        print(f"\n[WARN] Large config forward pass > 10ms target")
