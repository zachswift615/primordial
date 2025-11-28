"""Tests for GenomeModulator."""
import pytest
import torch

from primordial.lrn.genome import GenomeModulator
from primordial.lrn.lrn_config import LRNConfig


@pytest.fixture
def config():
    """Create test config."""
    return LRNConfig(
        genome_dim=100,
        hidden_dim=128,
    )


@pytest.fixture
def modulator(config):
    """Create GenomeModulator instance."""
    return GenomeModulator(config)


def test_output_shape_preservation(modulator):
    """Test that output shape matches input shape."""
    batch_size = 4
    seq_len = 64
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    genome = torch.randn(batch_size, genome_dim)

    output = modulator(x, genome)

    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"


def test_different_genomes_produce_different_outputs(modulator):
    """Test that different genomes produce different outputs."""
    batch_size = 2
    seq_len = 64
    hidden_dim = 128
    genome_dim = 100

    # Same input, different genomes
    x = torch.randn(1, seq_len, hidden_dim).repeat(batch_size, 1, 1)
    genome1 = torch.randn(1, genome_dim)
    genome2 = torch.randn(1, genome_dim)

    # Ensure genomes are actually different
    assert not torch.allclose(genome1, genome2)

    output1 = modulator(x[0:1], genome1)
    output2 = modulator(x[1:2], genome2)

    # Outputs should be different with different genomes
    assert not torch.allclose(output1, output2, atol=1e-4), \
        "Different genomes should produce different outputs"


def test_scale_is_in_valid_range(modulator):
    """Test that scale is in [0.5, 1.5] range."""
    batch_size = 4
    genome_dim = 100

    genome = torch.randn(batch_size, genome_dim)

    # Extract scale by looking at the internal computation
    genome_features = modulator.genome_projection(genome)
    scale = torch.sigmoid(modulator.scale_layer(genome_features))
    scale = 0.5 + scale  # Apply the same transformation as in forward

    # Verify scale is in [0.5, 1.5]
    assert torch.all(scale >= 0.5), \
        f"Scale should be >= 0.5, got min {scale.min().item()}"
    assert torch.all(scale <= 1.5), \
        f"Scale should be <= 1.5, got max {scale.max().item()}"


def test_gradient_flow_through_genome(modulator):
    """Test that gradients flow through genome parameter."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    genome = torch.randn(batch_size, genome_dim, requires_grad=True)

    output = modulator(x, genome)
    loss = output.sum()
    loss.backward()

    # Check that genome has gradients
    assert genome.grad is not None, "Gradient should flow through genome"
    assert not torch.all(genome.grad == 0), \
        "Genome gradients should be non-zero"


def test_gradient_flow_through_activations(modulator):
    """Test that gradients flow through activations."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    genome = torch.randn(batch_size, genome_dim)

    output = modulator(x, genome)
    loss = output.sum()
    loss.backward()

    # Check that x has gradients
    assert x.grad is not None, "Gradient should flow through activations"
    assert not torch.all(x.grad == 0), \
        "Activation gradients should be non-zero"


def test_batch_processing(modulator):
    """Test that batch processing works correctly."""
    seq_len = 32
    hidden_dim = 128
    genome_dim = 100

    # Test with different batch sizes
    for batch_size in [1, 2, 8, 16]:
        x = torch.randn(batch_size, seq_len, hidden_dim)
        genome = torch.randn(batch_size, genome_dim)

        output = modulator(x, genome)

        assert output.shape == (batch_size, seq_len, hidden_dim), \
            f"Failed for batch_size={batch_size}"


def test_same_genome_produces_same_output(modulator):
    """Test that the same genome applied to same input produces same output."""
    batch_size = 2
    seq_len = 64
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    genome = torch.randn(batch_size, genome_dim)

    # Run twice with same inputs
    output1 = modulator(x, genome)
    output2 = modulator(x, genome)

    assert torch.allclose(output1, output2), \
        "Same inputs should produce same outputs"


def test_genome_broadcast_over_sequence(modulator):
    """Test that genome modulation is broadcasted correctly over sequence."""
    batch_size = 2
    seq_len = 64
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    genome = torch.randn(batch_size, genome_dim)

    output = modulator(x, genome)

    # Get scale and shift parameters
    genome_features = modulator.genome_projection(genome)
    scale = 0.5 + torch.sigmoid(modulator.scale_layer(genome_features))
    shift = modulator.shift_layer(genome_features)

    # Manually compute expected output
    expected = x * scale.unsqueeze(1) + shift.unsqueeze(1)

    assert torch.allclose(output, expected, atol=1e-5), \
        "Output should match manual computation"


def test_affine_transformation_properties(modulator):
    """Test that the modulation is truly an affine transformation."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    genome = torch.randn(batch_size, genome_dim)

    output = modulator(x, genome)

    # Get internal parameters
    genome_features = modulator.genome_projection(genome)
    scale = 0.5 + torch.sigmoid(modulator.scale_layer(genome_features))
    shift = modulator.shift_layer(genome_features)

    # Verify affine property: f(x) = a*x + b
    expected = x * scale.unsqueeze(1) + shift.unsqueeze(1)
    assert torch.allclose(output, expected, atol=1e-5), \
        "Output should follow affine transformation"


def test_modulator_with_zero_genome(modulator):
    """Test behavior with zero genome vector."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    genome = torch.zeros(batch_size, genome_dim)

    # Should not crash and should produce valid output
    output = modulator(x, genome)

    assert output.shape == x.shape
    assert torch.isfinite(output).all(), "Output should be finite"


def test_modulator_with_extreme_genome(modulator):
    """Test stability with extreme genome values."""
    batch_size = 2
    seq_len = 32
    hidden_dim = 128
    genome_dim = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Test with large positive values
    genome_large = torch.ones(batch_size, genome_dim) * 10.0
    output_large = modulator(x, genome_large)
    assert torch.isfinite(output_large).all(), \
        "Output should be finite with large genome values"

    # Test with large negative values
    genome_small = torch.ones(batch_size, genome_dim) * -10.0
    output_small = modulator(x, genome_small)
    assert torch.isfinite(output_small).all(), \
        "Output should be finite with small genome values"


def test_config_parameters_used(config):
    """Test that config parameters are correctly used."""
    modulator = GenomeModulator(config)

    assert modulator.genome_dim == config.genome_dim
    assert modulator.hidden_dim == config.hidden_dim
    assert modulator.genome_projection[0].in_features == config.genome_dim
    assert modulator.genome_projection[0].out_features == config.hidden_dim
    assert modulator.scale_layer.in_features == config.hidden_dim
    assert modulator.scale_layer.out_features == config.hidden_dim
    assert modulator.shift_layer.in_features == config.hidden_dim
    assert modulator.shift_layer.out_features == config.hidden_dim
