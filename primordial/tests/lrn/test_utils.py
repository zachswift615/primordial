"""Tests for FFT utility functions."""
import torch
import pytest

from primordial.lrn.utils import (
    init_spectral_filter,
    complex_to_real,
    real_to_complex,
)


class TestInitSpectralFilter:
    """Tests for spectral filter initialization with bias."""

    def test_shape(self):
        """Test filter has correct shape."""
        seq_len = 164
        freq_bins = 83
        filter_init = init_spectral_filter(seq_len, freq_bins)

        assert filter_init.shape == (seq_len, freq_bins, 2)
        assert filter_init.dtype == torch.float32

    def test_spectral_bias_low_freq_larger(self):
        """Test low frequencies have larger magnitudes than high frequencies."""
        seq_len = 100
        freq_bins = 51
        filter_init = init_spectral_filter(seq_len, freq_bins)

        # Compute magnitude of complex filter
        filter_complex = torch.view_as_complex(filter_init.contiguous())
        magnitudes = torch.abs(filter_complex)

        # Average magnitude across sequence dimension for each frequency
        avg_mag_per_freq = magnitudes.mean(dim=0)

        # Low frequencies (first 10) should have higher avg magnitude
        low_freq_avg = avg_mag_per_freq[:10].mean()
        high_freq_avg = avg_mag_per_freq[-10:].mean()

        assert low_freq_avg > high_freq_avg, (
            f"Low freq avg ({low_freq_avg:.4f}) should be > "
            f"high freq avg ({high_freq_avg:.4f})"
        )

    def test_spectral_bias_decay_ratio(self):
        """Test low frequencies are ~50x larger than high frequencies."""
        seq_len = 100
        freq_bins = 51
        filter_init = init_spectral_filter(seq_len, freq_bins)

        # Compute magnitude
        filter_complex = torch.view_as_complex(filter_init.contiguous())
        magnitudes = torch.abs(filter_complex)

        # Average across sequence dimension
        avg_mag_per_freq = magnitudes.mean(dim=0)

        # Ratio between first and last frequency
        ratio = avg_mag_per_freq[0] / (avg_mag_per_freq[-1] + 1e-8)

        # With decay = exp(-freq / (freq_bins / 4)), at freq=freq_bins-1:
        # decay_last = exp(-(freq_bins-1) / (freq_bins/4))
        # For freq_bins=51: exp(-50 / 12.75) ≈ exp(-3.92) ≈ 0.02
        # So ratio should be around 1 / 0.02 = 50
        assert ratio > 10, f"Ratio {ratio:.2f} should be > 10"
        assert ratio < 100, f"Ratio {ratio:.2f} should be < 100"

    def test_monotonic_decay(self):
        """Test magnitude decreases monotonically with frequency."""
        seq_len = 100
        freq_bins = 51
        filter_init = init_spectral_filter(seq_len, freq_bins)

        filter_complex = torch.view_as_complex(filter_init.contiguous())
        magnitudes = torch.abs(filter_complex)
        avg_mag_per_freq = magnitudes.mean(dim=0)

        # Check that magnitude generally decreases
        # (allow some variance due to random initialization)
        for i in range(0, freq_bins - 10, 10):
            low_avg = avg_mag_per_freq[i : i + 5].mean()
            high_avg = avg_mag_per_freq[i + 5 : i + 10].mean()
            assert low_avg > high_avg, f"Non-monotonic at freq {i}"

    def test_different_dimensions(self):
        """Test with various seq_len and freq_bins."""
        test_cases = [
            (50, 26),  # Small
            (164, 83),  # Default LRN
            (200, 101),  # Medium
            (512, 257),  # Large
        ]

        for seq_len, freq_bins in test_cases:
            filter_init = init_spectral_filter(seq_len, freq_bins)
            assert filter_init.shape == (seq_len, freq_bins, 2)

    def test_random_variation(self):
        """Test that filters are randomized (not all same)."""
        seq_len = 100
        freq_bins = 51
        filter1 = init_spectral_filter(seq_len, freq_bins)
        filter2 = init_spectral_filter(seq_len, freq_bins)

        # Should be different due to random initialization
        assert not torch.allclose(filter1, filter2)


class TestComplexConversions:
    """Tests for complex ↔ real conversions."""

    def test_complex_to_real_shape(self):
        """Test complex_to_real produces correct shape."""
        c = torch.randn(10, 20, dtype=torch.complex64)
        r = complex_to_real(c)

        assert r.shape == (10, 20, 2)
        assert r.dtype == torch.float32

    def test_real_to_complex_shape(self):
        """Test real_to_complex produces correct shape."""
        r = torch.randn(10, 20, 2)
        c = real_to_complex(r)

        assert c.shape == (10, 20)
        assert c.dtype == torch.complex64

    def test_roundtrip_complex_to_real_to_complex(self):
        """Test complex → real → complex preserves values."""
        c_original = torch.randn(10, 20, dtype=torch.complex64)
        r = complex_to_real(c_original)
        c_recovered = real_to_complex(r)

        assert torch.allclose(c_original, c_recovered, atol=1e-6)

    def test_roundtrip_real_to_complex_to_real(self):
        """Test real → complex → real preserves values."""
        r_original = torch.randn(10, 20, 2)
        c = real_to_complex(r_original)
        r_recovered = complex_to_real(c)

        assert torch.allclose(r_original, r_recovered, atol=1e-6)

    def test_real_imaginary_components(self):
        """Test that real/imag components are correctly extracted."""
        # Create known complex tensor
        real = torch.tensor([1.0, 2.0, 3.0])
        imag = torch.tensor([4.0, 5.0, 6.0])
        c = torch.complex(real, imag)

        r = complex_to_real(c)

        assert torch.allclose(r[:, 0], real)
        assert torch.allclose(r[:, 1], imag)

    def test_multidimensional_tensors(self):
        """Test conversions work with various tensor shapes."""
        shapes = [
            (5,),
            (5, 10),
            (5, 10, 15),
            (2, 3, 4, 5),
        ]

        for shape in shapes:
            # Complex to real
            c = torch.randn(*shape, dtype=torch.complex64)
            r = complex_to_real(c)
            assert r.shape == (*shape, 2)

            # Real to complex
            r_input = torch.randn(*shape, 2)
            c_output = real_to_complex(r_input)
            assert c_output.shape == shape

    def test_gradients_flow_through_conversions(self):
        """Test that gradients flow through conversions."""
        # Real to complex to real (with computation)
        r = torch.randn(5, 10, 2, requires_grad=True)
        c = real_to_complex(r)

        # Some operation in complex domain
        c_modified = c * 2.0

        # Back to real
        r_out = complex_to_real(c_modified)

        # Compute loss and backward
        loss = r_out.sum()
        loss.backward()

        assert r.grad is not None
        assert r.grad.shape == r.shape


class TestSpectralFilterIntegration:
    """Integration tests using spectral filter with conversions."""

    def test_filter_can_be_used_as_parameter(self):
        """Test filter can be used as nn.Parameter."""
        import torch.nn as nn

        filter_init = init_spectral_filter(100, 51)
        filter_param = nn.Parameter(filter_init)

        assert filter_param.requires_grad
        assert filter_param.shape == (100, 51, 2)

    def test_filter_conversion_for_fft(self):
        """Test filter can be converted for FFT multiplication."""
        seq_len = 100
        freq_bins = 51
        filter_init = init_spectral_filter(seq_len, freq_bins)

        # Convert to complex
        filter_complex = real_to_complex(filter_init)

        assert filter_complex.shape == (seq_len, freq_bins)
        assert filter_complex.dtype == torch.complex64

    def test_filter_multiplication_with_fft(self):
        """Test filter can multiply with FFT output."""
        batch = 2
        hidden_dim = 64
        seq_len = 100
        freq_bins = 51

        # Simulated FFT output
        x_fft = torch.randn(batch, hidden_dim, freq_bins, dtype=torch.complex64)

        # Spectral filter
        filter_init = init_spectral_filter(seq_len, freq_bins)
        filter_complex = real_to_complex(filter_init)

        # Slice filter to match hidden_dim
        filter_slice = filter_complex[:hidden_dim, :]

        # Multiplication
        x_filtered = x_fft * filter_slice.unsqueeze(0)

        assert x_filtered.shape == (batch, hidden_dim, freq_bins)
        assert x_filtered.dtype == torch.complex64

    def test_gradient_through_filter_multiplication(self):
        """Test gradients flow through filter multiplication."""
        batch = 2
        hidden_dim = 64
        seq_len = 100
        freq_bins = 51

        # FFT output
        x_fft = torch.randn(batch, hidden_dim, freq_bins, dtype=torch.complex64)

        # Learnable filter
        filter_init = init_spectral_filter(seq_len, freq_bins)
        filter_param = torch.nn.Parameter(filter_init)
        filter_complex = real_to_complex(filter_param)
        filter_slice = filter_complex[:hidden_dim, :]

        # Forward
        x_filtered = x_fft * filter_slice.unsqueeze(0)

        # Loss (only real part for simplicity)
        loss = x_filtered.real.sum()
        loss.backward()

        # Check gradients
        assert filter_param.grad is not None
        assert filter_param.grad.shape == filter_init.shape
