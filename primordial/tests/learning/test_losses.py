"""
Tests for loss functions.
"""

import pytest
import torch
from primordial.learning.losses import PredictionLoss


def test_prediction_loss_zero_error():
    """Test that identical predictions give zero loss"""
    loss_fn = PredictionLoss()
    predicted = torch.randn(1, 10)
    actual = predicted.clone()

    loss = loss_fn(predicted, actual)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_prediction_loss_positive():
    """Test that different predictions give positive loss"""
    loss_fn = PredictionLoss()
    predicted = torch.randn(1, 10)
    actual = torch.randn(1, 10)

    loss = loss_fn(predicted, actual)
    assert loss > 0


def test_prediction_loss_shape():
    """Test loss computation with different shapes"""
    loss_fn = PredictionLoss()

    # Test 1D case
    predicted = torch.randn(1, 5)
    actual = torch.randn(1, 5)
    loss = loss_fn(predicted, actual)
    assert loss.dim() == 0  # Should be scalar

    # Test 2D case
    predicted = torch.randn(1, 10, 10)
    actual = torch.randn(1, 10, 10)
    loss = loss_fn(predicted, actual)
    assert loss.dim() == 0  # Should be scalar


def test_prediction_loss_reduction_none():
    """Test loss computation with reduction='none'"""
    loss_fn = PredictionLoss(reduction='none')
    predicted = torch.randn(2, 5)
    actual = torch.randn(2, 5)

    loss = loss_fn(predicted, actual)
    assert loss.shape == (2, 5)  # Should match input shape


def test_prediction_loss_gradient():
    """Test that loss is differentiable"""
    loss_fn = PredictionLoss()
    predicted = torch.randn(1, 10, requires_grad=True)
    actual = torch.randn(1, 10)

    loss = loss_fn(predicted, actual)
    loss.backward()

    assert predicted.grad is not None
    assert predicted.grad.shape == predicted.shape
