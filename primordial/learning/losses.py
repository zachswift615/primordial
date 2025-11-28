"""
Loss functions for online learning.

Implements prediction loss (MSE) for sensory prediction and reward prediction tasks.
"""

import torch
import torch.nn as nn


class PredictionLoss(nn.Module):
    """
    Mean Squared Error loss for prediction tasks.

    Used for both:
    - Sensory prediction: MSE(predicted_senses, actual_senses)
    - Reward prediction: MSE(predicted_rewards, actual_rewards)

    Args:
        reduction: Specifies the reduction to apply to the output:
            'mean': the sum of the output will be divided by the number of elements
            'none': no reduction will be applied
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predicted, actual):
        """
        Compute MSE loss between predicted and actual values.

        Args:
            predicted: Predicted values (batch, *)
            actual: Actual values (batch, *)

        Returns:
            Scalar loss if reduction='mean', otherwise tensor of per-element losses
        """
        return self.mse(predicted, actual)
