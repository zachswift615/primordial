"""Genome modulation for Living Resonance Network."""
import torch
import torch.nn as nn

from .lrn_config import LRNConfig


class GenomeModulator(nn.Module):
    """
    Modulates LRN activations based on genome vector.

    Genome encodes architectural hyperparameters that can evolve.
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.genome_dim = config.genome_dim  # 100
        self.hidden_dim = config.hidden_dim  # 128

        # Project genome to modulation parameters
        self.genome_projection = nn.Sequential(
            nn.Linear(self.genome_dim, self.hidden_dim),
            nn.Tanh()
        )

        # Scale and shift parameters
        self.scale_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.shift_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor, genome: torch.Tensor) -> torch.Tensor:
        """
        Apply genome-based modulation.

        Args:
            x: (batch, seq_len, hidden_dim) - activations
            genome: (batch, genome_dim) - genome vector

        Returns:
            (batch, seq_len, hidden_dim) - modulated activations
        """
        # Project genome
        genome_features = self.genome_projection(genome)  # (B, hidden_dim)

        # Compute scale and shift
        scale = torch.sigmoid(self.scale_layer(genome_features))  # (B, hidden_dim) in [0, 1]
        shift = self.shift_layer(genome_features)  # (B, hidden_dim)

        # Apply affine transformation
        # scale in [0.5, 1.5] for stability
        scale = 0.5 + scale

        # Broadcast over sequence dimension
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)

        return x
