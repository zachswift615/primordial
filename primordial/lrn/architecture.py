"""Main Living Resonance Network (LRN) architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .lrn_config import LRNConfig
from .encoders import VisionEncoder, AudioEncoder, ProprioEncoder, TouchEncoder
from .lrn_mixing import LRNFourierMixingLayer
from .lrn_heads import PredictionHead, LRNRewardHead, ActionHead
from .genome import GenomeModulator


class LivingResonanceNetwork(nn.Module):
    """
    Main LRN architecture combining all components.

    Fourier-based neural network for embodied AI agents.
    Combines multi-modal sensory encoding with Fourier mixing layers
    to predict future sensory states, rewards, and actions.
    """

    def __init__(self, config: LRNConfig):
        super().__init__()
        self.config = config

        # Modality encoders
        self.vision_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.proprio_encoder = ProprioEncoder(config)
        self.touch_encoder = TouchEncoder(config)

        # Fourier mixing layers (6 layers)
        self.mixing_layers = nn.ModuleList([
            LRNFourierMixingLayer(config)
            for _ in range(config.num_mixing_layers)
        ])

        # Output heads
        self.prediction_head = PredictionHead(config)
        self.reward_head = LRNRewardHead(config)
        self.action_head = ActionHead(config)

        # Optional genome modulation
        if config.use_genome_modulation:
            self.genome_modulator = GenomeModulator(config)
        else:
            self.genome_modulator = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable online learning."""

        # 1. Encoder projections: Xavier uniform (balanced variance)
        for encoder in [self.vision_encoder, self.audio_encoder,
                        self.proprio_encoder, self.touch_encoder]:
            for module in encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # 2. Spectral filters: Low-frequency bias
        for layer in self.mixing_layers:
            freq_bins = layer.freq_bins
            seq_len = layer.seq_len

            # Frequency decay
            freqs = torch.arange(freq_bins, dtype=torch.float32)
            decay = torch.exp(-freqs / (freq_bins / 4))

            # Initialize with decay
            with torch.no_grad():
                for i in range(seq_len):
                    layer.spectral_filter.data[i, :, 0] = torch.randn(freq_bins) * decay * 0.1
                    layer.spectral_filter.data[i, :, 1] = torch.randn(freq_bins) * decay * 0.1

        # 3. LayerNorm: Standard initialization
        for layer in self.mixing_layers:
            nn.init.ones_(layer.norm.weight)
            nn.init.zeros_(layer.norm.bias)

        # 4. Output heads: Small initialization to prevent large initial predictions
        for head in [self.prediction_head, self.reward_head, self.action_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)  # Small gain
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        vision: torch.Tensor,
        audio: torch.Tensor,
        proprio: torch.Tensor,
        touch: torch.Tensor,
        genome: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LRN.

        Args:
            vision: (batch, 32, 4)
            audio: (batch, 100, 2)
            proprio: (batch, 7)
            touch: (batch, 8)
            genome: (batch, 100) - optional genome vector

        Returns:
            predictions: (batch, 343) - predicted next sensory state
            reward_preds: (batch, 5) - predicted upcoming rewards
            actions: (batch, 5) - action outputs
        """
        # Encode each modality
        vision_enc = self.vision_encoder(vision)      # (B, 32, hidden_dim)
        audio_enc = self.audio_encoder(audio)          # (B, 100, hidden_dim)
        proprio_enc = self.proprio_encoder(proprio)    # (B, 16, hidden_dim)
        touch_enc = self.touch_encoder(touch)          # (B, 16, hidden_dim)

        # Concatenate along sequence dimension
        x = torch.cat([vision_enc, audio_enc, proprio_enc, touch_enc], dim=1)
        # x: (B, 164, hidden_dim)

        # Apply genome modulation if available
        if self.genome_modulator is not None and genome is not None:
            x = self.genome_modulator(x, genome)

        # Fourier mixing layers
        for layer in self.mixing_layers:
            x = layer(x)  # (B, 164, hidden_dim)

        # Pooling for output heads
        mean_pool = x.mean(dim=1)  # (B, hidden_dim)
        max_pool, _ = x.max(dim=1)  # (B, hidden_dim)
        last = x[:, -1, :]          # (B, hidden_dim)

        # Concatenate pooled features
        pooled = torch.cat([mean_pool, max_pool, last], dim=1)  # (B, 3*hidden_dim)

        # Output heads
        predictions = self.prediction_head(pooled)
        reward_preds = self.reward_head(pooled)
        actions = self.action_head(pooled)

        return predictions, reward_preds, actions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        reward_preds: torch.Tensor,
        next_sensory: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        actual_rewards: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with multi-task reward prediction.

        Args:
            predictions: (batch, 343) - predicted next sensory state
            reward_preds: (batch, 5) - predicted future rewards
            next_sensory: Dict with actual next sensory state
            actions: (batch, 5)
            actual_rewards: (batch, 5) - actual rewards for next N steps

        Returns:
            Dictionary with loss components
        """
        # Split sensory predictions
        pred_split = self.prediction_head.split_prediction(predictions, self.config)

        # Sensory prediction loss (MSE on each modality)
        vision_loss = F.mse_loss(pred_split['vision'], next_sensory['vision'])
        audio_loss = F.mse_loss(pred_split['audio'], next_sensory['audio'])
        proprio_loss = F.mse_loss(pred_split['proprio'], next_sensory['proprio'])
        touch_loss = F.mse_loss(pred_split['touch'], next_sensory['touch'])

        sensory_loss = vision_loss + audio_loss + proprio_loss + touch_loss

        # REWARD PREDICTION LOSS (creates survival gradient!)
        if actual_rewards is not None:
            # MSE between predicted and actual rewards
            reward_loss = F.mse_loss(reward_preds, actual_rewards)
        else:
            reward_loss = torch.tensor(0.0, device=actions.device)

        # Combined loss: sensory + reward prediction
        # Both contribute to learning representations that understand
        # the world AND survival value
        total_loss = sensory_loss + self.config.reward_loss_weight * reward_loss

        return {
            'total': total_loss,
            'sensory': sensory_loss,
            'reward': reward_loss,
            'vision': vision_loss,
            'audio': audio_loss,
            'proprio': proprio_loss,
            'touch': touch_loss
        }
