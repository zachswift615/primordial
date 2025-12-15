"""Loss functions for SPARC articulatory training.

Losses:
- EMA MSE: Articulator position accuracy (most important)
- Pitch MSE: Prosody melody matching (with unvoiced masking)
- Loudness MSE: Energy envelope matching
- Smoothness: Temporal regularization to prevent jitter
"""
import torch
import torch.nn.functional as F
from typing import Dict

from .config import SpeechConfig


def ema_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """MSE loss on EMA articulator positions.

    Args:
        pred: (batch, n_frames, 12) predicted EMA
        target: (batch, n_frames, 12) target EMA
        mask: (batch, n_frames) optional frame mask

    Returns:
        Scalar loss
    """
    if mask is not None:
        # Expand mask to match EMA dimensions
        mask = mask.unsqueeze(-1).expand_as(pred)
        diff = (pred - target) ** 2
        return (diff * mask).sum() / mask.sum().clamp(min=1)
    else:
        return F.mse_loss(pred, target)


def pitch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    unvoiced_threshold: float = 50.0,
) -> torch.Tensor:
    """MSE loss on pitch (F0), masking unvoiced frames.

    Args:
        pred: (batch, n_frames, 1) predicted pitch in Hz
        target: (batch, n_frames, 1) target pitch in Hz
        unvoiced_threshold: Frames with target < this are considered unvoiced

    Returns:
        Scalar loss
    """
    # Create mask for voiced frames
    voiced_mask = target > unvoiced_threshold

    if voiced_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Compute MSE only on voiced frames
    diff = (pred - target) ** 2
    return (diff * voiced_mask).sum() / voiced_mask.sum()


def loudness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on loudness envelope.

    Args:
        pred: (batch, n_frames, 1) predicted loudness in [0, 1]
        target: (batch, n_frames, 1) target loudness in [0, 1]

    Returns:
        Scalar loss
    """
    return F.mse_loss(pred, target)


def smoothness_loss(
    trajectory: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Temporal smoothness regularization to prevent jittery output.

    Penalizes rapid frame-to-frame changes in the trajectory.

    Args:
        trajectory: (batch, n_frames, features) any temporal sequence
        order: 1 for velocity penalty, 2 for acceleration penalty

    Returns:
        Scalar loss
    """
    if order == 1:
        # First-order: penalize velocity (frame differences)
        diff = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        return (diff ** 2).mean()
    elif order == 2:
        # Second-order: penalize acceleration (difference of differences)
        diff1 = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
        return (diff2 ** 2).mean()
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


def sparc_combined_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    config: SpeechConfig,
) -> Dict[str, torch.Tensor]:
    """Combined SPARC loss with all components.

    Args:
        pred: Dict with 'ema', 'pitch', 'loudness' predictions
        target: Dict with 'ema', 'pitch', 'loudness' targets
        config: SpeechConfig with loss weights

    Returns:
        Dict with 'total', 'ema', 'pitch', 'loudness', 'smoothness' losses
    """
    # Individual losses
    loss_ema = ema_loss(pred['ema'], target['ema'])
    loss_pitch = pitch_loss(pred['pitch'], target['pitch'])
    loss_loudness = loudness_loss(pred['loudness'], target['loudness'])
    loss_smooth = smoothness_loss(pred['ema'])

    # Weighted combination
    total = (
        config.ema_loss_weight * loss_ema +
        config.sparc_pitch_loss_weight * loss_pitch +
        config.sparc_loudness_loss_weight * loss_loudness +
        config.smoothness_loss_weight * loss_smooth
    )

    return {
        'total': total,
        'ema': loss_ema,
        'pitch': loss_pitch,
        'loudness': loss_loudness,
        'smoothness': loss_smooth,
    }
