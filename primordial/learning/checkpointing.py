"""Death handling and checkpointing for online learning."""

import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn


class DeathHandler:
    """Handle agent death in online learning.

    When agent dies:
    1. Save current model weights to checkpoint
    2. Reset optimizer state (clear momentum, etc.)
    3. Optionally reduce learning rate
    4. Respawn agent with same weights
    """

    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        reset_optimizer: bool = True,
        lr_reduction_factor: float = 0.5,
        min_lr: float = 1e-6
    ):
        """
        Args:
            checkpoint_dir: Directory to save death checkpoints
            reset_optimizer: Whether to reset optimizer state on death
            lr_reduction_factor: Reduce LR by this factor on death
            min_lr: Minimum learning rate
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.reset_optimizer = reset_optimizer
        self.lr_reduction_factor = lr_reduction_factor
        self.min_lr = min_lr
        self.death_count = 0

    def on_death(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler=None
    ) -> Dict:
        """Called when agent dies.

        Args:
            model: Agent's neural network
            optimizer: Optimizer instance
            lr_scheduler: Optional learning rate scheduler

        Returns:
            dict with death_count and checkpoint_path
        """
        self.death_count += 1

        # 1. Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'death_{self.death_count}.pt'
        torch.save({
            'death_count': self.death_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        # 2. Reset optimizer state (clear momentum)
        if self.reset_optimizer:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    # Reset Adam momentum
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()

        # 3. Reduce learning rate
        if lr_scheduler is not None:
            for group in optimizer.param_groups:
                current_lr = group['lr']
                new_lr = max(
                    current_lr * self.lr_reduction_factor,
                    self.min_lr
                )
                group['lr'] = new_lr

        return {
            'death_count': self.death_count,
            'checkpoint_path': str(checkpoint_path)
        }

    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Optional[Dict]:
        """Load most recent death checkpoint.

        Args:
            model: Agent's neural network
            optimizer: Optimizer instance

        Returns:
            checkpoint dict if found, None otherwise
        """
        checkpoints = sorted(self.checkpoint_dir.glob('death_*.pt'))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        checkpoint = torch.load(latest)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint


class DeathReplay:
    """Replay recent experiences on death before respawn.

    Optional: Helps consolidate recent learning by replaying
    experiences that led to death.
    """

    def __init__(
        self,
        replay_buffer_size: int = 100,
        replay_iterations: int = 10
    ):
        """
        Args:
            replay_buffer_size: Number of recent experiences to keep
            replay_iterations: How many times to replay on death
        """
        self.buffer = []
        self.buffer_size = replay_buffer_size
        self.replay_iterations = replay_iterations

    def add_experience(
        self,
        senses: torch.Tensor,
        action: torch.Tensor,
        prediction: torch.Tensor,
        next_senses: torch.Tensor,
        reward: float
    ):
        """Add experience to replay buffer.

        Args:
            senses: Current sensory input
            action: Action taken
            prediction: Model's prediction
            next_senses: Actual next sensory state
            reward: Reward received
        """
        self.buffer.append({
            'senses': senses.detach().clone(),
            'action': action.detach().clone(),
            'prediction': prediction.detach().clone(),
            'next_senses': next_senses.detach().clone(),
            'reward': reward
        })

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def replay_on_death(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module
    ):
        """Replay recent experiences when agent dies.

        Args:
            model: Agent's neural network
            optimizer: Optimizer instance
            loss_fn: Loss function
        """
        if len(self.buffer) < 10:
            return  # Not enough experiences

        for _ in range(self.replay_iterations):
            # Sample random experience
            exp = random.choice(self.buffer)

            # Re-forward through model to get fresh prediction with gradients
            prediction = model(exp['senses'])

            # Recompute loss and update
            loss = loss_fn(prediction, exp['next_senses'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def clear(self):
        """Clear replay buffer."""
        self.buffer.clear()
