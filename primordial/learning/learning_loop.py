"""
Main online learning loop for Primordial agents.

Integrates loss functions, optimizers, rewards, stability measures, and metrics
into a cohesive continual learning system.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from .losses import PredictionLoss
from .optimizer import RewardModulatedOptimizer, OnlineLRScheduler
from .rewards import RewardCombiner
from .stability import GradientClipper, GradientMonitor, ExponentialMovingAverage


class OnlineLearningLoop:
    """
    Main online learning loop for Primordial agents.

    Coordinates prediction-based learning with reward modulation,
    stability measures, and death handling for continual learning.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: Dict[str, Any],
        loss_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        stability_config: Optional[Dict[str, Any]] = None,
        death_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize online learning loop.

        Args:
            model: Agent's neural network
            optimizer_config: Optimizer configuration dict
            loss_config: Loss function configuration (optional)
            reward_config: Reward system configuration (optional)
            stability_config: Stability measures configuration (optional)
            death_config: Death handling configuration (optional)
        """
        self.model = model

        # Use defaults if not provided
        loss_config = loss_config or {}
        reward_config = reward_config or {}
        stability_config = stability_config or {}
        death_config = death_config or {}

        # Initialize components
        self.loss_fn = self._create_loss_fn(loss_config)
        self.base_optimizer = self._create_optimizer(optimizer_config)

        modulation_config = reward_config.get('modulation', {})
        self.optimizer = RewardModulatedOptimizer(
            self.base_optimizer,
            **modulation_config
        )

        combiner_config = reward_config.get('combiner', {})
        self.reward_combiner = RewardCombiner(**combiner_config)

        lr_schedule_config = optimizer_config.get('lr_schedule', {})
        self.lr_scheduler = OnlineLRScheduler(
            self.base_optimizer,
            **lr_schedule_config
        )

        # Stability components
        clipping_config = stability_config.get('clipping', {})
        self.gradient_clipper = GradientClipper(**clipping_config)

        monitoring_config = stability_config.get('monitoring', {})
        self.grad_monitor = GradientMonitor(**monitoring_config)

        ema_config = stability_config.get('ema', {})
        self.ema = ExponentialMovingAverage(model, **ema_config)

        # Death handling
        self.checkpoint_dir = Path(death_config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.reset_optimizer_on_death = death_config.get('reset_optimizer', True)
        self.lr_reduction_factor = death_config.get('lr_reduction_factor', 0.5)
        self.min_lr = death_config.get('min_lr', 1e-6)
        self.death_count = 0

        # State
        self.prev_prediction = None
        self.prev_state = None
        self.step_count = 0

    def step(
        self,
        senses: torch.Tensor,
        prev_senses: torch.Tensor,
        agent_state: Any,
        prev_agent_state: Any,
        events: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of online learning.

        Args:
            senses: Current sensory input (batch=1, sense_dim)
            prev_senses: Previous sensory input
            agent_state: Current agent state (health, energy, etc.)
            prev_agent_state: Previous agent state
            events: List of events that occurred this step

        Returns:
            action: Selected action
            prediction: Predicted next senses
        """
        self.step_count += 1

        # 1. Learn from previous prediction (if available)
        if self.prev_prediction is not None:
            # Compute prediction loss
            loss = self.loss_fn(self.prev_prediction, senses)

            # Compute reward
            total_reward, survival_reward, teaching_reward = \
                self.reward_combiner.compute_total_reward(
                    prev_agent_state, agent_state, events
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = self.gradient_clipper.clip(self.model)

            # Reward-modulated optimizer step
            modulation_factor = self.optimizer.step(total_reward)

            # Update EMA
            self.ema.update()

            # Learning rate schedule
            lr = self.lr_scheduler.step()

            # Record metrics
            self.grad_monitor.record(self.model)

        # 2. Forward pass to get action (inference with EMA) and prediction (for learning)
        # Get action using EMA weights (no grad for action)
        with torch.no_grad():
            self.ema.apply_shadow()  # Use EMA weights for inference
            output = self.model(senses)
            self.ema.restore()

            # Handle both tuple (action, prediction) or single output
            if isinstance(output, tuple):
                action, _ = output
            else:
                # For simple test models that only return prediction
                action = output

        # Get prediction WITH gradients for learning
        output = self.model(senses)
        if isinstance(output, tuple):
            _, prediction = output
        else:
            prediction = output

        # 3. Store for next iteration
        self.prev_prediction = prediction
        self.prev_state = agent_state

        return action, prediction

    def on_death(self) -> Dict[str, Any]:
        """
        Handle agent death.

        Saves checkpoint, optionally resets optimizer state, and reduces learning rate.

        Returns:
            Dict with death count and checkpoint path
        """
        self.death_count += 1

        # 1. Save checkpoint
        checkpoint_path = self.checkpoint_dir / f'death_{self.death_count}.pt'
        torch.save({
            'death_count': self.death_count,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.base_optimizer.state_dict(),
        }, checkpoint_path)

        # 2. Reset optimizer state (clear momentum)
        if self.reset_optimizer_on_death:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    state = self.base_optimizer.state[p]
                    # Reset Adam momentum
                    if 'exp_avg' in state:
                        state['exp_avg'].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'].zero_()

        # 3. Reduce learning rate
        for group in self.base_optimizer.param_groups:
            current_lr = group['lr']
            new_lr = max(
                current_lr * self.lr_reduction_factor,
                self.min_lr
            )
            group['lr'] = new_lr

        # Reset state
        self.prev_prediction = None
        self.prev_state = None

        return {
            'death_count': self.death_count,
            'checkpoint_path': str(checkpoint_path)
        }

    def save_checkpoint(self, path: str):
        """
        Save full learning state.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.base_optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'step_count': self.step_count,
            'death_count': self.death_count,
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load full learning state.

        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.step_count = checkpoint['step_count']
        self.death_count = checkpoint.get('death_count', 0)

    def _create_loss_fn(self, config: Dict[str, Any]) -> nn.Module:
        """Create loss function from config."""
        loss_type = config.get('type', 'mse')
        if loss_type == 'mse':
            return PredictionLoss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _create_optimizer(self, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_type = config.get('type', 'adamw')
        params = config.get('params', {})

        # Set defaults if not provided
        if 'lr' not in params:
            params['lr'] = 1e-4

        if opt_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                **params
            )
        elif opt_type == 'rmsprop':
            return torch.optim.RMSprop(
                self.model.parameters(),
                **params
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
