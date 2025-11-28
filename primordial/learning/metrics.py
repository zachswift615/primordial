"""Learning metrics and visualization for online learning."""

from typing import Dict, Optional

import numpy as np
import torch


class LearningMetrics:
    """Track and log online learning metrics."""

    def __init__(self, log_interval: int = 10):
        """
        Args:
            log_interval: Log metrics every N steps
        """
        self.log_interval = log_interval
        self.step_count = 0

        # Metrics buffers
        self.prediction_losses = []
        self.rewards_survival = []
        self.rewards_teaching = []
        self.rewards_total = []
        self.gradient_norms = []
        self.modulation_factors = []
        self.learning_rates = []

    def record_step(
        self,
        prediction_loss: float,
        survival_reward: float,
        teaching_reward: float,
        total_reward: float,
        gradient_norm: float,
        modulation_factor: float,
        learning_rate: float
    ):
        """Record metrics for one step.

        Args:
            prediction_loss: Prediction loss value
            survival_reward: Survival reward component
            teaching_reward: Human teaching reward component
            total_reward: Combined total reward
            gradient_norm: Gradient norm
            modulation_factor: Reward modulation factor
            learning_rate: Current learning rate
        """
        self.step_count += 1

        self.prediction_losses.append(prediction_loss)
        self.rewards_survival.append(survival_reward)
        self.rewards_teaching.append(teaching_reward)
        self.rewards_total.append(total_reward)
        self.gradient_norms.append(gradient_norm)
        self.modulation_factors.append(modulation_factor)
        self.learning_rates.append(learning_rate)

    def should_log(self) -> bool:
        """Check if we should log now.

        Returns:
            True if it's time to log metrics
        """
        return self.step_count % self.log_interval == 0

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics.

        Returns:
            dict of metric name -> value
        """
        if not self.prediction_losses:
            return {}

        return {
            'step': self.step_count,
            'loss/prediction_mean': np.mean(self.prediction_losses),
            'loss/prediction_std': np.std(self.prediction_losses),
            'reward/survival_mean': np.mean(self.rewards_survival),
            'reward/teaching_mean': np.mean(self.rewards_teaching),
            'reward/total_mean': np.mean(self.rewards_total),
            'training/gradient_norm_mean': np.mean(self.gradient_norms),
            'training/modulation_factor_mean': np.mean(self.modulation_factors),
            'training/learning_rate': self.learning_rates[-1] if self.learning_rates else 0.0,
        }

    def clear_buffers(self):
        """Clear metric buffers after logging."""
        self.prediction_losses.clear()
        self.rewards_survival.clear()
        self.rewards_teaching.clear()
        self.rewards_total.clear()
        self.gradient_norms.clear()
        self.modulation_factors.clear()
        self.learning_rates.clear()


class LearningVisualizer:
    """Visualize learning progress in real-time.

    Supports TensorBoard and Weights & Biases logging.
    Both are optional dependencies.
    """

    def __init__(
        self,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        log_dir: str = 'runs/online_learning'
    ):
        """
        Args:
            use_tensorboard: Log to TensorBoard
            use_wandb: Log to Weights & Biases
            log_dir: Directory for TensorBoard logs
        """
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.tb_writer = None
        self.wandb = None

        # Try to import TensorBoard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False

        # Try to import wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project='primordial', name='online_learning')
            except ImportError:
                print("Warning: Weights & Biases not available. Install with: pip install wandb")
                self.use_wandb = False

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to configured backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Current training step
        """
        if self.use_tensorboard and self.tb_writer is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def log_model_predictions(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
        step: int
    ):
        """Log prediction visualizations.

        Args:
            predicted: Predicted values
            actual: Actual values
            step: Current training step
        """
        if self.use_tensorboard and self.tb_writer is not None:
            # Log prediction error histogram
            error = (predicted - actual).abs()
            self.tb_writer.add_histogram('predictions/error', error, step)

    def close(self):
        """Close logging backends."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
