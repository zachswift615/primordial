"""Main wrapper for Minecraft agent training."""
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import time

from .config import MinecraftConfig
from .observation import ObservationProcessor
from .actions import ActionConverter
from .rewards import MinecraftRewardComputer, RewardInfo

from primordial.lrn.lrn_config import LRNConfig
from primordial.lrn.architecture import LivingResonanceNetwork
from primordial.learning.optimizer import RewardModulatedOptimizer
from primordial.learning.stability import GradientClipper, ExponentialMovingAverage


class MinecraftAgentWrapper:
    """Wraps LRN model for training in Minecraft environments.

    Provides the same interface as AgentWrapper but for Minecraft:
    - step() performs sense -> think -> act -> learn cycle
    - Handles observation processing and action conversion
    - Computes shaped rewards from game state
    """

    def __init__(
        self,
        config: MinecraftConfig,
        env: Optional[Any] = None,
        device: str = "cpu",
    ):
        """
        Args:
            config: Minecraft configuration
            env: MineRL/MineDojo environment (created if None)
            device: Device to run model on ("cpu", "cuda", "mps")
        """
        self.config = config
        self.device = device

        # Create environment if not provided
        if env is None:
            self.env = self._create_environment()
        else:
            self.env = env

        # Create LRN model with Minecraft configuration
        lrn_config = LRNConfig(
            environment="minecraft",
            mc_rgb_size=config.rgb_size,
        )
        self.model = LivingResonanceNetwork(lrn_config).to(device)

        # Observation and action processors
        self.obs_processor = ObservationProcessor(rgb_size=config.rgb_size)
        self.action_converter = ActionConverter()
        self.reward_computer = MinecraftRewardComputer(
            health_scale=config.health_reward_scale,
            food_scale=config.food_reward_scale,
            navigation_scale=config.navigation_reward_scale,
            movement_bonus=config.movement_bonus,
            idle_penalty=config.idle_penalty,
            death_penalty=config.death_penalty,
        )

        # Optimizer and stability
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        self.reward_modulated_optimizer = RewardModulatedOptimizer(
            self.optimizer,
            reward_scale=0.1
        )
        self.gradient_clipper = GradientClipper(max_norm=1.0)
        self.ema = ExponentialMovingAverage(self.model, decay=0.999)

        # State tracking
        self.prev_vision = None
        self.prev_audio = None
        self.prev_proprio = None
        self.prev_touch = None
        self.prev_prediction = None

        # Metrics
        self.step_count = 0
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_steps = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_environment(self):
        """Create MineRL environment."""
        try:
            import minerl
            import gym
            env = gym.make(self.config.env_name)
            return env
        except ImportError:
            raise ImportError(
                "MineRL not installed. Install with: pip install minerl\n"
                "Also ensure you have Java 8 installed."
            )

    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode.

        Returns:
            Initial observation
        """
        obs = self.env.reset()
        self.reward_computer.reset()

        # Reset state
        self.prev_vision = None
        self.prev_audio = None
        self.prev_proprio = None
        self.prev_touch = None
        self.prev_prediction = None

        # Reset episode metrics
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_count += 1

        return obs

    def step(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool, Dict[str, Any]]:
        """Perform one step: sense -> think -> learn -> act.

        Args:
            obs: Current observation from environment

        Returns:
            action: Action dict to send to environment
            metrics: Learning metrics for this step
        """
        metrics = {}
        self.step_count += 1
        self.episode_steps += 1

        # 1. SENSE: Process observation
        vision, audio, proprio, touch = self.obs_processor.process(obs)
        vision = vision.to(self.device)
        audio = audio.to(self.device)
        proprio = proprio.to(self.device)
        touch = touch.to(self.device)

        # 2. LEARN: If we have previous state, compute loss and update
        if self.prev_prediction is not None:
            # Compute prediction loss
            # Flatten current sensory for comparison
            current_flat = torch.cat([
                vision.flatten(1),
                audio.flatten(1),
                proprio,
                touch
            ], dim=1)

            # Prediction loss (MSE)
            pred_loss = F.mse_loss(self.prev_prediction, current_flat)

            # Backward pass
            self.optimizer.zero_grad()
            pred_loss.backward()

            # Gradient clipping
            grad_norm = self.gradient_clipper.clip(self.model)

            # Optimizer step (reward modulation applied externally)
            self.optimizer.step()

            # Update EMA
            self.ema.update()

            metrics['pred_loss'] = pred_loss.item()
            metrics['grad_norm'] = grad_norm

        # 3. THINK: Forward pass to get action and prediction
        with torch.no_grad():
            self.ema.apply_shadow()
            _, _, action_tensor = self.model(vision, audio, proprio, touch)
            self.ema.restore()

        # Get prediction WITH gradients for next learning step
        prediction, reward_pred, _ = self.model(vision, audio, proprio, touch)

        # 4. Store for next step
        self.prev_vision = vision
        self.prev_audio = audio
        self.prev_proprio = proprio
        self.prev_touch = touch
        self.prev_prediction = prediction

        # 5. ACT: Convert to Minecraft action
        action = self.action_converter.convert(action_tensor)

        # Add exploration noise early in training
        if self.step_count < 10000:
            noise_scale = 0.3 * (1.0 - self.step_count / 10000)
            action = self.action_converter.add_exploration_noise(action, noise_scale)

        metrics['step'] = self.step_count
        metrics['episode'] = self.episode_count

        return action, metrics

    def on_step_result(
        self,
        obs: Dict[str, Any],
        env_reward: float,
        done: bool,
        info: Dict[str, Any]
    ) -> RewardInfo:
        """Process step result and compute shaped reward.

        Call this after env.step() with the results.

        Args:
            obs: New observation
            env_reward: Reward from environment
            done: Whether episode ended
            info: Additional info

        Returns:
            RewardInfo with breakdown
        """
        reward_info = self.reward_computer.compute(obs, env_reward, done, info)
        self.episode_reward += reward_info.total
        return reward_info

    def run_episode(self, render: bool = True, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run a complete episode.

        Args:
            render: Whether to render the game window
            max_steps: Maximum steps (uses config default if None)

        Returns:
            Episode statistics
        """
        max_steps = max_steps or self.config.max_episode_steps
        obs = self.reset()

        start_time = time.time()

        for step in range(max_steps):
            # Get action from model
            action, metrics = self.step(obs)

            # Execute in environment
            next_obs, env_reward, done, info = self.env.step(action)

            # Compute shaped reward
            reward_info = self.on_step_result(next_obs, env_reward, done, info)

            # Render if requested
            if render:
                self.env.render()

            # Log periodically
            if self.config.verbose and step % self.config.log_every_n_steps == 0:
                health = self.obs_processor.get_health(next_obs)
                food = self.obs_processor.get_food(next_obs)
                print(f"Step {step}: reward={reward_info.total:.3f}, "
                      f"health={health:.2f}, food={food:.2f}, "
                      f"loss={metrics.get('pred_loss', 0):.4f}")

            obs = next_obs

            if done:
                break

        elapsed = time.time() - start_time

        return {
            'episode': self.episode_count,
            'steps': self.episode_steps,
            'reward': self.episode_reward,
            'fps': self.episode_steps / elapsed,
            'elapsed': elapsed,
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model and training state."""
        if path is None:
            path = self.checkpoint_dir / f"minecraft_agent_ep{self.episode_count}.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config,
        }, path)

        if self.config.verbose:
            print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']

        if self.config.verbose:
            print(f"Loaded checkpoint from {path}")

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()
