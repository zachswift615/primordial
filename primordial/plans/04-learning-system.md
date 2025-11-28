# Online Learning System Implementation Plan

## Overview

The Online Learning System implements continual learning for Primordial agents through real-time experience, without batch training. Agents learn by minimizing prediction error while survival and human teaching signals modulate the learning process.

### Core Principles

1. **Multi-Task Prediction**: Agents learn by predicting BOTH future sensory input AND future rewards
2. **Dual Learning Signals**: Sensory prediction teaches world dynamics; reward prediction teaches survival value
3. **Online Updates**: Single-sample gradient updates after each action
4. **Reward History Tracking**: Maintains buffer of recent rewards as prediction targets
5. **Catastrophic Forgetting Mitigation**: Stabilization techniques for continuous learning

### Key Innovation: Multi-Task Reward Prediction

Unlike previous predictive coding approaches that only minimize sensory prediction error,
our system predicts upcoming rewards (both positive and negative). This creates a **direct
gradient toward survival** — the agent learns representations that encode not just "what
will I sense?" but "will this help or hurt me?"

```
Sensory Prediction → Learns world dynamics ("fire is orange and flickering")
Reward Prediction  → Learns survival value ("fire causes pain")
```

Both losses contribute equally to learning, creating representations that understand
reality AND its survival implications.

### Key Research Foundation

- **Continual Learning**: https://arxiv.org/html/2403.05175v1
- **Predictive Coding**: Minimizing prediction error as primary learning signal
- **Online Gradient Descent**: Single-sample optimization in non-stationary environments

### Design Goals

- Stable learning over hours of continuous operation
- Immediate adaptation to environment changes
- Integration of multiple reward signals (survival + human teaching)
- Graceful handling of agent death and respawn
- Observable learning progress through metrics

## File Structure

```
primordial/
├── learning/
│   ├── __init__.py
│   ├── losses.py                    # Loss function implementations
│   ├── rewards.py                   # Reward computation and modulation
│   ├── optimizer.py                 # Custom optimizer with gradient modulation
│   ├── learning_loop.py             # Main online learning loop
│   ├── stability.py                 # Gradient clipping, normalization, etc.
│   ├── metrics.py                   # Learning metrics and tracking
│   └── checkpointing.py             # Death handling and weight persistence
├── config/
│   └── learning_config.yaml         # Hyperparameters and configuration
└── tests/
    └── test_learning/
        ├── test_losses.py
        ├── test_rewards.py
        ├── test_stability.py
        └── test_integration.py
```

## Loss Functions

### Multi-Task Loss (Primary)

The fundamental learning signal combines sensory prediction AND reward prediction:

```python
# Total loss = Sensory Prediction + Reward Prediction
L_total = L_sensory + λ * L_reward

Where:
  L_sensory = MSE(predicted_senses, actual_senses)
  L_reward = MSE(predicted_rewards, actual_rewards)
  λ = reward_loss_weight (default: 1.0, equal weight)
```

### Component 1: Sensory Prediction Loss

Predicts next sensory state (teaches world dynamics):

```python
# Sensory prediction loss (MSE)
L_sensory = (1/N) * Σ(s_predicted[i] - s_actual[i])²

Where:
  s_predicted = model's prediction of next sensory state
  s_actual = actual sensory state after action
  N = number of sensory dimensions
```

### Component 2: Reward Prediction Loss (NEW - Creates Survival Gradient)

Predicts upcoming rewards over a horizon (teaches survival value):

```python
# Reward prediction loss (MSE over horizon)
L_reward = (1/H) * Σ(r_predicted[t] - r_actual[t])²

Where:
  r_predicted = model's prediction of rewards for t+1 to t+H
  r_actual = actual rewards that occurred (from history buffer)
  H = reward_horizon (default: 5 steps)

Example:
  Predicted: [-0.5, -1.0, -0.5, 0.0, 0.0]  # "pain coming soon"
  Actual:    [-2.0, 0.0, 0.0, 0.0, 0.0]    # predator hit at t+1
  L_reward = MSE = 0.85
```

**Why this creates a survival gradient:**
- If agent predicts reward incorrectly, the loss backpropagates through the network
- Representations that better predict rewards → lower loss → reinforced
- "Seeing predator" pattern becomes associated with "negative reward prediction"
- This is exactly how dopamine prediction error works in biological brains

**Implementation**:

```python
class PredictionLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, predicted_senses, actual_senses):
        """
        Args:
            predicted_senses: (batch=1, sense_dim)
            actual_senses: (batch=1, sense_dim)
        Returns:
            scalar loss
        """
        return self.mse(predicted_senses, actual_senses)
```

### Alternative Loss Functions

**1. Huber Loss (Robust to Outliers)**

```python
L_huber = {
    0.5 * (y - ŷ)²           if |y - ŷ| ≤ δ
    δ * (|y - ŷ| - 0.5*δ)    otherwise
}

# More stable for noisy single-sample updates
loss = nn.SmoothL1Loss(beta=1.0)
```

**2. Per-Sense Weighted Loss**

```python
# Weight different sensory modalities differently
L_weighted = Σ w_i * (s_predicted[i] - s_actual[i])²

weights = {
    'vision': 1.0,
    'proprioception': 2.0,  # More important for survival
    'energy': 3.0,           # Critical signal
    'damage': 3.0
}
```

**3. Temporal Difference Loss (Optional)**

```python
# For multi-step predictions
L_td = Σ γ^t * (s_predicted[t] - s_actual[t])²

Where:
  γ = discount factor (0.95-0.99)
  t = prediction horizon (1-5 steps)
```

### Loss Configuration

```python
class LossConfig:
    loss_type: str = 'mse'  # 'mse', 'huber', 'weighted'
    huber_delta: float = 1.0
    sense_weights: Dict[str, float] = None
    use_temporal_difference: bool = False
    td_gamma: float = 0.97
    td_horizon: int = 3
```

## Reward System

### Reward History Buffer (NEW - For Multi-Task Learning)

To train the RewardHead, we need to know what rewards *actually* occurred after
each prediction. This requires a history buffer:

```python
class RewardHistoryBuffer:
    """
    Tracks reward history for multi-task reward prediction.

    When agent makes a prediction at time t, it predicts rewards for
    t+1, t+2, ..., t+horizon. We need to store actual rewards to compute
    the prediction loss once those timesteps occur.

    Uses dict for O(1) reward lookup instead of O(n) deque iteration.
    Handles stale predictions that exceed max age.
    """

    def __init__(
        self,
        horizon: int = 5,
        max_pending: int = 100,
        max_stale_steps: int = 50  # Discard predictions older than this
    ):
        self.horizon = horizon
        self.max_pending = max_pending
        self.max_stale_steps = max_stale_steps

        # Dict for O(1) reward lookup: step -> reward
        self.reward_history: Dict[int, float] = {}

        # Pending predictions awaiting actual rewards
        self.pending_predictions: List[Dict] = []

        # Track oldest step for cleanup
        self._oldest_step = 0

    def record_prediction(self, step: int, reward_preds: torch.Tensor):
        """
        Record a reward prediction for later loss computation.

        Args:
            step: Current timestep
            reward_preds: (horizon,) predicted rewards for next H steps
        """
        self.pending_predictions.append({
            'step': step,
            'predictions': reward_preds.detach().clone(),
            'steps_remaining': self.horizon
        })

        # Enforce max_pending limit
        if len(self.pending_predictions) > self.max_pending:
            self.pending_predictions.pop(0)

    def record_actual_reward(self, step: int, reward: float):
        """
        Record an actual reward that occurred. O(1) insertion.

        Args:
            step: Current timestep
            reward: Actual reward value
        """
        self.reward_history[step] = reward

        # Cleanup old entries to prevent unbounded growth
        self._cleanup_old_entries(step)

    def _cleanup_old_entries(self, current_step: int):
        """Remove reward history entries older than needed."""
        # Keep rewards from (current_step - horizon - max_stale_steps) onwards
        cutoff = current_step - self.horizon - self.max_stale_steps

        if cutoff > self._oldest_step:
            # Remove old entries
            old_keys = [k for k in self.reward_history if k < cutoff]
            for k in old_keys:
                del self.reward_history[k]
            self._oldest_step = cutoff

    def get_ready_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get prediction/actual pairs ready for loss computation.

        Returns predictions that now have enough actual reward history
        to compute loss against. Discards stale predictions.

        Returns:
            List of (predicted_rewards, actual_rewards) tensors
        """
        ready_pairs = []
        remaining = []

        for pending in self.pending_predictions:
            pending['steps_remaining'] -= 1
            age = self.horizon - pending['steps_remaining']

            # Check for stale predictions (too old, discard)
            if age > self.max_stale_steps:
                # Prediction is stale, skip without computing loss
                continue

            if pending['steps_remaining'] <= 0:
                # This prediction has waited long enough
                # Gather actual rewards using O(1) dict lookup
                pred_step = pending['step']
                actual_rewards = []

                for i in range(1, self.horizon + 1):
                    target_step = pred_step + i
                    # O(1) lookup instead of O(n) iteration
                    reward = self.reward_history.get(target_step, 0.0)
                    actual_rewards.append(reward)

                ready_pairs.append((
                    pending['predictions'],
                    torch.tensor(actual_rewards)
                ))
            else:
                remaining.append(pending)

        self.pending_predictions = remaining
        return ready_pairs

    def on_death(self):
        """
        Clear buffer state on agent death.

        Stale predictions from before death should not affect
        learning after respawn.
        """
        self.pending_predictions.clear()
        self.reward_history.clear()
        self._oldest_step = 0
```

**How it works:**
1. At step t, agent predicts rewards for steps t+1, t+2, ..., t+H
2. Buffer stores this prediction
3. As steps pass, actual rewards are recorded
4. After H steps, we have actual rewards for t+1 through t+H
5. Now we can compute MSE(predicted, actual) for that prediction
6. This loss backprops to improve future predictions

### Survival Rewards

Automatically generated from environment interactions:

```python
class SurvivalRewards:
    """Intrinsic reward signals from survival events"""

    # Event-based rewards
    EATING_FOOD = +1.0
    TAKING_DAMAGE = -2.0
    DEATH = -10.0

    # Continuous rewards (per step)
    STARVING = -0.1        # When energy < 30%
    LOW_HEALTH = -0.05     # When health < 50%
    HEALTHY = +0.01        # When health > 80% and energy > 50%

    @staticmethod
    def compute_reward(prev_state, current_state, events):
        """
        Args:
            prev_state: AgentState before action
            current_state: AgentState after action
            events: List of events that occurred

        Returns:
            float: total survival reward
        """
        reward = 0.0

        # Event-based
        if 'ate_food' in events:
            reward += SurvivalRewards.EATING_FOOD
        if 'took_damage' in events:
            reward += SurvivalRewards.TAKING_DAMAGE
        if 'died' in events:
            reward += SurvivalRewards.DEATH

        # Continuous
        if current_state.energy < 0.3 * current_state.max_energy:
            reward += SurvivalRewards.STARVING
        if current_state.health < 0.5 * current_state.max_health:
            reward += SurvivalRewards.LOW_HEALTH
        if (current_state.health > 0.8 * current_state.max_health and
            current_state.energy > 0.5 * current_state.max_energy):
            reward += SurvivalRewards.HEALTHY

        return reward
```

### Human Teaching Signal

Direct feedback from human observers:

```python
class HumanTeaching:
    """Human-provided reward signals"""

    REWARD_BUTTON = +1.0
    PUNISH_BUTTON = -1.0

    def __init__(self, window_size=10):
        """
        Args:
            window_size: Number of steps to apply human feedback
        """
        self.pending_rewards = []
        self.window_size = window_size

    def add_teaching_signal(self, reward_value):
        """Called when human presses reward/punish button"""
        self.pending_rewards.append({
            'value': reward_value,
            'steps_remaining': self.window_size
        })

    def get_current_reward(self):
        """Get aggregated human teaching reward for current step"""
        total = 0.0

        # Update and sum all pending rewards
        active_rewards = []
        for r in self.pending_rewards:
            if r['steps_remaining'] > 0:
                total += r['value'] / self.window_size
                r['steps_remaining'] -= 1
                active_rewards.append(r)

        self.pending_rewards = active_rewards
        return total
```

### Combined Reward Signal

```python
class RewardCombiner:
    """Combines survival and human teaching rewards"""

    def __init__(self, survival_weight=1.0, teaching_weight=1.5):
        """
        Args:
            survival_weight: Scaling factor for survival rewards
            teaching_weight: Scaling factor for human teaching
                           (higher = human feedback more important)
        """
        self.survival_weight = survival_weight
        self.teaching_weight = teaching_weight
        self.human_teaching = HumanTeaching()

    def compute_total_reward(self, prev_state, current_state, events):
        """Compute combined reward signal"""
        survival = SurvivalRewards.compute_reward(
            prev_state, current_state, events
        )
        teaching = self.human_teaching.get_current_reward()

        total = (
            self.survival_weight * survival +
            self.teaching_weight * teaching
        )

        return total, survival, teaching  # Return components for logging
```

## Gradient Modulation

### Reward-Modulated Learning

Rewards modulate the effective learning rate, not the loss directly:

```python
# Core equation:
effective_lr = base_lr * modulation_factor(reward)

# Modulation factor options:

# Option 1: Linear scaling
modulation_factor = 1.0 + α * reward
  where α = reward_scale (e.g., 0.1)

# Option 2: Sigmoid scaling (bounded)
modulation_factor = σ(β * reward)
  where β = reward_sensitivity (e.g., 2.0)

# Option 3: Exponential scaling
modulation_factor = exp(γ * reward)
  where γ = reward_coefficient (e.g., 0.05)
```

**Mathematical Justification**:

```
Standard gradient update:
θ_{t+1} = θ_t - η * ∇L(θ_t)

Reward-modulated update:
θ_{t+1} = θ_t - η * m(r_t) * ∇L(θ_t)

Where:
  η = base learning rate
  m(r_t) = modulation factor based on reward r_t
  ∇L(θ_t) = gradient of prediction loss

Effect:
- Positive reward (r > 0): m(r) > 1, learn faster
- Negative reward (r < 0): m(r) < 1, learn slower (or unlearn)
- Zero reward (r = 0): m(r) = 1, normal learning
```

### Implementation

```python
class RewardModulatedOptimizer:
    """Optimizer wrapper that modulates gradients by reward"""

    def __init__(
        self,
        optimizer,
        modulation_type='linear',
        reward_scale=0.1,
        min_modulation=0.1,
        max_modulation=3.0
    ):
        """
        Args:
            optimizer: Base PyTorch optimizer
            modulation_type: 'linear', 'sigmoid', or 'exponential'
            reward_scale: Sensitivity to reward
            min_modulation: Lower bound on modulation factor
            max_modulation: Upper bound on modulation factor
        """
        self.optimizer = optimizer
        self.modulation_type = modulation_type
        self.reward_scale = reward_scale
        self.min_modulation = min_modulation
        self.max_modulation = max_modulation

    def compute_modulation(self, reward):
        """Compute gradient modulation factor from reward"""
        if self.modulation_type == 'linear':
            mod = 1.0 + self.reward_scale * reward
        elif self.modulation_type == 'sigmoid':
            mod = torch.sigmoid(torch.tensor(self.reward_scale * reward)).item()
            mod = 0.1 + 2.9 * mod  # Map [0,1] to [0.1, 3.0]
        elif self.modulation_type == 'exponential':
            mod = torch.exp(torch.tensor(self.reward_scale * reward)).item()
        else:
            raise ValueError(f"Unknown modulation type: {self.modulation_type}")

        # Clamp to prevent extreme values
        return np.clip(mod, self.min_modulation, self.max_modulation)

    def step(self, reward):
        """
        Perform optimizer step with reward-modulated gradients

        Args:
            reward: Scalar reward value
        """
        modulation = self.compute_modulation(reward)

        # Scale all gradients by modulation factor
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad *= modulation

        # Perform standard optimizer step
        self.optimizer.step()

        return modulation  # Return for logging

    def zero_grad(self):
        """Pass through to base optimizer"""
        self.optimizer.zero_grad()
```

## Optimizer Configuration

### Base Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,              # Conservative for single-sample updates
    betas=(0.9, 0.999),   # Standard Adam betas
    eps=1e-8,
    weight_decay=1e-5     # Light regularization
)
```

**Rationale**:
- AdamW handles noisy single-sample gradients better than SGD
- Adaptive learning rates per parameter
- Decoupled weight decay prevents interference with reward modulation

### Alternative: RMSprop

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=5e-5,
    alpha=0.99,      # Decay rate for moving average
    eps=1e-8,
    momentum=0.9
)
```

**Use when**: More stable for very noisy environments

### Learning Rate Schedule

```python
class OnlineLRScheduler:
    """Custom LR scheduler for online learning"""

    def __init__(
        self,
        optimizer,
        warmup_steps=1000,
        base_lr=1e-4,
        min_lr=1e-6,
        decay_rate=0.9999  # Very slow decay
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.step_count = 0

    def step(self):
        """Update learning rate"""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.step_count / self.warmup_steps)
        else:
            # Exponential decay
            steps_after_warmup = self.step_count - self.warmup_steps
            lr = self.base_lr * (self.decay_rate ** steps_after_warmup)

        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
```

## Stability Measures

### 1. Gradient Clipping

Essential for single-sample updates:

```python
class GradientClipper:
    """Gradient clipping for stability"""

    def __init__(
        self,
        clip_type='norm',
        max_norm=1.0,
        max_value=10.0
    ):
        """
        Args:
            clip_type: 'norm' or 'value'
            max_norm: Maximum gradient norm (for clip_type='norm')
            max_value: Maximum gradient value (for clip_type='value')
        """
        self.clip_type = clip_type
        self.max_norm = max_norm
        self.max_value = max_value

    def clip(self, model):
        """Clip gradients in-place"""
        if self.clip_type == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_norm
            )
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(
                model.parameters(),
                self.max_value
            )
            grad_norm = self._compute_grad_norm(model)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")

        return grad_norm

    def _compute_grad_norm(self, model):
        """Compute total gradient norm"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
```

### 2. Gradient Accumulation

Optional: Reduce noise by accumulating gradients over multiple steps:

```python
class GradientAccumulator:
    """Accumulate gradients over multiple steps before update"""

    def __init__(self, accumulation_steps=4):
        """
        Args:
            accumulation_steps: Number of steps to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def should_update(self):
        """Check if we should perform optimizer step"""
        self.step_count += 1
        if self.step_count >= self.accumulation_steps:
            self.step_count = 0
            return True
        return False

    def scale_loss(self, loss):
        """Scale loss for accumulation"""
        return loss / self.accumulation_steps
```

### 3. Moving Average of Weights (EMA)

Stabilize model predictions:

```python
class ExponentialMovingAverage:
    """EMA of model weights for stable predictions"""

    def __init__(self, model, decay=0.999):
        """
        Args:
            model: PyTorch model
            decay: EMA decay rate (higher = slower update)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Temporarily replace model weights with EMA weights"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
```

### 4. Gradient Noise Tracking

Monitor gradient statistics for debugging:

```python
class GradientMonitor:
    """Track gradient statistics for stability monitoring"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.grad_norms = []
        self.grad_vars = []

    def record(self, model):
        """Record gradient statistics"""
        grad_norm = 0.0
        grad_values = []

        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
                grad_values.extend(p.grad.data.flatten().cpu().numpy())

        grad_norm = grad_norm ** 0.5
        grad_var = np.var(grad_values) if len(grad_values) > 0 else 0.0

        self.grad_norms.append(grad_norm)
        self.grad_vars.append(grad_var)

        # Keep only recent history
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)
            self.grad_vars.pop(0)

    def get_statistics(self):
        """Get gradient statistics"""
        return {
            'grad_norm_mean': np.mean(self.grad_norms),
            'grad_norm_std': np.std(self.grad_norms),
            'grad_var_mean': np.mean(self.grad_vars),
            'is_stable': self._check_stability()
        }

    def _check_stability(self):
        """Check if gradients are stable"""
        if len(self.grad_norms) < 10:
            return True

        recent_std = np.std(self.grad_norms[-10:])
        overall_mean = np.mean(self.grad_norms)

        # Unstable if recent variance is too high
        return recent_std < 2 * overall_mean
```

## Death Handling

### Strategy: Persist and Reset

When agent dies:
1. Save current model weights to checkpoint
2. Reset optimizer state (clear momentum, etc.)
3. Optionally reduce learning rate
4. Respawn agent with same weights

```python
class DeathHandler:
    """Handle agent death in online learning"""

    def __init__(
        self,
        checkpoint_dir='./checkpoints',
        reset_optimizer=True,
        lr_reduction_factor=0.5,
        min_lr=1e-6
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

    def on_death(self, model, optimizer, lr_scheduler=None):
        """Called when agent dies"""
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

    def load_latest_checkpoint(self, model, optimizer):
        """Load most recent death checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob('death_*.pt'))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        checkpoint = torch.load(latest)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint
```

### Alternative Strategy: Experience Replay on Death

```python
class DeathReplay:
    """Replay recent experiences on death before respawn"""

    def __init__(self, replay_buffer_size=100, replay_iterations=10):
        """
        Args:
            replay_buffer_size: Number of recent experiences to keep
            replay_iterations: How many times to replay on death
        """
        self.buffer = []
        self.buffer_size = replay_buffer_size
        self.replay_iterations = replay_iterations

    def add_experience(self, senses, action, prediction, next_senses, reward):
        """Add experience to replay buffer"""
        self.buffer.append({
            'senses': senses,
            'action': action,
            'prediction': prediction,
            'next_senses': next_senses,
            'reward': reward
        })

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def replay_on_death(self, model, optimizer, loss_fn):
        """Replay recent experiences when agent dies"""
        if len(self.buffer) < 10:
            return  # Not enough experiences

        for _ in range(self.replay_iterations):
            # Sample random experience
            exp = random.choice(self.buffer)

            # Recompute loss and update
            loss = loss_fn(exp['prediction'], exp['next_senses'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Metrics & Logging

### Core Metrics

```python
class LearningMetrics:
    """Track and log online learning metrics"""

    def __init__(self, log_interval=10):
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
        prediction_loss,
        survival_reward,
        teaching_reward,
        total_reward,
        gradient_norm,
        modulation_factor,
        learning_rate
    ):
        """Record metrics for one step"""
        self.step_count += 1

        self.prediction_losses.append(prediction_loss)
        self.rewards_survival.append(survival_reward)
        self.rewards_teaching.append(teaching_reward)
        self.rewards_total.append(total_reward)
        self.gradient_norms.append(gradient_norm)
        self.modulation_factors.append(modulation_factor)
        self.learning_rates.append(learning_rate)

    def should_log(self):
        """Check if we should log now"""
        return self.step_count % self.log_interval == 0

    def get_summary(self):
        """Get summary statistics"""
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
            'training/learning_rate': self.learning_rates[-1],
        }

    def clear_buffers(self):
        """Clear metric buffers after logging"""
        self.prediction_losses.clear()
        self.rewards_survival.clear()
        self.rewards_teaching.clear()
        self.rewards_total.clear()
        self.gradient_norms.clear()
        self.modulation_factors.clear()
        self.learning_rates.clear()
```

### Visualization

```python
class LearningVisualizer:
    """Visualize learning progress in real-time"""

    def __init__(self, use_tensorboard=True, use_wandb=False):
        """
        Args:
            use_tensorboard: Log to TensorBoard
            use_wandb: Log to Weights & Biases
        """
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter('runs/online_learning')

        if use_wandb:
            import wandb
            wandb.init(project='primordial', name='online_learning')

    def log_metrics(self, metrics, step):
        """Log metrics to configured backends"""
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def log_model_predictions(self, predicted, actual, step):
        """Log prediction visualizations"""
        if self.use_tensorboard:
            # Log prediction error heatmap
            error = (predicted - actual).abs()
            self.tb_writer.add_histogram('predictions/error', error, step)

    def close(self):
        """Close logging backends"""
        if self.use_tensorboard:
            self.tb_writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
```

## API Specification

### Main Learning Loop

```python
class OnlineLearningLoop:
    """Main online learning loop for Primordial agents"""

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: dict,
        loss_config: dict,
        reward_config: dict,
        stability_config: dict,
        death_config: dict,
        metrics_config: dict
    ):
        """
        Initialize online learning loop

        Args:
            model: Agent's neural network
            optimizer_config: Optimizer configuration
            loss_config: Loss function configuration
            reward_config: Reward system configuration
            stability_config: Stability measures configuration
            death_config: Death handling configuration
            metrics_config: Metrics and logging configuration
        """
        self.model = model

        # Initialize components
        self.loss_fn = self._create_loss_fn(loss_config)
        self.base_optimizer = self._create_optimizer(optimizer_config)
        self.optimizer = RewardModulatedOptimizer(
            self.base_optimizer,
            **reward_config.get('modulation', {})
        )
        self.reward_combiner = RewardCombiner(**reward_config.get('combiner', {}))
        self.lr_scheduler = OnlineLRScheduler(
            self.base_optimizer,
            **optimizer_config.get('lr_schedule', {})
        )

        # Stability components
        self.gradient_clipper = GradientClipper(**stability_config.get('clipping', {}))
        self.grad_monitor = GradientMonitor(**stability_config.get('monitoring', {}))
        self.ema = ExponentialMovingAverage(
            model,
            **stability_config.get('ema', {})
        )

        # Death handling
        self.death_handler = DeathHandler(**death_config)

        # Metrics
        self.metrics = LearningMetrics(**metrics_config.get('metrics', {}))
        self.visualizer = LearningVisualizer(**metrics_config.get('viz', {}))

        # State
        self.prev_prediction = None
        self.prev_state = None
        self.step_count = 0

    def step(
        self,
        senses: torch.Tensor,
        prev_senses: torch.Tensor,
        agent_state: 'AgentState',
        prev_agent_state: 'AgentState',
        events: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of online learning

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
            self.metrics.record_step(
                prediction_loss=loss.item(),
                survival_reward=survival_reward,
                teaching_reward=teaching_reward,
                total_reward=total_reward,
                gradient_norm=grad_norm,
                modulation_factor=modulation_factor,
                learning_rate=lr
            )

            # Log if needed
            if self.metrics.should_log():
                summary = self.metrics.get_summary()
                self.visualizer.log_metrics(summary, self.step_count)
                self.metrics.clear_buffers()

        # 2. Forward pass to get action and prediction
        with torch.no_grad():
            self.ema.apply_shadow()  # Use EMA weights for inference
            action, prediction = self.model(senses)
            self.ema.restore()

        # 3. Store for next iteration
        self.prev_prediction = prediction
        self.prev_state = agent_state

        return action, prediction

    def on_death(self):
        """Handle agent death"""
        result = self.death_handler.on_death(
            self.model,
            self.base_optimizer,
            self.lr_scheduler
        )

        # Reset state
        self.prev_prediction = None
        self.prev_state = None

        return result

    def save_checkpoint(self, path: str):
        """Save full learning state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.base_optimizer.state_dict(),
            'ema_shadow': self.ema.shadow,
            'step_count': self.step_count,
        }, path)

    def load_checkpoint(self, path: str):
        """Load full learning state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ema.shadow = checkpoint['ema_shadow']
        self.step_count = checkpoint['step_count']

    def _create_loss_fn(self, config):
        """Create loss function from config"""
        loss_type = config.get('type', 'mse')
        if loss_type == 'mse':
            return PredictionLoss()
        elif loss_type == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _create_optimizer(self, config):
        """Create optimizer from config"""
        opt_type = config.get('type', 'adamw')
        if opt_type == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                **config.get('params', {})
            )
        elif opt_type == 'rmsprop':
            return torch.optim.RMSprop(
                self.model.parameters(),
                **config.get('params', {})
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
```

### Human Teaching Interface

```python
class TeachingInterface:
    """API for human teaching interactions"""

    def __init__(self, learning_loop: OnlineLearningLoop):
        """
        Args:
            learning_loop: Main learning loop to interface with
        """
        self.learning_loop = learning_loop
        self.teaching_history = []

    def reward(self):
        """Called when human presses reward button"""
        self.learning_loop.reward_combiner.human_teaching.add_teaching_signal(
            HumanTeaching.REWARD_BUTTON
        )
        self.teaching_history.append({
            'timestamp': time.time(),
            'type': 'reward',
            'value': HumanTeaching.REWARD_BUTTON
        })

    def punish(self):
        """Called when human presses punish button"""
        self.learning_loop.reward_combiner.human_teaching.add_teaching_signal(
            HumanTeaching.PUNISH_BUTTON
        )
        self.teaching_history.append({
            'timestamp': time.time(),
            'type': 'punish',
            'value': HumanTeaching.PUNISH_BUTTON
        })

    def get_recent_teaching(self, n=10):
        """Get recent teaching signals"""
        return self.teaching_history[-n:]
```

## Testing Strategy

### 1. Unit Tests

```python
# tests/test_learning/test_losses.py

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

# tests/test_learning/test_rewards.py

def test_survival_rewards_eating():
    """Test reward for eating food"""
    prev_state = AgentState(health=50, energy=50)
    curr_state = AgentState(health=50, energy=70)
    events = ['ate_food']

    reward = SurvivalRewards.compute_reward(prev_state, curr_state, events)
    assert reward == SurvivalRewards.EATING_FOOD

def test_reward_combiner():
    """Test combining survival and teaching rewards"""
    combiner = RewardCombiner(survival_weight=1.0, teaching_weight=1.5)
    combiner.human_teaching.add_teaching_signal(1.0)

    prev_state = AgentState(health=50, energy=50)
    curr_state = AgentState(health=50, energy=70)
    events = ['ate_food']

    total, survival, teaching = combiner.compute_total_reward(
        prev_state, curr_state, events
    )

    assert survival == 1.0  # Eating food
    assert teaching > 0     # Human reward
    assert total == survival + 1.5 * teaching

# tests/test_learning/test_stability.py

def test_gradient_clipping():
    """Test gradient clipping prevents explosion"""
    model = SimpleModel()
    clipper = GradientClipper(clip_type='norm', max_norm=1.0)

    # Create large gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 100

    norm_before = compute_total_norm(model)
    assert norm_before > 1.0

    clipper.clip(model)
    norm_after = compute_total_norm(model)
    assert norm_after <= 1.0

def test_ema_stability():
    """Test EMA provides smoother weights"""
    model = SimpleModel()
    ema = ExponentialMovingAverage(model, decay=0.99)

    # Make several large updates
    for _ in range(10):
        for p in model.parameters():
            p.data += torch.randn_like(p) * 0.1
        ema.update()

    # EMA weights should be less extreme
    ema.apply_shadow()
    shadow_norm = compute_total_norm(model)
    ema.restore()
    actual_norm = compute_total_norm(model)

    assert shadow_norm < actual_norm
```

### 2. Integration Tests

```python
# tests/test_learning/test_integration.py

def test_full_learning_step():
    """Test complete learning step"""
    model = create_test_model()
    learning_loop = create_test_learning_loop(model)

    # Create synthetic data
    senses = torch.randn(1, 10)
    prev_senses = torch.randn(1, 10)
    agent_state = AgentState(health=80, energy=60)
    prev_agent_state = AgentState(health=70, energy=50)
    events = ['ate_food']

    # Perform two steps (need previous prediction)
    action1, pred1 = learning_loop.step(
        prev_senses, prev_senses, prev_agent_state, prev_agent_state, []
    )
    action2, pred2 = learning_loop.step(
        senses, prev_senses, agent_state, prev_agent_state, events
    )

    # Verify outputs
    assert action2 is not None
    assert pred2 is not None
    assert learning_loop.step_count == 2

def test_death_handling():
    """Test learning continues after death"""
    model = create_test_model()
    learning_loop = create_test_learning_loop(model)

    # Get initial weights
    initial_weights = {
        name: p.clone() for name, p in model.named_parameters()
    }

    # Train for some steps
    for _ in range(10):
        learning_loop.step(*create_random_step_data())

    # Trigger death
    result = learning_loop.on_death()
    assert result['death_count'] == 1

    # Weights should be preserved
    for name, p in model.named_parameters():
        assert not torch.equal(p, initial_weights[name])  # Changed

    # Should be able to continue learning
    learning_loop.step(*create_random_step_data())
```

### 3. Synthetic Scenarios

```python
def test_scenario_learn_to_eat():
    """Test agent learns to associate food-eating with reward"""
    model = create_test_model()
    learning_loop = create_test_learning_loop(model)

    # Scenario: Agent sees food, eats it, gets reward
    # Repeat 100 times

    losses_before = []
    losses_after = []

    for i in range(100):
        # Create food-nearby senses
        senses = create_food_nearby_senses()

        # Agent should predict reward will come
        action, prediction = learning_loop.step(
            senses,
            prev_senses=senses,
            agent_state=AgentState(health=80, energy=60),
            prev_agent_state=AgentState(health=80, energy=50),
            events=['ate_food']  # Positive reward
        )

        # Record prediction error
        if i < 10:
            losses_before.append(learning_loop.metrics.prediction_losses[-1])
        elif i > 90:
            losses_after.append(learning_loop.metrics.prediction_losses[-1])

    # Prediction error should decrease
    assert np.mean(losses_after) < np.mean(losses_before)

def test_scenario_avoid_damage():
    """Test agent learns to avoid damage"""
    model = create_test_model()
    learning_loop = create_test_learning_loop(model)

    # Scenario: Agent takes damage, receives negative reward
    # Should learn to predict negative outcomes

    for i in range(100):
        senses = create_danger_nearby_senses()

        learning_loop.step(
            senses,
            prev_senses=senses,
            agent_state=AgentState(health=60, energy=60),
            prev_agent_state=AgentState(health=80, energy=60),
            events=['took_damage']  # Negative reward
        )

    # After training, prediction when seeing danger should indicate bad outcome
    # (Implementation depends on how predictions encode value)
```

## Hyperparameters

### Default Configuration

| Category | Parameter | Default | Range | Description |
|----------|-----------|---------|-------|-------------|
| **Optimizer** | type | adamw | {adamw, rmsprop} | Base optimizer |
| | learning_rate | 1e-4 | [1e-6, 1e-3] | Base learning rate |
| | weight_decay | 1e-5 | [0, 1e-3] | L2 regularization |
| | betas | (0.9, 0.999) | - | Adam momentum terms |
| **LR Schedule** | warmup_steps | 1000 | [100, 5000] | Linear warmup steps |
| | decay_rate | 0.9999 | [0.999, 1.0] | Exponential decay |
| | min_lr | 1e-6 | [1e-8, 1e-5] | Minimum learning rate |
| **Loss** | type | mse | {mse, huber} | Loss function |
| | huber_delta | 1.0 | [0.1, 10.0] | Huber loss threshold |
| **Rewards** | survival_weight | 1.0 | [0.5, 2.0] | Survival reward scaling |
| | teaching_weight | 1.5 | [1.0, 3.0] | Human teaching scaling |
| | teaching_window | 10 | [5, 20] | Steps to apply teaching |
| **Modulation** | type | linear | {linear, sigmoid, exp} | Modulation function |
| | reward_scale | 0.1 | [0.01, 0.5] | Reward sensitivity |
| | min_modulation | 0.1 | [0.01, 0.5] | Min gradient scaling |
| | max_modulation | 3.0 | [1.5, 5.0] | Max gradient scaling |
| **Stability** | clip_type | norm | {norm, value} | Gradient clipping type |
| | max_norm | 1.0 | [0.5, 5.0] | Max gradient norm |
| | ema_decay | 0.999 | [0.99, 0.9999] | EMA weight decay |
| **Death** | reset_optimizer | True | {True, False} | Reset on death |
| | lr_reduction | 0.5 | [0.3, 0.9] | LR reduction on death |
| **Metrics** | log_interval | 10 | [1, 100] | Steps between logs |
| | use_tensorboard | True | {True, False} | Enable TensorBoard |

### YAML Configuration Example

```yaml
# config/learning_config.yaml

optimizer:
  type: adamw
  params:
    lr: 1.0e-4
    betas: [0.9, 0.999]
    eps: 1.0e-8
    weight_decay: 1.0e-5
  lr_schedule:
    warmup_steps: 1000
    base_lr: 1.0e-4
    min_lr: 1.0e-6
    decay_rate: 0.9999

loss:
  type: mse

reward:
  combiner:
    survival_weight: 1.0
    teaching_weight: 1.5
  modulation:
    modulation_type: linear
    reward_scale: 0.1
    min_modulation: 0.1
    max_modulation: 3.0

stability:
  clipping:
    clip_type: norm
    max_norm: 1.0
  ema:
    decay: 0.999
  monitoring:
    window_size: 100

death:
  checkpoint_dir: ./checkpoints
  reset_optimizer: true
  lr_reduction_factor: 0.5
  min_lr: 1.0e-6

metrics:
  metrics:
    log_interval: 10
  viz:
    use_tensorboard: true
    use_wandb: false
```

## Implementation Order

### Phase 1: Core Infrastructure (Week 1)

1. **Loss Functions** (`losses.py`)
   - Implement `PredictionLoss` with MSE
   - Add Huber loss variant
   - Unit tests for loss functions

2. **Reward System** (`rewards.py`)
   - Implement `SurvivalRewards` class
   - Implement `HumanTeaching` class
   - Implement `RewardCombiner`
   - Unit tests for reward computation

3. **Basic Optimizer** (`optimizer.py`)
   - Implement `RewardModulatedOptimizer` wrapper
   - Linear modulation function
   - Unit tests for gradient scaling

### Phase 2: Stability & Learning Loop (Week 2)

4. **Stability Measures** (`stability.py`)
   - Implement `GradientClipper`
   - Implement `GradientMonitor`
   - Implement `ExponentialMovingAverage`
   - Unit tests for each component

5. **Learning Loop** (`learning_loop.py`)
   - Implement `OnlineLearningLoop` class
   - Integrate all components
   - Single-step learning test

6. **LR Scheduler** (`optimizer.py`)
   - Implement `OnlineLRScheduler`
   - Warmup and decay logic
   - Unit tests

### Phase 3: Auxiliary Systems (Week 3)

7. **Death Handling** (`checkpointing.py`)
   - Implement `DeathHandler`
   - Checkpoint saving/loading
   - Optimizer reset logic
   - Integration tests

8. **Metrics & Logging** (`metrics.py`)
   - Implement `LearningMetrics`
   - Implement `LearningVisualizer`
   - TensorBoard integration
   - Unit tests

9. **Teaching Interface** (`learning_loop.py`)
   - Implement `TeachingInterface`
   - Button press handling
   - History tracking

### Phase 4: Integration & Testing (Week 4)

10. **Configuration System**
    - Create YAML config structure
    - Config loading/validation
    - Default hyperparameters

11. **Integration Tests**
    - Full learning loop test
    - Death handling test
    - Multi-step learning test
    - Synthetic scenario tests

12. **Documentation & Examples**
    - API documentation
    - Usage examples
    - Hyperparameter tuning guide

### Phase 5: Advanced Features (Week 5+)

13. **Advanced Modulation**
    - Sigmoid modulation
    - Exponential modulation
    - Comparison benchmarks

14. **Experience Replay** (Optional)
    - Implement `DeathReplay`
    - Buffer management
    - Replay integration

15. **Optimization**
    - Profile learning loop
    - Optimize hot paths
    - Memory efficiency

## Dependencies

```
# requirements.txt
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
tensorboard>=2.13.0
pytest>=7.3.0
```

## Success Criteria

The online learning system is complete when:

1. ✓ Agent can learn from prediction error over 1000+ steps
2. ✓ Reward modulation demonstrably affects learning speed
3. ✓ System remains stable (no gradient explosion) for 1+ hour runs
4. ✓ Human teaching signals visibly influence agent behavior
5. ✓ Agent recovers from death without catastrophic forgetting
6. ✓ Metrics clearly show learning progress
7. ✓ All unit tests pass
8. ✓ Synthetic scenarios demonstrate learning

## Notes

- Start conservative with learning rates (1e-4)
- Monitor gradient norms closely in early testing
- Death handling may need tuning based on how often agents die
- Consider adding gradient accumulation if single-sample updates too noisy
- Human teaching weight may need tuning based on user feedback frequency
