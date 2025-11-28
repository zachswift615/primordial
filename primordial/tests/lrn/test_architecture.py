"""Tests for the main LivingResonanceNetwork architecture."""
import pytest
import torch

from primordial.lrn import LivingResonanceNetwork, LRNConfig


@pytest.fixture
def config():
    """Create default LRN configuration."""
    return LRNConfig()


@pytest.fixture
def config_no_genome():
    """Create LRN configuration without genome modulation."""
    return LRNConfig(use_genome_modulation=False)


@pytest.fixture
def batch_inputs():
    """Create a batch of test inputs."""
    batch_size = 4
    return {
        'vision': torch.randn(batch_size, 32, 4),
        'audio': torch.randn(batch_size, 100, 2),
        'proprio': torch.randn(batch_size, 7),
        'touch': torch.randn(batch_size, 8),
        'genome': torch.randn(batch_size, 100)
    }


@pytest.fixture
def single_inputs():
    """Create single sample inputs."""
    return {
        'vision': torch.randn(1, 32, 4),
        'audio': torch.randn(1, 100, 2),
        'proprio': torch.randn(1, 7),
        'touch': torch.randn(1, 8),
        'genome': torch.randn(1, 100)
    }


def test_lrn_initialization(config):
    """Test that LRN initializes correctly."""
    model = LivingResonanceNetwork(config)

    # Check components exist
    assert model.vision_encoder is not None
    assert model.audio_encoder is not None
    assert model.proprio_encoder is not None
    assert model.touch_encoder is not None

    # Check 6 mixing layers created
    assert len(model.mixing_layers) == 6
    assert len(model.mixing_layers) == config.num_mixing_layers

    # Check output heads
    assert model.prediction_head is not None
    assert model.reward_head is not None
    assert model.action_head is not None

    # Check genome modulator (should be present with default config)
    assert model.genome_modulator is not None


def test_lrn_initialization_no_genome(config_no_genome):
    """Test that LRN initializes correctly without genome modulation."""
    model = LivingResonanceNetwork(config_no_genome)

    # Check genome modulator is None
    assert model.genome_modulator is None


def test_forward_pass_shapes(config, batch_inputs):
    """Test that forward pass produces correct output shapes."""
    model = LivingResonanceNetwork(config)

    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        batch_inputs['genome']
    )

    batch_size = batch_inputs['vision'].shape[0]

    # Check predictions shape (flattened sensory state)
    assert predictions.shape == (batch_size, 343)
    assert predictions.shape == (batch_size, config.total_sensory_dim)

    # Check reward predictions shape
    assert reward_preds.shape == (batch_size, 5)
    assert reward_preds.shape == (batch_size, config.reward_horizon)

    # Check actions shape
    assert actions.shape == (batch_size, 5)
    assert actions.shape == (batch_size, config.action_dim)


def test_forward_pass_without_genome(config, batch_inputs):
    """Test forward pass without genome input."""
    model = LivingResonanceNetwork(config)

    # Call without genome (should still work)
    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        genome=None
    )

    batch_size = batch_inputs['vision'].shape[0]

    # Check shapes still correct
    assert predictions.shape == (batch_size, 343)
    assert reward_preds.shape == (batch_size, 5)
    assert actions.shape == (batch_size, 5)


def test_forward_pass_no_genome_config(config_no_genome, batch_inputs):
    """Test forward pass with genome modulation disabled in config."""
    model = LivingResonanceNetwork(config_no_genome)

    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        genome=None
    )

    batch_size = batch_inputs['vision'].shape[0]

    # Check shapes still correct
    assert predictions.shape == (batch_size, 343)
    assert reward_preds.shape == (batch_size, 5)
    assert actions.shape == (batch_size, 5)


def test_single_sample_forward(config, single_inputs):
    """Test that model handles single samples (for online learning)."""
    model = LivingResonanceNetwork(config)

    predictions, reward_preds, actions = model(
        single_inputs['vision'],
        single_inputs['audio'],
        single_inputs['proprio'],
        single_inputs['touch'],
        single_inputs['genome']
    )

    # Check shapes for batch size 1
    assert predictions.shape == (1, 343)
    assert reward_preds.shape == (1, 5)
    assert actions.shape == (1, 5)


def test_compute_loss_with_rewards(config, batch_inputs):
    """Test compute_loss returns correct loss components with rewards."""
    model = LivingResonanceNetwork(config)

    # Forward pass
    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        batch_inputs['genome']
    )

    # Create target sensory state
    next_sensory = {
        'vision': torch.randn_like(batch_inputs['vision']),
        'audio': torch.randn_like(batch_inputs['audio']),
        'proprio': torch.randn_like(batch_inputs['proprio']),
        'touch': torch.randn_like(batch_inputs['touch'])
    }

    # Create target rewards
    batch_size = batch_inputs['vision'].shape[0]
    actual_rewards = torch.randn(batch_size, 5)

    # Compute loss
    loss_dict = model.compute_loss(
        predictions,
        reward_preds,
        next_sensory,
        actions,
        actual_rewards
    )

    # Check all loss components exist
    assert 'total' in loss_dict
    assert 'sensory' in loss_dict
    assert 'reward' in loss_dict
    assert 'vision' in loss_dict
    assert 'audio' in loss_dict
    assert 'proprio' in loss_dict
    assert 'touch' in loss_dict

    # Check losses are scalars
    for key, value in loss_dict.items():
        assert value.dim() == 0, f"{key} loss should be scalar"

    # Check total loss is sum of components
    expected_total = (
        loss_dict['sensory'] +
        config.reward_loss_weight * loss_dict['reward']
    )
    assert torch.allclose(loss_dict['total'], expected_total, atol=1e-5)

    # Check sensory loss is sum of modality losses
    expected_sensory = (
        loss_dict['vision'] +
        loss_dict['audio'] +
        loss_dict['proprio'] +
        loss_dict['touch']
    )
    assert torch.allclose(loss_dict['sensory'], expected_sensory, atol=1e-5)


def test_compute_loss_without_rewards(config, batch_inputs):
    """Test compute_loss handles missing rewards gracefully."""
    model = LivingResonanceNetwork(config)

    # Forward pass
    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        batch_inputs['genome']
    )

    # Create target sensory state
    next_sensory = {
        'vision': torch.randn_like(batch_inputs['vision']),
        'audio': torch.randn_like(batch_inputs['audio']),
        'proprio': torch.randn_like(batch_inputs['proprio']),
        'touch': torch.randn_like(batch_inputs['touch'])
    }

    # Compute loss without rewards
    loss_dict = model.compute_loss(
        predictions,
        reward_preds,
        next_sensory,
        actions,
        actual_rewards=None
    )

    # Check reward loss is zero
    assert loss_dict['reward'] == 0.0

    # Check total loss equals sensory loss when no rewards
    assert torch.allclose(loss_dict['total'], loss_dict['sensory'], atol=1e-5)


def test_gradient_flow(config, batch_inputs):
    """Test that gradients flow through entire network."""
    model = LivingResonanceNetwork(config)

    # Forward pass
    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        batch_inputs['genome']
    )

    # Create target sensory state
    next_sensory = {
        'vision': torch.randn_like(batch_inputs['vision']),
        'audio': torch.randn_like(batch_inputs['audio']),
        'proprio': torch.randn_like(batch_inputs['proprio']),
        'touch': torch.randn_like(batch_inputs['touch'])
    }

    # Create target rewards
    batch_size = batch_inputs['vision'].shape[0]
    actual_rewards = torch.randn(batch_size, 5)

    # Compute loss
    loss_dict = model.compute_loss(
        predictions,
        reward_preds,
        next_sensory,
        actions,
        actual_rewards
    )

    # Backward pass
    loss_dict['total'].backward()

    # Check gradients exist for all encoder parameters
    for encoder in [model.vision_encoder, model.audio_encoder,
                    model.proprio_encoder, model.touch_encoder]:
        for param in encoder.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    # Check gradients exist for mixing layers
    for layer in model.mixing_layers:
        for param in layer.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    # Check gradients exist for prediction and reward heads
    # (action_head won't have gradients since actions aren't used in loss)
    for head in [model.prediction_head, model.reward_head]:
        for param in head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_parameter_count(config):
    """Test that parameter count is reasonable for the architecture."""
    model = LivingResonanceNetwork(config)

    total_params = sum(p.numel() for p in model.parameters())

    # Should be around 500K-800K parameters
    # (Actual count depends on encoder and head sizes)
    assert 400_000 <= total_params <= 900_000, \
        f"Parameter count {total_params} outside expected range [400K, 900K]"

    print(f"\nTotal parameters: {total_params:,}")


def test_trainable_parameters(config):
    """Test that all parameters are trainable by default."""
    model = LivingResonanceNetwork(config)

    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} is not trainable"


def test_batch_independence(config):
    """Test that batch samples are processed independently."""
    model = LivingResonanceNetwork(config)
    model.eval()  # Disable any stochastic behavior

    # Create two identical samples
    vision = torch.randn(1, 32, 4)
    audio = torch.randn(1, 100, 2)
    proprio = torch.randn(1, 7)
    touch = torch.randn(1, 8)
    genome = torch.randn(1, 100)

    # Process individually
    with torch.no_grad():
        pred1, reward1, action1 = model(vision, audio, proprio, touch, genome)
        pred2, reward2, action2 = model(vision, audio, proprio, touch, genome)

    # Should be identical (deterministic)
    assert torch.allclose(pred1, pred2, atol=1e-5)
    assert torch.allclose(reward1, reward2, atol=1e-5)
    assert torch.allclose(action1, action2, atol=1e-5)

    # Process as batch
    vision_batch = torch.cat([vision, vision], dim=0)
    audio_batch = torch.cat([audio, audio], dim=0)
    proprio_batch = torch.cat([proprio, proprio], dim=0)
    touch_batch = torch.cat([touch, touch], dim=0)
    genome_batch = torch.cat([genome, genome], dim=0)

    with torch.no_grad():
        pred_batch, reward_batch, action_batch = model(
            vision_batch, audio_batch, proprio_batch, touch_batch, genome_batch
        )

    # Batch results should match individual results
    assert torch.allclose(pred_batch[0], pred1[0], atol=1e-5)
    assert torch.allclose(pred_batch[1], pred1[0], atol=1e-5)
    assert torch.allclose(reward_batch[0], reward1[0], atol=1e-5)
    assert torch.allclose(action_batch[0], action1[0], atol=1e-5)


def test_output_finite(config, batch_inputs):
    """Test that outputs are finite (no NaN or Inf)."""
    model = LivingResonanceNetwork(config)

    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        batch_inputs['genome']
    )

    # Check all outputs are finite
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(reward_preds).all()
    assert torch.isfinite(actions).all()


def test_six_mixing_layers(config):
    """Test that exactly 6 mixing layers are created."""
    model = LivingResonanceNetwork(config)

    # Count mixing layers
    assert len(model.mixing_layers) == 6

    # Verify they're all LRNFourierMixingLayer instances
    from primordial.lrn.lrn_mixing import LRNFourierMixingLayer
    for layer in model.mixing_layers:
        assert isinstance(layer, LRNFourierMixingLayer)


def test_weight_initialization(config):
    """Test that weights are properly initialized."""
    model = LivingResonanceNetwork(config)

    # Check that spectral filters are initialized with low-frequency bias
    for layer in model.mixing_layers:
        # Check spectral filter exists and has correct shape
        assert hasattr(layer, 'spectral_filter')
        assert layer.spectral_filter.shape[2] == 2  # Real and imaginary components

        # Check values are small (scaled by 0.1)
        assert layer.spectral_filter.abs().mean() < 0.5

    # Check encoder biases are zero
    for encoder in [model.vision_encoder, model.audio_encoder,
                    model.proprio_encoder, model.touch_encoder]:
        for module in encoder.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    # Check LayerNorm initialization
    for layer in model.mixing_layers:
        assert torch.allclose(layer.norm.weight, torch.ones_like(layer.norm.weight))
        assert torch.allclose(layer.norm.bias, torch.zeros_like(layer.norm.bias))


def test_config_preservation(config):
    """Test that config is stored correctly."""
    model = LivingResonanceNetwork(config)

    assert model.config is config
    assert model.config.num_mixing_layers == 6
    assert model.config.hidden_dim == 128
    assert model.config.reward_horizon == 5


def test_different_batch_sizes(config):
    """Test model with various batch sizes."""
    model = LivingResonanceNetwork(config)

    for batch_size in [1, 2, 4, 8, 16]:
        inputs = {
            'vision': torch.randn(batch_size, 32, 4),
            'audio': torch.randn(batch_size, 100, 2),
            'proprio': torch.randn(batch_size, 7),
            'touch': torch.randn(batch_size, 8),
            'genome': torch.randn(batch_size, 100)
        }

        predictions, reward_preds, actions = model(
            inputs['vision'],
            inputs['audio'],
            inputs['proprio'],
            inputs['touch'],
            inputs['genome']
        )

        assert predictions.shape == (batch_size, 343)
        assert reward_preds.shape == (batch_size, 5)
        assert actions.shape == (batch_size, 5)


def test_compile_parameter(config):
    """Test that compile parameter works correctly."""
    # Create model with compile=False (default)
    model_no_compile = LivingResonanceNetwork(config, compile=False)
    assert model_no_compile._compile_enabled is False
    assert model_no_compile._compiled_forward is None

    # Create model with compile=True
    model_with_compile = LivingResonanceNetwork(config, compile=True)

    # If torch.compile is available, it should be enabled
    if hasattr(torch, 'compile'):
        assert model_with_compile._compile_enabled is True
        assert model_with_compile._compiled_forward is not None
    else:
        # If torch.compile is not available, it should fall back gracefully
        assert model_with_compile._compile_enabled is False


def test_compiled_model_forward(config, batch_inputs):
    """Test that compiled model produces correct outputs."""
    # Skip if torch.compile not available
    if not hasattr(torch, 'compile'):
        pytest.skip("torch.compile not available")

    model = LivingResonanceNetwork(config, compile=True)

    predictions, reward_preds, actions = model(
        batch_inputs['vision'],
        batch_inputs['audio'],
        batch_inputs['proprio'],
        batch_inputs['touch'],
        batch_inputs['genome']
    )

    batch_size = batch_inputs['vision'].shape[0]

    # Check shapes are correct
    assert predictions.shape == (batch_size, 343)
    assert reward_preds.shape == (batch_size, 5)
    assert actions.shape == (batch_size, 5)

    # Check outputs are finite
    assert torch.isfinite(predictions).all()
    assert torch.isfinite(reward_preds).all()
    assert torch.isfinite(actions).all()


def test_compiled_vs_non_compiled_equivalence(config, single_inputs):
    """Test that compiled and non-compiled models produce same results."""
    # Skip if torch.compile not available
    if not hasattr(torch, 'compile'):
        pytest.skip("torch.compile not available")

    # Create two models with same initialization
    torch.manual_seed(42)
    model_no_compile = LivingResonanceNetwork(config, compile=False)

    torch.manual_seed(42)
    model_compiled = LivingResonanceNetwork(config, compile=True)

    # Set to eval mode for deterministic behavior
    model_no_compile.eval()
    model_compiled.eval()

    with torch.no_grad():
        # Get outputs from non-compiled model
        pred1, reward1, action1 = model_no_compile(
            single_inputs['vision'],
            single_inputs['audio'],
            single_inputs['proprio'],
            single_inputs['touch'],
            single_inputs['genome']
        )

        # Get outputs from compiled model
        pred2, reward2, action2 = model_compiled(
            single_inputs['vision'],
            single_inputs['audio'],
            single_inputs['proprio'],
            single_inputs['touch'],
            single_inputs['genome']
        )

    # Outputs should be very close (allowing for minor numerical differences)
    assert torch.allclose(pred1, pred2, atol=1e-4)
    assert torch.allclose(reward1, reward2, atol=1e-4)
    assert torch.allclose(action1, action2, atol=1e-4)


def test_inference_mode_context_manager(config, batch_inputs):
    """Test that inference_mode context manager works correctly."""
    model = LivingResonanceNetwork(config)

    # Initially in training mode
    assert model.training is True

    # Enter inference mode
    with model.inference_mode() as m:
        # Should be in eval mode
        assert m.training is False
        assert model.training is False

        # Forward pass should work
        predictions, reward_preds, actions = model(
            batch_inputs['vision'],
            batch_inputs['audio'],
            batch_inputs['proprio'],
            batch_inputs['touch'],
            batch_inputs['genome']
        )

        # Check shapes are correct
        batch_size = batch_inputs['vision'].shape[0]
        assert predictions.shape == (batch_size, 343)
        assert reward_preds.shape == (batch_size, 5)
        assert actions.shape == (batch_size, 5)

        # Gradients should not be computed
        assert not predictions.requires_grad
        assert not reward_preds.requires_grad
        assert not actions.requires_grad

    # After exiting, should be back in training mode
    assert model.training is True


def test_inference_mode_restores_state(config):
    """Test that inference_mode restores original training state."""
    model = LivingResonanceNetwork(config)

    # Test from training mode
    model.train()
    assert model.training is True
    with model.inference_mode():
        pass
    assert model.training is True

    # Test from eval mode
    model.eval()
    assert model.training is False
    with model.inference_mode():
        pass
    assert model.training is False


def test_inference_mode_no_gradients(config, batch_inputs):
    """Test that inference_mode prevents gradient computation."""
    model = LivingResonanceNetwork(config)

    with model.inference_mode():
        predictions, reward_preds, actions = model(
            batch_inputs['vision'],
            batch_inputs['audio'],
            batch_inputs['proprio'],
            batch_inputs['touch'],
            batch_inputs['genome']
        )

        # Attempt to compute gradients should fail or do nothing
        # (no requires_grad, so backward would fail)
        assert not predictions.requires_grad
        assert not reward_preds.requires_grad
        assert not actions.requires_grad
