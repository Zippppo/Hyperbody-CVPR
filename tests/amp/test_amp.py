"""Tests for AMP (Automatic Mixed Precision) training support."""

import pytest
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from config import Config
from models.unet3d import UNet3D
from models.losses import CombinedLoss, DiceLoss


class TestAMPConfig:
    """Test AMP configuration."""

    def test_config_has_use_amp(self):
        """Config should have use_amp field."""
        cfg = Config()
        assert hasattr(cfg, 'use_amp')
        assert isinstance(cfg.use_amp, bool)

    def test_config_use_amp_default_true(self):
        """use_amp should default to True."""
        cfg = Config()
        assert cfg.use_amp is True


class TestAMPForwardPass:
    """Test model forward pass with AMP."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return UNet3D(
            in_channels=1,
            num_classes=10,
            base_channels=8,
            growth_rate=8,
            dense_layers=2,
            bn_size=2,
        ).cuda()

    @pytest.fixture
    def inputs(self):
        """Create test inputs."""
        return torch.randn(1, 1, 32, 32, 32).cuda()

    @pytest.fixture
    def targets(self):
        """Create test targets."""
        return torch.randint(0, 10, (1, 32, 32, 32)).cuda()

    def test_forward_with_autocast(self, model, inputs):
        """Model forward pass should work with autocast."""
        with autocast():
            outputs = model(inputs)
        assert outputs.shape == (1, 10, 32, 32, 32)
        # Output should be float16 inside autocast
        assert outputs.dtype == torch.float16

    def test_loss_with_autocast(self, model, inputs, targets):
        """Loss computation should work with autocast."""
        criterion = CombinedLoss(num_classes=10)
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        assert loss.dtype == torch.float32  # Loss should be float32

    def test_backward_with_scaler(self, model, inputs, targets):
        """Backward pass should work with GradScaler."""
        criterion = CombinedLoss(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler()

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Should complete without error
        assert scaler.get_scale() > 0


class TestDiceLossFloat32:
    """Test DiceLoss float32 safety."""

    def test_dice_loss_outputs_float32(self):
        """DiceLoss should output float32 even with float16 input."""
        loss_fn = DiceLoss()
        # Simulate float16 input (as would happen inside autocast)
        logits = torch.randn(1, 10, 8, 8, 8, dtype=torch.float16).cuda()
        targets = torch.randint(0, 10, (1, 8, 8, 8)).cuda()

        loss = loss_fn(logits, targets)
        assert loss.dtype == torch.float32


class TestGradScalerState:
    """Test GradScaler state save/load."""

    def test_scaler_state_dict(self):
        """GradScaler should have saveable state."""
        scaler = GradScaler()
        state = scaler.state_dict()
        assert 'scale' in state
        assert '_growth_tracker' in state

    def test_scaler_load_state_dict(self):
        """GradScaler should restore from state dict."""
        scaler1 = GradScaler()
        # Simulate some training that changes scale
        scaler1._scale = torch.tensor(1024.0)

        state = scaler1.state_dict()

        scaler2 = GradScaler()
        scaler2.load_state_dict(state)

        assert scaler2.get_scale() == 1024.0
