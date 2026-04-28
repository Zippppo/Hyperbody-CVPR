import torch
import pytest


class TestUNet3DFeatures:
    """Test UNet3D return_features functionality."""

    def test_default_returns_only_logits(self):
        """By default, forward should return only logits."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 32, 24, 32)

        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 70

    def test_return_features_gives_tuple(self):
        """With return_features=True, should return (logits, features)."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 32, 24, 32)

        out = model(x, return_features=True)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_features_shape_matches_output(self):
        """Features should have same spatial dimensions as output."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 32, 24, 32)

        logits, features = model(x, return_features=True)

        # Features are d2 with base_channels
        assert features.shape[0] == 1
        assert features.shape[1] == 32  # base_channels
        assert features.shape[2:] == logits.shape[2:]

    def test_backward_compatible(self):
        """Existing code using model(x) should still work."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 16, 12, 16)

        # Old usage
        logits = model(x)
        assert logits.shape == (1, 70, 16, 12, 16)
