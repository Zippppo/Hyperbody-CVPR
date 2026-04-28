import torch
import pytest


class TestLorentzProjectionHead:
    """Test LorentzProjectionHead module."""

    def test_output_shape(self):
        """Output should preserve spatial dimensions."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=32, embed_dim=32)
        x = torch.randn(2, 32, 8, 16, 12)  # (B, C, D, H, W)

        out = head(x)
        assert out.shape == (2, 32, 8, 16, 12), f"Expected (2, 32, 8, 16, 12), got {out.shape}"

    def test_output_is_finite(self):
        """Output should not contain inf or nan."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=32, embed_dim=32)
        x = torch.randn(2, 32, 8, 6, 4)

        out = head(x)
        assert torch.isfinite(out).all(), "Output contains inf or nan"

    def test_different_embed_dim(self):
        """Should work with different embedding dimensions."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=64, embed_dim=16)
        x = torch.randn(2, 64, 8, 6, 4)

        out = head(x)
        assert out.shape == (2, 16, 8, 6, 4)

    def test_gradient_flow(self):
        """Gradients should flow through the head."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=32, embed_dim=32)
        x = torch.randn(2, 32, 4, 4, 4, requires_grad=True)

        out = head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert head.conv.weight.grad is not None
