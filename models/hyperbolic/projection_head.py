"""
Projection head for mapping decoder features to Lorentz hyperbolic space.
"""
import torch
import torch.nn as nn
from torch import Tensor

from models.hyperbolic.lorentz_ops import exp_map0


class LorentzProjectionHead(nn.Module):
    """
    Projects 3D feature maps to Lorentz hyperbolic space.

    Architecture: 1x1x1 Conv3d -> exp_map0
    """

    def __init__(
        self,
        in_channels: int = 32,
        embed_dim: int = 32,
        curv: float = 1.0,
    ):
        """
        Args:
            in_channels: Number of input channels from decoder
            embed_dim: Embedding dimension in Lorentz space
            curv: Fixed curvature
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.curv = curv

        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Project decoder features to Lorentz space.

        Args:
            x: Decoder features (B, C, D, H, W)

        Returns:
            Lorentz embeddings (B, embed_dim, D, H, W)
        """
        # 1x1x1 convolution
        x = self.conv(x)  # (B, embed_dim, D, H, W)

        # Permute for exp_map0: (B, D, H, W, embed_dim)
        x = x.permute(0, 2, 3, 4, 1)

        # Map to Lorentz manifold
        x = exp_map0(x, self.curv)

        # Permute back: (B, embed_dim, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)

        return x
