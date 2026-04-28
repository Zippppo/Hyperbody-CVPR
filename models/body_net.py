"""
BodyNet: UNet3D with hyperbolic embedding head.

Combines segmentation and hyperbolic geometry for hierarchy-aware
organ segmentation.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple

from models.unet3d import UNet3D
from models.hyperbolic.projection_head import LorentzProjectionHead
from models.hyperbolic.label_embedding import LorentzLabelEmbedding


class BodyNet(nn.Module):
    """
    UNet3D with Lorentz hyperbolic embedding branch.

    Returns:
        - logits: Segmentation logits (B, num_classes, D, H, W)
        - voxel_emb: Lorentz voxel embeddings (B, embed_dim, D, H, W)
        - label_emb: Lorentz class embeddings (num_classes, embed_dim)
    """

    def __init__(
        self,
        # UNet3D params
        in_channels: int = 1,
        num_classes: int = 70,
        base_channels: int = 32,
        growth_rate: int = 32,
        dense_layers: int = 4,
        bn_size: int = 4,
        # Hyperbolic params
        embed_dim: int = 32,
        curv: float = 1.0,
        class_depths: Optional[Dict[int, int]] = None,
        min_radius: float = 0.1,
        max_radius: float = 2.0,
        direction_mode: str = "random",
        text_embedding_path: Optional[str] = None,
    ):
        """
        Args:
            in_channels: Input channels for UNet3D
            num_classes: Number of segmentation classes
            base_channels: Base channels for UNet3D
            growth_rate: Dense block growth rate
            dense_layers: Number of dense layers
            bn_size: Bottleneck size multiplier
            embed_dim: Hyperbolic embedding dimension
            curv: Hyperbolic curvature
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Min tangent norm for label embedding init
            max_radius: Max tangent norm for label embedding init
            direction_mode: "random" or "semantic" for label embedding init
            text_embedding_path: Path to text embeddings (required for semantic mode)
        """
        super().__init__()

        # Segmentation backbone
        self.unet = UNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            growth_rate=growth_rate,
            dense_layers=dense_layers,
            bn_size=bn_size,
        )

        # Hyperbolic projection head (from decoder features to Lorentz space)
        self.hyp_head = LorentzProjectionHead(
            in_channels=base_channels,
            embed_dim=embed_dim,
            curv=curv,
        )

        # Learnable class embeddings in Lorentz space
        self.label_emb = LorentzLabelEmbedding(
            num_classes=num_classes,
            embed_dim=embed_dim,
            curv=curv,
            class_depths=class_depths,
            min_radius=min_radius,
            max_radius=max_radius,
            direction_mode=direction_mode,
            text_embedding_path=text_embedding_path,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input volume (B, in_channels, D, H, W)

        Returns:
            Tuple of:
                - logits: (B, num_classes, D, H, W)
                - voxel_emb: (B, embed_dim, D, H, W) in Lorentz space
                - label_emb: (num_classes, embed_dim) in Lorentz space
        """
        # Get segmentation logits and decoder features
        logits, d2 = self.unet(x, return_features=True)

        # Project decoder features to Lorentz space
        voxel_emb = self.hyp_head(d2)

        # Get class embeddings in Lorentz space
        label_emb = self.label_emb()

        return logits, voxel_emb, label_emb
