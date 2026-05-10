"""
Learnable label embeddings in Lorentz (hyperbolic) space.

Embeddings are stored as tangent vectors at the origin and mapped to the
Lorentz manifold via exp_map0 during forward pass.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from models.hyperbolic.lorentz_ops import exp_map0


class LorentzLabelEmbedding(nn.Module):
    """
    Learnable class embeddings in Lorentz hyperbolic space.

    Stores tangent vectors at origin, applies exp_map0 in forward pass.
    Initialization uses random unit directions and hierarchy-depth-based norms:
    deeper classes start farther from the origin.
    """

    def __init__(
        self,
        num_classes: int = 70,
        embed_dim: int = 32,
        curv: float = 1.0,
        class_depths: Optional[Dict[int, int]] = None,
        min_radius: float = 0.1,
        max_radius: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension
            curv: Fixed curvature (positive value for negative curvature -curv)
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Tangent norm for shallowest classes
            max_radius: Tangent norm for deepest classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.curv = curv

        tangent_vectors = self._init_tangent_vectors(
            num_classes, embed_dim, class_depths, min_radius, max_radius
        )
        self.tangent_embeddings = nn.Parameter(tangent_vectors)

    def _init_tangent_vectors(
        self,
        num_classes: int,
        embed_dim: int,
        class_depths: Optional[Dict[int, int]],
        min_radius: float,
        max_radius: float,
    ) -> Tensor:
        """Random unit directions scaled by depth-based norms."""
        directions = torch.randn(num_classes, embed_dim)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        norms = self._get_depth_norms(num_classes, class_depths, min_radius, max_radius)

        return directions * norms.unsqueeze(-1)

    def _get_depth_norms(
        self,
        num_classes: int,
        class_depths: Optional[Dict[int, int]],
        min_radius: float,
        max_radius: float,
    ) -> Tensor:
        if class_depths is None:
            return torch.ones(num_classes) * (min_radius + max_radius) / 2

        depths = list(class_depths.values())
        min_depth = min(depths)
        max_depth = max(depths)
        depth_range = max_depth - min_depth if max_depth > min_depth else 1

        norms = torch.zeros(num_classes)
        for class_idx in range(num_classes):
            depth = class_depths.get(class_idx, min_depth)
            normalized_depth = (depth - min_depth) / depth_range
            norms[class_idx] = min_radius + (max_radius - min_radius) * normalized_depth

        return norms

    def forward(self) -> Tensor:
        """
        Get Lorentz embeddings for all classes.

        Returns:
            Tensor of shape [num_classes, embed_dim] in Lorentz space
        """
        return exp_map0(self.tangent_embeddings, self.curv)

    def get_embedding(self, class_idx: int) -> Tensor:
        """Get Lorentz embedding for a single class."""
        tangent = self.tangent_embeddings[class_idx]
        return exp_map0(tangent.unsqueeze(0), self.curv).squeeze(0)
