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
    Initialization uses hierarchy depth: deeper classes start farther from origin.

    Supports two direction modes:
    - "random": random unit directions (default, backward compatible)
    - "semantic": directions from PCA-projected text embeddings
    """

    def __init__(
        self,
        num_classes: int = 70,
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
            num_classes: Number of classes
            embed_dim: Embedding dimension
            curv: Fixed curvature (positive value for negative curvature -curv)
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Tangent norm for shallowest classes
            max_radius: Tangent norm for deepest classes
            direction_mode: "random" or "semantic"
            text_embedding_path: Path to text embeddings .pt file (required for semantic mode)
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.curv = curv
        self.direction_mode = direction_mode
        self.text_embedding_path = text_embedding_path

        # Validate direction_mode
        if direction_mode not in ("random", "semantic"):
            raise ValueError(f"Unknown direction_mode: {direction_mode}. Must be 'random' or 'semantic'.")

        if direction_mode == "semantic" and text_embedding_path is None:
            raise ValueError("text_embedding_path required for semantic direction_mode")

        # Initialize tangent vectors
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
        """
        Initialize tangent vectors with configurable direction and depth-based norms.

        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Tangent norm for shallowest classes
            max_radius: Tangent norm for deepest classes

        Returns:
            Tensor of shape [num_classes, embed_dim]
        """
        # Get directions based on mode
        directions = self._get_directions(num_classes, embed_dim)  # [N, D] unit vectors

        # Get depth-based norms
        norms = self._get_depth_norms(num_classes, class_depths, min_radius, max_radius)  # [N]

        # Combine: tangent_vector = direction * norm
        tangent_vectors = directions * norms.unsqueeze(-1)  # [N, D]

        return tangent_vectors

    def _get_directions(self, num_classes: int, embed_dim: int) -> Tensor:
        """
        Get unit direction vectors for each class.

        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension

        Returns:
            Tensor of shape [num_classes, embed_dim] with unit vectors
        """
        if self.direction_mode == "random":
            directions = torch.randn(num_classes, embed_dim)
            directions = directions / directions.norm(dim=-1, keepdim=True)

        elif self.direction_mode == "semantic":
            directions = self._load_semantic_directions(embed_dim)

        return directions

    def _get_depth_norms(
        self,
        num_classes: int,
        class_depths: Optional[Dict[int, int]],
        min_radius: float,
        max_radius: float,
    ) -> Tensor:
        """
        Compute depth-based norms for each class.

        Args:
            num_classes: Number of classes
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Tangent norm for shallowest classes
            max_radius: Tangent norm for deepest classes

        Returns:
            Tensor of shape [num_classes] with norm values
        """
        if class_depths is None:
            # Fallback: uniform norm at midpoint
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

    def _load_semantic_directions(self, embed_dim: int) -> Tensor:
        """
        Load text embeddings and project to embed_dim via PCA.

        Args:
            embed_dim: Target embedding dimension

        Returns:
            Tensor of shape [num_classes, embed_dim] with unit vectors
        """
        data = torch.load(self.text_embedding_path, weights_only=False)
        embeddings = data['embeddings']  # [N, 768]
        label_ids = data['label_ids']    # [N]

        num_samples = embeddings.shape[0]

        # Validate num_classes matches text embedding file
        if num_samples != self.num_classes:
            raise ValueError(
                f"Text embedding file has {num_samples} classes, "
                f"but num_classes={self.num_classes}"
            )

        # Safety check: PCA can only extract min(n_samples, n_features) components
        max_components = min(num_samples, embeddings.shape[1])
        if embed_dim > max_components:
            raise ValueError(
                f"embed_dim ({embed_dim}) > max PCA components ({max_components}). "
                f"With {num_samples} classes, max embed_dim is {num_samples}."
            )

        # PCA projection
        centered = embeddings - embeddings.mean(dim=0)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        # Vh shape: [min(N, 768), 768]
        # Vh[:embed_dim] shape: [embed_dim, 768]
        # centered @ Vh[:embed_dim].T shape: [N, embed_dim]
        projected = centered @ Vh[:embed_dim].T

        # Normalize to unit vectors
        directions = projected / projected.norm(dim=-1, keepdim=True)

        # Reorder by label_id to match class indices (vectorized)
        ordered = torch.zeros(num_samples, embed_dim)
        ordered[label_ids] = directions

        return ordered

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
