"""
Lorentz space ranking loss for hyperbolic embeddings.

Uses triplet margin loss: pull voxel embeddings toward their class embeddings,
push away from other class embeddings.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from models.hyperbolic.lorentz_ops import pointwise_dist, pairwise_dist


def _normalize_sampling_weights(weights: Tensor, neg_mask: Tensor) -> Tensor:
    """Convert raw negative-sampling weights into valid multinomial probabilities."""
    probs = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)

    row_sums = probs.sum(dim=1, keepdim=True)
    invalid_rows = row_sums.squeeze(1) <= 0
    if invalid_rows.any():
        # Fallback to uniform sampling over valid negatives for pathological rows.
        fallback = neg_mask[invalid_rows].to(dtype=probs.dtype)
        fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp(min=1e-8)
        probs[invalid_rows] = fallback
        row_sums = probs.sum(dim=1, keepdim=True)

    return probs / row_sums.clamp(min=1e-8)


class LorentzRankingLoss(nn.Module):
    """
    Triplet ranking loss in Lorentz hyperbolic space.

    For each sampled voxel:
    - anchor = voxel embedding
    - positive = class embedding of voxel's true class
    - negatives = M embeddings of random other classes

    Loss = mean(max(0, margin + d(anchor, positive) - d(anchor, negative)))
    """

    def __init__(
        self,
        margin: float = 0.1,
        curv: float = 1.0,
        num_samples_per_class: int = 64,
        num_negatives: int = 8,
        # Curriculum Negative Mining parameters
        t_start: float = 2.0,
        t_end: float = 0.1,
        warmup_epochs: int = 5,
        curriculum_epochs: int = 50,
    ):
        """
        Args:
            margin: Triplet margin
            curv: Curvature (for distance computation)
            num_samples_per_class: Max voxels to sample per class
            num_negatives: Number of negative classes per anchor
            t_start: Initial temperature for curriculum sampling (high = more random)
            t_end: Final temperature for curriculum sampling (low = more hard negatives)
            warmup_epochs: Number of epochs to use uniform random sampling before curriculum
            curriculum_epochs: Total epochs for easy->hard curriculum (decoupled from total training epochs)
        """
        super().__init__()
        self.margin = margin
        self.curv = curv
        self.num_samples_per_class = num_samples_per_class
        self.num_negatives = num_negatives
        self.t_start = t_start
        self.t_end = t_end
        self.warmup_epochs = warmup_epochs
        self.curriculum_epochs = curriculum_epochs

        # Register buffer for epoch tracking (saved/loaded with model state)
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum scheduling."""
        self.current_epoch.fill_(epoch)

    def get_temperature(self) -> float:
        """
        Get current temperature for curriculum negative mining.

        During warmup (epoch < warmup_epochs): returns t_start
        After warmup: exponential decay from t_start to t_end over curriculum_epochs
        After curriculum completes: stays at t_end
        """
        epoch = self.current_epoch.item()

        # During warmup, return t_start
        if epoch < self.warmup_epochs:
            return self.t_start

        # progress based on fixed curriculum_epochs, not total training epochs
        progress = (epoch - self.warmup_epochs) / max(self.curriculum_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)

        # t = t_start * (t_end / t_start)^progress
        temperature = self.t_start * (self.t_end / self.t_start) ** progress
        return temperature

    def forward(
        self,
        voxel_emb: Tensor,
        labels: Tensor,
        label_emb: Tensor,
    ) -> Tensor:
        """
        Compute ranking loss.

        Args:
            voxel_emb: (B, C, D, H, W) Lorentz voxel embeddings (C=embed_dim)
            labels: (B, D, H, W) ground truth labels (int64)
            label_emb: (num_classes, C) Lorentz class embeddings

        Returns:
            Scalar loss
        """
        # Keep hyperbolic distance computation in FP32: AMP autocast can downcast
        # matrix multiplications in pairwise_dist to FP16 and overflow to inf/nan.
        with torch.autocast(device_type=voxel_emb.device.type, enabled=False):
            voxel_emb = voxel_emb.float()
            label_emb = label_emb.float()

            device = voxel_emb.device
            B, C, D, H, W = voxel_emb.shape
            num_classes = label_emb.shape[0]

            # Reshape: (B, C, D, H, W) -> (N, C) where N = B*D*H*W
            voxel_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N, C)
            labels_flat = labels.reshape(-1)  # (N)

            # Fully vectorized sampling: sample up to num_samples_per_class per class
            N = labels_flat.shape[0]

            # Create random priorities for sampling
            random_priorities = torch.rand(N, device=device)

            # Sort by (class, random_priority) to group by class with random order within
            # Use composite key: class * 2 + random (since random in [0,1))
            sort_key = labels_flat.float() * 2.0 + random_priorities
            sorted_indices = torch.argsort(sort_key)
            sorted_labels = labels_flat[sorted_indices]

            # Compute position within each class using cumsum trick
            # label_changes[i] = 1 if sorted_labels[i] != sorted_labels[i-1], else 0
            label_changes = torch.cat([
                torch.ones(1, device=device, dtype=torch.long),
                (sorted_labels[1:] != sorted_labels[:-1]).long()
            ])
            # cumsum gives group id, subtract to get position within group
            group_ids = torch.cumsum(label_changes, dim=0) - 1
            # Position within class: for each element, count how many before it have same label
            # Use scatter to count positions
            positions = torch.zeros(N, device=device, dtype=torch.long)
            # For each group, positions should be 0, 1, 2, ...
            # We can compute this by: position[i] = i - first_index_of_group[group_ids[i]]
            unique_groups, inverse_indices = torch.unique(group_ids, return_inverse=True)
            # Get first occurrence of each group
            first_occurrence = torch.zeros(len(unique_groups), device=device, dtype=torch.long)
            # scatter_reduce to get min index for each group
            first_occurrence.scatter_reduce_(
                0, inverse_indices,
                torch.arange(N, device=device, dtype=torch.long),
                reduce='amin', include_self=False
            )
            positions = torch.arange(N, device=device, dtype=torch.long) - first_occurrence[inverse_indices]

            # Select samples where position < num_samples_per_class
            sample_mask = positions < self.num_samples_per_class
            sampled_indices = sorted_indices[sample_mask]  # [K]
            sampled_classes = sorted_labels[sample_mask]   # [K]
            K = sampled_indices.shape[0]

            if K == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            # Get anchor embeddings
            anchors = voxel_flat[sampled_indices]  # (K, C)

            # Get positive embeddings (class embedding for each anchor's true class)
            positives = label_emb[sampled_classes]  # (K, C)

            # Compute positive distances
            d_pos = pointwise_dist(anchors, positives, self.curv)  # [K]

            # Vectorized negative sampling and loss computation
            # Compute all pairwise distances: anchors to all class embeddings
            all_dists = pairwise_dist(anchors, label_emb, self.curv)  # [K, num_classes]

            # Create mask for valid negatives (exclude true class for each anchor)
            # neg_mask[i, j] = True if class j is a valid negative for anchor i
            class_indices = torch.arange(num_classes, device=device)  # [num_classes]
            neg_mask = class_indices.unsqueeze(0) != sampled_classes.unsqueeze(1)  # [K, num_classes]

            # Curriculum negative sampling
            # Get top-k negative indices for each anchor
            n_neg = min(self.num_negatives, num_classes - 1)
            if n_neg <= 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            with torch.no_grad():
                all_dists_f32 = all_dists.float()
                epoch = self.current_epoch.item()

                if epoch < self.warmup_epochs:
                    # Warmup: uniform random sampling
                    neg_weights = torch.where(
                        neg_mask,
                        torch.ones_like(all_dists_f32),
                        torch.zeros_like(all_dists_f32)
                    )
                else:
                    # Curriculum: distance-weighted sampling with temperature
                    # Lower temperature = prefer harder negatives (closer ones)
                    temperature = self.get_temperature()
                    neg_weights = torch.where(
                        neg_mask,
                        torch.exp(-all_dists_f32 / temperature),
                        torch.zeros_like(all_dists_f32)
                    )

                # Stabilize probabilities to avoid multinomial failures on pathological rows.
                neg_weights = _normalize_sampling_weights(neg_weights, neg_mask)
                neg_indices = torch.multinomial(neg_weights, n_neg, replacement=False)

            # Gather negative distances
            d_neg = torch.gather(all_dists, 1, neg_indices)  # [K, n_neg]

            # Compute triplet loss: max(0, margin + d_pos - d_neg)
            # d_pos: [K], d_neg: [K, n_neg]
            triplet_loss = torch.clamp(self.margin + d_pos.unsqueeze(1) - d_neg, min=0)  # [K, n_neg]

            # Average over negatives, then over anchors
            loss = triplet_loss.mean()

            return loss


class LorentzTreeRankingLoss(nn.Module):
    """
    Triplet ranking loss in Lorentz hyperbolic space with tree-based negative sampling.

    Similar to LorentzRankingLoss, but uses tree distance (from hierarchy)
    for negative sampling weights instead of hyperbolic distance.

    Key differences from LorentzRankingLoss:
    1. Takes tree_dist_matrix in constructor, registers as buffer
    2. Sampling weights use tree distance (per-class) instead of hyperbolic distance (per-anchor)
    3. Triplet loss computation (d_pos, d_neg) still uses hyperbolic distance

    For each sampled voxel:
    - anchor = voxel embedding
    - positive = class embedding of voxel's true class
    - negatives = M embeddings of classes sampled based on tree distance

    Loss = mean(max(0, margin + d(anchor, positive) - d(anchor, negative)))
    """

    def __init__(
        self,
        tree_dist_matrix: Tensor,
        margin: float = 0.1,
        curv: float = 1.0,
        num_samples_per_class: int = 64,
        num_negatives: int = 8,
        # Curriculum Negative Mining parameters
        t_start: float = 2.0,
        t_end: float = 0.1,
        warmup_epochs: int = 5,
        curriculum_epochs: int = 50,
    ):
        """
        Args:
            tree_dist_matrix: (num_classes, num_classes) pairwise tree distances
            margin: Triplet margin
            curv: Curvature (for distance computation)
            num_samples_per_class: Max voxels to sample per class
            num_negatives: Number of negative classes per anchor
            t_start: Initial temperature for curriculum sampling (high = more random)
            t_end: Final temperature for curriculum sampling (low = more hard negatives)
            warmup_epochs: Number of epochs to use uniform random sampling before curriculum
            curriculum_epochs: Total epochs for easy->hard curriculum (decoupled from total training epochs)
        """
        super().__init__()
        self.margin = margin
        self.curv = curv
        self.num_samples_per_class = num_samples_per_class
        self.num_negatives = num_negatives
        self.t_start = t_start
        self.t_end = t_end
        self.warmup_epochs = warmup_epochs
        self.curriculum_epochs = curriculum_epochs

        # Register tree distance matrix as buffer (saved/loaded with model state)
        self.register_buffer('tree_dist_matrix', tree_dist_matrix.float())

        # Register buffer for epoch tracking (saved/loaded with model state)
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum scheduling."""
        self.current_epoch.fill_(epoch)

    def get_temperature(self) -> float:
        """
        Get current temperature for curriculum negative mining.

        During warmup (epoch < warmup_epochs): returns t_start
        After warmup: exponential decay from t_start to t_end over curriculum_epochs
        After curriculum completes: stays at t_end
        """
        epoch = self.current_epoch.item()

        # During warmup, return t_start
        if epoch < self.warmup_epochs:
            return self.t_start

        # progress based on fixed curriculum_epochs, not total training epochs
        progress = (epoch - self.warmup_epochs) / max(self.curriculum_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)

        # t = t_start * (t_end / t_start)^progress
        temperature = self.t_start * (self.t_end / self.t_start) ** progress
        return temperature

    def forward(
        self,
        voxel_emb: Tensor,
        labels: Tensor,
        label_emb: Tensor,
    ) -> Tensor:
        """
        Compute ranking loss with tree-distance-based negative sampling.

        Args:
            voxel_emb: (B, C, D, H, W) Lorentz voxel embeddings (C=embed_dim)
            labels: (B, D, H, W) ground truth labels (int64)
            label_emb: (num_classes, C) Lorentz class embeddings

        Returns:
            Scalar loss
        """
        # Keep hyperbolic distance computation in FP32: AMP autocast can downcast
        # matrix multiplications in pairwise_dist to FP16 and overflow to inf/nan.
        with torch.autocast(device_type=voxel_emb.device.type, enabled=False):
            voxel_emb = voxel_emb.float()
            label_emb = label_emb.float()

            device = voxel_emb.device
            B, C, D, H, W = voxel_emb.shape
            num_classes = label_emb.shape[0]

            # Reshape: (B, C, D, H, W) -> (N, C) where N = B*D*H*W
            voxel_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N, C)
            labels_flat = labels.reshape(-1)  # (N)

            # Fully vectorized sampling: sample up to num_samples_per_class per class
            N = labels_flat.shape[0]

            # Create random priorities for sampling
            random_priorities = torch.rand(N, device=device)

            # Sort by (class, random_priority) to group by class with random order within
            # Use composite key: class * 2 + random (since random in [0,1))
            sort_key = labels_flat.float() * 2.0 + random_priorities
            sorted_indices = torch.argsort(sort_key)
            sorted_labels = labels_flat[sorted_indices]

            # Compute position within each class using cumsum trick
            # label_changes[i] = 1 if sorted_labels[i] != sorted_labels[i-1], else 0
            label_changes = torch.cat([
                torch.ones(1, device=device, dtype=torch.long),
                (sorted_labels[1:] != sorted_labels[:-1]).long()
            ])
            # cumsum gives group id, subtract to get position within group
            group_ids = torch.cumsum(label_changes, dim=0) - 1
            # Position within class: for each element, count how many before it have same label
            # Use scatter to count positions
            positions = torch.zeros(N, device=device, dtype=torch.long)
            # For each group, positions should be 0, 1, 2, ...
            # We can compute this by: position[i] = i - first_index_of_group[group_ids[i]]
            unique_groups, inverse_indices = torch.unique(group_ids, return_inverse=True)
            # Get first occurrence of each group
            first_occurrence = torch.zeros(len(unique_groups), device=device, dtype=torch.long)
            # scatter_reduce to get min index for each group
            first_occurrence.scatter_reduce_(
                0, inverse_indices,
                torch.arange(N, device=device, dtype=torch.long),
                reduce='amin', include_self=False
            )
            positions = torch.arange(N, device=device, dtype=torch.long) - first_occurrence[inverse_indices]

            # Select samples where position < num_samples_per_class
            sample_mask = positions < self.num_samples_per_class
            sampled_indices = sorted_indices[sample_mask]  # [K]
            sampled_classes = sorted_labels[sample_mask]   # [K]
            K = sampled_indices.shape[0]

            if K == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            # Get anchor embeddings
            anchors = voxel_flat[sampled_indices]  # (K, C)

            # Get positive embeddings (class embedding for each anchor's true class)
            positives = label_emb[sampled_classes]  # (K, C)

            # Compute positive distances using hyperbolic distance
            d_pos = pointwise_dist(anchors, positives, self.curv)  # [K]

            # Compute all pairwise hyperbolic distances for triplet loss (d_neg)
            # This is needed for the loss computation, not for sampling
            all_hyp_dists = pairwise_dist(anchors, label_emb, self.curv)  # [K, num_classes]

            # Create mask for valid negatives (exclude true class for each anchor)
            # neg_mask[i, j] = True if class j is a valid negative for anchor i
            class_indices = torch.arange(num_classes, device=device)  # [num_classes]
            neg_mask = class_indices.unsqueeze(0) != sampled_classes.unsqueeze(1)  # [K, num_classes]

            # Curriculum negative sampling using TREE DISTANCE (per-class, not per-anchor)
            # Key difference from LorentzRankingLoss: use tree_dist_matrix[sampled_classes]
            n_neg = min(self.num_negatives, num_classes - 1)
            if n_neg <= 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            with torch.no_grad():
                epoch = self.current_epoch.item()

                # Get tree distances for each anchor's class: [K, num_classes]
                # tree_dist_matrix[sampled_classes] gives distances from each anchor's class to all classes
                tree_dists = self.tree_dist_matrix[sampled_classes]  # [K, num_classes]

                if epoch < self.warmup_epochs:
                    # Warmup: uniform random sampling
                    neg_weights = torch.where(
                        neg_mask,
                        torch.ones_like(tree_dists),
                        torch.zeros_like(tree_dists)
                    )
                else:
                    # Curriculum: tree-distance-weighted sampling with temperature
                    # Lower temperature = prefer harder negatives (closer in tree)
                    temperature = self.get_temperature()
                    neg_weights = torch.where(
                        neg_mask,
                        torch.exp(-tree_dists / temperature),
                        torch.zeros_like(tree_dists)
                    )

                # Stabilize probabilities to avoid multinomial failures on pathological rows.
                neg_weights = _normalize_sampling_weights(neg_weights, neg_mask)
                neg_indices = torch.multinomial(neg_weights, n_neg, replacement=False)

            # Gather negative distances (using hyperbolic distance for loss computation)
            d_neg = torch.gather(all_hyp_dists, 1, neg_indices)  # [K, n_neg]

            # Compute triplet loss: max(0, margin + d_pos - d_neg)
            # d_pos: [K], d_neg: [K, n_neg]
            triplet_loss = torch.clamp(self.margin + d_pos.unsqueeze(1) - d_neg, min=0)  # [K, n_neg]

            # Average over negatives, then over anchors
            loss = triplet_loss.mean()

            return loss
