"""
Lorentz space ranking loss for hyperbolic embeddings.

Triplet ranking with negative sampling driven by a precomputed pairwise
distance matrix (e.g. graph distance over the organ adjacency graph).
The triplet distances themselves are computed in Lorentz hyperbolic space.
"""
import torch
import torch.nn as nn
from torch import Tensor

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


class LorentzMatrixRankingLoss(nn.Module):
    """
    Triplet ranking loss in Lorentz hyperbolic space with matrix-distance-based
    negative sampling.

    For each sampled voxel:
    - anchor = voxel embedding
    - positive = class embedding of voxel's true class
    - negatives = M class embeddings sampled by precomputed inter-class distance

    Sampling weights come from `dist_matrix` (per-class), while the triplet
    distances (d_pos, d_neg) are computed in Lorentz space.

    Loss = mean(max(0, margin + d(anchor, positive) - d(anchor, negative)))
    """

    def __init__(
        self,
        dist_matrix: Tensor,
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
            dist_matrix: (num_classes, num_classes) pairwise inter-class distances
                used to weight negative sampling (e.g. graph or tree distance).
            margin: Triplet margin
            curv: Curvature (for Lorentz distance computation)
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

        self.register_buffer('dist_matrix', dist_matrix.float())
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum scheduling."""
        self.current_epoch.fill_(epoch)

    def get_temperature(self) -> float:
        """
        Curriculum temperature schedule:
        - During warmup (epoch < warmup_epochs): returns t_start
        - After warmup: exponential decay from t_start to t_end over curriculum_epochs
        - After curriculum completes: stays at t_end
        """
        epoch = self.current_epoch.item()

        if epoch < self.warmup_epochs:
            return self.t_start

        progress = (epoch - self.warmup_epochs) / max(self.curriculum_epochs - self.warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)

        return self.t_start * (self.t_end / self.t_start) ** progress

    def forward(
        self,
        voxel_emb: Tensor,
        labels: Tensor,
        label_emb: Tensor,
    ) -> Tensor:
        """
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
            voxel_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, C)
            labels_flat = labels.reshape(-1)

            # Fully vectorized sampling: sample up to num_samples_per_class per class
            N = labels_flat.shape[0]

            random_priorities = torch.rand(N, device=device)

            # Sort by (class, random_priority) to group by class with random order within
            sort_key = labels_flat.float() * 2.0 + random_priorities
            sorted_indices = torch.argsort(sort_key)
            sorted_labels = labels_flat[sorted_indices]

            # Compute position within each class using cumsum trick
            label_changes = torch.cat([
                torch.ones(1, device=device, dtype=torch.long),
                (sorted_labels[1:] != sorted_labels[:-1]).long()
            ])
            group_ids = torch.cumsum(label_changes, dim=0) - 1
            unique_groups, inverse_indices = torch.unique(group_ids, return_inverse=True)
            first_occurrence = torch.zeros(len(unique_groups), device=device, dtype=torch.long)
            first_occurrence.scatter_reduce_(
                0, inverse_indices,
                torch.arange(N, device=device, dtype=torch.long),
                reduce='amin', include_self=False
            )
            positions = torch.arange(N, device=device, dtype=torch.long) - first_occurrence[inverse_indices]

            sample_mask = positions < self.num_samples_per_class
            sampled_indices = sorted_indices[sample_mask]
            sampled_classes = sorted_labels[sample_mask]
            K = sampled_indices.shape[0]

            if K == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            anchors = voxel_flat[sampled_indices]
            positives = label_emb[sampled_classes]

            d_pos = pointwise_dist(anchors, positives, self.curv)

            all_hyp_dists = pairwise_dist(anchors, label_emb, self.curv)

            class_indices = torch.arange(num_classes, device=device)
            neg_mask = class_indices.unsqueeze(0) != sampled_classes.unsqueeze(1)

            n_neg = min(self.num_negatives, num_classes - 1)
            if n_neg <= 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            with torch.no_grad():
                epoch = self.current_epoch.item()

                # Per-anchor inter-class distances driven by the precomputed matrix.
                anchor_class_dists = self.dist_matrix[sampled_classes]

                if epoch < self.warmup_epochs:
                    neg_weights = torch.where(
                        neg_mask,
                        torch.ones_like(anchor_class_dists),
                        torch.zeros_like(anchor_class_dists)
                    )
                else:
                    # Lower temperature = prefer harder (closer) negatives.
                    temperature = self.get_temperature()
                    neg_weights = torch.where(
                        neg_mask,
                        torch.exp(-anchor_class_dists / temperature),
                        torch.zeros_like(anchor_class_dists)
                    )

                neg_weights = _normalize_sampling_weights(neg_weights, neg_mask)
                neg_indices = torch.multinomial(neg_weights, n_neg, replacement=False)

            d_neg = torch.gather(all_hyp_dists, 1, neg_indices)

            triplet_loss = torch.clamp(self.margin + d_pos.unsqueeze(1) - d_neg, min=0)
            return triplet_loss.mean()
