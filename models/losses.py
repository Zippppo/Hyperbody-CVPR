import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss for 3D segmentation"""

    def __init__(self, smooth: float = 1.0, ignore_index: int = None):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Class index to exclude from Dice averaging (None = use all)
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)

        Returns:
            Scalar Dice loss (1 - mean_dice)
        """
        num_classes = logits.shape[1]

        # Force float32 for numerical stability in AMP
        logits = logits.float()

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, D, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.shape[0], num_classes, -1)  # (B, C, N)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)  # (B, C, N)

        # Compute Dice per class
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # Average over classes and batch, optionally excluding ignore_index
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, dtype=torch.bool, device=dice_per_class.device)
            mask[self.ignore_index] = False
            mean_dice = dice_per_class[:, mask].mean()
        else:
            mean_dice = dice_per_class.mean()

        return 1.0 - mean_dice


class MemoryEfficientDiceLoss(nn.Module):
    """Multi-class Dice loss for 3D segmentation (memory-efficient).

    Note: Using gather/scatter_add for memory efficiency, replaces
    O(B*C*D*H*W) one-hot tensor with O(B*D*H*W) gather operations.
    Mathematically equivalent to DiceLoss (one-hot version).
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = None):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_index: Class index to exclude from Dice averaging (None = use all)
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)

        Returns:
            Scalar Dice loss (1 - mean_dice)
        """
        B, num_classes = logits.shape[0], logits.shape[1]
        N = targets[0].numel()  # D * H * W

        # Force float32 for numerical stability in AMP
        logits = logits.float()

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)

        # Flatten spatial dimensions
        probs_flat = probs.view(B, num_classes, N)  # (B, C, N)
        targets_flat = targets.view(B, N)  # (B, N)

        # --- Intersection via gather + scatter_add ---
        # Gather probabilities at ground-truth class for each voxel
        targets_idx = targets_flat.unsqueeze(1)  # (B, 1, N)
        probs_at_target = probs_flat.gather(1, targets_idx).squeeze(1)  # (B, N)

        # Accumulate per-class intersection
        intersection = torch.zeros(
            B, num_classes, device=logits.device, dtype=logits.dtype
        )
        intersection.scatter_add_(1, targets_flat, probs_at_target)  # (B, C)

        # --- Union ---
        # Probs side: sum of predicted probabilities per class
        probs_sum = probs_flat.sum(dim=2)  # (B, C)

        # Targets side: count of voxels per class via bincount
        # Encode batch index into targets to vectorize bincount
        offsets = torch.arange(B, device=targets.device) * num_classes  # (B,)
        offset_targets = targets_flat + offsets.unsqueeze(1)  # (B, N)
        targets_count = torch.bincount(
            offset_targets.reshape(-1), minlength=B * num_classes
        ).view(B, num_classes).to(dtype=logits.dtype)  # (B, C)

        union = probs_sum + targets_count  # (B, C)

        # Dice per class
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # Average over classes and batch, optionally excluding ignore_index
        if self.ignore_index is not None:
            mask = torch.ones(num_classes, dtype=torch.bool, device=dice_per_class.device)
            mask[self.ignore_index] = False
            mean_dice = dice_per_class[:, mask].mean()
        else:
            mean_dice = dice_per_class.mean()

        return 1.0 - mean_dice


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice loss for 3D segmentation"""

    def __init__(
        self,
        num_classes: int = 70,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor = None,
        smooth: float = 1.0,
        dice_ignore_index: int = None,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for Dice loss
            class_weights: (C,) tensor of per-class weights for CE loss
            smooth: Smoothing factor for Dice loss
            dice_ignore_index: Class index to exclude from Dice averaging (None = use all)
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = MemoryEfficientDiceLoss(smooth=smooth, ignore_index=dice_ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)

        Returns:
            Combined loss scalar
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.ce_weight * ce + self.dice_weight * dice


def compute_class_weights(
    dataset,
    num_classes: int = 70,
    num_samples: int = 100,
    method: str = "inverse_sqrt",
    cache_path: str = None,
) -> torch.Tensor:
    """
    Compute per-class weights from dataset samples.

    Args:
        dataset: PyTorch Dataset with (input, label) items
        num_classes: Number of segmentation classes
        num_samples: Number of samples to use for weight computation
        method: Weight computation method ('inverse_sqrt' or 'inverse')
        cache_path: Path to cache the weights. If provided and file exists,
                    loads weights from cache instead of computing.

    Returns:
        (num_classes,) tensor of normalized class weights
    """
    import os
    import random

    # Try to load from cache
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=True)
        # Validate cached weights match current config
        if (
            cached.get("num_classes") == num_classes
            and cached.get("num_samples") == num_samples
            and cached.get("method") == method
        ):
            return cached["weights"]

    # Sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Count class frequencies
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for idx in indices:
        _, labels = dataset[idx]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()

    # Compute weights
    total_voxels = class_counts.sum()
    class_freq = class_counts / total_voxels

    # Avoid division by zero for absent classes
    class_freq = torch.clamp(class_freq, min=1e-8)

    if method == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(class_freq)
    elif method == "inverse":
        weights = 1.0 / class_freq
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(
            {
                "weights": weights,
                "num_classes": num_classes,
                "num_samples": num_samples,
                "method": method,
            },
            cache_path,
        )

    return weights
