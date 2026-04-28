import torch
from typing import Tuple, Optional


class DiceMetric:
    """Accumulate per-class Dice scores across batches (GPU-optimized)"""

    def __init__(
        self,
        num_classes: int = 70,
        smooth: float = 1e-5,
        target_ignore_index: Optional[int] = None,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            smooth: Smoothing factor to avoid division by zero
            target_ignore_index: Voxel label value excluded from all counts
        """
        self.num_classes = num_classes
        self.smooth = smooth
        self.target_ignore_index = target_ignore_index
        # Accumulators initialized lazily to inherit device from input
        self._intersection = None
        self._pred_sum = None
        self._target_sum = None

    def reset(self):
        """Reset accumulators for new epoch"""
        if self._intersection is not None:
            self._intersection.zero_()
            self._pred_sum.zero_()
            self._target_sum.zero_()

    def _init_accumulators(self, device: torch.device):
        """Initialize accumulators on the given device"""
        self._intersection = torch.zeros(
            self.num_classes, dtype=torch.float64, device=device
        )
        self._pred_sum = torch.zeros(
            self.num_classes, dtype=torch.float64, device=device
        )
        self._target_sum = torch.zeros(
            self.num_classes, dtype=torch.float64, device=device
        )

    @property
    def intersection(self):
        if self._intersection is None:
            # Return CPU tensor for backward compatibility before first update
            return torch.zeros(self.num_classes, dtype=torch.float64)
        return self._intersection

    @property
    def pred_sum(self):
        if self._pred_sum is None:
            return torch.zeros(self.num_classes, dtype=torch.float64)
        return self._pred_sum

    @property
    def target_sum(self):
        if self._target_sum is None:
            return torch.zeros(self.num_classes, dtype=torch.float64)
        return self._target_sum

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate statistics from a batch (GPU-optimized, no Python loops).

        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)
        """
        device = logits.device

        # Lazy initialization of accumulators on correct device
        if self._intersection is None:
            self._init_accumulators(device)

        # Get predictions: (B, D, H, W)
        preds = logits.argmax(dim=1)

        # Flatten to 1D for bincount
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        if self.target_ignore_index is not None:
            keep = targets_flat != self.target_ignore_index
            preds_flat = preds_flat[keep]
            targets_flat = targets_flat[keep]

        # Compute per-class statistics using bincount (vectorized, no Python loop)
        # pred_sum[c] = count of pixels predicted as class c
        batch_pred_sum = torch.bincount(
            preds_flat, minlength=self.num_classes
        ).to(torch.float64)

        # target_sum[c] = count of pixels with ground truth class c
        batch_target_sum = torch.bincount(
            targets_flat, minlength=self.num_classes
        ).to(torch.float64)

        # intersection[c] = count of pixels where pred==c AND target==c
        # Create combined index: only count where pred == target
        match_mask = (preds_flat == targets_flat)
        matched_classes = targets_flat[match_mask]
        batch_intersection = torch.bincount(
            matched_classes, minlength=self.num_classes
        ).to(torch.float64)

        # Accumulate (all operations stay on GPU)
        self._intersection += batch_intersection
        self._pred_sum += batch_pred_sum
        self._target_sum += batch_target_sum

    def compute(self) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Compute Dice scores from accumulated statistics.

        Returns:
            dice_per_class: (num_classes,) Dice score for each class (CPU tensor)
            mean_dice: Mean Dice across classes present in targets
            valid_mask: (num_classes,) boolean mask of classes present in targets (CPU tensor)
        """
        # Get accumulators (handles None case via properties)
        intersection = self.intersection
        pred_sum = self.pred_sum
        target_sum = self.target_sum

        # Compute Dice per class
        dice_per_class = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )

        # Mask for classes present in targets (avoid inflating mean with absent classes)
        valid_mask = target_sum > 0

        # Mean Dice only over present classes
        if valid_mask.sum() > 0:
            mean_dice = dice_per_class[valid_mask].mean().item()
        else:
            mean_dice = 0.0

        # Return CPU tensors for compatibility with logging/visualization
        return dice_per_class.float().cpu(), mean_dice, valid_mask.cpu()

    def compute_iou(self) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Compute IoU (Jaccard index) from accumulated statistics.

        Returns:
            iou_per_class: (num_classes,) IoU for each class (CPU tensor)
            mean_iou: Mean IoU across classes present in targets
            valid_mask: (num_classes,) boolean mask of classes present in targets (CPU tensor)
        """
        intersection = self.intersection
        pred_sum = self.pred_sum
        target_sum = self.target_sum

        union = pred_sum + target_sum - intersection
        iou_per_class = (intersection + self.smooth) / (union + self.smooth)

        valid_mask = target_sum > 0
        if valid_mask.sum() > 0:
            mean_iou = iou_per_class[valid_mask].mean().item()
        else:
            mean_iou = 0.0

        return iou_per_class.float().cpu(), mean_iou, valid_mask.cpu()

    def compute_per_class_dict(self, class_names: list = None) -> dict:
        """
        Compute Dice scores as a dictionary.

        Args:
            class_names: Optional list of class names

        Returns:
            Dictionary mapping class index/name to Dice score
        """
        dice_per_class, mean_dice, valid_mask = self.compute()

        result = {"mean_dice": mean_dice}

        for c in range(self.num_classes):
            if valid_mask[c]:
                key = class_names[c] if class_names else f"class_{c}"
                result[key] = dice_per_class[c].item()

        return result

    def sync_across_processes(self):
        """Synchronize accumulators across all distributed processes.

        This method should be called before compute() when using DDP
        to aggregate metrics from all processes.
        """
        import torch.distributed as dist

        if self._intersection is None:
            return

        if not (dist.is_available() and dist.is_initialized()):
            return

        # All-reduce sums across all processes
        dist.all_reduce(self._intersection, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._pred_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._target_sum, op=dist.ReduceOp.SUM)
