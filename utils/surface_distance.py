"""Surface-distance utilities for 3D segmentation metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch


def extract_surface_voxels(volume: np.ndarray, class_id: int) -> np.ndarray:
    """Return `(N, 3)` coordinates for boundary voxels of `class_id` in `volume`."""
    binary_mask = volume == class_id
    if not np.any(binary_mask):
        return np.empty((0, 3), dtype=np.float64)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(binary_mask, structure=structure)
    surface_mask = binary_mask & ~eroded

    return np.argwhere(surface_mask).astype(np.float64)


def compute_surface_distances(
    pred_surface: np.ndarray,
    gt_surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute symmetric nearest-neighbor distances between two surface point sets."""
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    gt_tree = cKDTree(gt_surface)
    pred_tree = cKDTree(pred_surface)

    d_pred_to_gt, _ = gt_tree.query(pred_surface)
    d_gt_to_pred, _ = pred_tree.query(gt_surface)

    return d_pred_to_gt.astype(np.float64), d_gt_to_pred.astype(np.float64)


class SurfaceDistanceMetric:
    """Accumulate per-class surface distances for HD95/NSD metrics."""

    def __init__(self, num_classes: int = 70, nsd_tolerance: float = 2.0):
        self.num_classes = num_classes
        self.nsd_tolerance = nsd_tolerance
        self._all_distances: list[list[tuple[np.ndarray, np.ndarray]]] = [
            [] for _ in range(num_classes)
        ]

    def reset(self):
        """Reset all per-class accumulators."""
        self._all_distances = [[] for _ in range(self.num_classes)]

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Compute and store per-sample symmetric surface distances.

        Args:
            logits: (B, C, D, H, W) model logits
            targets: (B, D, H, W) label volume
        """
        preds = logits.argmax(dim=1)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for batch_index in range(preds_np.shape[0]):
            pred_volume = preds_np[batch_index]
            gt_volume = targets_np[batch_index]

            present_classes = set(np.unique(pred_volume)) | set(np.unique(gt_volume))
            for class_id in present_classes:
                class_index = int(class_id)
                if class_index <= 0 or class_index >= self.num_classes:
                    continue

                pred_surface = extract_surface_voxels(pred_volume, class_index)
                gt_surface = extract_surface_voxels(gt_volume, class_index)

                if len(pred_surface) == 0 and len(gt_surface) == 0:
                    continue

                if len(pred_surface) == 0:
                    d_pred_to_gt = np.array([], dtype=np.float64)
                    d_gt_to_pred = np.full(len(gt_surface), np.inf, dtype=np.float64)
                elif len(gt_surface) == 0:
                    d_pred_to_gt = np.full(len(pred_surface), np.inf, dtype=np.float64)
                    d_gt_to_pred = np.array([], dtype=np.float64)
                else:
                    d_pred_to_gt, d_gt_to_pred = compute_surface_distances(
                        pred_surface, gt_surface
                    )

                self._all_distances[class_index].append((d_pred_to_gt, d_gt_to_pred))

    def compute_hd95(self) -> tuple[list[float], float, list[bool]]:
        """
        Compute HD95 per class as mean of per-sample 95th-percentile distances.

        Returns:
            hd95_per_class: list of class HD95 values (nan for absent classes)
            mean_hd95: mean over valid classes with finite HD95
            valid_mask: list indicating if a class has any observations
        """
        hd95_per_class = [float("nan")] * self.num_classes
        valid_mask = [False] * self.num_classes

        for class_id in range(self.num_classes):
            class_distances = self._all_distances[class_id]
            if not class_distances:
                continue

            per_sample_hd95 = []
            for d_pred_to_gt, d_gt_to_pred in class_distances:
                symmetric = np.concatenate([d_pred_to_gt, d_gt_to_pred])
                if symmetric.size == 0 or np.all(np.isinf(symmetric)):
                    per_sample_hd95.append(float("inf"))
                else:
                    per_sample_hd95.append(float(np.percentile(symmetric, 95)))

            finite_values = [value for value in per_sample_hd95 if np.isfinite(value)]
            hd95_per_class[class_id] = (
                float(np.mean(finite_values)) if finite_values else float("inf")
            )
            valid_mask[class_id] = True

        valid_values = [
            value
            for value, is_valid in zip(hd95_per_class, valid_mask)
            if is_valid and np.isfinite(value)
        ]
        mean_hd95 = float(np.mean(valid_values)) if valid_values else 0.0
        return hd95_per_class, mean_hd95, valid_mask

    def compute_nsd(self) -> tuple[list[float], float, list[bool]]:
        """
        Compute normalized surface Dice per class.

        NSD is the fraction of surface points with distance <= `nsd_tolerance`.

        Returns:
            nsd_per_class: list of class NSD values (nan for absent classes)
            mean_nsd: mean over valid classes
            valid_mask: list indicating if a class has any observations
        """
        nsd_per_class = [float("nan")] * self.num_classes
        valid_mask = [False] * self.num_classes

        for class_id in range(self.num_classes):
            class_distances = self._all_distances[class_id]
            if not class_distances:
                continue

            total_within = 0
            total_points = 0
            for d_pred_to_gt, d_gt_to_pred in class_distances:
                total_within += int(np.sum(d_pred_to_gt <= self.nsd_tolerance))
                total_within += int(np.sum(d_gt_to_pred <= self.nsd_tolerance))
                total_points += len(d_pred_to_gt) + len(d_gt_to_pred)

            nsd_value = (total_within / total_points) if total_points > 0 else 0.0
            nsd_per_class[class_id] = float(nsd_value)
            valid_mask[class_id] = True

        valid_values = [
            value
            for value, is_valid in zip(nsd_per_class, valid_mask)
            if is_valid and not np.isnan(value)
        ]
        mean_nsd = float(np.mean(valid_values)) if valid_values else 0.0
        return nsd_per_class, mean_nsd, valid_mask
