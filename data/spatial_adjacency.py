"""Compute spatial adjacency (contact) matrix between organs from GT labels."""

import logging

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


logger = logging.getLogger(__name__)


_IGNORED_SPATIAL_CLASS_NAMES = {
    "inside_body_empty",
}


def infer_ignored_spatial_class_indices(class_names: Sequence[str]) -> Tuple[int, ...]:
    """Infer class indices that should be excluded from spatial adjacency computation."""
    ignored_indices = []
    for class_idx, class_name in enumerate(class_names):
        if class_name.strip().lower() in _IGNORED_SPATIAL_CLASS_NAMES:
            ignored_indices.append(class_idx)
    return tuple(ignored_indices)


def _normalize_ignored_class_indices(
    num_classes: int,
    ignored_class_indices: Optional[Iterable[int]],
) -> Tuple[int, ...]:
    """Validate and canonicalize ignored class indices."""
    if ignored_class_indices is None:
        return ()

    normalized = []
    for class_idx in ignored_class_indices:
        class_idx = int(class_idx)
        if class_idx < 0 or class_idx >= num_classes:
            raise ValueError(
                f"Ignored class index {class_idx} is out of range for num_classes={num_classes}"
            )
        normalized.append(class_idx)

    return tuple(sorted(set(normalized)))


def _apply_ignored_class_mask(
    overlap: torch.Tensor,
    volume: torch.Tensor,
    ignored_class_indices: Tuple[int, ...],
) -> None:
    """Zero-out ignored classes in overlap + volume tensors in-place."""
    if not ignored_class_indices:
        return

    ignored_tensor = torch.as_tensor(ignored_class_indices, dtype=torch.long, device=overlap.device)
    overlap.index_fill_(0, ignored_tensor, 0.0)
    overlap.index_fill_(1, ignored_tensor, 0.0)
    volume.index_fill_(0, ignored_tensor, 0.0)


def _apply_ignored_pair_mask(matrix: torch.Tensor, ignored_class_indices: Tuple[int, ...]) -> None:
    """Zero-out rows/cols of ignored classes in a pairwise matrix in-place."""
    if not ignored_class_indices:
        return

    ignored_tensor = torch.as_tensor(ignored_class_indices, dtype=torch.long, device=matrix.device)
    matrix.index_fill_(0, ignored_tensor, 0.0)
    matrix.index_fill_(1, ignored_tensor, 0.0)


def _compute_single_sample_overlap_chunked(
    labels: torch.Tensor,
    num_classes: int,
    dilation_radius: int,
    class_batch_size: int,
    ignored_class_indices: Tuple[int, ...] = (),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-safe chunked overlap computation when full one-hot is too large."""
    kernel = 2 * dilation_radius + 1
    overlap = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=labels.device)
    volume = torch.bincount(labels.reshape(-1), minlength=num_classes).to(torch.float32)

    class_ids = torch.arange(num_classes, device=labels.device)
    labels_view = labels.unsqueeze(0)

    for src_start in range(0, num_classes, class_batch_size):
        src_ids = class_ids[src_start:src_start + class_batch_size]
        src_masks = (labels_view == src_ids.view(-1, 1, 1, 1)).to(torch.float32)
        src_dilated = F.max_pool3d(
            src_masks.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=dilation_radius,
        ).squeeze(1)

        for tgt_start in range(0, num_classes, class_batch_size):
            tgt_ids = class_ids[tgt_start:tgt_start + class_batch_size]
            tgt_masks = (labels_view == tgt_ids.view(-1, 1, 1, 1)).to(torch.float32)

            sub_overlap = torch.einsum("sdhw,tdhw->st", src_dilated, tgt_masks)
            overlap[
                src_start:src_start + src_ids.numel(),
                tgt_start:tgt_start + tgt_ids.numel(),
            ] = sub_overlap

    _apply_ignored_class_mask(overlap, volume, ignored_class_indices)
    return overlap, volume


def _compute_single_sample_overlap(
    labels: torch.Tensor,
    num_classes: int,
    dilation_radius: int = 3,
    class_batch_size: int = 0,
    ignored_class_indices: Optional[Iterable[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise overlap between dilated organ masks for one sample."""
    if labels.dim() != 3:
        raise ValueError(f"Expected labels with shape (D, H, W), got {tuple(labels.shape)}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    labels = labels.long()
    ignored_class_indices = _normalize_ignored_class_indices(num_classes, ignored_class_indices)

    if class_batch_size > 0 and class_batch_size < num_classes:
        return _compute_single_sample_overlap_chunked(
            labels=labels,
            num_classes=num_classes,
            dilation_radius=dilation_radius,
            class_batch_size=class_batch_size,
            ignored_class_indices=ignored_class_indices,
        )

    kernel = 2 * dilation_radius + 1

    try:
        # (D,H,W) -> (C,D,H,W)
        one_hot = F.one_hot(labels, num_classes=num_classes).permute(3, 0, 1, 2).float()

        volume = one_hot.sum(dim=(1, 2, 3))

        # Dilate each class mask independently with a cube kernel.
        dilated = F.max_pool3d(
            one_hot.unsqueeze(1),
            kernel_size=kernel,
            stride=1,
            padding=dilation_radius,
        ).squeeze(1)

        # overlap[u, v] = sum(dilated_u * original_v)
        overlap = torch.einsum("cdhw,kdhw->ck", dilated, one_hot)
        _apply_ignored_class_mask(overlap, volume, ignored_class_indices)
        return overlap, volume
    except RuntimeError as err:
        if "out of memory" not in str(err).lower():
            raise
        if labels.device.type == "cuda":
            torch.cuda.empty_cache()
        fallback_batch_size = min(8, num_classes)
        return _compute_single_sample_overlap_chunked(
            labels=labels,
            num_classes=num_classes,
            dilation_radius=dilation_radius,
            class_batch_size=fallback_batch_size,
            ignored_class_indices=ignored_class_indices,
        )


def compute_contact_matrix_from_dataset(
    dataset: Dataset,
    num_classes: int,
    dilation_radius: int = 3,
    num_workers: int = 0,
    class_batch_size: int = 0,
    ignored_class_indices: Optional[Iterable[int]] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Aggregate asymmetric contact matrix over all samples in a dataset."""
    ignored_class_indices = _normalize_ignored_class_indices(num_classes, ignored_class_indices)

    global_overlap = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    global_volume = torch.zeros((num_classes,), dtype=torch.float64)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    total = len(loader)
    last_logged_decile = -1

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, total=total, desc="Contact matrix", unit="sample")

    for index, (_, lbl) in enumerate(iterator):
        labels = lbl.squeeze(0).long()
        overlap, volume = _compute_single_sample_overlap(
            labels=labels,
            num_classes=num_classes,
            dilation_radius=dilation_radius,
            class_batch_size=class_batch_size,
            ignored_class_indices=ignored_class_indices,
        )
        global_overlap += overlap.double().cpu()
        global_volume += volume.double().cpu()

        if total > 0:
            progress_percent = (100 * (index + 1)) // total
            decile = progress_percent // 10
            if decile > last_logged_decile or progress_percent == 100:
                logger.info(
                    "Contact matrix: %d/%d samples (%d%%)",
                    index + 1,
                    total,
                    progress_percent,
                )
                last_logged_decile = decile

    contact_matrix = global_overlap / global_volume.unsqueeze(1).clamp(min=1.0)
    contact_matrix = contact_matrix.clamp(min=0.0, max=1.0)
    contact_matrix.fill_diagonal_(0.0)
    _apply_ignored_pair_mask(contact_matrix, ignored_class_indices)
    return contact_matrix.float()


def compute_graph_distance_matrix(
    D_tree: torch.Tensor,
    contact_matrix: torch.Tensor,
    lambda_: float = 1.0,
    epsilon: float = 0.01,
    ignored_class_indices: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Fuse tree distance and spatial contact with per-pair minimum."""
    if D_tree.shape != contact_matrix.shape:
        raise ValueError(
            f"D_tree shape {tuple(D_tree.shape)} must match contact shape {tuple(contact_matrix.shape)}"
        )

    ignored_class_indices = _normalize_ignored_class_indices(
        num_classes=D_tree.shape[0],
        ignored_class_indices=ignored_class_indices,
    )

    D_spatial = lambda_ / (contact_matrix + epsilon)
    D_final = torch.min(D_tree, D_spatial)

    if ignored_class_indices:
        ignored_tensor = torch.as_tensor(
            ignored_class_indices,
            dtype=torch.long,
            device=D_final.device,
        )
        D_final[ignored_tensor, :] = D_tree[ignored_tensor, :]
        D_final[:, ignored_tensor] = D_tree[:, ignored_tensor]

    D_final.fill_diagonal_(0.0)
    return D_final.float()
