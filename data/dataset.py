## PyTorch Dataset for HyperBody voxel data

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from data.voxelizer import pad_labels, voxelize_point_cloud


def fold_outside_label(
    voxel_labels: np.ndarray,
    outside_label: Optional[int],
    label_pad_value: int,
) -> np.ndarray:
    """Fold a raw dataset outside marker into the configured empty/pad label."""
    if outside_label is None:
        return voxel_labels
    return np.where(voxel_labels == outside_label, label_pad_value, voxel_labels)


class HyperBodyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split_file: str,
        split: str,
        volume_size: tuple,
        label_pad_value: int = 0,
        outside_label: Optional[int] = None,
    ):
        """
        Args:
            data_dir: path to Dataset/voxel_data/
            split_file: path to dataset_split.json
            split: 'train', 'val', or 'test'
            volume_size: (128, 96, 256)
            label_pad_value: label assigned to voxels outside the original volume
                             when padding to ``volume_size``.
            outside_label: if set, any voxel in the raw on-disk ``voxel_labels``
                           equal to this value will be remapped to
                           ``label_pad_value``. Use this to fold a dataset-level
                           "outside body" / ignore marker (e.g. 255) into an
                           ordinary class (e.g. 0 = inside_body_empty), matching
                           the PaSCo-style behaviour where outside_empty has no
                           independent class or loss.
        """
        with open(split_file) as f:
            splits = json.load(f)

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        self.filenames = splits[split]
        self.data_dir = data_dir
        self.volume_size = volume_size
        self.label_pad_value = label_pad_value
        self.outside_label = outside_label

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.filenames[idx])
        data = np.load(path)

        # Voxelize point cloud -> binary occupancy (X, Y, Z)
        occupancy = voxelize_point_cloud(
            data["sensor_pc"],
            data["grid_world_min"],
            data["grid_voxel_size"],
            self.volume_size,
        )

        # Optionally fold a dataset-level "outside body" marker into the pad
        # value (e.g. 255 -> 0 to match the PaSCo-style behaviour where
        # outside_empty has no independent class).
        raw_voxel_labels = fold_outside_label(
            data["voxel_labels"],
            self.outside_label,
            self.label_pad_value,
        )

        # Pad labels -> (X, Y, Z) int64
        labels = pad_labels(
            raw_voxel_labels,
            self.volume_size,
            fill_value=self.label_pad_value,
        )

        # Convert to tensors: input has channel dim (1, X, Y, Z)
        inp = torch.from_numpy(occupancy).unsqueeze(0)
        lbl = torch.from_numpy(labels)

        return inp, lbl
