## PyTorch Dataset for HyperBody voxel data

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data.voxelizer import pad_labels, voxelize_point_cloud


class HyperBodyDataset(Dataset):
    def __init__(self, data_dir: str, split_file: str, split: str, volume_size: tuple):
        """
        Args:
            data_dir: path to Dataset/voxel_data/
            split_file: path to dataset_split.json
            split: 'train', 'val', or 'test'
            volume_size: (128, 96, 256)
        """
        with open(split_file) as f:
            splits = json.load(f)

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

        self.filenames = splits[split]
        self.data_dir = data_dir
        self.volume_size = volume_size

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

        # Pad labels -> (X, Y, Z) int64
        labels = pad_labels(data["voxel_labels"], self.volume_size)

        # Convert to tensors: input has channel dim (1, X, Y, Z)
        inp = torch.from_numpy(occupancy).unsqueeze(0)
        lbl = torch.from_numpy(labels)

        return inp, lbl
