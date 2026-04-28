## 用于将输入的点云数据体素化。可视化结果已经检查，没有任何问题

import numpy as np


def voxelize_point_cloud(
    sensor_pc: np.ndarray,
    grid_world_min: np.ndarray,
    grid_voxel_size: np.ndarray,
    volume_size: tuple,
) -> np.ndarray:
    """Convert point cloud to binary occupancy grid.

    Args:
        sensor_pc: (N, 3) float32 point coordinates
        grid_world_min: (3,) float32 world-space origin
        grid_voxel_size: (3,) float32 voxel dimensions
        volume_size: (X, Y, Z) target volume shape

    Returns:
        Binary occupancy grid, shape=volume_size, dtype=float32
    """
    # Compute voxel indices
    idx = np.floor((sensor_pc - grid_world_min) / grid_voxel_size).astype(np.int64)

    # Clip to valid range
    for d in range(3):
        idx[:, d] = np.clip(idx[:, d], 0, volume_size[d] - 1)

    # Create binary volume
    volume = np.zeros(volume_size, dtype=np.float32)
    volume[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return volume


def pad_labels(
    voxel_labels: np.ndarray,
    volume_size: tuple,
) -> np.ndarray:
    """Crop and/or pad voxel labels to fixed volume size.

    Args:
        voxel_labels: (X, Y, Z) uint8, variable size
        volume_size: (X, Y, Z) target shape

    Returns:
        Cropped/padded labels, shape=volume_size, dtype=int64
    """
    padded = np.zeros(volume_size, dtype=np.int64)
    x, y, z = voxel_labels.shape
    # 取输入和目标的最小值，实现裁剪
    cx, cy, cz = min(x, volume_size[0]), min(y, volume_size[1]), min(z, volume_size[2])
    padded[:cx, :cy, :cz] = voxel_labels[:cx, :cy, :cz].astype(np.int64)
    return padded
