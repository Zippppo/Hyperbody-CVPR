"""Tests for surface distance helper utilities."""

import numpy as np

from utils.surface_distance import compute_surface_distances, extract_surface_voxels


class TestSurfaceExtraction:
    """Test surface voxel extraction and point distance helpers."""

    def test_cube_surface(self):
        """A solid cube should return only boundary voxels as surface."""
        volume = np.zeros((16, 16, 16), dtype=np.int64)
        volume[4:12, 4:12, 4:12] = 1

        surface = extract_surface_voxels(volume, class_id=1)

        assert surface.shape[1] == 3
        assert len(surface) > 0
        assert len(surface) < 8 ** 3
        assert np.all(surface >= 4)
        assert np.all(surface < 12)

    def test_single_voxel_is_all_surface(self):
        """A single voxel object has itself as the full surface."""
        volume = np.zeros((8, 8, 8), dtype=np.int64)
        volume[4, 4, 4] = 1

        surface = extract_surface_voxels(volume, class_id=1)

        assert len(surface) == 1
        assert np.array_equal(surface[0], [4, 4, 4])

    def test_empty_class_returns_empty(self):
        """Absent class should produce an empty coordinate array."""
        volume = np.zeros((8, 8, 8), dtype=np.int64)

        surface = extract_surface_voxels(volume, class_id=1)

        assert len(surface) == 0
        assert surface.shape == (0, 3)

    def test_surface_distances_identical(self):
        """Identical point sets should have zero nearest-neighbor distances."""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)

        d_pred_to_gt, d_gt_to_pred = compute_surface_distances(points, points)

        assert np.allclose(d_pred_to_gt, 0.0)
        assert np.allclose(d_gt_to_pred, 0.0)

    def test_surface_distances_known_offset(self):
        """Shifted collinear point sets should have known asymmetric distances."""
        gt_points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        pred_points = gt_points + np.array([3, 0, 0], dtype=np.float64)

        d_pred_to_gt, d_gt_to_pred = compute_surface_distances(pred_points, gt_points)

        assert np.allclose(d_pred_to_gt, [1.0, 2.0, 3.0])
        assert np.allclose(d_gt_to_pred, [3.0, 2.0, 1.0])
