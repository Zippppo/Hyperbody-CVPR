"""Tests for HD95 metric aggregation."""

import pytest
import torch

from utils.surface_distance import SurfaceDistanceMetric


class TestHD95Metric:
    """Test HD95 computation on toy volumes."""

    def test_perfect_prediction_hd95_is_zero(self):
        """Perfect prediction should give HD95 = 0 for valid classes."""
        num_classes = 3
        metric = SurfaceDistanceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        targets[0, 2:8, 2:8, 2:8] = 1
        targets[0, 9:14, 9:14, 9:14] = 2

        logits = torch.zeros(1, num_classes, 16, 16, 16)
        logits.scatter_(1, targets.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        hd95_per_class, mean_hd95, valid_mask = metric.compute_hd95()

        for c in range(num_classes):
            if valid_mask[c]:
                assert hd95_per_class[c] == pytest.approx(0.0, abs=1e-4)
        assert mean_hd95 == pytest.approx(0.0, abs=1e-4)

    def test_shifted_prediction_hd95(self):
        """A one-voxel shift should produce HD95 around 1."""
        num_classes = 2
        metric = SurfaceDistanceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        targets[0, 4:12, 4:12, 4:12] = 1

        preds = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        preds[0, 5:13, 4:12, 4:12] = 1

        logits = torch.zeros(1, num_classes, 16, 16, 16)
        logits.scatter_(1, preds.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        hd95_per_class, _, valid_mask = metric.compute_hd95()

        assert valid_mask[1]
        assert 0.5 < hd95_per_class[1] < 2.0

    def test_multi_sample_accumulation(self):
        """HD95 should accumulate over multiple update calls."""
        num_classes = 2
        metric = SurfaceDistanceMetric(num_classes=num_classes)

        for _ in range(3):
            targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
            targets[0, 2:6, 2:6, 2:6] = 1
            logits = torch.zeros(1, num_classes, 8, 8, 8)
            logits.scatter_(1, targets.unsqueeze(1), 1.0)
            metric.update(logits, targets)

        hd95_per_class, mean_hd95, valid_mask = metric.compute_hd95()

        assert valid_mask[1]
        assert hd95_per_class[1] == pytest.approx(0.0, abs=1e-4)
        assert mean_hd95 == pytest.approx(0.0, abs=1e-4)

    def test_hd95_return_shapes(self):
        """Return values should match expected shape/types."""
        num_classes = 4
        metric = SurfaceDistanceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        targets[0, :4, :, :] = 1
        logits = torch.zeros(1, num_classes, 8, 8, 8)
        logits.scatter_(1, targets.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        hd95_per_class, mean_hd95, valid_mask = metric.compute_hd95()

        assert len(hd95_per_class) == num_classes
        assert len(valid_mask) == num_classes
        assert isinstance(mean_hd95, float)
