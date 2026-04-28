"""Tests for NSD (normalized surface Dice) metric."""

import torch

from utils.surface_distance import SurfaceDistanceMetric


class TestNSDMetric:
    """Test NSD computation."""

    def test_perfect_prediction_nsd_is_one(self):
        """Perfect prediction should give NSD = 1.0 for valid classes."""
        num_classes = 3
        metric = SurfaceDistanceMetric(num_classes=num_classes, nsd_tolerance=2.0)

        targets = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        targets[0, 2:8, 2:8, 2:8] = 1
        targets[0, 9:14, 9:14, 9:14] = 2

        logits = torch.zeros(1, num_classes, 16, 16, 16)
        logits.scatter_(1, targets.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        nsd_per_class, mean_nsd, valid_mask = metric.compute_nsd()

        for c in range(num_classes):
            if valid_mask[c]:
                assert nsd_per_class[c] == 1.0
        assert mean_nsd == 1.0

    def test_shifted_within_tolerance(self):
        """Small shifts within tolerance should keep NSD high."""
        num_classes = 2
        metric = SurfaceDistanceMetric(num_classes=num_classes, nsd_tolerance=2.0)

        targets = torch.zeros(1, 20, 20, 20, dtype=torch.long)
        targets[0, 5:15, 5:15, 5:15] = 1

        preds = torch.zeros(1, 20, 20, 20, dtype=torch.long)
        preds[0, 6:16, 5:15, 5:15] = 1

        logits = torch.zeros(1, num_classes, 20, 20, 20)
        logits.scatter_(1, preds.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        nsd_per_class, _, valid_mask = metric.compute_nsd()

        assert valid_mask[1]
        assert nsd_per_class[1] > 0.7

    def test_large_shift_beyond_tolerance(self):
        """Large shifts beyond tolerance should produce low NSD."""
        num_classes = 2
        metric = SurfaceDistanceMetric(num_classes=num_classes, nsd_tolerance=1.0)

        targets = torch.zeros(1, 32, 32, 32, dtype=torch.long)
        targets[0, 2:10, 2:10, 2:10] = 1

        preds = torch.zeros(1, 32, 32, 32, dtype=torch.long)
        preds[0, 20:28, 20:28, 20:28] = 1

        logits = torch.zeros(1, num_classes, 32, 32, 32)
        logits.scatter_(1, preds.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        nsd_per_class, _, valid_mask = metric.compute_nsd()

        assert valid_mask[1]
        assert nsd_per_class[1] < 0.1

    def test_nsd_range_zero_to_one(self):
        """NSD should always be in [0, 1]."""
        num_classes = 2
        metric = SurfaceDistanceMetric(num_classes=num_classes, nsd_tolerance=2.0)

        targets = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        targets[0, 2:8, 2:8, 2:8] = 1

        preds = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        preds[0, 4:10, 4:10, 4:10] = 1

        logits = torch.zeros(1, num_classes, 16, 16, 16)
        logits.scatter_(1, preds.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        nsd_per_class, _, valid_mask = metric.compute_nsd()

        for c in range(num_classes):
            if valid_mask[c]:
                assert 0.0 <= nsd_per_class[c] <= 1.0
