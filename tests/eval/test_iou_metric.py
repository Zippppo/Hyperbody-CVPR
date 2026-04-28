"""Tests for IoU metric computed from Dice accumulators."""

import pytest
import torch

from utils.metrics import DiceMetric


class TestIoUMetric:
    """Test IoU computation within DiceMetric."""

    def test_perfect_prediction_iou_is_one(self):
        """Perfect prediction should give IoU = 1.0 for all present classes."""
        num_classes = 4
        metric = DiceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        targets[0, :4, :, :] = 1
        targets[0, 4:, :4, :] = 2

        logits = torch.zeros(1, num_classes, 8, 8, 8)
        logits.scatter_(1, targets.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        iou_per_class, mean_iou, valid_mask = metric.compute_iou()

        for c in range(num_classes):
            if valid_mask[c]:
                assert iou_per_class[c].item() == pytest.approx(1.0, abs=1e-4)
        assert mean_iou == pytest.approx(1.0, abs=1e-4)

    def test_zero_overlap_iou_is_zero_for_target_class(self):
        """No overlap should give IoU ≈ 0 for the target class."""
        num_classes = 3
        metric = DiceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        targets[0, :4, :, :] = 1

        preds = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        preds[0, :4, :, :] = 2

        logits = torch.zeros(1, num_classes, 8, 8, 8)
        logits.scatter_(1, preds.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        iou_per_class, mean_iou, valid_mask = metric.compute_iou()

        assert valid_mask[1]
        assert iou_per_class[1].item() < 0.01
        assert mean_iou < 0.7

    def test_iou_dice_relationship(self):
        """IoU and Dice should satisfy: IoU = Dice / (2 - Dice)."""
        num_classes = 3
        metric = DiceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        targets[0, :8, :, :] = 1
        targets[0, 8:, :8, :] = 2

        preds = torch.zeros(1, 16, 16, 16, dtype=torch.long)
        preds[0, :10, :, :] = 1
        preds[0, 10:, :10, :] = 2

        logits = torch.zeros(1, num_classes, 16, 16, 16)
        logits.scatter_(1, preds.unsqueeze(1), 1.0)

        metric.update(logits, targets)
        dice_per_class, _, valid_mask = metric.compute()
        iou_per_class, _, _ = metric.compute_iou()

        for c in range(num_classes):
            if valid_mask[c]:
                dice_value = dice_per_class[c].item()
                expected_iou = dice_value / (2.0 - dice_value)
                assert iou_per_class[c].item() == pytest.approx(expected_iou, abs=1e-3)

    def test_iou_shape_and_dtype(self):
        """Return type should match Dice compute output conventions."""
        num_classes = 5
        metric = DiceMetric(num_classes=num_classes)

        targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        logits = torch.zeros(1, num_classes, 4, 4, 4)
        logits[:, 0, :, :, :] = 1.0
        metric.update(logits, targets)

        iou_per_class, mean_iou, valid_mask = metric.compute_iou()

        assert iou_per_class.shape == (num_classes,)
        assert valid_mask.shape == (num_classes,)
        assert isinstance(mean_iou, float)
        assert iou_per_class.device == torch.device("cpu")
