"""Integration tests for multi-metric evaluation in eval_all.py."""

import numpy as np

from eval.eval_all import evaluate_model


class TestEvalAllMetrics:
    """Test output fields for evaluate_model."""

    def test_result_contains_all_metrics(self, tmp_path):
        """evaluate_model should expose all requested mean metrics."""
        num_classes = 4
        volume_size = (8, 8, 8)

        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        gt_labels = np.zeros(volume_size, dtype=np.int64)
        gt_labels[1:5, 1:5, 1:5] = 1
        gt_labels[5:7, 5:7, 5:7] = 2
        np.savez(gt_dir / "sample_001.npz", voxel_labels=gt_labels)

        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        np.savez(
            pred_dir / "sample_001.npz",
            pred_labels=gt_labels,
            original_filename="sample_001.npz",
        )

        result = evaluate_model(str(pred_dir), str(gt_dir), num_classes)

        assert "mean_dice" in result
        assert "mean_iou" in result
        assert "mean_hd95" in result
        assert "mean_nsd" in result
        assert "rib_mean_dice" in result

        assert result["mean_dice"] > 0.99
        assert result["mean_iou"] > 0.99
        assert result["mean_hd95"] < 0.1
        assert result["mean_nsd"] > 0.99

    def test_result_per_class_contains_all_metrics(self, tmp_path):
        """Each valid class should include dice/iou/hd95/nsd entries."""
        num_classes = 3
        volume_size = (8, 8, 8)

        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        gt_labels = np.zeros(volume_size, dtype=np.int64)
        gt_labels[2:6, 2:6, 2:6] = 1
        np.savez(gt_dir / "s1.npz", voxel_labels=gt_labels)

        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        np.savez(
            pred_dir / "s1.npz",
            pred_labels=gt_labels,
            original_filename="s1.npz",
        )

        result = evaluate_model(str(pred_dir), str(gt_dir), num_classes)

        for class_metrics in result["per_class"].values():
            assert "dice" in class_metrics
            assert "iou" in class_metrics
            assert "hd95" in class_metrics
            assert "nsd" in class_metrics
