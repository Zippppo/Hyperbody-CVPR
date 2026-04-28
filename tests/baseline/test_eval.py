"""
Tests for evaluation pipeline.
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class TestPredBaseline:
    """Test prediction script for baseline model."""

    def test_load_unet3d_checkpoint(self):
        """Test loading UNet3D checkpoint (baseline model without hyperbolic)."""
        from models.unet3d import UNet3D

        cfg = Config.from_yaml("configs/baseline.yaml")
        device = torch.device("cpu")

        model = UNet3D(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
        )

        ckpt_path = "checkpoints/best.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Loaded checkpoint from {ckpt_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best Dice: {checkpoint.get('best_dice', 'N/A'):.4f}")

        assert checkpoint["best_dice"] > 0.3, "Expected best_dice > 0.3"

    def test_inference_single_sample(self):
        """Test inference on a single test sample."""
        from models.unet3d import UNet3D
        from data.dataset import HyperBodyDataset

        cfg = Config.from_yaml("configs/baseline.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model = UNet3D(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
        )
        checkpoint = torch.load("checkpoints/best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Load one test sample
        test_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "test", cfg.volume_size)
        inp, gt = test_dataset[0]
        inp = inp.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(inp)

        pred = logits.argmax(dim=1).squeeze(0).cpu()

        print(f"Input shape: {inp.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Pred shape: {pred.shape}")
        print(f"GT shape: {gt.shape}")
        print(f"Unique pred classes: {torch.unique(pred).tolist()[:10]}...")

        assert pred.shape == gt.shape, f"Shape mismatch: {pred.shape} vs {gt.shape}"
        assert pred.dtype == torch.int64, f"Expected int64, got {pred.dtype}"

    def test_save_prediction_format(self):
        """Test that prediction is saved in correct format."""
        from models.unet3d import UNet3D
        from data.dataset import HyperBodyDataset

        cfg = Config.from_yaml("configs/baseline.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model = UNet3D(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
        )
        checkpoint = torch.load("checkpoints/best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        # Load one test sample
        test_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "test", cfg.volume_size)
        inp, _ = test_dataset[0]
        inp = inp.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(inp)
        pred_labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # Load original data for metadata
        filename = test_dataset.filenames[0]
        original_path = os.path.join(cfg.data_dir, filename)
        original_data = np.load(original_path)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_path = f.name

        np.savez_compressed(
            temp_path,
            pred_labels=pred_labels.astype(np.int64),
            grid_world_min=original_data["grid_world_min"],
            grid_voxel_size=original_data["grid_voxel_size"],
            original_filename=filename,
        )

        # Verify saved file
        saved = np.load(temp_path)
        print(f"Saved keys: {list(saved.keys())}")
        print(f"pred_labels shape: {saved['pred_labels'].shape}")
        print(f"pred_labels dtype: {saved['pred_labels'].dtype}")
        print(f"original_filename: {saved['original_filename']}")

        assert "pred_labels" in saved
        assert "grid_world_min" in saved
        assert "grid_voxel_size" in saved
        assert "original_filename" in saved
        assert "voxel_labels" not in saved, "GT labels should NOT be in prediction file!"

        os.unlink(temp_path)


class TestEvalAll:
    """Test unified evaluation script."""

    def test_dice_metric_from_predictions(self):
        """Test computing Dice from saved predictions."""
        from utils.metrics import DiceMetric

        num_classes = 70
        volume_size = (144, 128, 268)

        # Create fake prediction and GT
        pred_labels = np.random.randint(0, num_classes, volume_size, dtype=np.int64)
        gt_labels = np.random.randint(0, num_classes, volume_size, dtype=np.int64)

        # Make some overlap to get non-zero Dice
        gt_labels[:50, :50, :50] = 1
        pred_labels[:50, :50, :50] = 1

        metric = DiceMetric(num_classes=num_classes)

        # Convert to tensors (DiceMetric expects logits, so create fake logits)
        pred_tensor = torch.from_numpy(pred_labels).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_labels).unsqueeze(0)

        fake_logits = torch.zeros(1, num_classes, *volume_size)
        fake_logits.scatter_(1, pred_tensor.unsqueeze(1), 1.0)

        metric.update(fake_logits, gt_tensor)
        dice_per_class, mean_dice, valid_mask = metric.compute()

        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Dice for class 1: {dice_per_class[1]:.4f}")
        print(f"Valid classes: {valid_mask.sum().item()}")

        assert mean_dice > 0, "Expected non-zero mean Dice"
        assert dice_per_class[1] > 0.5, "Expected high Dice for class 1 (forced overlap)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
