"""
Tests for lorentz_random inference script (pred_lorentz_random.py).

Tests:
1. Load BodyNet model with hyperbolic parameters
2. Handle DDP checkpoint loading (strip module. prefix)
3. Forward pass returns correct output shapes
4. Prediction saving format matches eval_all.py expectations
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from data.organ_hierarchy import load_organ_hierarchy
from models.body_net import BodyNet


def get_gpu_free_memory():
    """Get free GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0
    try:
        free_mem = torch.cuda.mem_get_info()[0] / (1024 * 1024)  # Convert to MB
        return free_mem
    except Exception:
        return 0


# Skip GPU tests if not enough memory (need at least 10GB for full volume)
GPU_MEMORY_THRESHOLD_MB = 10000
skip_if_low_gpu_memory = pytest.mark.skipif(
    get_gpu_free_memory() < GPU_MEMORY_THRESHOLD_MB,
    reason=f"Not enough GPU memory (need {GPU_MEMORY_THRESHOLD_MB}MB, have {get_gpu_free_memory():.0f}MB)"
)


class TestLoadBodyNet:
    """Test loading BodyNet model for lorentz_random config."""

    def test_load_class_depths(self):
        """Test loading class_depths from tree.json."""
        cfg = Config.from_yaml("configs/lorentz_random.yaml")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]

        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        print(f"Number of classes: {len(class_names)}")
        print(f"Number of class_depths: {len(class_depths)}")
        print(f"Sample depths: {list(class_depths.items())[:5]}")
        print(f"Depth range: [{min(class_depths.values())}, {max(class_depths.values())}]")

        assert len(class_depths) == cfg.num_classes
        assert all(d >= 1 for d in class_depths.values())

    def test_create_body_net(self):
        """Test creating BodyNet with lorentz_random config."""
        cfg = Config.from_yaml("configs/lorentz_random.yaml")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        model = BodyNet(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
            embed_dim=cfg.hyp_embed_dim,
            curv=cfg.hyp_curv,
            class_depths=class_depths,
            min_radius=cfg.hyp_min_radius,
            max_radius=cfg.hyp_max_radius,
            direction_mode=cfg.hyp_direction_mode,
            text_embedding_path=cfg.hyp_text_embedding_path,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

        assert model is not None
        assert hasattr(model, "unet")
        assert hasattr(model, "hyp_head")
        assert hasattr(model, "label_emb")


class TestLoadCheckpoint:
    """Test loading checkpoint for BodyNet."""

    def test_checkpoint_exists(self):
        """Test that checkpoint file exists."""
        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")

        print(f"Checkpoint path: {ckpt_path}")
        print(f"Exists: {os.path.exists(ckpt_path)}")

        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    def test_checkpoint_keys(self):
        """Test checkpoint contains expected keys."""
        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Best Dice: {checkpoint.get('best_dice', 'N/A')}")

        assert "model_state_dict" in checkpoint
        assert "epoch" in checkpoint

    def test_load_ddp_checkpoint(self):
        """Test loading DDP checkpoint (handles module. prefix)."""
        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        model = BodyNet(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
            embed_dim=cfg.hyp_embed_dim,
            curv=cfg.hyp_curv,
            class_depths=class_depths,
            min_radius=cfg.hyp_min_radius,
            max_radius=cfg.hyp_max_radius,
            direction_mode=cfg.hyp_direction_mode,
            text_embedding_path=cfg.hyp_text_embedding_path,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]

        # Check if DDP wrapped (keys start with "module.")
        is_ddp = any(k.startswith("module.") for k in state_dict.keys())
        print(f"Is DDP checkpoint: {is_ddp}")
        print(f"Sample keys: {list(state_dict.keys())[:5]}")

        # Strip "module." prefix if DDP
        if is_ddp:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            print(f"Stripped module. prefix")
            print(f"Sample keys after strip: {list(state_dict.keys())[:5]}")

        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint!")

        assert True  # If we get here, loading succeeded


class TestForwardPass:
    """Test forward pass of BodyNet."""

    @skip_if_low_gpu_memory
    def test_forward_output_shapes(self):
        """Test forward pass returns correct output shapes (with AMP for memory efficiency)."""
        from torch.amp import autocast

        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        model = BodyNet(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
            embed_dim=cfg.hyp_embed_dim,
            curv=cfg.hyp_curv,
            class_depths=class_depths,
            min_radius=cfg.hyp_min_radius,
            max_radius=cfg.hyp_max_radius,
            direction_mode=cfg.hyp_direction_mode,
            text_embedding_path=cfg.hyp_text_embedding_path,
        )
        model = model.to(device)
        model.eval()

        # Create dummy input: (B, C, D, H, W)
        B = 1
        D, H, W = cfg.volume_size  # (144, 128, 268)
        x = torch.randn(B, cfg.in_channels, D, H, W, device=device)

        # Use AMP for memory efficiency
        with torch.no_grad():
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits, voxel_emb, label_emb = model(x)
            else:
                logits, voxel_emb, label_emb = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Voxel embedding shape: {voxel_emb.shape}")
        print(f"Label embedding shape: {label_emb.shape}")

        # Expected shapes
        # logits: (B, num_classes, D, H, W)
        # voxel_emb: (B, embed_dim, D, H, W)
        # label_emb: (num_classes, embed_dim)
        assert logits.shape == (B, cfg.num_classes, D, H, W)
        assert voxel_emb.shape == (B, cfg.hyp_embed_dim, D, H, W)
        assert label_emb.shape == (cfg.num_classes, cfg.hyp_embed_dim)

    @skip_if_low_gpu_memory
    def test_inference_with_checkpoint(self):
        """Test inference with loaded checkpoint (with AMP for memory efficiency)."""
        from torch.amp import autocast

        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        model = BodyNet(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
            embed_dim=cfg.hyp_embed_dim,
            curv=cfg.hyp_curv,
            class_depths=class_depths,
            min_radius=cfg.hyp_min_radius,
            max_radius=cfg.hyp_max_radius,
            direction_mode=cfg.hyp_direction_mode,
            text_embedding_path=cfg.hyp_text_embedding_path,
        )

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        # Create dummy input
        D, H, W = cfg.volume_size
        x = torch.randn(1, cfg.in_channels, D, H, W, device=device)

        # Use AMP for memory efficiency
        with torch.no_grad():
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits, voxel_emb, label_emb = model(x)
            else:
                logits, voxel_emb, label_emb = model(x)

        pred_labels = logits.argmax(dim=1).squeeze(0)

        print(f"Pred labels shape: {pred_labels.shape}")
        print(f"Pred labels dtype: {pred_labels.dtype}")
        print(f"Unique classes: {torch.unique(pred_labels).tolist()[:10]}...")

        assert pred_labels.shape == (D, H, W)


class TestInferenceOnRealData:
    """Test inference on real test data."""

    @skip_if_low_gpu_memory
    def test_inference_single_sample(self):
        """Test inference on a single test sample (with AMP for memory efficiency)."""
        from torch.amp import autocast
        from data.dataset import HyperBodyDataset

        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        # Load model
        model = BodyNet(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
            embed_dim=cfg.hyp_embed_dim,
            curv=cfg.hyp_curv,
            class_depths=class_depths,
            min_radius=cfg.hyp_min_radius,
            max_radius=cfg.hyp_max_radius,
            direction_mode=cfg.hyp_direction_mode,
            text_embedding_path=cfg.hyp_text_embedding_path,
        )

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Load test dataset
        test_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "test", cfg.volume_size)
        print(f"Test samples: {len(test_dataset)}")

        # Inference on first sample
        inp, gt = test_dataset[0]
        inp = inp.unsqueeze(0).to(device)

        # Use AMP for memory efficiency
        with torch.no_grad():
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits, voxel_emb, label_emb = model(inp)
            else:
                logits, voxel_emb, label_emb = model(inp)

        pred = logits.argmax(dim=1).squeeze(0).cpu()

        print(f"Input shape: {inp.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Pred shape: {pred.shape}")
        print(f"GT shape: {gt.shape}")
        print(f"Unique pred classes: {torch.unique(pred).tolist()[:10]}...")

        assert pred.shape == gt.shape


class TestSavePrediction:
    """Test saving prediction in correct format."""

    @skip_if_low_gpu_memory
    def test_save_prediction_format(self):
        """Test that prediction is saved in format compatible with eval_all.py (with AMP)."""
        from torch.amp import autocast
        from data.dataset import HyperBodyDataset

        cfg = Config.from_yaml("configs/lorentz_random.yaml")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pth")

        with open(cfg.dataset_info_file) as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

        # Load model
        model = BodyNet(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            base_channels=cfg.base_channels,
            growth_rate=cfg.growth_rate,
            dense_layers=cfg.dense_layers,
            bn_size=cfg.bn_size,
            embed_dim=cfg.hyp_embed_dim,
            curv=cfg.hyp_curv,
            class_depths=class_depths,
            min_radius=cfg.hyp_min_radius,
            max_radius=cfg.hyp_max_radius,
            direction_mode=cfg.hyp_direction_mode,
            text_embedding_path=cfg.hyp_text_embedding_path,
        )

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Load test dataset
        test_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "test", cfg.volume_size)
        inp, _ = test_dataset[0]
        inp = inp.unsqueeze(0).to(device)

        # Inference with AMP
        with torch.no_grad():
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits, _, _ = model(inp)
            else:
                logits, _, _ = model(inp)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
