"""TDD tests for config.py - Step 1"""
import pytest


def test_config_import():
    """Config module can be imported and instantiated."""
    from config import Config
    cfg = Config()
    assert cfg is not None


def test_config_default_values():
    """Config has correct default values matching the design plan."""
    from config import Config
    cfg = Config()

    # Data settings
    assert cfg.data_dir == "Dataset/voxel_data"
    assert cfg.split_file == "Dataset/dataset_split.json"
    assert cfg.num_classes == 70
    assert cfg.voxel_size == 4.0
    assert cfg.volume_size == (128, 96, 256)

    # Model settings
    assert cfg.in_channels == 1
    assert cfg.base_channels == 32
    assert cfg.num_levels == 4

    # Training settings
    assert cfg.batch_size == 4
    assert cfg.num_workers == 4
    assert cfg.epochs == 120
    assert cfg.lr == 1e-3
    assert cfg.weight_decay == 1e-5
    assert cfg.grad_clip == 1.0

    # Loss settings
    assert cfg.ce_weight == 0.5
    assert cfg.dice_weight == 0.5

    # LR scheduler settings
    assert cfg.lr_patience == 10
    assert cfg.lr_factor == 0.5

    # Checkpoint settings
    assert cfg.checkpoint_dir == "checkpoints"
    assert cfg.save_every == 10
    assert cfg.log_dir == "runs"

    # GPU settings
    assert cfg.gpu_ids == [0, 1]

    # Resume
    assert cfg.resume == ""


def test_config_volume_size_covers_max_dimensions():
    """Volume size (128, 96, 256) must cover max sample dims (117, 92, 241)."""
    from config import Config
    cfg = Config()
    max_dims = (117, 92, 241)
    assert cfg.volume_size[0] >= max_dims[0]
    assert cfg.volume_size[1] >= max_dims[1]
    assert cfg.volume_size[2] >= max_dims[2]


def test_config_is_dataclass():
    """Config should be a dataclass so fields are easily overridable."""
    from dataclasses import fields
    from config import Config
    cfg = Config()
    field_names = [f.name for f in fields(cfg)]
    assert "data_dir" in field_names
    assert "volume_size" in field_names
    assert "gpu_ids" in field_names


def test_config_override():
    """Config fields can be overridden at construction."""
    from config import Config
    cfg = Config(batch_size=2, epochs=10, gpu_ids=[0])
    assert cfg.batch_size == 2
    assert cfg.epochs == 10
    assert cfg.gpu_ids == [0]
