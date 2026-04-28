import importlib
import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _prefer_current_repo_modules():
    while ROOT in sys.path:
        sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)

    for module_name in list(sys.modules):
        if (
            module_name == "config"
            or module_name in {"data", "models", "utils"}
            or module_name.startswith(("data.", "models.", "utils."))
        ):
            module = sys.modules[module_name]
            module_file = getattr(module, "__file__", "") or ""
            if module_file and not os.path.abspath(module_file).startswith(ROOT):
                del sys.modules[module_name]


_prefer_current_repo_modules()


@pytest.fixture(autouse=True)
def _current_repo_imports():
    _prefer_current_repo_modules()


def test_s2i_config_loads_ignore_fields_and_existing_paths():
    from config import Config

    cfg = Config.from_yaml("configs/s2i.yaml")

    assert cfg.target_ignore_index == 255
    assert cfg.label_pad_value == 255
    assert cfg.dataset_info_file == "S2I_Dataset/dataset_info.json"
    assert os.path.exists(cfg.dataset_info_file)
    assert cfg.hyp_distance_mode == "hyperbolic"
    assert cfg.hyp_direction_mode == "random"
    assert cfg.graph_distance_matrix == ""
    assert cfg.hyp_text_embedding_path == ""


def test_build_s2i_split_script_writes_disjoint_train_val_test(tmp_path, monkeypatch):
    data_root = tmp_path / "S2I_Dataset"
    train_dir = data_root / "train"
    train_dir.mkdir(parents=True)
    for i in range(20):
        (train_dir / f"BDMAP_{i:08d}.npz").write_bytes(b"placeholder")

    mod = importlib.import_module("scripts.build_s2i_split")
    out = data_root / "dataset_split.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_s2i_split.py",
            "--data-root",
            str(data_root),
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.1",
            "--seed",
            "7",
            "--output",
            str(out),
        ],
    )

    mod.main()

    payload = json.loads(out.read_text())
    train, val, test = map(set, (payload["train"], payload["val"], payload["test"]))
    assert len(train) == 14
    assert len(val) == 4
    assert len(test) == 2
    assert not (train & val or train & test or val & test)
    assert all(p.startswith("train/") for p in train | val | test)


def test_pad_labels_supports_custom_fill_value():
    from data.voxelizer import pad_labels

    labels = np.ones((2, 2, 2), dtype=np.uint8) * 3
    result = pad_labels(labels, (3, 4, 2), fill_value=255)

    assert result.dtype == np.int64
    np.testing.assert_array_equal(result[:2, :2, :], labels.astype(np.int64))
    assert np.all(result[2:, :, :] == 255)
    assert np.all(result[:, 2:, :] == 255)


def test_dataset_passes_label_pad_value_to_pad_labels(tmp_path):
    from data.dataset import HyperBodyDataset

    data_root = tmp_path / "S2I_Dataset"
    train_dir = data_root / "train"
    train_dir.mkdir(parents=True)
    sample_name = "train/sample.npz"
    np.savez(
        data_root / sample_name,
        sensor_pc=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        grid_world_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        grid_voxel_size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        voxel_labels=np.ones((1, 1, 1), dtype=np.uint8) * 4,
    )
    split_file = data_root / "dataset_split.json"
    split_file.write_text(json.dumps({"train": [sample_name], "val": [], "test": []}))

    ds = HyperBodyDataset(
        str(data_root),
        str(split_file),
        "train",
        (2, 2, 2),
        label_pad_value=255,
    )
    _, labels = ds[0]

    assert labels[0, 0, 0].item() == 4
    assert labels[1, 1, 1].item() == 255


def _manual_masked_dice_loss(logits, targets, target_ignore_index, dice_ignore_index=None, smooth=1.0):
    num_classes = logits.shape[1]
    probs = F.softmax(logits.float(), dim=1)
    valid = targets != target_ignore_index
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    valid_f = valid.to(probs.dtype).unsqueeze(1)
    one_hot = one_hot * valid_f
    probs = probs * valid_f
    dims = tuple(range(2, probs.dim()))
    intersection = (probs * one_hot).sum(dim=dims)
    union = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    if dice_ignore_index is not None:
        keep = torch.ones(num_classes, dtype=torch.bool, device=logits.device)
        keep[dice_ignore_index] = False
        dice = dice[:, keep]
    return 1.0 - dice.mean()


def test_memory_efficient_dice_masks_voxel_ignore_without_counting_as_class_zero():
    from models.losses import MemoryEfficientDiceLoss

    logits = torch.tensor(
        [
            [
                [[[8.0, 8.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [8.0, 8.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        ]
    )
    targets = torch.tensor([[[[0, 255], [1, 255]]]], dtype=torch.long)

    loss_fn = MemoryEfficientDiceLoss(
        smooth=1.0,
        ignore_index=0,
        target_ignore_index=255,
    )
    expected = _manual_masked_dice_loss(
        logits,
        targets,
        target_ignore_index=255,
        dice_ignore_index=0,
        smooth=1.0,
    )

    assert torch.allclose(loss_fn(logits, targets), expected, atol=1e-6)


def test_combined_loss_all_ignore_batch_is_zero_not_nan():
    from models.losses import CombinedLoss

    logits = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    targets = torch.full((1, 2, 2, 2), 255, dtype=torch.long)
    loss_fn = CombinedLoss(num_classes=3, target_ignore_index=255)

    loss = loss_fn(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert logits.grad is not None


def test_compute_class_weights_filters_ignore_and_invalidates_cache_by_signature(tmp_path):
    from models.losses import compute_class_weights

    class TinyDataset:
        def __init__(self, labels):
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return torch.empty(1), self.labels[idx]

    cache = tmp_path / "weights.pt"
    first = TinyDataset([torch.tensor([0, 255]), torch.tensor([0, 255])])
    second = TinyDataset([torch.tensor([1, 1]), torch.tensor([1, 255])])

    w1 = compute_class_weights(
        first,
        num_classes=2,
        num_samples=2,
        cache_path=str(cache),
        target_ignore_index=255,
        dataset_signature="first",
    )
    w2 = compute_class_weights(
        second,
        num_classes=2,
        num_samples=2,
        cache_path=str(cache),
        target_ignore_index=255,
        dataset_signature="second",
    )
    cached = torch.load(cache, weights_only=True)

    assert not torch.equal(w1, w2)
    assert cached["target_ignore_index"] == 255
    assert cached["dataset_signature"] == "second"


def test_dice_metric_filters_target_ignore_index_from_all_counts():
    from utils.metrics import DiceMetric

    metric = DiceMetric(num_classes=3, smooth=0.0, target_ignore_index=255)
    targets = torch.tensor([[[[1, 255]]]], dtype=torch.long)
    logits = torch.full((1, 3, 1, 1, 2), -10.0)
    logits[:, 1, 0, 0, 0] = 10.0
    logits[:, 2, 0, 0, 1] = 10.0

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()

    assert metric.target_sum.sum().item() == 1
    assert metric.pred_sum.sum().item() == 1
    assert valid_mask.tolist() == [False, True, False]
    assert dice_per_class[1].item() == 1.0
    assert mean_dice == 1.0


@pytest.mark.parametrize("loss_name", ["LorentzRankingLoss", "LorentzTreeRankingLoss"])
def test_lorentz_losses_filter_out_of_range_labels(loss_name):
    from models.hyperbolic.lorentz_loss import LorentzRankingLoss, LorentzTreeRankingLoss
    from models.hyperbolic.lorentz_ops import exp_map0

    torch.manual_seed(0)
    voxel_emb = exp_map0(torch.randn(1, 8, 2, 2, 2) * 0.1)
    labels = torch.tensor([[[[0, 1], [255, -1]], [[2, 3], [255, 1]]]], dtype=torch.long)
    label_emb = exp_map0(torch.randn(4, 8) * 0.1)

    if loss_name == "LorentzRankingLoss":
        loss_fn = LorentzRankingLoss(num_samples_per_class=2, num_negatives=2)
    else:
        loss_fn = LorentzTreeRankingLoss(
            tree_dist_matrix=torch.ones(4, 4),
            num_samples_per_class=2,
            num_negatives=2,
        )

    loss = loss_fn(voxel_emb, labels, label_emb)

    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("loss_name", ["LorentzRankingLoss", "LorentzTreeRankingLoss"])
def test_lorentz_losses_return_zero_when_all_labels_are_ignored(loss_name):
    from models.hyperbolic.lorentz_loss import LorentzRankingLoss, LorentzTreeRankingLoss
    from models.hyperbolic.lorentz_ops import exp_map0

    voxel_emb = exp_map0(torch.randn(1, 8, 1, 1, 2) * 0.1)
    labels = torch.full((1, 1, 1, 2), 255, dtype=torch.long)
    label_emb = exp_map0(torch.randn(4, 8) * 0.1)

    if loss_name == "LorentzRankingLoss":
        loss_fn = LorentzRankingLoss(num_samples_per_class=2, num_negatives=2)
    else:
        loss_fn = LorentzTreeRankingLoss(
            tree_dist_matrix=torch.ones(4, 4),
            num_samples_per_class=2,
            num_negatives=2,
        )

    loss = loss_fn(voxel_emb, labels, label_emb)

    assert loss.item() == 0.0
    assert loss.requires_grad
