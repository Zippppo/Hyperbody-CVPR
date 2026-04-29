import json
import logging
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _prefer_current_repo_modules():
    while ROOT in sys.path:
        sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)

    for module_name in list(sys.modules):
        if module_name == "config" or module_name == "train":
            module = sys.modules[module_name]
            module_file = getattr(module, "__file__", "") or ""
            if module_file and not os.path.abspath(module_file).startswith(ROOT):
                del sys.modules[module_name]


_prefer_current_repo_modules()


@pytest.fixture(autouse=True)
def _current_repo_imports():
    _prefer_current_repo_modules()


def _tree_leaves(node):
    if isinstance(node, str):
        return {node}
    if isinstance(node, dict):
        leaves = set()
        for value in node.values():
            leaves.update(_tree_leaves(value))
        return leaves
    raise TypeError(f"Unsupported tree node type: {type(node)!r}")


def test_s2i_stage_b_final_config_matches_graph_random_training_shape():
    from config import Config

    cfg = Config.from_yaml("configs/s2i_021201-19.yaml")
    source = Config.from_yaml("configs/021201-19.yaml")

    assert cfg.checkpoint_dir == "checkpoints/s2i_021201-19"
    assert cfg.log_dir == "runs/s2i_021201-19"
    assert cfg.save_every == source.save_every

    assert cfg.data_dir == "S2I_Dataset"
    assert cfg.split_file == "S2I_Dataset/dataset_split.json"
    assert cfg.tree_file == "S2I_Dataset/tree.json"
    assert cfg.dataset_info_file == "S2I_Dataset/dataset_info.json"
    assert cfg.num_classes == 121
    assert cfg.volume_size == source.volume_size

    # Final S2I semantics match the original single-empty-class setup:
    # raw 255 outside-body voxels are folded into class 0 at load time.
    assert cfg.target_ignore_index is None
    assert cfg.label_pad_value == 0
    assert cfg.outside_label == 255
    assert cfg.dice_ignore_index == 0

    assert cfg.hyp_distance_mode == "graph"
    assert cfg.graph_distance_matrix == "S2I_Dataset/graph_distance_matrix.pt"
    assert cfg.hyp_direction_mode == "random"
    assert cfg.hyp_text_embedding_path == ""
    assert cfg.hyp_freeze_epochs == source.hyp_freeze_epochs == 5

    for field in (
        "hyp_embed_dim",
        "hyp_curv",
        "hyp_margin",
        "hyp_samples_per_class",
        "hyp_num_negatives",
        "hyp_t_start",
        "hyp_t_end",
        "hyp_warmup_epochs",
        "hyp_min_radius",
        "hyp_max_radius",
        "hyp_text_lr_ratio",
        "hyp_text_grad_clip",
        "num_workers",
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip",
        "use_amp",
        "ce_weight",
        "dice_weight",
        "hyp_weight",
        "lr_scheduler",
        "lr_warmup_epochs",
        "lr_phase1_end",
        "lr_phase1_min",
        "lr_phase2_end",
        "lr_phase2_min",
    ):
        assert getattr(cfg, field) == getattr(source, field), field
    assert cfg.batch_size == 3


def test_s2i_stage_b_resources_are_consistent_with_config_and_class_names():
    from config import Config
    from train import load_precomputed_graph_distance_matrix

    cfg = Config.from_yaml("configs/s2i_021201-19.yaml")

    with open(cfg.dataset_info_file, encoding="utf-8") as f:
        dataset_info = json.load(f)
    class_names = dataset_info["class_names"]
    with open(cfg.tree_file, encoding="utf-8") as f:
        tree = json.load(f)

    assert dataset_info["num_classes"] == cfg.num_classes == len(class_names)
    assert dataset_info["special_labels"]["inside_body_empty"] == 0
    assert dataset_info["special_labels"]["outside_body_background"] == cfg.outside_label
    assert _tree_leaves(tree) == set(class_names)

    for path in (cfg.split_file, cfg.graph_distance_matrix, "S2I_Dataset/contact_matrix.pt"):
        assert os.path.exists(path), path

    graph = load_precomputed_graph_distance_matrix(
        cfg.graph_distance_matrix,
        logging.getLogger("test_s2i_stage_b"),
    )
    contact = torch.load("S2I_Dataset/contact_matrix.pt", map_location="cpu")

    assert tuple(graph.shape) == (cfg.num_classes, cfg.num_classes)
    assert tuple(contact.shape) == (cfg.num_classes, cfg.num_classes)
    assert torch.isfinite(graph).all()
    assert torch.isfinite(contact).all()
    assert torch.equal(graph.diag(), torch.zeros(cfg.num_classes))
    assert torch.equal(contact.diag(), torch.zeros(cfg.num_classes))


def test_s2i_stage_b_precomputed_class_weights_resource_is_finite():
    from config import Config

    cfg = Config.from_yaml("configs/s2i_021201-19.yaml")
    payload = torch.load("S2I_Dataset/s2i_class_weights.pt", map_location="cpu")

    assert payload["num_classes"] == cfg.num_classes
    assert payload["dataset"] == cfg.data_dir
    # Existing resource was generated from raw S2I labels with 255 treated as
    # the raw outside marker. Final training recomputes its own fold-to-0 cache.
    assert payload.get("ignore_index") == cfg.outside_label
    assert cfg.target_ignore_index is None
    assert tuple(payload["weights"].shape) == (cfg.num_classes,)
    assert torch.isfinite(payload["weights"]).all()


def test_s2i_stage_b_dataset_folds_raw_outside_label_to_empty_class(tmp_path):
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
        voxel_labels=np.array([[[255, 7]]], dtype=np.uint8),
    )
    split_file = data_root / "dataset_split.json"
    split_file.write_text(json.dumps({"train": [sample_name], "val": [], "test": []}))

    ds = HyperBodyDataset(
        str(data_root),
        str(split_file),
        "train",
        (2, 1, 2),
        label_pad_value=0,
        outside_label=255,
    )
    _, labels = ds[0]

    assert labels[0, 0, 0].item() == 0
    assert labels[0, 0, 1].item() == 7
    assert labels[1, 0, 0].item() == 0
    assert 255 not in labels.unique().tolist()
