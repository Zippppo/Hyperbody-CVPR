"""Startup validation for config consistency."""

import json
import os

import torch


def validate_config_consistency(cfg) -> None:
    """Validate that num_classes, graph_distance_matrix, and dataset_info agree.

    Raises ValueError on mismatch, FileNotFoundError on missing files.
    """
    if not os.path.exists(cfg.graph_distance_matrix):
        raise FileNotFoundError(
            f"graph_distance_matrix not found: {cfg.graph_distance_matrix}"
        )

    mat = torch.load(cfg.graph_distance_matrix, map_location="cpu", weights_only=True)
    if mat.shape != (cfg.num_classes, cfg.num_classes):
        raise ValueError(
            f"num_classes={cfg.num_classes} but graph_distance_matrix shape is {tuple(mat.shape)}"
        )

    if not os.path.exists(cfg.dataset_info_file):
        raise FileNotFoundError(
            f"dataset_info_file not found: {cfg.dataset_info_file}"
        )

    with open(cfg.dataset_info_file) as f:
        info = json.load(f)

    class_names = info.get("class_names", [])
    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"num_classes={cfg.num_classes} but dataset_info has {len(class_names)} class names"
        )
