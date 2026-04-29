"""
Inference script for the S2I 120-class foreground model.

The S2I setup uses 121 output labels in the checkpoint:
class 0 is ``inside_body_empty`` and classes 1-120 are anatomical labels.

Usage:
    python eval/eval_120cls.py
    python eval/eval_120cls.py --gpuids 0
    python eval/eval_120cls.py --ckpt checkpoints/s2i_021201-19/epoch_45.pth
    python eval/eval_120cls.py --output eval/pred/s2i_120cls_epoch45
"""
import argparse
import json
import os
import sys
from typing import Optional

# Add project root to path so the script works when launched as eval/eval_120cls.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm

from config import Config
from data.dataset import HyperBodyDataset
from data.organ_hierarchy import load_organ_hierarchy
from models.body_net import BodyNet


DEFAULT_CONFIG = "configs/s2i_021201-19.yaml"
DEFAULT_CKPT = "checkpoints/s2i_021201-19/epoch_45.pth"
DEFAULT_OUTPUT = "eval/pred/s2i_120cls_epoch45"


def parse_args():
    parser = argparse.ArgumentParser(description="Run S2I 120-class inference")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to YAML config")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="Checkpoint path or filename")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="Dataset split")
    parser.add_argument("--gpuids", type=int, default=0, help="GPU device ID to use when CUDA is available")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples for a smoke run")
    parser.add_argument(
        "--preserve-subdirs",
        action="store_true",
        help="Mirror split-relative subdirectories under the output directory",
    )
    return parser.parse_args()


def resolve_checkpoint_path(cfg: Config, ckpt_arg: str) -> str:
    """Resolve either a direct checkpoint path or a filename inside cfg.checkpoint_dir."""
    if os.path.exists(ckpt_arg):
        return ckpt_arg

    ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_arg)
    if os.path.exists(ckpt_path):
        return ckpt_path

    raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg} or {ckpt_path}")


def strip_module_prefix(state_dict):
    """Handle checkpoints saved from DistributedDataParallel."""
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def checkpoint_num_classes(state_dict) -> Optional[int]:
    """Infer output class count from common BodyNet checkpoint keys."""
    for key in ("unet.final.weight", "module.unet.final.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    for key in ("label_emb.tangent_embeddings", "module.label_emb.tangent_embeddings"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    return None


def format_metric(value):
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return str(value)


def load_model(cfg: Config, ckpt_path: str, device: torch.device) -> BodyNet:
    """Build BodyNet from the S2I config and load the checkpoint."""
    with open(cfg.dataset_info_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]

    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"dataset_info class count ({len(class_names)}) != cfg.num_classes ({cfg.num_classes})"
        )

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
    ckpt_classes = checkpoint_num_classes(state_dict)
    if ckpt_classes is not None and ckpt_classes != cfg.num_classes:
        raise ValueError(
            f"Checkpoint has {ckpt_classes} classes, but config has {cfg.num_classes}. "
            "Use the S2I config for this checkpoint."
        )

    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best Dice: {format_metric(checkpoint.get('best_dice', 'N/A'))}")
    print(f"  Classes: {cfg.num_classes} total (0 background/empty + 120 foreground)")

    return model


def output_path_for(output_dir: str, filename: str, preserve_subdirs: bool) -> str:
    if preserve_subdirs:
        return os.path.join(output_dir, filename)
    return os.path.join(output_dir, os.path.basename(filename))


@torch.no_grad()
def run_inference(
    model: BodyNet,
    dataset: HyperBodyDataset,
    output_dir: str,
    data_dir: str,
    device: torch.device,
    use_amp: bool = True,
    limit: Optional[int] = None,
    preserve_subdirs: bool = False,
):
    """Run inference and save compressed npz predictions."""
    os.makedirs(output_dir, exist_ok=True)

    num_samples = len(dataset) if limit is None else min(limit, len(dataset))
    used_outputs = {}

    for idx in tqdm(range(num_samples), desc="Inference"):
        inp, _ = dataset[idx]
        inp = inp.unsqueeze(0).to(device)

        if use_amp and device.type == "cuda":
            with autocast(device_type="cuda"):
                logits, _, _ = model(inp)
        else:
            logits, _, _ = model(inp)

        pred_labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        filename = dataset.filenames[idx]
        original_path = os.path.join(data_dir, filename)
        original_data = np.load(original_path)

        output_path = output_path_for(output_dir, filename, preserve_subdirs)
        previous = used_outputs.get(output_path)
        if previous is not None and previous != filename:
            raise RuntimeError(
                f"Output filename collision: {previous} and {filename} both map to {output_path}. "
                "Rerun with --preserve-subdirs."
            )
        used_outputs[output_path] = filename

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            pred_labels=pred_labels.astype(np.int64),
            grid_world_min=original_data["grid_world_min"],
            grid_voxel_size=original_data["grid_voxel_size"],
            original_filename=filename,
        )

    print(f"Saved {num_samples} predictions to {output_dir}")


def main():
    args = parse_args()

    cfg = Config.from_yaml(args.config)
    ckpt_path = resolve_checkpoint_path(cfg, args.ckpt)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpuids}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")

    model = load_model(cfg, ckpt_path, device)

    label_pad_value = cfg.label_pad_value if cfg.label_pad_value is not None else 0
    dataset = HyperBodyDataset(
        cfg.data_dir,
        cfg.split_file,
        args.split,
        cfg.volume_size,
        label_pad_value=label_pad_value,
        outside_label=cfg.outside_label,
    )
    print(f"{args.split.capitalize()} samples: {len(dataset)}")

    run_inference(
        model,
        dataset,
        args.output,
        cfg.data_dir,
        device,
        use_amp=cfg.use_amp,
        limit=args.limit,
        preserve_subdirs=args.preserve_subdirs,
    )


if __name__ == "__main__":
    main()
