#!/usr/bin/env python3
"""Compute class weights for the current S2I-Dataset schema.

Defaults are intentionally tied to the current 121-class S2I-Dataset. The old
70-class Dataset layout is no longer treated as a supported default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.voxelizer import pad_labels


DEFAULT_DATASET_INFO = "S2I-Dataset/dataset_info.json"
DEFAULT_DATA_DIR = "S2I-Dataset/data"
DEFAULT_SPLIT_FILE = "S2I-Dataset/info/dataset_split.json"
DEFAULT_OUTPUT = "S2I-Dataset/info/s2i_class_weights.pt"
DEFAULT_VOLUME_SIZE = (144, 128, 268)


class LabelOnlyDataset(Dataset):
    """Load only padded labels from the split file."""

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        split: str,
        volume_size: Sequence[int],
    ):
        with open(split_file, "r", encoding="utf-8") as f:
            splits = json.load(f)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}. Must be train, val, or test.")

        self.filenames = splits[split]
        self.data_dir = Path(data_dir)
        self.split_file = split_file
        self.split = split
        self.volume_size = tuple(int(v) for v in volume_size)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.data_dir / self.filenames[idx]
        with np.load(path) as data:
            labels = pad_labels(data["voxel_labels"], self.volume_size)
        return torch.from_numpy(labels).long()


def load_class_names(dataset_info_path: str) -> list[str]:
    """Load class names from dataset_info.json."""
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    class_names = info["class_names"]
    declared_num_classes = info.get("num_classes")
    if declared_num_classes is not None and declared_num_classes != len(class_names):
        raise ValueError(
            f"num_classes={declared_num_classes} but class_names has {len(class_names)} entries"
        )
    return class_names


def compute_weights_inverse_sqrt(class_counts: torch.Tensor) -> Dict:
    """Compute inverse square-root frequency weights."""
    num_classes = len(class_counts)
    total_voxels = class_counts.sum()
    class_freq = torch.clamp(class_counts / total_voxels, min=1e-8)
    weights = 1.0 / torch.sqrt(class_freq)
    weights = weights / weights.sum() * num_classes
    return {
        "weights": weights.float(),
        "num_classes": num_classes,
        "method": "inverse_sqrt",
    }


def compute_weights_effective_number(
    class_counts: torch.Tensor,
    beta: float = 0.99,
) -> Dict:
    """Compute weights using the effective number of samples."""
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    num_classes = len(class_counts)
    nonzero_mask = class_counts > 0
    if nonzero_mask.any():
        min_nonzero = class_counts[nonzero_mask].min()
        scaled_counts = class_counts / min_nonzero
    else:
        scaled_counts = class_counts.clone()

    log_beta = torch.tensor(beta, dtype=torch.float64).log()
    beta_pow_n = torch.exp(scaled_counts * log_beta)
    effective_num = (1.0 - beta_pow_n) / (1.0 - beta)
    effective_num = torch.clamp(effective_num, min=1e-8)

    weights = 1.0 / effective_num
    weights = weights / weights.sum() * num_classes
    return {
        "weights": weights.float(),
        "num_classes": num_classes,
        "method": f"effective_number_beta{beta}",
    }


def compute_weights_log_dampened(
    class_counts: torch.Tensor,
    dampening_factor: float = 10.0,
) -> Dict:
    """Compute log-dampened inverse-frequency weights."""
    if dampening_factor <= 0:
        raise ValueError(f"dampening_factor must be positive, got {dampening_factor}")

    num_classes = len(class_counts)
    total_voxels = class_counts.sum()
    class_counts_safe = torch.clamp(class_counts, min=1.0)
    weights = torch.log(total_voxels / class_counts_safe + dampening_factor)
    weights = weights / weights.sum() * num_classes
    return {
        "weights": weights.float(),
        "num_classes": num_classes,
        "method": f"log_dampened_factor{dampening_factor}",
    }


def sample_indices(dataset_size: int, num_samples: int, seed: int) -> list[int]:
    """Return deterministic sample indices; num_samples <= 0 means all samples."""
    if dataset_size <= 0:
        raise ValueError("Dataset split is empty.")
    if num_samples <= 0 or num_samples >= dataset_size:
        return list(range(dataset_size))

    rng = random.Random(seed)
    return sorted(rng.sample(range(dataset_size), num_samples))


def count_classes_from_dataset(
    data_dir: str,
    split_file: str,
    split: str,
    volume_size: Sequence[int],
    num_classes: int,
    num_samples: int = 0,
    seed: int = 42,
) -> tuple[torch.Tensor, int, int]:
    """Count class frequencies from padded label volumes."""
    dataset = LabelOnlyDataset(
        data_dir=data_dir,
        split_file=split_file,
        split=split,
        volume_size=volume_size,
    )
    indices = sample_indices(len(dataset), num_samples=num_samples, seed=seed)
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    total = len(indices)
    log_interval = max(total // 20, 1)
    print(f"Counting class frequencies from {total}/{len(dataset)} {split} samples...")

    for i, idx in enumerate(indices):
        if (i + 1) % log_interval == 0 or (i + 1) == total:
            print(f"  Processing sample {i + 1}/{total}")

        labels = dataset[idx].reshape(-1)
        label_min = int(labels.min())
        label_max = int(labels.max())
        if label_min < 0 or label_max >= num_classes:
            filename = dataset.filenames[idx]
            raise ValueError(
                f"{filename} has label range {label_min}..{label_max}, "
                f"expected 0..{num_classes - 1}"
            )

        counts = torch.bincount(labels, minlength=num_classes).double()
        class_counts += counts[:num_classes]

    return class_counts, total, len(dataset)


def save_weights(
    result: Dict,
    output_path: str,
    class_counts: torch.Tensor,
    samples_used: int,
    total_split_samples: int,
    args: argparse.Namespace,
) -> None:
    """Save weights and provenance metadata."""
    output_data = {
        "weights": result["weights"],
        "class_counts": class_counts,
        "num_classes": result["num_classes"],
        "num_samples": samples_used,
        "total_split_samples": total_split_samples,
        "method": result["method"],
        "dataset": "S2I-Dataset",
        "dataset_info": args.dataset_info,
        "data_dir": args.data_dir,
        "split_file": args.split_file,
        "split": args.split,
        "volume_size": list(args.volume_size),
        "random_seed": args.seed,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    print(f"Saved weights to: {output_path}")


def infer_highlight_indices(class_names: Sequence[str]) -> list[int]:
    """Select useful classes for compact weight display."""
    indices = [
        i
        for i, name in enumerate(class_names)
        if name.startswith("rib_") or name in {"inside_body_empty", "liver", "heart"}
    ]
    return indices


def view_weights(
    weight_path: str,
    dataset_info_path: str,
    show_all: bool = False,
) -> None:
    """View weights from a .pt file with class names."""
    data = torch.load(weight_path, map_location="cpu")
    weights = data["weights"]

    metadata_info = data.get("dataset_info")
    info_path = metadata_info if metadata_info and Path(metadata_info).exists() else dataset_info_path
    class_names = load_class_names(info_path)
    if len(class_names) != len(weights):
        raise ValueError(
            f"{info_path} has {len(class_names)} class names but weights has {len(weights)} entries"
        )

    print(f"\n{'=' * 70}")
    print(f"File: {weight_path}")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Samples: {data.get('num_samples', 'unknown')} / {data.get('total_split_samples', 'unknown')}")
    print(f"Dataset info: {info_path}")
    print(f"{'=' * 70}\n")

    print("Statistics:")
    print(f"  min:  {weights.min():.4f}")
    print(f"  max:  {weights.max():.4f}")
    print(f"  mean: {weights.mean():.4f}")
    print(f"  sum:  {weights.sum():.4f}")
    print(f"  ratio (max/min): {weights.max() / weights.min():.2f}")
    print()

    highlight_indices = set(infer_highlight_indices(class_names))
    print(f"{'Idx':<4} {'Class Name':<35} {'Weight':>10}")
    print("-" * 55)
    for i, (name, weight) in enumerate(zip(class_names, weights)):
        if show_all or i in highlight_indices or weight > 1.5 or weight < 0.2:
            print(f"{i:<4} {name:<35} {weight.item():>10.4f}")

    if not show_all:
        print("\n(Use --all to show all classes)")


def compare_methods(args: argparse.Namespace) -> None:
    """Compare weight computation methods for the configured dataset."""
    class_names = load_class_names(args.dataset_info)
    class_counts, samples_used, total_split_samples = count_classes_from_dataset(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split=args.split,
        volume_size=args.volume_size,
        num_classes=len(class_names),
        num_samples=args.num_samples,
        seed=args.seed,
    )

    methods = {
        "inverse_sqrt": compute_weights_inverse_sqrt(class_counts),
        "effective_number_beta0.9": compute_weights_effective_number(class_counts, beta=0.9),
        "effective_number_beta0.99": compute_weights_effective_number(class_counts, beta=0.99),
        "effective_number_beta0.999": compute_weights_effective_number(class_counts, beta=0.999),
        "log_dampened_factor10": compute_weights_log_dampened(class_counts, dampening_factor=10.0),
        "log_dampened_factor100": compute_weights_log_dampened(class_counts, dampening_factor=100.0),
    }

    print("\n" + "=" * 92)
    print(f"Method comparison ({samples_used}/{total_split_samples} {args.split} samples)")
    print("=" * 92)
    print(f"{'Method':<32} {'Min':>8} {'Max':>8} {'Ratio':>8} {'BG (0)':>10} {'Rib Mean':>10}")
    print("-" * 92)

    rib_indices = [i for i, name in enumerate(class_names) if name.startswith("rib_")]
    for name, result in methods.items():
        weights = result["weights"]
        rib_mean = weights[rib_indices].mean().item() if rib_indices else float("nan")
        print(
            f"{name:<32} {weights.min().item():>8.4f} {weights.max().item():>8.4f} "
            f"{(weights.max() / weights.min()).item():>8.2f} {weights[0].item():>10.4f} "
            f"{rib_mean:>10.4f}"
        )
    print("=" * 92)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-info", default=DEFAULT_DATASET_INFO)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--volume-size", type=int, nargs=3, default=list(DEFAULT_VOLUME_SIZE))
    parser.add_argument("--seed", type=int, default=42)


def main() -> None:
    parser = argparse.ArgumentParser(description="S2I-Dataset class weight calculator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    compute_parser = subparsers.add_parser("compute", help="Compute weights from S2I-Dataset")
    add_dataset_args(compute_parser)
    compute_parser.add_argument(
        "--method",
        choices=["effective_number", "log_dampened", "inverse_sqrt"],
        default="effective_number",
    )
    compute_parser.add_argument("--beta", type=float, default=0.99)
    compute_parser.add_argument("--dampening", type=float, default=10.0)
    compute_parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to use; 0 means the full split.",
    )
    compute_parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)

    compare_parser = subparsers.add_parser("compare", help="Compare weight methods")
    add_dataset_args(compare_parser)
    compare_parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to use; 0 means the full split.",
    )

    view_parser = subparsers.add_parser("view", help="View weights from a .pt file")
    view_parser.add_argument("weight_file", nargs="?", default=DEFAULT_OUTPUT)
    view_parser.add_argument("--dataset-info", default=DEFAULT_DATASET_INFO)
    view_parser.add_argument("--all", action="store_true", help="Show all classes")

    args = parser.parse_args()

    if args.command == "compute":
        class_names = load_class_names(args.dataset_info)
        class_counts, samples_used, total_split_samples = count_classes_from_dataset(
            data_dir=args.data_dir,
            split_file=args.split_file,
            split=args.split,
            volume_size=args.volume_size,
            num_classes=len(class_names),
            num_samples=args.num_samples,
            seed=args.seed,
        )

        if args.method == "effective_number":
            result = compute_weights_effective_number(class_counts, beta=args.beta)
        elif args.method == "log_dampened":
            result = compute_weights_log_dampened(class_counts, dampening_factor=args.dampening)
        else:
            result = compute_weights_inverse_sqrt(class_counts)

        save_weights(
            result=result,
            output_path=args.output,
            class_counts=class_counts,
            samples_used=samples_used,
            total_split_samples=total_split_samples,
            args=args,
        )

        weights = result["weights"]
        print("\nWeight statistics:")
        print(f"  min:   {weights.min():.4f}")
        print(f"  max:   {weights.max():.4f}")
        print(f"  sum:   {weights.sum():.4f}")
        print(f"  ratio: {weights.max() / weights.min():.2f}")

    elif args.command == "compare":
        compare_methods(args)

    elif args.command == "view":
        view_weights(args.weight_file, dataset_info_path=args.dataset_info, show_all=args.all)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
