"""
S2I_Dataset Class Weight Calculator

Computes class weights for the S2I_Dataset (121 classes).
Raw outside-body label 255 is folded into class 0, matching final training
semantics where there is one empty/background class.

Methods:
    - effective_number: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    - log_dampened:    log-dampened inverse frequency

Usage:
    # Compute weights using effective number method
    python tools/s2i_class_weight_calculator.py compute --method effective_number --beta 0.99 \
        --output checkpoints/class_weight_store/s2i_class_weights.pt

    # Compute weights using log-dampened method
    python tools/s2i_class_weight_calculator.py compute --method log_dampened --dampening 10.0 \
        --output checkpoints/class_weight_store/s2i_class_weights.pt

    # Compare different methods
    python tools/s2i_class_weight_calculator.py compare --num-samples 1000

    # View weights from a file
    python tools/s2i_class_weight_calculator.py view checkpoints/class_weight_store/s2i_class_weights.pt
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import fold_outside_label

DATASET_ROOT = Path("S2I_Dataset")
DATASET_INFO_PATH = DATASET_ROOT / "dataset_info.json"
TRAIN_DIR = DATASET_ROOT / "train"
OUTSIDE_LABEL = 255
LABEL_PAD_VALUE = 0
VOLUME_SIZE = (144, 128, 268)

# Class index ranges in S2I_Dataset (121 classes total)
RIB_INDICES = list(range(52, 76))           # rib_left_1 .. rib_right_12
VERTEBRAE_INDICES = list(range(26, 52))     # vertebrae_C1 .. sacrum
SELECTED_CLASSES_FOR_COMPARE = [0, 1, 10, 11, 14, 26, 52, 75, 96, 120]


def load_dataset_info() -> Dict:
    with open(DATASET_INFO_PATH) as f:
        return json.load(f)


def compute_weights_inverse_sqrt(class_counts: torch.Tensor) -> Dict:
    """Original inverse sqrt method (for comparison only)."""
    num_classes = len(class_counts)
    total_voxels = class_counts.sum()
    class_freq = class_counts / total_voxels
    class_freq = torch.clamp(class_freq, min=1e-8)

    weights = 1.0 / torch.sqrt(class_freq)
    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    return {"weights": weights, "num_classes": num_classes, "method": "inverse_sqrt"}


def compute_weights_effective_number(class_counts: torch.Tensor, beta: float = 0.99) -> Dict:
    """
    Compute weights using Effective Number of Samples (CVPR 2019).

    E_n = (1 - beta^n) / (1 - beta).  Voxel counts are rescaled so the smallest
    non-zero class has count=1, otherwise beta^n underflows to 0 for all classes.
    """
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
    weights = weights.float()

    return {"weights": weights, "num_classes": num_classes, "method": f"effective_number_beta{beta}"}


def compute_weights_log_dampened(class_counts: torch.Tensor, dampening_factor: float = 10.0) -> Dict:
    """weight_c = log(total / count_c + dampening_factor)."""
    num_classes = len(class_counts)
    total_voxels = class_counts.sum()

    class_counts_safe = torch.clamp(class_counts, min=1.0)
    weights = torch.log(total_voxels / class_counts_safe + dampening_factor)

    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    return {"weights": weights, "num_classes": num_classes, "method": f"log_dampened_factor{dampening_factor}"}


def count_classes_from_dataset(
    num_classes: int,
    num_samples: int,
    volume_size=VOLUME_SIZE,
    label_pad_value: int = LABEL_PAD_VALUE,
    outside_label: int = OUTSIDE_LABEL,
) -> torch.Tensor:
    """
    Count voxel-level class frequencies by reading .npz files from S2I_Dataset/train/.

    The raw outside-body marker is folded into ``label_pad_value`` before counting,
    and padding up to ``volume_size`` is counted as ``label_pad_value`` as well.
    """
    npz_files = sorted(TRAIN_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {TRAIN_DIR}")

    sample_files = random.sample(npz_files, min(num_samples, len(npz_files)))
    total = len(sample_files)
    log_interval = max(total // 20, 1)
    print(f"Counting class frequencies from {total} samples in {TRAIN_DIR}...")

    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for i, fp in enumerate(sample_files):
        if (i + 1) % log_interval == 0 or (i + 1) == total:
            print(f"  Processing sample {i + 1}/{total}")

        with np.load(fp) as data:
            labels = fold_outside_label(
                data["voxel_labels"],
                outside_label,
                label_pad_value,
            )

        x, y, z = labels.shape
        cx = min(x, volume_size[0])
        cy = min(y, volume_size[1])
        cz = min(z, volume_size[2])
        labels_t = torch.from_numpy(labels[:cx, :cy, :cz].astype(np.int64)).flatten()
        max_label = int(labels_t.max().item()) if labels_t.numel() > 0 else 0
        minlength = max(num_classes, max_label + 1)
        counts = torch.bincount(labels_t, minlength=minlength).double()
        class_counts += counts[:num_classes]

        padded_voxels = int(np.prod(volume_size)) - cx * cy * cz
        if 0 <= label_pad_value < num_classes:
            class_counts[label_pad_value] += padded_voxels

    return class_counts


def save_weights(result: Dict, output_path: str, num_samples: int):
    output_data = {
        "weights": result["weights"],
        "num_classes": result["num_classes"],
        "num_samples": num_samples,
        "method": result["method"],
        "dataset": "S2I_Dataset",
        "target_ignore_index": None,
        "outside_label": OUTSIDE_LABEL,
        "label_pad_value": LABEL_PAD_VALUE,
        "volume_size": VOLUME_SIZE,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    print(f"Saved weights to: {output_path}")


def view_weights(weight_path: str, show_all: bool = False):
    info = load_dataset_info()
    class_names = info["class_names"]
    data = torch.load(weight_path, weights_only=True)
    weights = data["weights"]

    print(f"\n{'=' * 70}")
    print(f"File: {weight_path}")
    print(f"Dataset: {data.get('dataset', 'unknown')}")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Num samples: {data.get('num_samples', 'unknown')}")
    print(f"Num classes: {len(weights)}")
    print(
        "Label semantics: "
        f"target_ignore_index={data.get('target_ignore_index', data.get('ignore_index', '-'))}, "
        f"outside_label={data.get('outside_label', '-')}, "
        f"label_pad_value={data.get('label_pad_value', '-')}"
    )
    print(f"{'=' * 70}\n")

    print(f"Statistics:")
    print(f"  min:  {weights.min():.4f}")
    print(f"  max:  {weights.max():.4f}")
    print(f"  mean: {weights.mean():.4f}")
    print(f"  sum:  {weights.sum():.4f}")
    print(f"  ratio (max/min): {weights.max() / weights.min():.2f}")
    print()

    print(f"{'Idx':<4} {'Class Name':<32} {'Weight':>10}")
    print("-" * 52)
    for i, (name, w) in enumerate(zip(class_names, weights)):
        if show_all or i in RIB_INDICES or w > 1.5 or w < 0.2:
            print(f"{i:<4} {name:<32} {w.item():>10.4f}")

    if not show_all:
        print("\n(Use --all to show all classes)")


def compare_methods(num_samples: int):
    info = load_dataset_info()
    class_names = info["class_names"]
    num_classes = info["num_classes"]

    print(f"Computing class counts from {num_samples} samples...")
    class_counts = count_classes_from_dataset(num_classes=num_classes, num_samples=num_samples)

    methods = {
        "inverse_sqrt (original)": compute_weights_inverse_sqrt(class_counts),
        "effective_number (beta=0.9)": compute_weights_effective_number(class_counts, beta=0.9),
        "effective_number (beta=0.99)": compute_weights_effective_number(class_counts, beta=0.99),
        "effective_number (beta=0.999)": compute_weights_effective_number(class_counts, beta=0.999),
        "log_dampened (factor=1.0)": compute_weights_log_dampened(class_counts, dampening_factor=1.0),
        "log_dampened (factor=10.0)": compute_weights_log_dampened(class_counts, dampening_factor=10.0),
        "log_dampened (factor=100.0)": compute_weights_log_dampened(class_counts, dampening_factor=100.0),
    }

    print("\n" + "=" * 100)
    print("Method Comparison")
    print("=" * 100)
    print(f"{'Method':<35} {'Min':>8} {'Max':>8} {'Ratio':>8} {'Cls0':>10} {'Rib Mean':>10} {'Vert Mean':>10}")
    print("-" * 100)

    for name, result in methods.items():
        w = result["weights"]
        min_w = w.min().item()
        max_w = w.max().item()
        ratio = max_w / min_w if min_w > 0 else float("inf")
        c0 = w[0].item()
        rib_mean = w[RIB_INDICES].mean().item()
        vert_mean = w[VERTEBRAE_INDICES].mean().item()
        print(f"{name:<35} {min_w:>8.4f} {max_w:>8.4f} {ratio:>8.2f} {c0:>10.4f} {rib_mean:>10.4f} {vert_mean:>10.4f}")

    print("=" * 100)

    print("\n" + "=" * 100)
    print("Weights for Selected Classes")
    print("=" * 100)
    header = f"{'Class':<28}"
    for name in methods.keys():
        short_name = name.split("(")[0].strip()[:12]
        header += f" {short_name:>12}"
    print(header)
    print("-" * 100)

    for idx in SELECTED_CLASSES_FOR_COMPARE:
        row = f"{class_names[idx]:<28}"
        for result in methods.values():
            row += f" {result['weights'][idx].item():>12.4f}"
        print(row)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="S2I_Dataset Class Weight Calculator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    compute_parser = subparsers.add_parser("compute", help="Compute weights from dataset")
    compute_parser.add_argument(
        "--method",
        choices=["effective_number", "log_dampened"],
        default="effective_number",
    )
    compute_parser.add_argument("--beta", type=float, default=0.99)
    compute_parser.add_argument("--dampening", type=float, default=10.0)
    compute_parser.add_argument("--num-samples", type=int, default=1000)
    compute_parser.add_argument("--output", "-o", required=True, help="Output path for weights .pt")
    compute_parser.add_argument("--seed", type=int, default=42)

    compare_parser = subparsers.add_parser("compare", help="Compare different methods")
    compare_parser.add_argument("--num-samples", type=int, default=1000)
    compare_parser.add_argument("--seed", type=int, default=42)

    view_parser = subparsers.add_parser("view", help="View weights from a file")
    view_parser.add_argument("weight_file", help="Path to .pt weight file")
    view_parser.add_argument("--all", action="store_true", help="Show all classes")

    args = parser.parse_args()

    if args.command == "compute":
        random.seed(args.seed)
        info = load_dataset_info()
        num_classes = info["num_classes"]
        class_counts = count_classes_from_dataset(num_classes=num_classes, num_samples=args.num_samples)

        if args.method == "effective_number":
            result = compute_weights_effective_number(class_counts, beta=args.beta)
        else:
            result = compute_weights_log_dampened(class_counts, dampening_factor=args.dampening)

        save_weights(result, args.output, args.num_samples)

        w = result["weights"]
        print(f"\nWeight statistics:")
        print(f"  min:  {w.min():.4f}")
        print(f"  max:  {w.max():.4f}")
        print(f"  ratio: {w.max() / w.min():.2f}")

    elif args.command == "compare":
        random.seed(args.seed)
        compare_methods(num_samples=args.num_samples)

    elif args.command == "view":
        view_weights(args.weight_file, show_all=args.all)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
