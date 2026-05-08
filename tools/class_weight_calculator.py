"""
Class Weight Calculator with Improved Methods

This tool computes class weights for imbalanced segmentation tasks using methods
that avoid extreme weight values.

Methods:
    - effective_number: Based on "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    - log_dampened: Log-dampened inverse frequency for smoother weight distribution

Usage:
    # Compute weights from dataset using effective number method
    python tools/class_weight_calculator.py compute --method effective_number --beta 0.99 \
        --output checkpoints/class_weight_store/class_weights.pt

    # Compute weights using log-dampened method
    python tools/class_weight_calculator.py compute --method log_dampened --dampening 10.0 \
        --output checkpoints/class_weight_store/class_weights.pt

    # Compare different methods on current dataset
    python tools/class_weight_calculator.py compare --num-samples 5000

    # View weights from a file
    python tools/class_weight_calculator.py view checkpoints/class_weight_store/class_weights.pt
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch

DATASET_INFO_PATH = "Dataset/dataset_info.json"


def load_class_names() -> list:
    """Load class names from dataset info"""
    with open(DATASET_INFO_PATH) as f:
        info = json.load(f)
    return info["class_names"]


def compute_weights_inverse_sqrt(class_counts: torch.Tensor) -> Dict:
    """
    Original inverse sqrt method (for comparison).

    Args:
        class_counts: (num_classes,) tensor of voxel counts per class

    Returns:
        Dict with weights and metadata
    """
    num_classes = len(class_counts)
    total_voxels = class_counts.sum()
    class_freq = class_counts / total_voxels
    class_freq = torch.clamp(class_freq, min=1e-8)

    weights = 1.0 / torch.sqrt(class_freq)
    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    return {
        "weights": weights,
        "num_classes": num_classes,
        "method": "inverse_sqrt",
    }


def compute_weights_effective_number(
    class_counts: torch.Tensor,
    beta: float = 0.99,
) -> Dict:
    """
    Compute weights using Effective Number of Samples.

    Reference: "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    The effective number of samples is defined as:
        E_n = (1 - beta^n) / (1 - beta)

    where n is the number of samples. As n -> inf, E_n -> 1/(1-beta).
    This provides a smooth transition that avoids extreme weights.

    Args:
        class_counts: (num_classes,) tensor of voxel counts per class
        beta: Hyperparameter in [0, 1). Higher values give more weight to rare classes.
              Typical values: 0.9, 0.99, 0.999, 0.9999

    Returns:
        Dict with weights and metadata
    """
    num_classes = len(class_counts)

    # Rescale counts: normalize so that the smallest non-zero class has count=1.
    # Raw voxel counts (billions) cause beta^n to underflow to 0 for all classes,
    # destroying all discrimination. Rescaling preserves relative ratios while
    # keeping values in a range where the formula is numerically effective.
    nonzero_mask = class_counts > 0
    if nonzero_mask.any():
        min_nonzero = class_counts[nonzero_mask].min()
        scaled_counts = class_counts / min_nonzero
    else:
        scaled_counts = class_counts.clone()

    # Compute effective number for each class
    # E_n = (1 - beta^n) / (1 - beta)
    # Use log-space to avoid underflow: beta^n = exp(n * log(beta))
    log_beta = torch.tensor(beta, dtype=torch.float64).log()
    beta_pow_n = torch.exp(scaled_counts * log_beta)
    effective_num = (1.0 - beta_pow_n) / (1.0 - beta)
    # Clamp to avoid division by zero for classes with very large counts
    effective_num = torch.clamp(effective_num, min=1e-8)

    # Weight is inverse of effective number
    weights = 1.0 / effective_num

    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    return {
        "weights": weights,
        "num_classes": num_classes,
        "method": f"effective_number_beta{beta}",
    }


def compute_weights_log_dampened(
    class_counts: torch.Tensor,
    dampening_factor: float = 10.0,
) -> Dict:
    """
    Compute weights using log-dampened inverse frequency.

    weight_c = log(total / count_c + dampening_factor)

    The log function compresses the range of weights, preventing extreme values
    for very rare or very common classes.

    Args:
        class_counts: (num_classes,) tensor of voxel counts per class
        dampening_factor: Added inside log to prevent extreme values for rare classes.
                         Higher values = more compression. Typical: 1.0 - 100.0

    Returns:
        Dict with weights and metadata
    """
    num_classes = len(class_counts)
    total_voxels = class_counts.sum()

    # Avoid division by zero
    class_counts_safe = torch.clamp(class_counts, min=1.0)

    # Log-dampened: log(total/count + dampening)
    # This gives higher weight to rare classes but compressed by log
    weights = torch.log(total_voxels / class_counts_safe + dampening_factor)

    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    return {
        "weights": weights,
        "num_classes": num_classes,
        "method": f"log_dampened_factor{dampening_factor}",
    }


def count_classes_from_dataset(
    num_classes: int = 70,
    num_samples: int = 5000,
) -> torch.Tensor:
    """
    Count class frequencies from dataset samples.

    Args:
        num_classes: Number of segmentation classes
        num_samples: Number of samples to use

    Returns:
        (num_classes,) tensor of voxel counts per class
    """
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from data.dataset import HyperBodyDataset

    dataset = HyperBodyDataset(
        data_dir="Dataset/voxel_data",
        split_file="Dataset/dataset_split.json",
        split="train",
        volume_size=(128, 96, 256),
    )
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    total = len(indices)
    log_interval = max(total // 20, 1)  # ~5% progress steps
    print(f"Counting class frequencies from {total} samples...")
    for i, idx in enumerate(indices):
        if (i + 1) % log_interval == 0 or (i + 1) == total:
            print(f"  Processing sample {i + 1}/{total}")

        _, labels = dataset[idx]
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)
        labels = labels.long().flatten()
        # Vectorized bincount instead of per-class loop
        counts = torch.bincount(labels, minlength=num_classes).double()
        class_counts += counts[:num_classes]

    return class_counts


def save_weights(result: Dict, output_path: str, num_samples: int):
    """Save weights to file in compatible format"""
    output_data = {
        "weights": result["weights"],
        "num_classes": result["num_classes"],
        "num_samples": num_samples,
        "method": result["method"],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    print(f"Saved weights to: {output_path}")


def view_weights(weight_path: str, show_all: bool = False):
    """View weights from a .pt file with class names"""
    class_names = load_class_names()
    data = torch.load(weight_path, weights_only=True)
    weights = data["weights"]

    print(f"\n{'=' * 70}")
    print(f"File: {weight_path}")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Num samples: {data.get('num_samples', 'unknown')}")
    print(f"{'=' * 70}\n")

    # Statistics
    print(f"Statistics:")
    print(f"  min:  {weights.min():.4f}")
    print(f"  max:  {weights.max():.4f}")
    print(f"  mean: {weights.mean():.4f}")
    print(f"  sum:  {weights.sum():.4f}")
    print(f"  ratio (max/min): {weights.max() / weights.min():.2f}")
    print()

    # Detailed view
    print(f"{'Idx':<4} {'Class Name':<25} {'Weight':>10}")
    print("-" * 45)

    rib_indices = list(range(23, 47))
    for i, (name, w) in enumerate(zip(class_names, weights)):
        if show_all or i in rib_indices or w > 1.5 or w < 0.2:
            print(f"{i:<4} {name:<25} {w.item():>10.4f}")

    if not show_all:
        print("\n(Use --all to show all classes)")


def compare_methods(num_samples: int = 100):
    """Compare different weight computation methods"""
    print(f"Computing class counts from {num_samples} samples...")
    class_counts = count_classes_from_dataset(num_classes=70, num_samples=num_samples)

    class_names = load_class_names()

    methods = {
        "inverse_sqrt (original)": compute_weights_inverse_sqrt(class_counts),
        "effective_number (beta=0.9)": compute_weights_effective_number(class_counts, beta=0.9),
        "effective_number (beta=0.99)": compute_weights_effective_number(class_counts, beta=0.99),
        "effective_number (beta=0.999)": compute_weights_effective_number(class_counts, beta=0.999),
        "log_dampened (factor=1.0)": compute_weights_log_dampened(class_counts, dampening_factor=1.0),
        "log_dampened (factor=10.0)": compute_weights_log_dampened(class_counts, dampening_factor=10.0),
        "log_dampened (factor=100.0)": compute_weights_log_dampened(class_counts, dampening_factor=100.0),
    }

    print("\n" + "=" * 90)
    print("Method Comparison")
    print("=" * 90)
    print(f"{'Method':<35} {'Min':>8} {'Max':>8} {'Ratio':>8} {'BG (0)':>10} {'Rib Mean':>10}")
    print("-" * 90)

    rib_indices = list(range(23, 47))

    for name, result in methods.items():
        w = result["weights"]
        min_w = w.min().item()
        max_w = w.max().item()
        ratio = max_w / min_w
        bg_weight = w[0].item()
        rib_mean = w[rib_indices].mean().item()
        print(f"{name:<35} {min_w:>8.4f} {max_w:>8.4f} {ratio:>8.2f} {bg_weight:>10.4f} {rib_mean:>10.4f}")

    print("=" * 90)

    # Show weights for specific classes
    print("\n" + "=" * 90)
    print("Weights for Selected Classes")
    print("=" * 90)

    selected_classes = [0, 1, 14, 23, 24, 35, 46]  # background, body, large organ, ribs
    header = f"{'Class':<20}"
    for name in methods.keys():
        short_name = name.split("(")[0].strip()[:12]
        header += f" {short_name:>10}"
    print(header)
    print("-" * 90)

    for idx in selected_classes:
        row = f"{class_names[idx]:<20}"
        for result in methods.values():
            row += f" {result['weights'][idx].item():>10.4f}"
        print(row)

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Class Weight Calculator with Improved Methods")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Compute command
    compute_parser = subparsers.add_parser("compute", help="Compute weights from dataset")
    compute_parser.add_argument(
        "--method",
        choices=["effective_number", "log_dampened"],
        default="effective_number",
        help="Weight computation method",
    )
    compute_parser.add_argument("--beta", type=float, default=0.99, help="Beta for effective_number method")
    compute_parser.add_argument("--dampening", type=float, default=10.0, help="Dampening factor for log_dampened")
    compute_parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to use")
    compute_parser.add_argument("--output", "-o", required=True, help="Output path for weights")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare different methods")
    compare_parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to use")

    # View command
    view_parser = subparsers.add_parser("view", help="View weights from a file")
    view_parser.add_argument("weight_file", help="Path to .pt weight file")
    view_parser.add_argument("--all", action="store_true", help="Show all classes")

    args = parser.parse_args()

    if args.command == "compute":
        class_counts = count_classes_from_dataset(num_classes=70, num_samples=args.num_samples)

        if args.method == "effective_number":
            result = compute_weights_effective_number(class_counts, beta=args.beta)
        else:
            result = compute_weights_log_dampened(class_counts, dampening_factor=args.dampening)

        save_weights(result, args.output, args.num_samples)

        # Show summary
        w = result["weights"]
        print(f"\nWeight statistics:")
        print(f"  min:  {w.min():.4f}")
        print(f"  max:  {w.max():.4f}")
        print(f"  ratio: {w.max() / w.min():.2f}")

    elif args.command == "compare":
        compare_methods(num_samples=args.num_samples)

    elif args.command == "view":
        view_weights(args.weight_file, show_all=args.all)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
