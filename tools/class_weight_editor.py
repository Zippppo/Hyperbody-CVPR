"""
Class Weight Editor Tool

Usage:
    # View weights from a file
    python tools/class_weight_editor.py view checkpoints/class_weight_store/class_weights-origin.pt

    # Create custom weights (edit CUSTOM_WEIGHTS dict below, then run)
    python tools/class_weight_editor.py create --output checkpoints/class_weight_store/class_weights-custom.pt

    # Apply multiplier to specific classes
    python tools/class_weight_editor.py boost checkpoints/class_weight_store/class_weights-origin.pt \
        --classes 23-46 --multiplier 2.0 --output checkpoints/class_weight_store/class_weights-rib_boost.pt
"""

import argparse
import json
import torch
from pathlib import Path

# ============================================================================
# CUSTOM WEIGHTS CONFIGURATION
# Edit this dictionary to set custom weights for each class
# Keys: class index (0-69) or class name
# Values: weight value
# ============================================================================
CUSTOM_WEIGHTS = {
    # Example: boost all rib classes
    # "rib_left_1": 3.0,
    # "rib_left_2": 3.0,
    # ... or use indices:
    # 23: 3.0,  # rib_left_1
    # 24: 3.0,  # rib_left_2
}

# Multiplier for rib classes (applied on top of base weights)
RIB_BOOST_MULTIPLIER = 2.0

# ============================================================================

DATASET_INFO_PATH = "Dataset/dataset_info.json"


def load_class_names():
    """Load class names from dataset info"""
    with open(DATASET_INFO_PATH) as f:
        info = json.load(f)
    return info["class_names"]


def get_rib_indices():
    """Return indices of all rib classes (23-46)"""
    return list(range(23, 47))


def view_weights(weight_path: str, show_all: bool = False):
    """View weights from a .pt file with class names"""
    class_names = load_class_names()
    data = torch.load(weight_path, weights_only=True)
    weights = data["weights"]

    print(f"\n{'='*70}")
    print(f"File: {weight_path}")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Num samples: {data.get('num_samples', 'unknown')}")
    print(f"{'='*70}\n")

    # Group by category
    rib_indices = get_rib_indices()

    # Statistics
    rib_weights = weights[rib_indices]
    non_rib_weights = torch.cat([weights[:23], weights[47:]])

    print(f"Overall Statistics:")
    print(f"  All classes:     min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    print(f"  Rib classes:     min={rib_weights.min():.4f}, max={rib_weights.max():.4f}, mean={rib_weights.mean():.4f}")
    print(f"  Non-rib classes: min={non_rib_weights.min():.4f}, max={non_rib_weights.max():.4f}, mean={non_rib_weights.mean():.4f}")
    print()

    # Detailed view
    print(f"{'Idx':<4} {'Class Name':<25} {'Weight':>10} {'Category':<10}")
    print("-" * 55)

    for i, (name, w) in enumerate(zip(class_names, weights)):
        category = "rib" if i in rib_indices else ""
        if show_all or i in rib_indices or w > 1.5:
            print(f"{i:<4} {name:<25} {w.item():>10.4f} {category:<10}")

    if not show_all:
        print("\n(Use --all to show all classes)")


def create_custom_weights(output_path: str, base_path: str = None):
    """Create custom weights file"""
    class_names = load_class_names()
    num_classes = len(class_names)

    # Start from base weights or uniform
    if base_path:
        data = torch.load(base_path, weights_only=True)
        weights = data["weights"].clone()
        print(f"Starting from base weights: {base_path}")
    else:
        weights = torch.ones(num_classes)
        print("Starting from uniform weights")

    # Apply custom weights
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for key, value in CUSTOM_WEIGHTS.items():
        if isinstance(key, str):
            if key in name_to_idx:
                idx = name_to_idx[key]
            else:
                print(f"Warning: Unknown class name '{key}'")
                continue
        else:
            idx = key

        weights[idx] = value
        print(f"  Set {class_names[idx]} (idx={idx}) = {value}")

    # Save (preserve original metadata so train.py cache validation passes)
    if base_path:
        output_data = {
            "weights": weights,
            "num_classes": data["num_classes"],
            "num_samples": data["num_samples"],
            "method": data.get("method", "inverse_sqrt"),
        }
    else:
        output_data = {
            "weights": weights,
            "num_classes": num_classes,
            "num_samples": 100,
            "method": "inverse_sqrt",
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    print(f"\nSaved to: {output_path}")


def boost_classes(
    input_path: str,
    output_path: str,
    class_range: str,
    multiplier: float,
):
    """Apply multiplier to specific class range"""
    class_names = load_class_names()
    data = torch.load(input_path, weights_only=True)
    weights = data["weights"].clone()

    # Parse class range (e.g., "23-46" or "23,24,25")
    if "-" in class_range:
        start, end = map(int, class_range.split("-"))
        indices = list(range(start, end + 1))
    else:
        indices = [int(x) for x in class_range.split(",")]

    print(f"Boosting classes {class_range} by {multiplier}x")
    print(f"Input: {input_path}")
    print()

    for idx in indices:
        old_val = weights[idx].item()
        weights[idx] *= multiplier
        print(f"  {class_names[idx]:<25} {old_val:.4f} -> {weights[idx].item():.4f}")

    # Save (preserve original metadata so train.py cache validation passes)
    output_data = {
        "weights": weights,
        "num_classes": data["num_classes"],
        "num_samples": data["num_samples"],
        "method": data.get("method", "inverse_sqrt"),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Class Weight Editor Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # View command
    view_parser = subparsers.add_parser("view", help="View weights from a file")
    view_parser.add_argument("weight_file", help="Path to .pt weight file")
    view_parser.add_argument("--all", action="store_true", help="Show all classes")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create custom weights")
    create_parser.add_argument("--output", "-o", required=True, help="Output path")
    create_parser.add_argument("--base", "-b", help="Base weight file to start from")

    # Boost command
    boost_parser = subparsers.add_parser("boost", help="Boost specific classes")
    boost_parser.add_argument("input_file", help="Input weight file")
    boost_parser.add_argument("--classes", "-c", required=True, help="Class range (e.g., 23-46)")
    boost_parser.add_argument("--multiplier", "-m", type=float, default=2.0, help="Multiplier")
    boost_parser.add_argument("--output", "-o", required=True, help="Output path")

    args = parser.parse_args()

    if args.command == "view":
        view_weights(args.weight_file, show_all=args.all)
    elif args.command == "create":
        create_custom_weights(args.output, base_path=args.base)
    elif args.command == "boost":
        boost_classes(args.input_file, args.output, args.classes, args.multiplier)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
