"""
View Class Weights

A simple tool to view class weights from .pt files.

Usage:
    # View a specific weight file
    python tools/view_class_weights.py checkpoints/class_weight_store/class_weights.pt

    # List all available weight files
    python tools/view_class_weights.py --list

    # Show all classes (default shows only notable ones)
    python tools/view_class_weights.py checkpoints/class_weight_store/class_weights.pt --all
"""

import argparse
import json
from pathlib import Path

import torch

DATASET_INFO_PATH = "Dataset/dataset_info.json"
WEIGHT_STORE_DIR = "checkpoints/class_weight_store"


def load_class_names() -> list:
    """Load class names from dataset info"""
    try:
        with open(DATASET_INFO_PATH) as f:
            info = json.load(f)
        return info["class_names"]
    except FileNotFoundError:
        return [f"class_{i}" for i in range(70)]


def list_weight_files() -> list:
    """List all available weight files"""
    store_dir = Path(WEIGHT_STORE_DIR)
    if not store_dir.exists():
        return []
    return sorted(store_dir.glob("*.pt"))


def view_weights(weight_path: str, show_all: bool = False):
    """View weights from a .pt file with class names"""
    class_names = load_class_names()
    data = torch.load(weight_path, weights_only=True)
    weights = data["weights"]

    print(f"\n{'=' * 70}")
    print(f"File: {weight_path}")
    print(f"Method: {data.get('method', 'unknown')}")
    print(f"Num classes: {data.get('num_classes', len(weights))}")
    print(f"Num samples: {data.get('num_samples', 'unknown')}")
    print(f"{'=' * 70}\n")

    # Statistics
    print("Statistics:")
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
    for i, w in enumerate(weights):
        name = class_names[i] if i < len(class_names) else f"class_{i}"
        if show_all or i in rib_indices or w > 1.5 or w < 0.2:
            print(f"{i:<4} {name:<25} {w.item():>10.4f}")

    if not show_all:
        print("\n(Use --all to show all classes)")


def main():
    parser = argparse.ArgumentParser(
        description="View class weights from .pt files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/view_class_weights.py class_weights.pt
    python tools/view_class_weights.py --list
    python tools/view_class_weights.py class_weights.pt --all
        """,
    )
    parser.add_argument("weight_file", nargs="?", help="Path to .pt weight file")
    parser.add_argument("--list", "-l", action="store_true", help="List all available weight files")
    parser.add_argument("--all", "-a", action="store_true", help="Show all classes")

    args = parser.parse_args()

    if args.list:
        files = list_weight_files()
        if files:
            print(f"\nAvailable weight files in {WEIGHT_STORE_DIR}:")
            print("-" * 50)
            for f in files:
                print(f"  {f.name}")
            print()
        else:
            print(f"No weight files found in {WEIGHT_STORE_DIR}")
        return

    if not args.weight_file:
        parser.print_help()
        return

    if not Path(args.weight_file).exists():
        print(f"Error: File not found: {args.weight_file}")
        return

    view_weights(args.weight_file, show_all=args.all)


if __name__ == "__main__":
    main()
