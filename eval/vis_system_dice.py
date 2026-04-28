"""
Extract and visualize per-system mean Dice scores from metrics.json.

Groups the 70 classes into anatomical systems (skeletal, muscular, digestive, etc.)
and computes the mean Dice for each system per model.

Usage:
    python eval/vis_system_dice.py
    python eval/vis_system_dice.py --metrics docs/exp/metrics.json --output _VIS/system_dice.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from data.organ_hierarchy import load_class_to_system


def parse_args():
    parser = argparse.ArgumentParser(description="Per-system Dice visualization")
    parser.add_argument("--metrics", type=str, default="docs/exp/metrics.json")
    parser.add_argument("--dataset_info", type=str, default="Dataset/dataset_info.json")
    parser.add_argument("--tree", type=str, default="Dataset/tree.json")
    parser.add_argument("--output", type=str, default="_VIS/system_dice.png")
    return parser.parse_args()


def compute_system_dice(per_class: dict, class_to_system: dict) -> dict:
    """Compute mean Dice per anatomical system (excluding class 0)."""
    system_dice = {}
    for cls_str, metrics in per_class.items():
        cls_idx = int(cls_str)
        if cls_idx == 0:
            continue
        system = class_to_system.get(cls_idx, "other")
        dice = metrics.get("dice", 0.0)
        if dice is None or (isinstance(dice, float) and dice < 1e-8):
            dice = 0.0
        system_dice.setdefault(system, []).append(dice)

    return {s: np.mean(v) for s, v in system_dice.items()}


def main():
    args = parse_args()

    with open(args.metrics) as f:
        all_metrics = json.load(f)
    with open(args.dataset_info) as f:
        class_names = json.load(f)["class_names"]

    class_to_system = load_class_to_system(args.tree, class_names)

    # Collect system dice for all models
    system_order = ["skeletal", "muscular", "digestive", "respiratory",
                    "urinary", "cardiovascular", "nervous", "other"]
    model_names = list(all_metrics.keys())

    results = {}
    for model, data in all_metrics.items():
        sd = compute_system_dice(data["per_class"], class_to_system)
        results[model] = sd

    # Print table
    header = f"{'System':<18}" + "".join(f"{m:<16}" for m in model_names)
    print(header)
    print("-" * len(header))
    for sys_name in system_order:
        row = f"{sys_name:<18}"
        for model in model_names:
            val = results[model].get(sys_name, 0.0)
            row += f"{val:<16.4f}"
        print(row)

    # Plot grouped bar chart
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    x = np.arange(len(system_order))
    n_models = len(model_names)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(model_names):
        vals = [results[model].get(s, 0.0) for s in system_order]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(system_order, rotation=30, ha="right")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Per-System Mean Dice by Model")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
