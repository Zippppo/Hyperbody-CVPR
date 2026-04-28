"""
Unified evaluation script for all model predictions.

Usage:
    python eval/eval_all.py ablation-w-hyp
    python eval/eval_all.py --pred_dir eval/pred/LR-GD-M04-LRP3 --gt_dir Dataset/voxel_data
"""
import argparse
import json
import math
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import DiceMetric
from utils.surface_distance import SurfaceDistanceMetric


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all model predictions")
    parser.add_argument("--pred_dir", type=str, default="eval/pred", help="Predictions root directory")
    parser.add_argument("--gt_dir", type=str, default="Dataset/voxel_data", help="Ground truth data directory")
    parser.add_argument("--num_classes", type=int, default=70, help="Number of classes")
    parser.add_argument("--model", type=str, default=None, help="Evaluate only this model (subdirectory name)")
    parser.add_argument("--output", type=str, default="eval/results/metrics.json", help="Output JSON path")
    return parser.parse_args()


# Rib class indices: rib_left_1 to rib_left_12 (23-34), rib_right_1 to rib_right_12 (35-46)
RIB_CLASS_INDICES = list(range(23, 47))  # 24 ribs total


def evaluate_model(pred_dir, gt_dir, num_classes):
    """Evaluate a single model's predictions against GT.

    Returns:
        dict with global and per-class metrics
    """
    dice_metric = DiceMetric(num_classes=num_classes)
    surface_metric = SurfaceDistanceMetric(num_classes=num_classes, nsd_tolerance=2.0)

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".npz")])

    if len(pred_files) == 0:
        print(f"  Warning: No prediction files found in {pred_dir}")
        return None

    evaluated_samples = 0

    for pred_file in tqdm(pred_files, desc="  Evaluating", leave=False):
        # Load prediction
        pred_path = os.path.join(pred_dir, pred_file)
        pred_data = np.load(pred_path)
        pred_labels = pred_data["pred_labels"]

        # Get original filename for GT lookup
        if "original_filename" in pred_data:
            gt_filename = str(pred_data["original_filename"])
        else:
            gt_filename = pred_file

        # Load GT
        gt_path = os.path.join(gt_dir, gt_filename)
        if not os.path.exists(gt_path):
            print(f"  Warning: GT not found for {gt_filename}, skipping")
            continue

        gt_data = np.load(gt_path)
        gt_labels = gt_data["voxel_labels"]

        # Pad GT if needed (same as dataset does)
        volume_size = pred_labels.shape
        if gt_labels.shape != volume_size:
            from data.voxelizer import pad_labels
            gt_labels = pad_labels(gt_labels, volume_size)

        # Convert to tensors and update metric
        pred_tensor = torch.from_numpy(pred_labels).long().unsqueeze(0)  # (1, X, Y, Z)
        gt_tensor = torch.from_numpy(gt_labels).long().unsqueeze(0)  # (1, X, Y, Z)

        # Create fake logits for DiceMetric (it expects logits and does argmax)
        # We already have argmax, so create one-hot-like logits
        fake_logits = torch.zeros(1, num_classes, *volume_size)
        fake_logits.scatter_(1, pred_tensor.unsqueeze(1), 1.0)

        dice_metric.update(fake_logits, gt_tensor)
        surface_metric.update(fake_logits, gt_tensor)
        evaluated_samples += 1

    # Compute final metrics
    dice_per_class, mean_dice, valid_mask = dice_metric.compute()
    hd95_per_class, mean_hd95, hd95_valid_mask = surface_metric.compute_hd95()

    # Compute rib mean dice
    rib_dice_values = []
    for c in RIB_CLASS_INDICES:
        if c >= num_classes:
            continue
        if valid_mask[c]:
            rib_dice_values.append(dice_per_class[c].item())
    rib_mean_dice = sum(rib_dice_values) / len(rib_dice_values) if rib_dice_values else 0.0

    result = {
        "mean_dice": mean_dice,
        "mean_hd95": mean_hd95,
        "rib_mean_dice": rib_mean_dice,
        "num_samples": evaluated_samples,
        "per_class": {}
    }

    for c in range(num_classes):
        if valid_mask[c]:
            result["per_class"][str(c)] = {
                "dice": dice_per_class[c].item(),
                "hd95": hd95_per_class[c] if hd95_valid_mask[c] else None,
            }

    return result


def main():
    args = parse_args()

    # Find all model directories
    if not os.path.exists(args.pred_dir):
        print(f"Prediction directory not found: {args.pred_dir}")
        return

    if args.model:
        model_path = os.path.join(args.pred_dir, args.model)
        if not os.path.isdir(model_path):
            print(f"Model directory not found: {model_path}")
            return
        model_dirs = [args.model]
    else:
        model_dirs = sorted([
            d for d in os.listdir(args.pred_dir)
            if os.path.isdir(os.path.join(args.pred_dir, d))
        ])

    if len(model_dirs) == 0:
        print(f"No model directories found in {args.pred_dir}")
        return

    print(f"Found {len(model_dirs)} model(s): {model_dirs}")
    print(f"GT directory: {args.gt_dir}")
    print("-" * 50)

    all_results = {}

    for model_name in model_dirs:
        print(f"\nEvaluating: {model_name}")
        pred_path = os.path.join(args.pred_dir, model_name)

        result = evaluate_model(pred_path, args.gt_dir, args.num_classes)

        if result is not None:
            all_results[model_name] = result
            hd95_display = (
                f"{result['mean_hd95']:.4f}"
                if math.isfinite(result["mean_hd95"])
                else "inf"
            )
            print(
                f"  Dice: {result['mean_dice']:.4f}, "
                f"HD95: {hd95_display}, "
                f"Rib Dice: {result['rib_mean_dice']:.4f} ({result['num_samples']} samples)"
            )

    # Print summary table
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(
        f"{'Model':<20} {'Dice':<8} {'HD95':<10} "
        f"{'RibDice':<8} {'Samples':<8}"
    )
    print("-" * 68)
    for model_name, result in sorted(all_results.items(), key=lambda x: -x[1]["mean_dice"]):
        hd95_display = (
            f"{result['mean_hd95']:.4f}"
            if math.isfinite(result["mean_hd95"])
            else "inf"
        )
        print(
            f"{model_name:<20} {result['mean_dice']:<8.4f} "
            f"{hd95_display:<10} {result['rib_mean_dice']:<8.4f} "
            f"{result['num_samples']:<8}"
        )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
