"""
Visualize model performance across classes as a heatmap.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Abbreviation mapping for class names
ABBREV_MAP = {
    "inside_body_empty": "Empty",
    "liver": "Liver",
    "spleen": "Spleen",
    "kidney_left": "KidL",
    "kidney_right": "KidR",
    "stomach": "Stomach",
    "pancreas": "Pancreas",
    "gallbladder": "Gallbldr",
    "urinary_bladder": "UrinBldr",
    "prostate": "Prostate",
    "heart": "Heart",
    "brain": "Brain",
    "thyroid_gland": "Thyroid",
    "spinal_cord": "SpinCord",
    "lung": "Lung",
    "esophagus": "Esoph",
    "trachea": "Trachea",
    "small_bowel": "SmBowel",
    "duodenum": "Duoden",
    "colon": "Colon",
    "adrenal_gland_left": "AdrenL",
    "adrenal_gland_right": "AdrenR",
    "spine": "Spine",
    "rib_left_1": "RibL1",
    "rib_left_2": "RibL2",
    "rib_left_3": "RibL3",
    "rib_left_4": "RibL4",
    "rib_left_5": "RibL5",
    "rib_left_6": "RibL6",
    "rib_left_7": "RibL7",
    "rib_left_8": "RibL8",
    "rib_left_9": "RibL9",
    "rib_left_10": "RibL10",
    "rib_left_11": "RibL11",
    "rib_left_12": "RibL12",
    "rib_right_1": "RibR1",
    "rib_right_2": "RibR2",
    "rib_right_3": "RibR3",
    "rib_right_4": "RibR4",
    "rib_right_5": "RibR5",
    "rib_right_6": "RibR6",
    "rib_right_7": "RibR7",
    "rib_right_8": "RibR8",
    "rib_right_9": "RibR9",
    "rib_right_10": "RibR10",
    "rib_right_11": "RibR11",
    "rib_right_12": "RibR12",
    "skull": "Skull",
    "sternum": "Sternum",
    "costal_cartilages": "CostCart",
    "scapula_left": "ScapL",
    "scapula_right": "ScapR",
    "clavicula_left": "ClavL",
    "clavicula_right": "ClavR",
    "humerus_left": "HumL",
    "humerus_right": "HumR",
    "hip_left": "HipL",
    "hip_right": "HipR",
    "femur_left": "FemL",
    "femur_right": "FemR",
    "gluteus_maximus_left": "GluMaxL",
    "gluteus_maximus_right": "GluMaxR",
    "gluteus_medius_left": "GluMedL",
    "gluteus_medius_right": "GluMedR",
    "gluteus_minimus_left": "GluMinL",
    "gluteus_minimus_right": "GluMinR",
    "autochthon_left": "AutoL",
    "autochthon_right": "AutoR",
    "iliopsoas_left": "IliopL",
    "iliopsoas_right": "IliopR",
}


def get_abbrev(name: str) -> str:
    """Get abbreviated class name."""
    return ABBREV_MAP.get(name, name)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize per-class metric heatmap")
    parser.add_argument(
        "--metric",
        type=str,
        default="dice",
        choices=["dice", "iou", "hd95", "nsd"],
        help="Metric to visualize from eval/results/metrics.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load metrics
    metrics_path = Path(__file__).parent / "results" / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Load class names
    dataset_info_path = Path(__file__).parent.parent / "Dataset" / "dataset_info.json"
    with open(dataset_info_path, "r") as f:
        class_names = json.load(f)["class_names"]

    # Extract model names and per-class dice scores
    model_names = list(metrics.keys())
    num_classes = len(metrics[model_names[0]]["per_class"])

    # Build matrix: rows = models, cols = classes
    data = np.zeros((len(model_names), num_classes))
    for i, model in enumerate(model_names):
        for j in range(num_classes):
            class_entry = metrics[model]["per_class"][str(j)]
            if isinstance(class_entry, dict):
                value = class_entry.get(args.metric)
                data[i, j] = np.nan if value is None else value
            else:
                # Backward compatibility with legacy format where per_class is Dice-only scalar.
                data[i, j] = class_entry if args.metric == "dice" else np.nan

    # Create abbreviated class labels
    class_labels = [get_abbrev(class_names[i]) for i in range(num_classes)]

    if args.metric == "hd95":
        finite_values = data[np.isfinite(data)]
        vmax = np.percentile(finite_values, 95) if finite_values.size > 0 else 1.0
        vmin = 0
        cmap = "RdYlGn_r"  # Lower HD95 is better.
    else:
        vmin = 0
        vmax = 1
        cmap = "RdYlGn"

    # Create figure
    fig, ax = plt.subplots(figsize=(28, 7))

    # Plot heatmap with annotations
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=class_labels,
        yticklabels=model_names,
        cbar_kws={"label": f"{args.metric.upper()} Score"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6}
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    ax.set_title(f"Per-Class {args.metric.upper()} Comparison Across Models")

    # Rotate x labels for readability
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / "docs" / "visualizations" / "metrics_heatmap.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved heatmap to {output_path}")

    # Also print summary statistics
    print("\n=== Summary ===")
    for i, model in enumerate(model_names):
        mean_dice = metrics[model]["mean_dice"]
        rib_mean_dice = metrics[model].get("rib_mean_dice", float("nan"))
        mean_key = f"mean_{args.metric}"
        mean_value = metrics[model].get(mean_key, float("nan"))
        print(
            f"{model}: mean_dice={mean_dice:.4f}, rib_mean_dice={rib_mean_dice:.4f}, "
            f"{mean_key}={mean_value:.4f}"
        )

if __name__ == "__main__":
    main()
