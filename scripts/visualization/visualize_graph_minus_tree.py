"""Visualize (Graph Distance - Tree Distance) as a PNG heatmap.

Negative values indicate pairs where spatial contact edges shortened the distance.
Zero means the tree distance was already optimal (no spatial shortcut).
"""
import json
import argparse
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.organ_hierarchy import compute_tree_distance_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default="Dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/visualization")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_dist = torch.load(dataset_dir / "graph_distance_matrix.pt", map_location="cpu")
    with open(dataset_dir / "dataset_info.json", encoding="utf-8") as f:
        info = json.load(f)

    class_names = info["class_names"]
    tree_dist = compute_tree_distance_matrix(str(dataset_dir / "tree.json"), class_names)

    diff = (graph_dist - tree_dist).numpy()  # negative = shortened by spatial edge

    n = diff.shape[0]
    short_names = [name.replace("_", " ") for name in class_names]

    fig, ax = plt.subplots(figsize=(18, 16))

    vmin = diff.min()
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=max(0.5, -vmin * 0.1))

    im = ax.imshow(diff, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    label_size = 4.5 if n > 80 else 5.5
    ax.set_xticklabels(short_names, rotation=90, fontsize=label_size, ha="center")
    ax.set_yticklabels(short_names, fontsize=label_size)

    ax.set_title(
        "Graph Distance - Tree Distance\n(negative = shortened by spatial contact edge)",
        fontsize=14,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Distance Difference", fontsize=11)

    for i in range(n):
        for j in range(n):
            v = diff[i, j]
            if v < -1.5:
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=3.5,
                    color="white",
                    fontweight="bold",
                )

    plt.tight_layout()
    output_path = output_dir / f"{dataset_dir.name}_graph_minus_tree_distance.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {output_path}")
    print(f"Diff range: [{diff.min():.1f}, {diff.max():.1f}]")
    print(f"Shortened pairs (diff < 0): {(diff < 0).sum() // 2}")


if __name__ == "__main__":
    main()
