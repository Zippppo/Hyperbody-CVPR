"""Deprecated: precompute spatial contact matrix only.

Prefer scripts/precompute_graph_distance.py to generate both
contact_matrix.pt and graph_distance_matrix.pt in one run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import HyperBodyDataset
from data.spatial_adjacency import (
    compute_contact_matrix_from_dataset,
    infer_ignored_spatial_class_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute spatial contact matrix")
    parser.add_argument(
        "--output",
        type=str,
        default="Dataset/contact_matrix.pt",
        help="Output path for contact matrix (.pt)",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=3,
        help="Cube dilation radius in voxels",
    )
    parser.add_argument(
        "--class-batch-size",
        type=int,
        default=0,
        help="Class chunk size for memory-safe overlap (0 = auto/full)",
    )
    parser.add_argument("--data-dir", type=str, default="Dataset/voxel_data")
    parser.add_argument("--split-file", type=str, default="Dataset/dataset_split.json")
    parser.add_argument("--dataset-info", type=str, default="Dataset/dataset_info.json")
    parser.add_argument("--volume-size", type=int, nargs=3, default=[144, 128, 268])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.dataset_info) as f:
        class_names = json.load(f)["class_names"]
    num_classes = len(class_names)
    ignored_class_indices = infer_ignored_spatial_class_indices(class_names)
    print(f"Classes: {num_classes}")
    if ignored_class_indices:
        print(f"Ignoring classes for spatial adjacency: {ignored_class_indices}")

    dataset = HyperBodyDataset(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split="train",
        volume_size=tuple(args.volume_size),
    )
    print(f"Training samples: {len(dataset)}")

    print(
        "Computing contact matrix "
        f"(radius={args.dilation_radius}, class_batch_size={args.class_batch_size})..."
    )
    start = time.time()
    contact = compute_contact_matrix_from_dataset(
        dataset=dataset,
        num_classes=num_classes,
        dilation_radius=args.dilation_radius,
        num_workers=args.num_workers,
        class_batch_size=args.class_batch_size,
        ignored_class_indices=ignored_class_indices,
        show_progress=True,
    )
    elapsed = time.time() - start

    nonzero = int((contact > 0).sum().item())
    total = num_classes * num_classes - num_classes
    print(f"Done in {elapsed:.1f}s")
    print(f"Non-zero contacts: {nonzero}/{total} ({100.0 * nonzero / max(total, 1):.2f}%)")
    print(f"Max contact: {contact.max().item():.6f}")

    nonzero_values = contact[contact > 0]
    if nonzero_values.numel() > 0:
        print(f"Mean non-zero contact: {nonzero_values.mean().item():.6f}")
    else:
        print("Mean non-zero contact: 0.000000")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(contact, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
