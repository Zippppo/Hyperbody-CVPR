"""Precompute graph distance matrix for graph-mode curriculum mining."""

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
from data.organ_hierarchy import compute_tree_distance_matrix
from data.spatial_adjacency import (
    compute_contact_matrix_from_dataset,
    compute_graph_distance_matrix,
    infer_ignored_spatial_class_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute graph distance matrix")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Dataset",
        help="Directory to save contact_matrix.pt and graph_distance_matrix.pt",
    )
    parser.add_argument("--tree-file", type=str, default="Dataset/tree.json")
    parser.add_argument("--data-dir", type=str, default="Dataset/voxel_data")
    parser.add_argument("--split-file", type=str, default="Dataset/dataset_split.json")
    parser.add_argument("--dataset-info", type=str, default="Dataset/dataset_info.json")
    parser.add_argument("--volume-size", type=int, nargs=3, default=[144, 128, 268])
    parser.add_argument("--dilation-radius", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--class-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--contact-matrix",
        type=str,
        default="",
        help="Optional path to existing contact_matrix.pt (skip dataset traversal)",
    )
    return parser.parse_args()


def _load_class_names(dataset_info_path: str) -> list[str]:
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def _load_contact_matrix(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Contact matrix file not found: {path}")
    contact = torch.load(path, map_location="cpu")
    print(f"Loaded contact matrix from {path}")
    return contact


def _compute_contact_matrix(args: argparse.Namespace, num_classes: int, ignored_indices: tuple[int, ...]) -> torch.Tensor:
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
        ignored_class_indices=ignored_indices,
        show_progress=True,
    )
    print(f"Contact matrix done in {time.time() - start:.1f}s")
    return contact


def main() -> None:
    args = parse_args()

    class_names = _load_class_names(args.dataset_info)
    num_classes = len(class_names)
    ignored_indices = infer_ignored_spatial_class_indices(class_names)

    print(f"Classes: {num_classes}")
    if ignored_indices:
        print(f"Ignoring classes for spatial adjacency: {ignored_indices}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_output_path = output_dir / "contact_matrix.pt"
    graph_output_path = output_dir / "graph_distance_matrix.pt"

    if args.contact_matrix and args.contact_matrix.strip():
        contact_matrix = _load_contact_matrix(Path(args.contact_matrix))
    else:
        contact_matrix = _compute_contact_matrix(args, num_classes, ignored_indices)

    if tuple(contact_matrix.shape) != (num_classes, num_classes):
        raise ValueError(
            f"contact_matrix shape {tuple(contact_matrix.shape)} does not match ({num_classes}, {num_classes})"
        )

    tree_dist_matrix = compute_tree_distance_matrix(args.tree_file, class_names)
    graph_dist_matrix = compute_graph_distance_matrix(
        tree_dist_matrix,
        contact_matrix,
        lambda_=args.lambda_,
        epsilon=args.epsilon,
        ignored_class_indices=ignored_indices,
    )

    nonzero_contacts = int((contact_matrix > 0).sum().item())
    total_pairs = num_classes * num_classes - num_classes
    shortened_pairs = int((tree_dist_matrix - graph_dist_matrix > 0).sum().item())

    print(f"Non-zero contacts: {nonzero_contacts}/{total_pairs}")
    print(f"Shortened pairs: {shortened_pairs}/{total_pairs}")

    torch.save(contact_matrix.float(), contact_output_path)
    torch.save(graph_dist_matrix.float(), graph_output_path)
    print(f"Saved contact matrix to {contact_output_path}")
    print(f"Saved graph distance matrix to {graph_output_path}")


if __name__ == "__main__":
    main()
