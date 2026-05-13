"""Precompute S2I-Dataset contact and graph distance matrices."""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Add project root to import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.organ_hierarchy import compute_tree_distance_matrix
from data.spatial_adjacency import (
    compute_contact_matrix_from_dataset,
    compute_graph_distance_matrix,
    infer_ignored_spatial_class_indices,
)
from data.voxelizer import pad_labels


class LabelOnlyDataset(Dataset):
    """Lightweight dataset that only loads labels, skipping occupancy construction."""

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        split: str,
        volume_size: tuple,
        label_ignore_index: int | None = None,
    ):
        with open(split_file) as f:
            splits = json.load(f)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}.")
        self.filenames = splits[split]
        self.data_dir = data_dir
        self.volume_size = volume_size
        self.label_ignore_index = label_ignore_index

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.filenames[idx])
        data = np.load(path)
        labels = pad_labels(data["voxel_labels"], self.volume_size)
        if self.label_ignore_index is not None:
            labels[labels == self.label_ignore_index] = 0
        return torch.from_numpy(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute graph distance matrix")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="S2I-Dataset/info",
        help="Directory to save contact_matrix.pt and graph_distance_matrix.pt",
    )
    parser.add_argument("--tree-file", type=str, default="S2I-Dataset/info/tree.json")
    parser.add_argument("--data-dir", type=str, default="S2I-Dataset/data")
    parser.add_argument("--split-file", type=str, default="S2I-Dataset/info/dataset_split.json")
    parser.add_argument("--dataset-info", type=str, default="S2I-Dataset/dataset_info.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--volume-size", type=int, nargs=3, default=[144, 128, 268])
    parser.add_argument("--dilation-radius", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument(
        "--label-ignore-index",
        type=int,
        default=None,
        help=(
            "Label value to remap to class 0 before contact computation; "
            "defaults to dataset_info.ignore_index or "
            "special_labels.outside_body_background if present"
        ),
    )
    parser.add_argument(
        "--class-batch-size",
        type=int,
        default=8,
        help="Process classes in chunks to control memory. Use 0 for full one-hot.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--contact-matrix",
        type=str,
        default="",
        help="Optional path to existing contact_matrix.pt (skip dataset traversal)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for contact matrix computation: auto, cpu, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


def _load_class_names(dataset_info_path: str) -> list[str]:
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def _load_label_ignore_index(dataset_info_path: str) -> int | None:
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    value = info.get("ignore_index")
    if value is None:
        value = info.get("special_labels", {}).get("outside_body_background")
    return int(value) if value is not None else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split_counts(split_file: str) -> dict:
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return {
        "split_info": splits.get("split_info", {}),
        "train_count": len(splits.get("train", [])),
        "val_count": len(splits.get("val", [])),
        "test_count": len(splits.get("test", [])),
    }


def _load_contact_matrix(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Contact matrix file not found: {path}")
    contact = torch.load(path, map_location="cpu")
    print(f"Loaded contact matrix from {path}")
    return contact


def _compute_contact_matrix(
    args: argparse.Namespace,
    num_classes: int,
    ignored_indices: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    dataset = LabelOnlyDataset(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split=args.split,
        volume_size=tuple(args.volume_size),
        label_ignore_index=args.label_ignore_index,
    )
    print(f"{args.split} samples: {len(dataset)}")
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
        device=device,
    )
    print(f"Contact matrix done in {time.time() - start:.1f}s")
    return contact


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")

    class_names = _load_class_names(args.dataset_info)
    if args.label_ignore_index is None:
        args.label_ignore_index = _load_label_ignore_index(args.dataset_info)
    num_classes = len(class_names)
    ignored_indices = infer_ignored_spatial_class_indices(class_names)

    print(f"Classes: {num_classes}")
    print(f"Device: {device}")
    if args.label_ignore_index is not None:
        print(f"Remapping label ignore index {args.label_ignore_index} to class 0")
    if ignored_indices:
        print(f"Ignoring classes for spatial adjacency: {ignored_indices}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_output_path = output_dir / "contact_matrix.pt"
    graph_output_path = output_dir / "graph_distance_matrix.pt"
    metadata_output_path = output_dir / "graph_distance_metadata.json"

    if args.contact_matrix and args.contact_matrix.strip():
        contact_matrix = _load_contact_matrix(Path(args.contact_matrix))
    else:
        contact_matrix = _compute_contact_matrix(args, num_classes, ignored_indices, device)

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

    metadata = {
        "dataset": "S2I-Dataset",
        "num_classes": num_classes,
        "dataset_info": args.dataset_info,
        "dataset_info_sha256": _sha256(Path(args.dataset_info)),
        "tree_file": args.tree_file,
        "tree_file_sha256": _sha256(Path(args.tree_file)),
        "data_dir": args.data_dir,
        "split_file": args.split_file,
        "split_file_sha256": _sha256(Path(args.split_file)),
        "split": args.split,
        "split_counts": _load_split_counts(args.split_file),
        "volume_size": list(args.volume_size),
        "dilation_radius": args.dilation_radius,
        "lambda": args.lambda_,
        "epsilon": args.epsilon,
        "class_batch_size": args.class_batch_size,
        "label_ignore_index": args.label_ignore_index,
        "ignored_spatial_class_indices": list(ignored_indices),
        "contact_matrix": str(contact_output_path),
        "graph_distance_matrix": str(graph_output_path),
        "contact_shape": list(contact_matrix.shape),
        "graph_distance_shape": list(graph_dist_matrix.shape),
        "contact_nonzero_pairs": nonzero_contacts,
        "graph_shortened_pairs": shortened_pairs,
    }
    with metadata_output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Saved metadata to {metadata_output_path}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU peak memory: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
