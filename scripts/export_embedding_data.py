"""
Export label embedding data from training checkpoints to reusable files.

Extracts tangent embeddings, Lorentz points, geodesic distance matrices,
and MDS projections, then saves them as .npz (arrays) + .json (metadata)
for downstream visualization without re-processing checkpoints.

Outputs:
    <output_dir>/
        embedding_data.npz   - all numerical arrays
        embedding_meta.json   - per-checkpoint metadata, class names, systems

Usage:
    python scripts/export_embedding_data.py \
        --checkpoint-dir checkpoints/021002 \
        --config configs/021002.yaml \
        --output-dir _VIS/embedding_export
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_embedding_evolution import (
    discover_checkpoints,
    extract_tangent_embeddings,
    compute_geodesic_distance_matrix,
    mds_project,
    procrustes_align,
    normalize_to_poincare_ball,
)
from models.hyperbolic.lorentz_ops import exp_map0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export embedding data from checkpoints to .npz + .json"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory containing epoch_*.pth and best.pth",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: <checkpoint-dir>/embedding_export)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from config import Config
    from data.organ_hierarchy import load_class_to_system

    cfg = Config.from_yaml(args.config)
    curv = cfg.hyp_curv

    with open(cfg.dataset_info_file) as f:
        info = json.load(f)
    class_names = info["class_names"]
    class_to_system = load_class_to_system(cfg.tree_file, class_names)

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    print(f"Found {len(checkpoints)} checkpoints: {[l for l, _ in checkpoints]}")

    # Extract per-checkpoint data
    labels = []
    metadata_list = []
    tangent_list = []
    lorentz_list = []
    dist_matrices = []
    projections_raw = []
    prev_proj = None

    for label, path in checkpoints:
        print(f"  Processing {label}...")
        tangent, metadata = extract_tangent_embeddings(path)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)
        proj = mds_project(D, init=prev_proj)

        labels.append(label)
        metadata_list.append(metadata)
        tangent_list.append(tangent.cpu().numpy())
        lorentz_list.append(lorentz.cpu().numpy())
        dist_matrices.append(D)
        projections_raw.append(proj)
        prev_proj = proj

    # Procrustes alignment
    aligned = [projections_raw[0]]
    for i in range(1, len(projections_raw)):
        aligned.append(procrustes_align(aligned[i - 1], projections_raw[i]))

    # Normalize to Poincare ball
    normalized = normalize_to_poincare_ball(aligned)

    # --- Save ---
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(args.checkpoint_dir) / "embedding_export")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # .npz: all numerical arrays
    npz_data = {}
    for i, label in enumerate(labels):
        key = label.replace(".", "_")
        npz_data[f"tangent_{key}"] = tangent_list[i]
        npz_data[f"lorentz_{key}"] = lorentz_list[i]
        npz_data[f"geodesic_dist_{key}"] = dist_matrices[i]
        npz_data[f"mds_raw_{key}"] = projections_raw[i]
        npz_data[f"mds_aligned_{key}"] = aligned[i]
        npz_data[f"mds_normalized_{key}"] = normalized[i]

    npz_path = str(Path(output_dir) / "embedding_data.npz")
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved arrays: {npz_path}")

    # .json: metadata + class info
    meta = {
        "labels": labels,
        "curv": curv,
        "class_names": class_names,
        "class_to_system": {str(k): v for k, v in class_to_system.items()},
        "checkpoints": [
            {
                "label": label,
                "epoch": m.get("epoch"),
                "best_dice": m.get("best_dice"),
                "train_loss": m.get("train_loss"),
                "val_loss": m.get("val_loss"),
                "mean_dice": m.get("mean_dice"),
            }
            for label, m in zip(labels, metadata_list)
        ],
        "array_key_pattern": [
            "tangent_{label}", "lorentz_{label}", "geodesic_dist_{label}",
            "mds_raw_{label}", "mds_aligned_{label}", "mds_normalized_{label}",
        ],
    }

    json_path = str(Path(output_dir) / "embedding_meta.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {json_path}")

    print(f"Done! Export directory: {output_dir}")


if __name__ == "__main__":
    main()
