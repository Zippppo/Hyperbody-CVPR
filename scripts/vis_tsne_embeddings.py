"""
t-SNE visualization of voxel embeddings from BodyNet.

Extracts voxel embeddings (Lorentz space) from the model, projects them
to tangent space via log_map0, then runs t-SNE for 2D visualization.
Voxels are colored by organ system.

Usage:
    python scripts/vis_tsne_embeddings.py \
        --config configs/021201.yaml \
        --ckpt checkpoints/021201/best.pth \
        --num_samples 3 \
        --voxels_per_class 300
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.amp import autocast
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from config import Config
from data.dataset import HyperBodyDataset
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system
from models.body_net import BodyNet
from models.hyperbolic.lorentz_ops import log_map0


SYSTEM_COLORS = {
    "skeletal": "#1f77b4",
    "muscular": "#d62728",
    "digestive": "#2ca02c",
    "respiratory": "#ff7f0e",
    "urinary": "#9467bd",
    "cardiovascular": "#e377c2",
    "nervous": "#8c564b",
    "other": "#7f7f7f",
}


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE visualization of voxel embeddings")
    p.add_argument("--config", type=str, default="configs/021201.yaml")
    p.add_argument("--ckpt", type=str, default="checkpoints/021201/best.pth")
    p.add_argument("--output", type=str, default="_VIS/tsne_embeddings.png")
    p.add_argument("--num_samples", type=int, default=3, help="Number of test samples to use")
    p.add_argument("--voxels_per_class", type=int, default=300, help="Max voxels to sample per class")
    p.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def load_model(cfg, ckpt_path, device):
    """Load BodyNet from checkpoint."""
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

    model = BodyNet(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        growth_rate=cfg.growth_rate,
        dense_layers=cfg.dense_layers,
        bn_size=cfg.bn_size,
        embed_dim=cfg.hyp_embed_dim,
        curv=cfg.hyp_curv,
        class_depths=class_depths,
        min_radius=cfg.hyp_min_radius,
        max_radius=cfg.hyp_max_radius,
        direction_mode=cfg.hyp_direction_mode,
        text_embedding_path=cfg.hyp_text_embedding_path,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()

    epoch = ckpt.get("epoch", "?")
    dice = ckpt.get("best_dice", 0)
    print(f"Loaded checkpoint: epoch={epoch}, best_dice={dice:.4f}")
    return model


@torch.no_grad()
def extract_embeddings(model, dataset, num_samples, voxels_per_class, device, curv):
    """
    Run inference and collect per-class voxel embeddings in tangent space.

    Returns:
        embeddings: (N, embed_dim) numpy array
        labels: (N,) numpy array of class indices
    """
    class_voxels = {}

    for idx in range(min(num_samples, len(dataset))):
        inp, gt = dataset[idx]
        inp = inp.unsqueeze(0).to(device)
        gt = gt.numpy()

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            _, voxel_emb, _ = model(inp)

        # (1, E, D, H, W) -> (D, H, W, E)
        emb = voxel_emb.squeeze(0).permute(1, 2, 3, 0).float()
        emb_tangent = log_map0(emb, curv=curv).cpu().numpy()

        for cls_idx in range(1, 70):
            mask = gt == cls_idx
            if not mask.any():
                continue
            if cls_idx not in class_voxels:
                class_voxels[cls_idx] = []
            class_voxels[cls_idx].append(emb_tangent[mask])

        print(f"  Sample {idx+1}/{num_samples}: collected embeddings")

    all_embs, all_labels = [], []
    for cls_idx, chunks in class_voxels.items():
        cat = np.concatenate(chunks, axis=0)
        if len(cat) > voxels_per_class:
            indices = np.random.choice(len(cat), voxels_per_class, replace=False)
            cat = cat[indices]
        all_embs.append(cat)
        all_labels.append(np.full(len(cat), cls_idx, dtype=np.int64))

    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total voxels: {len(labels)} across {len(class_voxels)} classes")
    return embeddings, labels


def run_tsne(embeddings, perplexity=30.0):
    """Run t-SNE on voxel embeddings only."""
    n = len(embeddings)
    print(f"Running t-SNE on {n} points (dim={embeddings.shape[1]})...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, n // 5)),
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def build_figure(coords_2d, labels, class_names, class_to_system, output_path):
    """Build matplotlib figure colored by organ system, with per-class centroid labels."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot voxels grouped by system
    for system in sorted(SYSTEM_COLORS.keys()):
        sys_classes = {c for c, s in class_to_system.items() if s == system and c > 0}
        mask = np.isin(labels, list(sys_classes))
        if not mask.any():
            continue
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            s=4, alpha=0.4, c=SYSTEM_COLORS[system], label=system, rasterized=True,
        )

    # Annotate centroid of each class
    for cls_idx in range(1, len(class_names)):
        mask = labels == cls_idx
        if not mask.any():
            continue
        cx, cy = coords_2d[mask, 0].mean(), coords_2d[mask, 1].mean()
        sys = class_to_system.get(cls_idx, "other")
        ax.annotate(
            class_names[cls_idx], (cx, cy),
            fontsize=5, ha="center", va="center",
            color=SYSTEM_COLORS.get(sys, "#333"),
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    ax.set_title("t-SNE of Voxel Embeddings (tangent space)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=4, fontsize=9, loc="best")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()
    np.random.seed(42)

    cfg = Config.from_yaml(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_to_system = load_class_to_system(cfg.tree_file, class_names)

    model = load_model(cfg, args.ckpt, device)
    dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "test", cfg.volume_size)
    print(f"Test samples: {len(dataset)}")

    embeddings, labels = extract_embeddings(
        model, dataset, args.num_samples, args.voxels_per_class, device, cfg.hyp_curv,
    )
    coords_2d = run_tsne(embeddings, args.perplexity)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save raw data for reuse
    data_path = os.path.splitext(args.output)[0] + "_data.npz"
    np.savez_compressed(
        data_path,
        embeddings=embeddings,
        labels=labels,
        coords_2d=coords_2d,
        class_names=class_names,
    )
    print(f"Saved raw data: {data_path}")

    build_figure(coords_2d, labels, class_names, class_to_system, args.output)


if __name__ == "__main__":
    main()
