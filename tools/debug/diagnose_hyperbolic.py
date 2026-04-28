"""
Diagnostic script to investigate why hyperbolic module degrades performance.

Checks:
1. Embedding distribution (are they collapsing?)
2. Distance statistics (positive vs negative distances)
3. Triplet loss behavior (is margin being satisfied trivially?)
4. Gradient flow through hyperbolic branch
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from data.dataset import HyperBodyDataset
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system
from models.body_net import BodyNet
from models.hyperbolic.lorentz_ops import pointwise_dist, pairwise_dist, distance_to_origin
from models.hyperbolic.lorentz_loss import LorentzRankingLoss


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load model from checkpoint."""
    cfg = Config.from_yaml(config_path)

    # Load class info
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, cfg


def analyze_label_embeddings(model, curv: float = 1.0):
    """Analyze the learned label embeddings."""
    print("\n" + "="*60)
    print("LABEL EMBEDDING ANALYSIS")
    print("="*60)

    label_emb = model.label_emb()  # [num_classes, embed_dim]
    num_classes, embed_dim = label_emb.shape

    # 1. Distance from origin for each class
    dist_from_origin = distance_to_origin(label_emb, curv)  # [num_classes]

    print(f"\n1. Distance from origin (hyperbolic):")
    print(f"   Min: {dist_from_origin.min().item():.4f}")
    print(f"   Max: {dist_from_origin.max().item():.4f}")
    print(f"   Mean: {dist_from_origin.mean().item():.4f}")
    print(f"   Std: {dist_from_origin.std().item():.4f}")

    # 2. Pairwise distances between class embeddings
    pairwise_dists = pairwise_dist(label_emb, label_emb, curv)  # [num_classes, num_classes]

    # Exclude diagonal (self-distances)
    mask = ~torch.eye(num_classes, dtype=torch.bool, device=label_emb.device)
    off_diag_dists = pairwise_dists[mask]

    print(f"\n2. Pairwise distances between classes:")
    print(f"   Min: {off_diag_dists.min().item():.4f}")
    print(f"   Max: {off_diag_dists.max().item():.4f}")
    print(f"   Mean: {off_diag_dists.mean().item():.4f}")
    print(f"   Std: {off_diag_dists.std().item():.4f}")

    # 3. Check if embeddings are collapsing (all similar)
    euclidean_norms = label_emb.norm(dim=-1)
    print(f"\n3. Euclidean norms of spatial components:")
    print(f"   Min: {euclidean_norms.min().item():.4f}")
    print(f"   Max: {euclidean_norms.max().item():.4f}")
    print(f"   Mean: {euclidean_norms.mean().item():.4f}")

    # 4. Check tangent embeddings (before exp_map)
    tangent_emb = model.label_emb.tangent_embeddings  # [num_classes, embed_dim]
    tangent_norms = tangent_emb.norm(dim=-1)
    print(f"\n4. Tangent vector norms (before exp_map):")
    print(f"   Min: {tangent_norms.min().item():.4f}")
    print(f"   Max: {tangent_norms.max().item():.4f}")
    print(f"   Mean: {tangent_norms.mean().item():.4f}")

    return label_emb, pairwise_dists


def analyze_voxel_embeddings(model, sample_input, sample_labels, curv: float = 1.0):
    """Analyze voxel embeddings for a sample."""
    print("\n" + "="*60)
    print("VOXEL EMBEDDING ANALYSIS")
    print("="*60)

    # Use smaller crop to avoid OOM
    crop_size = 64
    sample_input = sample_input[:, :, :crop_size, :crop_size, :crop_size]
    sample_labels = sample_labels[:, :crop_size, :crop_size, :crop_size]
    print(f"Using cropped input: {sample_input.shape}")

    with torch.no_grad():
        logits, voxel_emb, label_emb = model(sample_input)

    # voxel_emb: [B, embed_dim, D, H, W]
    B, C, D, H, W = voxel_emb.shape

    # Reshape to [N, C]
    voxel_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, C)
    labels_flat = sample_labels.reshape(-1)

    # 1. Distance from origin
    voxel_dist_origin = distance_to_origin(voxel_flat, curv)

    print(f"\n1. Voxel distance from origin:")
    print(f"   Min: {voxel_dist_origin.min().item():.4f}")
    print(f"   Max: {voxel_dist_origin.max().item():.4f}")
    print(f"   Mean: {voxel_dist_origin.mean().item():.4f}")
    print(f"   Std: {voxel_dist_origin.std().item():.4f}")

    # 2. Euclidean norms
    voxel_norms = voxel_flat.norm(dim=-1)
    print(f"\n2. Voxel Euclidean norms:")
    print(f"   Min: {voxel_norms.min().item():.4f}")
    print(f"   Max: {voxel_norms.max().item():.4f}")
    print(f"   Mean: {voxel_norms.mean().item():.4f}")

    # 3. Sample some voxels and check distances to their true class vs other classes
    unique_classes = labels_flat.unique()
    print(f"\n3. Unique classes in sample: {len(unique_classes)}")

    # For each class, sample some voxels and compute distances
    print("\n4. Distance analysis per class (sample of 100 voxels per class):")

    for cls_idx in unique_classes[:5]:  # First 5 classes
        cls_mask = labels_flat == cls_idx
        cls_voxels = voxel_flat[cls_mask]

        if len(cls_voxels) == 0:
            continue

        # Sample up to 100 voxels
        n_sample = min(100, len(cls_voxels))
        sample_idx = torch.randperm(len(cls_voxels))[:n_sample]
        sampled_voxels = cls_voxels[sample_idx]

        # Distance to true class embedding
        true_class_emb = label_emb[cls_idx].unsqueeze(0)  # [1, C]
        d_positive = pointwise_dist(sampled_voxels, true_class_emb.expand(n_sample, -1), curv)

        # Distance to all other class embeddings
        all_dists = pairwise_dist(sampled_voxels, label_emb, curv)  # [n_sample, num_classes]

        # Mask out true class
        neg_mask = torch.ones(label_emb.shape[0], dtype=torch.bool, device=label_emb.device)
        neg_mask[cls_idx] = False
        d_negative = all_dists[:, neg_mask]  # [n_sample, num_classes-1]

        print(f"\n   Class {cls_idx.item()} ({cls_mask.sum().item()} voxels):")
        print(f"     d(voxel, true_class):  mean={d_positive.mean().item():.4f}, std={d_positive.std().item():.4f}")
        print(f"     d(voxel, other_class): mean={d_negative.mean().item():.4f}, std={d_negative.std().item():.4f}")
        print(f"     Margin gap (neg - pos): {(d_negative.mean() - d_positive.mean()).item():.4f}")

    return voxel_emb, logits


def analyze_triplet_loss(model, sample_input, sample_labels, cfg):
    """Analyze triplet loss behavior."""
    print("\n" + "="*60)
    print("TRIPLET LOSS ANALYSIS")
    print("="*60)

    # Use smaller crop to avoid OOM
    crop_size = 64
    sample_input = sample_input[:, :, :crop_size, :crop_size, :crop_size]
    sample_labels = sample_labels[:, :crop_size, :crop_size, :crop_size]

    hyp_criterion = LorentzRankingLoss(
        margin=cfg.hyp_margin,
        curv=cfg.hyp_curv,
        num_samples_per_class=cfg.hyp_samples_per_class,
        num_negatives=cfg.hyp_num_negatives,
    )

    with torch.no_grad():
        logits, voxel_emb, label_emb = model(sample_input)
        loss = hyp_criterion(voxel_emb, sample_labels, label_emb)

    print(f"\n1. Triplet loss value: {loss.item():.6f}")
    print(f"   Margin: {cfg.hyp_margin}")
    print(f"   hyp_weight: {cfg.hyp_weight}")
    print(f"   Effective contribution: {loss.item() * cfg.hyp_weight:.6f}")

    # Check if loss is near zero (margin already satisfied)
    if loss.item() < 0.01:
        print("\n   WARNING: Loss is very small!")
        print("   This could mean:")
        print("   - Margin is already satisfied (good)")
        print("   - Embeddings collapsed to similar values (bad)")
        print("   - Positive and negative distances are similar (bad)")

    return loss


def analyze_gradient_flow(model, sample_input, sample_labels, cfg, device):
    """Check if gradients flow through hyperbolic branch."""
    print("\n" + "="*60)
    print("GRADIENT FLOW ANALYSIS")
    print("="*60)

    # Use smaller crop to avoid OOM
    crop_size = 64
    sample_input = sample_input[:, :, :crop_size, :crop_size, :crop_size]
    sample_labels = sample_labels[:, :crop_size, :crop_size, :crop_size]

    model.train()

    # Enable gradients
    for param in model.parameters():
        param.requires_grad_(True)

    hyp_criterion = LorentzRankingLoss(
        margin=cfg.hyp_margin,
        curv=cfg.hyp_curv,
        num_samples_per_class=cfg.hyp_samples_per_class,
        num_negatives=cfg.hyp_num_negatives,
    )

    # Forward pass
    logits, voxel_emb, label_emb = model(sample_input)
    hyp_loss = hyp_criterion(voxel_emb, sample_labels, label_emb)

    # Backward pass
    hyp_loss.backward()

    # Check gradients
    print("\n1. Gradient norms for key components:")

    # Label embeddings
    if model.label_emb.tangent_embeddings.grad is not None:
        grad_norm = model.label_emb.tangent_embeddings.grad.norm().item()
        print(f"   label_emb.tangent_embeddings: {grad_norm:.6f}")
    else:
        print("   label_emb.tangent_embeddings: NO GRADIENT!")

    # Projection head
    if model.hyp_head.conv.weight.grad is not None:
        grad_norm = model.hyp_head.conv.weight.grad.norm().item()
        print(f"   hyp_head.conv.weight: {grad_norm:.6f}")
    else:
        print("   hyp_head.conv.weight: NO GRADIENT!")

    # UNet decoder (should have gradients from hyp_loss)
    unet_grad_found = False
    for name, param in model.unet.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            unet_grad_found = True
            break

    print(f"   UNet has gradients from hyp_loss: {unet_grad_found}")

    model.eval()
    model.zero_grad()


def compare_with_baseline_predictions(model, sample_input, sample_labels, cfg):
    """Compare segmentation predictions with baseline behavior."""
    print("\n" + "="*60)
    print("SEGMENTATION PREDICTION ANALYSIS")
    print("="*60)

    # Use smaller crop to avoid OOM
    crop_size = 64
    sample_input = sample_input[:, :, :crop_size, :crop_size, :crop_size]
    sample_labels = sample_labels[:, :crop_size, :crop_size, :crop_size]

    with torch.no_grad():
        logits, voxel_emb, label_emb = model(sample_input)

    # Get predictions
    preds = logits.argmax(dim=1)  # [B, D, H, W]

    # Class distribution in predictions vs ground truth
    pred_flat = preds.reshape(-1)
    gt_flat = sample_labels.reshape(-1)

    # Count class 0 (background/empty)
    pred_class0 = (pred_flat == 0).sum().item()
    gt_class0 = (gt_flat == 0).sum().item()
    total_voxels = pred_flat.numel()

    print(f"\n1. Class 0 (background) distribution:")
    print(f"   Ground truth: {gt_class0}/{total_voxels} ({100*gt_class0/total_voxels:.1f}%)")
    print(f"   Prediction:   {pred_class0}/{total_voxels} ({100*pred_class0/total_voxels:.1f}%)")

    if pred_class0 < gt_class0 * 0.8:
        print("\n   WARNING: Model is under-predicting class 0 (filling gaps)!")

    # Accuracy per class
    print("\n2. Per-class accuracy (first 10 classes):")
    for cls_idx in range(min(10, cfg.num_classes)):
        cls_mask = gt_flat == cls_idx
        if cls_mask.sum() > 0:
            correct = (pred_flat[cls_mask] == cls_idx).sum().item()
            total = cls_mask.sum().item()
            acc = 100 * correct / total
            print(f"   Class {cls_idx}: {acc:.1f}% ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Diagnose hyperbolic module issues")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to analyze")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, args.config, device)

    # Load a sample
    print(f"\nLoading sample {args.sample_idx} from validation set...")
    val_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "val", cfg.volume_size)
    sample_input, sample_labels = val_dataset[args.sample_idx]
    sample_input = sample_input.unsqueeze(0).to(device)  # [1, 1, D, H, W]
    sample_labels = sample_labels.unsqueeze(0).to(device)  # [1, D, H, W]

    print(f"Input shape: {sample_input.shape}")
    print(f"Labels shape: {sample_labels.shape}")

    # Run analyses
    analyze_label_embeddings(model, cfg.hyp_curv)
    analyze_voxel_embeddings(model, sample_input, sample_labels, cfg.hyp_curv)
    analyze_triplet_loss(model, sample_input, sample_labels, cfg)
    analyze_gradient_flow(model, sample_input.clone(), sample_labels.clone(), cfg, device)
    compare_with_baseline_predictions(model, sample_input, sample_labels, cfg)

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
