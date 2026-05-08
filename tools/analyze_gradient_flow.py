"""
Detailed analysis of gradient flow through the hyperbolic loss.

This script traces how gradients flow from the loss back to tangent_embeddings
to understand why embeddings collapse to center.
"""
import torch
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.body_net import BodyNet
from models.hyperbolic.lorentz_loss import LorentzRankingLoss
from models.hyperbolic.lorentz_ops import distance_to_origin, exp_map0
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system
from config import Config


def analyze_loss_gradient_direction(config_path: str):
    """
    Analyze the direction of gradients on label embeddings.

    Key question: Are gradients pushing embeddings toward or away from origin?
    """
    print(f"\n{'='*60}")
    print(f"Analyzing gradient direction on label embeddings")
    print(f"{'='*60}\n")

    cfg = Config.from_yaml(config_path)

    # Load class info
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

    # Create model
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

    # Create dummy data
    B, D, H, W = 2, 67, 36, 32  # D=Depth, spatial dimensions
    C = cfg.hyp_embed_dim  # C=Channels (embed_dim)

    # Random voxel embeddings
    voxel_emb = torch.randn(B, C, D, H, W) * 0.5

    # Random labels (ensure all classes appear)
    labels = torch.randint(0, cfg.num_classes, (B, D, H, W))

    # Get initial tangent embeddings
    tangent_before = model.label_emb.tangent_embeddings.data.clone()
    norms_before = torch.norm(tangent_before, dim=-1)

    # Get label embeddings (on manifold)
    label_emb = model.label_emb()

    # Create loss
    hyp_criterion = LorentzRankingLoss(
        margin=cfg.hyp_margin,
        curv=cfg.hyp_curv,
        num_samples_per_class=cfg.hyp_samples_per_class,
        num_negatives=cfg.hyp_num_negatives,
    )

    # Forward pass
    loss = hyp_criterion(voxel_emb, labels, label_emb)

    # Backward pass
    loss.backward()

    # Get gradients
    tangent_grad = model.label_emb.tangent_embeddings.grad

    if tangent_grad is None:
        print("ERROR: No gradients on tangent_embeddings!")
        return

    # Analyze gradient direction relative to current position
    # If gradient points toward origin, dot product with position is negative
    # If gradient points away from origin, dot product is positive

    # Normalize both vectors for direction analysis
    tangent_normalized = tangent_before / (torch.norm(tangent_before, dim=-1, keepdim=True) + 1e-8)
    grad_normalized = tangent_grad / (torch.norm(tangent_grad, dim=-1, keepdim=True) + 1e-8)

    # Dot product: positive means same direction, negative means opposite
    dot_products = (tangent_normalized * grad_normalized).sum(dim=-1)

    print(f"Loss value: {loss.item():.4f}")
    print(f"\nGradient direction analysis:")
    print(f"  Dot product (tangent · gradient):")
    print(f"    Mean: {dot_products.mean():.4f}")
    print(f"    Std: {dot_products.std():.4f}")
    print(f"    Range: [{dot_products.min():.4f}, {dot_products.max():.4f}]")

    # Count how many gradients point toward vs away from origin
    toward_origin = (dot_products < 0).sum().item()
    away_from_origin = (dot_products > 0).sum().item()

    print(f"\n  Direction count:")
    print(f"    Toward origin (negative dot): {toward_origin}/{cfg.num_classes}")
    print(f"    Away from origin (positive dot): {away_from_origin}/{cfg.num_classes}")

    # After gradient update, embeddings move in -gradient direction
    # So if dot(tangent, grad) < 0, update moves toward origin
    # If dot(tangent, grad) > 0, update moves away from origin

    if dot_products.mean() < -0.1:
        print(f"\n  FINDING: Gradients predominantly point AWAY from current position")
        print(f"           Since optimizer does -gradient, embeddings move TOWARD origin")
        print(f"           This explains the collapse!")
    elif dot_products.mean() > 0.1:
        print(f"\n  FINDING: Gradients predominantly point TOWARD current position")
        print(f"           Since optimizer does -gradient, embeddings move AWAY from origin")
    else:
        print(f"\n  FINDING: Gradients are mixed/perpendicular to current position")

    # Analyze gradient magnitude vs distance from origin
    print(f"\nGradient magnitude vs distance from origin:")
    distances = distance_to_origin(label_emb, cfg.hyp_curv).detach()
    grad_norms = torch.norm(tangent_grad, dim=-1)

    # Correlation
    correlation = torch.corrcoef(torch.stack([distances, grad_norms]))[0, 1]
    print(f"  Correlation: {correlation:.4f}")

    if correlation > 0.3:
        print(f"  FINDING: Larger gradients for embeddings farther from origin")
        print(f"           This creates a 'pull toward center' effect")
    elif correlation < -0.3:
        print(f"  FINDING: Larger gradients for embeddings closer to origin")

    # Check if loss is dominated by positive or negative pairs
    print(f"\nLoss component analysis:")
    with torch.no_grad():
        # Recompute to inspect internals
        from models.hyperbolic.lorentz_ops import pointwise_dist, pairwise_dist

        B, C, D, H, W = voxel_emb.shape
        num_classes = label_emb.shape[0]

        voxel_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1)

        # Sample some anchors
        sample_indices = torch.randperm(voxel_flat.shape[0])[:1000]
        anchors = voxel_flat[sample_indices]
        anchor_classes = labels_flat[sample_indices]

        # Get positive distances
        positives = label_emb[anchor_classes]
        d_pos = pointwise_dist(anchors, positives, cfg.hyp_curv)

        # Get negative distances (to all other classes)
        all_dists = pairwise_dist(anchors, label_emb, cfg.hyp_curv)

        # For each anchor, get distance to a random negative
        neg_classes = torch.randint(0, num_classes, (len(anchors),))
        # Make sure it's not the true class
        neg_classes = (anchor_classes + 1 + neg_classes) % num_classes
        d_neg = all_dists[torch.arange(len(anchors)), neg_classes]

        print(f"  Positive distances: mean={d_pos.mean():.4f}, std={d_pos.std():.4f}")
        print(f"  Negative distances: mean={d_neg.mean():.4f}, std={d_neg.std():.4f}")
        print(f"  Margin: {cfg.hyp_margin}")

        # Triplet loss: max(0, margin + d_pos - d_neg)
        violations = (cfg.hyp_margin + d_pos - d_neg > 0).sum().item()
        print(f"  Margin violations: {violations}/{len(anchors)} ({100*violations/len(anchors):.1f}%)")

        if d_pos.mean() > d_neg.mean():
            print(f"\n  PROBLEM: Positive pairs are FARTHER than negative pairs!")
            print(f"           Loss will push ALL embeddings closer together")
            print(f"           This causes collapse to center")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    analyze_loss_gradient_direction(args.config)


if __name__ == '__main__':
    main()
