"""
Debug script to investigate why text embeddings collapse to center.

Checks:
1. Gradient magnitudes on tangent_embeddings
2. Learning rate applied to embeddings
3. Distance changes per epoch
4. Loss gradient flow
"""
import torch
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.body_net import BodyNet
from models.hyperbolic.lorentz_loss import LorentzRankingLoss
from models.hyperbolic.lorentz_ops import distance_to_origin
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system
from config import Config


def analyze_checkpoint_embeddings(checkpoint_path: str, config_path: str):
    """Load checkpoint and analyze embedding state."""
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    # Load config
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']

    # Get tangent embeddings (before exp_map)
    tangent_emb = model.label_emb.tangent_embeddings.data  # [N, D]

    # Get manifold embeddings (after exp_map)
    with torch.no_grad():
        manifold_emb = model.label_emb()  # [N, D]

    # Compute statistics
    tangent_norms = torch.norm(tangent_emb, dim=-1)  # [N]
    manifold_distances = distance_to_origin(manifold_emb, cfg.hyp_curv)  # [N]

    print(f"Epoch: {epoch}")
    print(f"\nTangent space statistics:")
    print(f"  Norm range: [{tangent_norms.min():.4f}, {tangent_norms.max():.4f}]")
    print(f"  Norm mean: {tangent_norms.mean():.4f}")
    print(f"  Norm std: {tangent_norms.std():.4f}")

    print(f"\nManifold space statistics:")
    print(f"  Distance from origin range: [{manifold_distances.min():.4f}, {manifold_distances.max():.4f}]")
    print(f"  Distance mean: {manifold_distances.mean():.4f}")
    print(f"  Distance std: {manifold_distances.std():.4f}")

    # Check if embeddings are collapsing (low std indicates collapse)
    if manifold_distances.std() < 0.1:
        print(f"\n⚠️  WARNING: Embeddings appear to be collapsing! (std={manifold_distances.std():.4f})")

    return {
        'epoch': epoch,
        'tangent_norms': tangent_norms,
        'manifold_distances': manifold_distances,
    }


def simulate_gradient_update(config_path: str):
    """Simulate one gradient update to check magnitude."""
    print(f"\n{'='*60}")
    print(f"Simulating gradient update")
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

    # Random voxel embeddings on manifold
    voxel_emb = torch.randn(B, C, D, H, W) * 0.5

    # Random labels
    labels = torch.randint(0, cfg.num_classes, (B, D, H, W))

    # Get label embeddings
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

    # Check gradients on tangent_embeddings
    tangent_grad = model.label_emb.tangent_embeddings.grad

    if tangent_grad is not None:
        grad_norms = torch.norm(tangent_grad, dim=-1)  # [N]

        print(f"Loss value: {loss.item():.4f}")
        print(f"\nGradient statistics on tangent_embeddings:")
        print(f"  Gradient norm range: [{grad_norms.min():.6f}, {grad_norms.max():.6f}]")
        print(f"  Gradient norm mean: {grad_norms.mean():.6f}")
        print(f"  Gradient norm std: {grad_norms.std():.6f}")

        # Simulate update with Adam (approximate)
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        hyp_weight = cfg.hyp_weight

        # Effective gradient after scaling by hyp_weight
        effective_grad = tangent_grad * hyp_weight
        effective_grad_norms = torch.norm(effective_grad, dim=-1)

        print(f"\nEffective gradient (after hyp_weight={hyp_weight}):")
        print(f"  Effective gradient norm range: [{effective_grad_norms.min():.6f}, {effective_grad_norms.max():.6f}]")
        print(f"  Effective gradient norm mean: {effective_grad_norms.mean():.6f}")

        # Estimate update magnitude (simplified, ignoring Adam momentum)
        update_magnitude = lr * effective_grad_norms

        print(f"\nEstimated update magnitude (lr={lr}):")
        print(f"  Update range: [{update_magnitude.min():.6f}, {update_magnitude.max():.6f}]")
        print(f"  Update mean: {update_magnitude.mean():.6f}")

        # Compare to current tangent norms
        tangent_norms = torch.norm(model.label_emb.tangent_embeddings.data, dim=-1)
        relative_change = update_magnitude / (tangent_norms + 1e-8)

        print(f"\nRelative change (update / current_norm):")
        print(f"  Relative change range: [{relative_change.min():.6f}, {relative_change.max():.6f}]")
        print(f"  Relative change mean: {relative_change.mean():.6f}")

        if relative_change.mean() > 0.1:
            print(f"\n⚠️  WARNING: Large relative changes detected! (mean={relative_change.mean():.4f})")
            print(f"   This could cause embeddings to collapse rapidly.")
            print(f"   Consider:")
            print(f"   1. Reducing hyp_weight (current: {hyp_weight})")
            print(f"   2. Reducing learning rate (current: {lr})")
            print(f"   3. Using separate optimizer with lower LR for embeddings")
    else:
        print("No gradients found on tangent_embeddings!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to analyze')
    parser.add_argument('--simulate', action='store_true', help='Simulate gradient update')
    args = parser.parse_args()

    if args.checkpoint:
        analyze_checkpoint_embeddings(args.checkpoint, args.config)

    if args.simulate:
        simulate_gradient_update(args.config)


if __name__ == '__main__':
    main()
