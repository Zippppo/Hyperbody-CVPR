"""
Compare initial vs trained label embeddings to see how they've changed.
"""
import json
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from data.organ_hierarchy import load_organ_hierarchy
from models.hyperbolic.label_embedding import LorentzLabelEmbedding

def main():
    # Load config
    cfg = Config.from_yaml("configs/lorentz_semantic.yaml")

    # Load class info
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

    print("="*60)
    print("INITIAL LABEL EMBEDDING (before training)")
    print("="*60)

    # Create fresh label embedding (same initialization as training)
    torch.manual_seed(42)  # Same seed as training
    initial_emb = LorentzLabelEmbedding(
        num_classes=cfg.num_classes,
        embed_dim=cfg.hyp_embed_dim,
        curv=cfg.hyp_curv,
        class_depths=class_depths,
        min_radius=cfg.hyp_min_radius,
        max_radius=cfg.hyp_max_radius,
        direction_mode=cfg.hyp_direction_mode,
        text_embedding_path=cfg.hyp_text_embedding_path,
    )

    initial_tangent = initial_emb.tangent_embeddings.detach()
    initial_norms = initial_tangent.norm(dim=-1)

    print(f"\nInitial tangent norms:")
    print(f"  Min: {initial_norms.min().item():.4f}")
    print(f"  Max: {initial_norms.max().item():.4f}")
    print(f"  Mean: {initial_norms.mean().item():.4f}")

    # Show norms for first 10 classes
    print(f"\nFirst 10 classes initial norms:")
    for i in range(10):
        print(f"  Class {i} ({class_names[i]}): {initial_norms[i].item():.4f}")

    print("\n" + "="*60)
    print("TRAINED LABEL EMBEDDING (from checkpoint)")
    print("="*60)

    # Load trained checkpoint
    checkpoint = torch.load("checkpoints/lorentz_semantic/latest.pth", map_location="cpu", weights_only=False)
    trained_state = checkpoint['model_state_dict']

    # Extract label embedding tangent vectors
    trained_tangent = trained_state['label_emb.tangent_embeddings']
    trained_norms = trained_tangent.norm(dim=-1)

    print(f"\nTrained tangent norms:")
    print(f"  Min: {trained_norms.min().item():.4f}")
    print(f"  Max: {trained_norms.max().item():.4f}")
    print(f"  Mean: {trained_norms.mean().item():.4f}")

    # Show norms for first 10 classes
    print(f"\nFirst 10 classes trained norms:")
    for i in range(10):
        print(f"  Class {i} ({class_names[i]}): {trained_norms[i].item():.4f}")

    print("\n" + "="*60)
    print("CHANGE ANALYSIS")
    print("="*60)

    # Compute change
    norm_change = trained_norms - initial_norms
    direction_change = torch.nn.functional.cosine_similarity(
        initial_tangent, trained_tangent, dim=-1
    )

    print(f"\nNorm change (trained - initial):")
    print(f"  Min: {norm_change.min().item():.4f}")
    print(f"  Max: {norm_change.max().item():.4f}")
    print(f"  Mean: {norm_change.mean().item():.4f}")

    print(f"\nDirection similarity (cosine):")
    print(f"  Min: {direction_change.min().item():.4f}")
    print(f"  Max: {direction_change.max().item():.4f}")
    print(f"  Mean: {direction_change.mean().item():.4f}")

    # Check if embeddings collapsed (all similar)
    print(f"\nCollapse check:")
    pairwise_cos = torch.nn.functional.cosine_similarity(
        trained_tangent.unsqueeze(0), trained_tangent.unsqueeze(1), dim=-1
    )
    # Exclude diagonal
    mask = ~torch.eye(cfg.num_classes, dtype=torch.bool)
    off_diag_cos = pairwise_cos[mask]
    print(f"  Pairwise cosine similarity (off-diagonal):")
    print(f"    Min: {off_diag_cos.min().item():.4f}")
    print(f"    Max: {off_diag_cos.max().item():.4f}")
    print(f"    Mean: {off_diag_cos.mean().item():.4f}")

    if off_diag_cos.mean() > 0.8:
        print("  WARNING: High mean cosine similarity suggests embeddings are collapsing!")


if __name__ == "__main__":
    main()
