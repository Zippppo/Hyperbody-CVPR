"""
Compare initial vs trained label embeddings for lorentz_random model.
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
    cfg = Config.from_yaml("configs/lorentz_random.yaml")

    # Load class info
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

    print("="*60)
    print("LORENTZ RANDOM MODEL")
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

    # Load trained checkpoint
    checkpoint = torch.load("checkpoints/lorentz/latest.pth", map_location="cpu", weights_only=False)
    trained_state = checkpoint['model_state_dict']

    # Extract label embedding tangent vectors
    trained_tangent = trained_state['label_emb.tangent_embeddings']
    trained_norms = trained_tangent.norm(dim=-1)

    print(f"\nTrained tangent norms:")
    print(f"  Min: {trained_norms.min().item():.4f}")
    print(f"  Max: {trained_norms.max().item():.4f}")
    print(f"  Mean: {trained_norms.mean().item():.4f}")

    # Compute change
    norm_change = trained_norms - initial_norms
    print(f"\nNorm change (trained - initial):")
    print(f"  Min: {norm_change.min().item():.4f}")
    print(f"  Max: {norm_change.max().item():.4f}")
    print(f"  Mean: {norm_change.mean().item():.4f}")


if __name__ == "__main__":
    main()
