"""
使用训练好的权重，来可视化text embedding在双曲空间中的分布情况。
原因：text embedding到双曲空间中间还有一个MLP的降维过程（是可训练的）；降维之后才使用exp_map映射到双曲空间。

Usage:
    conda activate pasco
    python scripts/body/text_embedding/visualize_trained_hyperbolic_embeddings.py

    # Custom curvature
    python scripts/body/text_embedding/visualize_trained_hyperbolic_embeddings.py --curv 0.5

Output:
    scripts/body/text_embedding/figures/trained_hyperbolic_poincare_{model}.png
    scripts/body/text_embedding/figures/trained_hyperbolic_comparison.png
    scripts/body/text_embedding/figures/trained_hyperbolic_distance_analysis.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Patch
from sklearn.decomposition import PCA

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pasco.models.hyperbolic.lorentz_ops import (
    exp_map0,
    hyperbolic_distance_to_origin,
    pairwise_dist,
)
from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

# Import category mappings from existing visualize script
from scripts.body.text_embedding.visualize_embeddings import (
    CATEGORY_COLORS,
    CATEGORY_MAPPING,
    get_category_for_label,
)


# Checkpoint paths
CHECKPOINT_PATHS = {
    'sat': project_root / 'logs/CE+Emb/SAT_bs2_lr0.0001_ch16_hyp0.1/checkpoints/best_model.ckpt',
    'clip': project_root / 'logs/CE+Emb/CLIP_bs2_lr0.0001_ch16_hyp0.1/checkpoints/best_model.ckpt',
}

# Text embedding dims
TEXT_DIMS = {
    'sat': 768,
    'clip': 768,
    'biomedclip': 512,
}


def lorentz_to_poincare(x_space: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """
    Convert from Lorentz (hyperboloid) model to Poincare ball model.

    Args:
        x_space: (N, D) spatial components on the hyperboloid
        curv: positive scalar, hyperbolic curvature

    Returns:
        (N, D) points in the Poincare ball (norm < 1)
    """
    # Compute time component: x_time = sqrt(1/curv + ||x_space||^2)
    x_time = torch.sqrt(1 / curv + torch.sum(x_space ** 2, dim=-1, keepdim=True))

    # Stereographic projection: p = x_space / (x_time + sqrt(1/curv))
    poincare = x_space / (x_time + (1 / curv) ** 0.5)

    return poincare


def load_trained_projector(
    checkpoint_path: Path,
    text_dim: int = 768,
    embed_dim: int = 32,
) -> Tuple[TextEmbeddingProjector, torch.Tensor, List[str]]:
    """
    Load trained projector weights and text embeddings from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        text_dim: Input text embedding dimension
        embed_dim: Output embedding dimension

    Returns:
        projector: TextEmbeddingProjector with loaded weights
        text_embeddings: Original text embeddings [N, text_dim]
        label_names: List of label names
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']

    # Create projector
    projector = TextEmbeddingProjector(
        text_dim=text_dim,
        embed_dim=embed_dim,
        hidden_dim=256,
    )

    # Extract projector weights from state_dict
    projector_state = {}
    for k, v in state_dict.items():
        if 'label_emb.projector.' in k:
            new_key = k.replace('label_emb.projector.', '')
            projector_state[new_key] = v

    projector.load_state_dict(projector_state)
    projector.eval()

    # Extract text embeddings
    text_embeddings = state_dict['label_emb.text_embeddings']  # [70, 768]

    # Get label names from the embedding file
    # We need to load the original embedding file to get label names
    model_name = 'sat' if 'SAT' in str(checkpoint_path) else 'clip'
    emb_path = project_root / f'Dataset/text_embeddings/{model_name}_labeled_embeddings.pt'
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    label_names = emb_data['label_names']

    return projector, text_embeddings, label_names


def project_to_hyperbolic(
    projector: TextEmbeddingProjector,
    text_embeddings: torch.Tensor,
    curv: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project text embeddings to hyperbolic space using trained projector.

    Args:
        projector: Trained TextEmbeddingProjector
        text_embeddings: Original text embeddings [N, text_dim]
        curv: Hyperbolic curvature

    Returns:
        poincare_emb: Points in Poincare disk [N, embed_dim]
        lorentz_emb: Space components in Lorentz model [N, embed_dim]
    """
    with torch.no_grad():
        # Project to tangent space (32-dim)
        tangent_vectors = projector(text_embeddings.float())

        # Zero out class 0 (outside_body)
        tangent_vectors[0] = 0.0

        # Map to hyperboloid
        lorentz_emb = exp_map0(tangent_vectors, curv=curv)

        # Convert to Poincare disk
        poincare_emb = lorentz_to_poincare(lorentz_emb, curv=curv)

    return poincare_emb, lorentz_emb


def reduce_to_2d_for_viz(
    embeddings: torch.Tensor,
    curv: float = 1.0,
) -> torch.Tensor:
    """
    Reduce 32-dim hyperbolic embeddings to 2D for visualization.

    Uses PCA on Poincare coordinates, then re-projects to ensure
    points stay within the disk.

    Args:
        embeddings: (N, D) points in Poincare disk
        curv: Hyperbolic curvature

    Returns:
        (N, 2) points in 2D Poincare disk
    """
    embeddings_np = embeddings.numpy()

    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)

    # Scale to preserve relative radii
    original_radii = np.linalg.norm(embeddings_np, axis=-1, keepdims=True)
    new_radii = np.linalg.norm(embeddings_2d, axis=-1, keepdims=True)

    # Avoid division by zero
    scale = np.where(
        new_radii > 1e-8,
        original_radii / (new_radii + 1e-8),
        1.0
    )

    # Apply scale and ensure within unit ball
    embeddings_2d_scaled = embeddings_2d * scale
    norms = np.linalg.norm(embeddings_2d_scaled, axis=-1, keepdims=True)
    # Clamp points outside the ball to 0.99 radius
    embeddings_2d_scaled = np.where(
        norms > 0.99,
        embeddings_2d_scaled * 0.99 / (norms + 1e-8),
        embeddings_2d_scaled
    )

    return torch.from_numpy(embeddings_2d_scaled).float()


def plot_poincare_disk(
    ax: plt.Axes,
    points: torch.Tensor,
    labels: List[str],
    title: str,
    curv: float = 1.0,
    show_labels: bool = True,
    fontsize: int = 6,
) -> None:
    """
    Plot points on the Poincare disk.

    Args:
        ax: matplotlib axes
        points: (N, 2) points in Poincare disk (norm < 1)
        labels: list of label names
        title: plot title
        curv: curvature (for display)
        show_labels: whether to show text labels
        fontsize: font size for labels
    """
    points_np = points.numpy()

    # Draw the boundary circle (ideal boundary at infinity)
    boundary = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(boundary)

    # Draw concentric circles to show distance from origin
    for r in [0.25, 0.5, 0.75]:
        circle = Circle((0, 0), r, fill=False, color='lightgray',
                        linewidth=0.5, linestyle='--', alpha=0.5)
        ax.add_patch(circle)

    # Get colors based on categories
    categories = [get_category_for_label(name) for name in labels]
    colors = [CATEGORY_COLORS.get(cat, '#808080') for cat in categories]

    # Compute distance from origin for sizing (closer = larger)
    radii = np.linalg.norm(points_np, axis=-1)
    sizes = 100 * (1 - radii * 0.5)  # Larger for closer points

    # Plot points
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        c=colors,
        s=sizes,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5,
        zorder=10,
    )

    # Add labels
    if show_labels:
        for i, name in enumerate(labels):
            short_name = name.replace('_left', '_L').replace('_right', '_R')
            short_name = short_name.replace('rib_L_', 'rL').replace('rib_R_', 'rR')

            # Offset label slightly outward
            offset_dir = points_np[i] / (np.linalg.norm(points_np[i]) + 1e-8)
            offset = offset_dir * 0.03

            ax.annotate(
                short_name,
                (points_np[i, 0] + offset[0], points_np[i, 1] + offset[1]),
                fontsize=fontsize,
                alpha=0.8,
                ha='center',
                va='center',
            )

    # Set equal aspect and limits
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n(Poincare Disk, curv={curv})', fontsize=12, fontweight='bold')

    # Hide axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add origin marker
    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=2, zorder=11)


def create_trained_visualization(
    model_name: str,
    checkpoint_path: Path,
    output_dir: Path,
    curv: float = 1.0,
) -> Dict:
    """
    Create visualization for a single trained model.

    Returns dict with embeddings data for further analysis.
    """
    text_dim = TEXT_DIMS[model_name]

    # Load trained projector and embeddings
    projector, text_embeddings, label_names = load_trained_projector(
        checkpoint_path,
        text_dim=text_dim,
    )

    print(f"Processing {model_name}...")
    print(f"  Text embeddings shape: {text_embeddings.shape}")
    print(f"  Number of labels: {len(label_names)}")

    # Project to hyperbolic space
    poincare_emb, lorentz_emb = project_to_hyperbolic(
        projector, text_embeddings, curv=curv
    )

    print(f"  Poincare embeddings shape: {poincare_emb.shape}")

    # Reduce to 2D for visualization
    poincare_2d = reduce_to_2d_for_viz(poincare_emb, curv=curv)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    fontsize = 5 if len(label_names) > 80 else 7

    plot_poincare_disk(
        ax,
        poincare_2d,
        label_names,
        f'{model_name.upper()} TRAINED Embeddings',
        curv=curv,
        show_labels=True,
        fontsize=fontsize,
    )

    # Add legend
    unique_categories = sorted(set(
        get_category_for_label(name) for name in label_names
    ))
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS.get(cat, '#808080'),
              label=cat.replace('_', ' '))
        for cat in unique_categories
    ]
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
    )

    plt.tight_layout()
    output_path = output_dir / f'trained_hyperbolic_poincare_{model_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return {
        'label_names': label_names,
        'poincare_emb': poincare_emb,
        'lorentz_emb': lorentz_emb,
        'poincare_2d': poincare_2d,
    }


def create_comparison_plot(
    results: Dict[str, Dict],
    output_path: Path,
    curv: float = 1.0,
) -> None:
    """Create side-by-side comparison of trained models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, results.items()):
        plot_poincare_disk(
            ax,
            data['poincare_2d'],
            data['label_names'],
            f'{model_name.upper()} (Trained)',
            curv=curv,
            show_labels=False,
            fontsize=5,
        )

    # Add legend
    all_labels = []
    for data in results.values():
        all_labels.extend(data['label_names'])

    unique_categories = sorted(set(
        get_category_for_label(name) for name in all_labels
    ))
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS.get(cat, '#808080'),
              label=cat.replace('_', ' '))
        for cat in unique_categories
    ]
    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(1.12, 0.5),
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def create_distance_analysis(
    results: Dict[str, Dict],
    output_path: Path,
    curv: float = 1.0,
) -> None:
    """Analyze and visualize hyperbolic distances from origin."""
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for col, (model_name, data) in enumerate(results.items()):
        lorentz_emb = data['lorentz_emb']
        label_names = data['label_names']

        # Compute distances from origin
        distances = hyperbolic_distance_to_origin(lorentz_emb, curv=curv).numpy()

        # Top plot: Distance histogram by category
        ax_hist = axes[0, col]
        categories = [get_category_for_label(name) for name in label_names]
        unique_cats = sorted(set(categories))

        cat_distances = {cat: [] for cat in unique_cats}
        for i, cat in enumerate(categories):
            cat_distances[cat].append(distances[i])

        x_positions = np.arange(len(unique_cats))
        bar_width = 0.6

        means = [np.mean(cat_distances[cat]) for cat in unique_cats]
        stds = [np.std(cat_distances[cat]) for cat in unique_cats]
        colors = [CATEGORY_COLORS.get(cat, '#808080') for cat in unique_cats]

        ax_hist.bar(x_positions, means, bar_width, yerr=stds, color=colors,
                    alpha=0.7, edgecolor='white', capsize=3)
        ax_hist.set_xticks(x_positions)
        ax_hist.set_xticklabels([c.replace('_', '\n') for c in unique_cats],
                                 fontsize=6, rotation=45, ha='right')
        ax_hist.set_ylabel('Hyperbolic Distance from Origin')
        ax_hist.set_title(f'{model_name.upper()} (Trained): Distance by Category')

        # Bottom plot: Sorted distances
        ax_sorted = axes[1, col]
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        actual_colors = [CATEGORY_COLORS.get(categories[i], '#808080')
                        for i in sorted_indices]
        ax_sorted.bar(range(len(sorted_distances)), sorted_distances,
                      color=actual_colors, alpha=0.7, width=1.0)
        ax_sorted.set_xlabel('Labels (sorted by distance)')
        ax_sorted.set_ylabel('Hyperbolic Distance from Origin')
        ax_sorted.set_title(f'{model_name.upper()} (Trained): Distance Distribution')

        # Annotate extremes
        n_show = min(5, len(label_names))
        for i in range(n_show):
            idx = sorted_indices[i]
            short_name = label_names[idx].replace('_left', '_L').replace('_right', '_R')
            ax_sorted.annotate(short_name, (i, sorted_distances[i]),
                              fontsize=5, rotation=45, ha='right')
            idx = sorted_indices[-(i+1)]
            short_name = label_names[idx].replace('_left', '_L').replace('_right', '_R')
            ax_sorted.annotate(short_name, (len(sorted_distances)-1-i, sorted_distances[-(i+1)]),
                              fontsize=5, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance analysis: {output_path}")


def print_trained_stats(results: Dict[str, Dict], curv: float = 1.0) -> None:
    """Print statistics about trained hyperbolic embeddings."""
    print("\n" + "=" * 60)
    print("TRAINED Hyperbolic Embedding Statistics")
    print("=" * 60)

    for model_name, data in results.items():
        lorentz_emb = data['lorentz_emb']
        label_names = data['label_names']

        dist_to_origin = hyperbolic_distance_to_origin(lorentz_emb, curv=curv)
        pairwise = pairwise_dist(lorentz_emb, lorentz_emb, curv=curv)

        print(f"\n{model_name.upper()} (Trained, curvature={curv}):")
        print(f"  Embedding shape: {lorentz_emb.shape}")
        print(f"  Distance to origin: min={dist_to_origin.min():.4f}, "
              f"max={dist_to_origin.max():.4f}, mean={dist_to_origin.mean():.4f}")

        # Exclude diagonal
        mask = ~torch.eye(len(label_names), dtype=torch.bool)
        pairwise_values = pairwise[mask]
        print(f"  Pairwise distances: min={pairwise_values.min():.4f}, "
              f"max={pairwise_values.max():.4f}, mean={pairwise_values.mean():.4f}")

        # Find closest and furthest
        # Skip class 0 (outside_body) which is at origin
        dist_no_zero = dist_to_origin.clone()
        dist_no_zero[0] = float('inf')

        closest_idx = dist_no_zero.argmin().item()
        furthest_idx = dist_to_origin.argmax().item()
        print(f"  Closest to origin (non-zero): {label_names[closest_idx]} "
              f"(d={dist_to_origin[closest_idx]:.4f})")
        print(f"  Furthest from origin: {label_names[furthest_idx]} "
              f"(d={dist_to_origin[furthest_idx]:.4f})")

        # Tangent vector norms (before exp_map0)
        poincare_norms = data['poincare_emb'].norm(dim=-1)
        print(f"  Poincare norms: min={poincare_norms.min():.4f}, "
              f"max={poincare_norms.max():.4f}, mean={poincare_norms.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TRAINED text embeddings in hyperbolic space.'
    )
    parser.add_argument(
        '--curv',
        type=float,
        default=1.0,
        help='Hyperbolic curvature (positive value, default: 1.0)'
    )
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    print(f"Curvature: {args.curv}")

    # Process available models
    results = {}
    for model_name, ckpt_path in CHECKPOINT_PATHS.items():
        if ckpt_path.exists():
            try:
                results[model_name] = create_trained_visualization(
                    model_name,
                    ckpt_path,
                    output_dir,
                    curv=args.curv,
                )
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
        else:
            print(f"Warning: Checkpoint not found for {model_name}: {ckpt_path}")

    if not results:
        print("No trained models found!")
        return

    # Print statistics
    print_trained_stats(results, curv=args.curv)

    # Create comparison and analysis plots
    print("\n" + "=" * 60)
    print("Creating Additional Visualizations")
    print("=" * 60)

    create_comparison_plot(
        results,
        output_dir / 'trained_hyperbolic_comparison.png',
        curv=args.curv,
    )

    create_distance_analysis(
        results,
        output_dir / 'trained_hyperbolic_distance_analysis.png',
        curv=args.curv,
    )

    print("\n" + "=" * 60)
    print("Done! Trained hyperbolic visualizations saved to:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
