"""
Visualize text embeddings in hyperbolic space (Poincare disk model).

This script projects high-dimensional text embeddings from different encoders
(CLIP, SAT, BioMedCLIP) into hyperbolic space and visualizes them in the
2D Poincare disk model.

Usage:
    conda activate pasco
    python scripts/body/text_embedding/visualize_hyperbolic_embeddings.py

    # Custom curvature
    python scripts/body/text_embedding/visualize_hyperbolic_embeddings.py --curv 0.5

    # Include category nodes
    python scripts/body/text_embedding/visualize_hyperbolic_embeddings.py --include-categories

Output:
    scripts/body/text_embedding/figures/hyperbolic_poincare_{model}.png
    scripts/body/text_embedding/figures/hyperbolic_comparison.png
    scripts/body/text_embedding/figures/hyperbolic_distance_analysis.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

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


# Import category mappings from existing visualize script
from scripts.body.text_embedding.visualize_embeddings import (
    CATEGORY_COLORS,
    CATEGORY_MAPPING,
    CATEGORY_NODE_MAPPING,
    get_category_for_label,
)


def lorentz_to_poincare(
    x_space: torch.Tensor, curv: float = 1.0
) -> torch.Tensor:
    """
    Convert from Lorentz (hyperboloid) model to Poincare ball model.

    The Poincare ball is the stereographic projection of the hyperboloid
    onto the plane tangent to its apex.

    Args:
        x_space: (N, D) spatial components on the hyperboloid
        curv: positive scalar, hyperbolic curvature

    Returns:
        (N, D) points in the Poincare ball (norm < 1)
    """
    # Compute time component: x_time = sqrt(1/curv + ||x_space||^2)
    x_time = torch.sqrt(1 / curv + torch.sum(x_space ** 2, dim=-1, keepdim=True))

    # Stereographic projection: p = x_space / (x_time + sqrt(1/curv))
    # This maps the hyperboloid to the unit ball
    poincare = x_space / (x_time + (1 / curv) ** 0.5)

    return poincare


def reduce_to_hyperbolic_2d(
    embeddings: torch.Tensor,
    curv: float = 1.0,
    pca_dim: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce high-dimensional embeddings to 2D hyperbolic space.

    Process:
    1. Apply PCA to reduce to intermediate dimension
    2. Project to 2D
    3. Map to hyperbolic space via exp_map0
    4. Convert to Poincare disk for visualization

    Args:
        embeddings: (N, D) input embeddings (e.g., 768-dim)
        curv: hyperbolic curvature
        pca_dim: intermediate PCA dimension before final 2D reduction

    Returns:
        poincare_2d: (N, 2) points in Poincare disk
        lorentz_2d: (N, 2) spatial components in Lorentz model
    """
    embeddings_np = embeddings.float().numpy()

    # Step 1: PCA to intermediate dimension
    pca_inter = PCA(n_components=min(pca_dim, embeddings_np.shape[0] - 1))
    embeddings_inter = pca_inter.fit_transform(embeddings_np)

    # Step 2: PCA to 2D
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings_inter)

    # Normalize to unit ball and scale
    # Higher norms in embedding space -> further from origin in hyperbolic space
    norms_original = np.linalg.norm(embeddings_np, axis=-1, keepdims=True)
    norms_2d = np.linalg.norm(embeddings_2d, axis=-1, keepdims=True)

    # Scale 2D embeddings to preserve relative distances
    scale_factor = (norms_original / (norms_original.max() + 1e-8)).mean(axis=0)
    embeddings_2d_scaled = embeddings_2d * scale_factor * 2.0  # Scale for better spread

    # Convert to tensor
    tangent_2d = torch.from_numpy(embeddings_2d_scaled).float()

    # Step 3: Map from tangent space to hyperboloid
    lorentz_2d = exp_map0(tangent_2d, curv=curv)

    # Step 4: Convert to Poincare disk
    poincare_2d = lorentz_to_poincare(lorentz_2d, curv=curv)

    return poincare_2d, lorentz_2d


def plot_poincare_disk(
    ax: plt.Axes,
    points: torch.Tensor,
    labels: List[str],
    title: str,
    curv: float = 1.0,
    show_labels: bool = True,
    fontsize: int = 6,
    show_geodesics: bool = False,
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
        show_geodesics: whether to draw geodesic connections
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
    scatter = ax.scatter(
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


def create_hyperbolic_visualization(
    embeddings_dict: Dict[str, Dict],
    output_dir: Path,
    curv: float = 1.0,
    suffix: str = '',
) -> None:
    """Create Poincare disk visualizations for each model."""
    for model_name, data in embeddings_dict.items():
        embeddings = data['embeddings']
        label_names = data['label_names']

        print(f"Processing {model_name}...")

        # Reduce to 2D hyperbolic space
        poincare_2d, lorentz_2d = reduce_to_hyperbolic_2d(embeddings, curv=curv)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Adjust fontsize based on number of labels
        fontsize = 5 if len(label_names) > 80 else 7

        plot_poincare_disk(
            ax,
            poincare_2d,
            label_names,
            f'{model_name.upper()} Text Embeddings',
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
        output_path = output_dir / f'hyperbolic_poincare_{model_name}{suffix}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def create_comparison_plot(
    embeddings_dict: Dict[str, Dict],
    output_path: Path,
    curv: float = 1.0,
) -> None:
    """Create side-by-side comparison of all models in hyperbolic space."""
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, embeddings_dict.items()):
        embeddings = data['embeddings']
        label_names = data['label_names']

        print(f"Computing hyperbolic projection for {model_name}...")
        poincare_2d, _ = reduce_to_hyperbolic_2d(embeddings, curv=curv)

        plot_poincare_disk(
            ax,
            poincare_2d,
            label_names,
            model_name.upper(),
            curv=curv,
            show_labels=False,  # Too crowded
            fontsize=5,
        )

    # Add legend
    all_labels = []
    for data in embeddings_dict.values():
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
    embeddings_dict: Dict[str, Dict],
    output_path: Path,
    curv: float = 1.0,
) -> None:
    """Analyze and visualize hyperbolic distances from origin."""
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for col, (model_name, data) in enumerate(embeddings_dict.items()):
        embeddings = data['embeddings']
        label_names = data['label_names']

        # Get 2D hyperbolic representation
        poincare_2d, lorentz_2d = reduce_to_hyperbolic_2d(embeddings, curv=curv)

        # Compute distances from origin in hyperbolic space
        distances = hyperbolic_distance_to_origin(lorentz_2d, curv=curv).numpy()

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
        ax_hist.set_title(f'{model_name.upper()}: Distance by Category')

        # Bottom plot: Sorted distances
        ax_sorted = axes[1, col]
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        # Color by category
        actual_colors = [CATEGORY_COLORS.get(categories[i], '#808080')
                        for i in sorted_indices]
        ax_sorted.bar(range(len(sorted_distances)), sorted_distances,
                      color=actual_colors, alpha=0.7, width=1.0)
        ax_sorted.set_xlabel('Labels (sorted by distance)')
        ax_sorted.set_ylabel('Hyperbolic Distance from Origin')
        ax_sorted.set_title(f'{model_name.upper()}: Distance Distribution')

        # Add some label annotations for extremes
        n_show = min(5, len(label_names))
        for i in range(n_show):
            # Closest to origin
            idx = sorted_indices[i]
            short_name = label_names[idx].replace('_left', '_L').replace('_right', '_R')
            ax_sorted.annotate(short_name, (i, sorted_distances[i]),
                              fontsize=5, rotation=45, ha='right')
            # Furthest from origin
            idx = sorted_indices[-(i+1)]
            short_name = label_names[idx].replace('_left', '_L').replace('_right', '_R')
            ax_sorted.annotate(short_name, (len(sorted_distances)-1-i, sorted_distances[-(i+1)]),
                              fontsize=5, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved distance analysis: {output_path}")


def print_hyperbolic_stats(
    embeddings_dict: Dict[str, Dict],
    curv: float = 1.0,
) -> None:
    """Print statistics about hyperbolic embeddings."""
    print("\n" + "=" * 60)
    print("Hyperbolic Space Statistics")
    print("=" * 60)

    for model_name, data in embeddings_dict.items():
        embeddings = data['embeddings']
        label_names = data['label_names']

        # Get hyperbolic representation
        _, lorentz_2d = reduce_to_hyperbolic_2d(embeddings, curv=curv)

        # Compute distances
        dist_to_origin = hyperbolic_distance_to_origin(lorentz_2d, curv=curv)
        pairwise = pairwise_dist(lorentz_2d, lorentz_2d, curv=curv)

        print(f"\n{model_name.upper()} (curvature={curv}):")
        print(f"  Input shape: {embeddings.shape}")
        print(f"  Distance to origin: min={dist_to_origin.min():.4f}, "
              f"max={dist_to_origin.max():.4f}, mean={dist_to_origin.mean():.4f}")

        # Exclude diagonal for pairwise stats
        mask = ~torch.eye(len(label_names), dtype=torch.bool)
        pairwise_values = pairwise[mask]
        print(f"  Pairwise distances: min={pairwise_values.min():.4f}, "
              f"max={pairwise_values.max():.4f}, mean={pairwise_values.mean():.4f}")

        # Find closest and furthest from origin
        closest_idx = dist_to_origin.argmin().item()
        furthest_idx = dist_to_origin.argmax().item()
        print(f"  Closest to origin: {label_names[closest_idx]} "
              f"(d={dist_to_origin[closest_idx]:.4f})")
        print(f"  Furthest from origin: {label_names[furthest_idx]} "
              f"(d={dist_to_origin[furthest_idx]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize text embeddings in hyperbolic space (Poincare disk).'
    )
    parser.add_argument(
        '--curv',
        type=float,
        default=1.0,
        help='Hyperbolic curvature (positive value, default: 1.0)'
    )
    parser.add_argument(
        '--include-categories',
        action='store_true',
        help='Include category nodes in visualization'
    )
    args = parser.parse_args()

    # Paths
    dataset_dir = project_root / 'Dataset' / 'text_embeddings'
    output_dir = Path(__file__).resolve().parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    print(f"Curvature: {args.curv}")
    print(f"Include categories: {args.include_categories}")

    # Determine which embedding files to load
    if args.include_categories:
        embedding_files = {
            'clip': dataset_dir / 'clip_label_embeddings.pt',
            'biomedclip': dataset_dir / 'biomedclip_label_embeddings.pt',
            'sat': dataset_dir / 'sat_label_embeddings.pt',
        }
        suffix = '_with_categories'
    else:
        embedding_files = {
            'clip': dataset_dir / 'clip_labeled_embeddings.pt',
            'biomedclip': dataset_dir / 'biomedclip_labeled_embeddings.pt',
            'sat': dataset_dir / 'sat_labeled_embeddings.pt',
        }
        suffix = ''

    # Load embeddings
    embeddings_dict = {}
    for model_name, file_path in embedding_files.items():
        if file_path.exists():
            print(f"Loading {model_name} embeddings from: {file_path}")
            embeddings_dict[model_name] = torch.load(
                file_path, map_location='cpu', weights_only=False
            )
        else:
            print(f"Warning: {file_path} not found, skipping {model_name}")

    if not embeddings_dict:
        print("No embedding files found!")
        return

    # Print statistics
    print_hyperbolic_stats(embeddings_dict, curv=args.curv)

    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Hyperbolic Visualizations")
    print("=" * 60)

    # 1. Individual Poincare disk plots
    create_hyperbolic_visualization(
        embeddings_dict, output_dir, curv=args.curv, suffix=suffix
    )

    # 2. Comparison plot
    create_comparison_plot(
        embeddings_dict,
        output_dir / f'hyperbolic_comparison{suffix}.png',
        curv=args.curv,
    )

    # 3. Distance analysis
    create_distance_analysis(
        embeddings_dict,
        output_dir / f'hyperbolic_distance_analysis{suffix}.png',
        curv=args.curv,
    )

    print("\n" + "=" * 60)
    print("Done! Hyperbolic visualizations saved to:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
