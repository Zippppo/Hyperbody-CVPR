"""
Visualize and compare label embeddings from different text encoders.

Usage:
    conda activate pasco
    # Visualize labeled classes only (default)
    python scripts/body/text_embedding/visualize_embeddings.py

    # Visualize all labels including category nodes
    python scripts/body/text_embedding/visualize_embeddings.py --include-categories

Output:
    scripts/body/text_embedding/figures/embedding_comparison.png
    scripts/body/text_embedding/figures/embedding_tsne_{model}.png
    scripts/body/text_embedding/figures/embedding_similarity_heatmap.png
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Define anatomical categories for coloring
# Aligned with label_definitions.yaml hierarchy
CATEGORY_MAPPING = {
    # --- body_cavities ---
    'inside_body_empty': 'body_cavities',

    # --- splanchnology: digestive_system > accessory_glands ---
    'liver': 'accessory_glands',
    'pancreas': 'accessory_glands',
    'gallbladder': 'accessory_glands',

    # --- splanchnology: digestive_system > alimentary_canal ---
    'stomach': 'alimentary_canal',
    'esophagus': 'alimentary_canal',
    'small_bowel': 'alimentary_canal',
    'duodenum': 'alimentary_canal',
    'colon': 'alimentary_canal',

    # --- splanchnology: respiratory_system ---
    'lung': 'respiratory_system',
    'trachea': 'respiratory_system',

    # --- splanchnology: urinary_system ---
    'kidney_left': 'urinary_system',
    'kidney_right': 'urinary_system',
    'urinary_bladder': 'urinary_system',

    # --- splanchnology: reproductive_system ---
    'prostate': 'reproductive_system',

    # --- splanchnology: endocrine_system ---
    'thyroid_gland': 'endocrine_system',
    'adrenal_gland_left': 'endocrine_system',
    'adrenal_gland_right': 'endocrine_system',

    # --- splanchnology: lymphatic_system ---
    'spleen': 'lymphatic_system',

    # --- cardiovascular_system ---
    'heart': 'cardiovascular_system',

    # --- nervous_system: central_nervous_system ---
    'brain': 'central_nervous_system',
    'spinal_cord': 'central_nervous_system',

    # --- skeletal_system: axial_skeleton ---
    'skull': 'axial_skeleton',
    'spine': 'axial_skeleton',

    # --- skeletal_system: axial_skeleton > thoracic_cage ---
    'sternum': 'thoracic_cage',
    'costal_cartilages': 'thoracic_cage',

    # --- skeletal_system: appendicular_skeleton > upper_limb_girdle ---
    'scapula_left': 'upper_limb_girdle',
    'scapula_right': 'upper_limb_girdle',
    'clavicula_left': 'upper_limb_girdle',
    'clavicula_right': 'upper_limb_girdle',

    # --- skeletal_system: appendicular_skeleton > free_upper_limb ---
    'humerus_left': 'free_upper_limb',
    'humerus_right': 'free_upper_limb',

    # --- skeletal_system: appendicular_skeleton > pelvic_girdle ---
    'hip_left': 'pelvic_girdle',
    'hip_right': 'pelvic_girdle',

    # --- skeletal_system: appendicular_skeleton > free_lower_limb ---
    'femur_left': 'free_lower_limb',
    'femur_right': 'free_lower_limb',

    # --- muscular_system: trunk_muscles > back_muscles_deep ---
    'autochthon_left': 'back_muscles_deep',
    'autochthon_right': 'back_muscles_deep',

    # --- muscular_system: lower_limb_muscles > gluteal_region ---
    'gluteus_maximus_left': 'gluteal_region',
    'gluteus_maximus_right': 'gluteal_region',
    'gluteus_medius_left': 'gluteal_region',
    'gluteus_medius_right': 'gluteal_region',
    'gluteus_minimus_left': 'gluteal_region',
    'gluteus_minimus_right': 'gluteal_region',

    # --- muscular_system: lower_limb_muscles > iliopsoas ---
    'iliopsoas_left': 'iliopsoas',
    'iliopsoas_right': 'iliopsoas',
}

# Ribs mapping (skeletal_system > axial_skeleton > thoracic_cage > ribs)
for i in range(1, 13):
    CATEGORY_MAPPING[f'rib_left_{i}'] = 'ribs'
    CATEGORY_MAPPING[f'rib_right_{i}'] = 'ribs'

# Category node mappings (virtual nodes from label_definitions.yaml categories section)
# These are parent/group nodes without numeric IDs
CATEGORY_NODE_MAPPING = {
    # root
    'human_body': 'root',
    'outside_body': 'root',  # legacy, may still exist in embedding files

    # skeletal_system hierarchy
    'skeletal_system': 'skeletal_system_cat',
    'axial_skeleton': 'axial_skeleton',
    'thoracic_cage': 'thoracic_cage',
    'ribs': 'ribs',
    'ribs_left': 'ribs',
    'ribs_right': 'ribs',
    'appendicular_skeleton': 'appendicular_skeleton_cat',
    'upper_limb_girdle': 'upper_limb_girdle',
    'scapula': 'upper_limb_girdle',
    'clavicula': 'upper_limb_girdle',
    'free_upper_limb': 'free_upper_limb',
    'humerus': 'free_upper_limb',
    'pelvic_girdle': 'pelvic_girdle',
    'hip': 'pelvic_girdle',
    'free_lower_limb': 'free_lower_limb',
    'femur': 'free_lower_limb',

    # muscular_system hierarchy
    'muscular_system': 'muscular_system_cat',
    'trunk_muscles': 'trunk_muscles_cat',
    'back_muscles_deep': 'back_muscles_deep',
    'autochthon': 'back_muscles_deep',
    'lower_limb_muscles': 'lower_limb_muscles_cat',
    'gluteal_region': 'gluteal_region',
    'gluteus_maximus': 'gluteal_region',
    'gluteus_medius': 'gluteal_region',
    'gluteus_minimus': 'gluteal_region',
    'iliopsoas': 'iliopsoas',

    # splanchnology hierarchy
    'splanchnology': 'splanchnology_cat',
    'digestive_system': 'digestive_system_cat',
    'alimentary_canal': 'alimentary_canal',
    'small_intestine': 'alimentary_canal',
    'large_intestine': 'alimentary_canal',
    'accessory_glands': 'accessory_glands',
    'respiratory_system': 'respiratory_system',
    'urinary_system': 'urinary_system',
    'kidney': 'urinary_system',
    'reproductive_system': 'reproductive_system',
    'endocrine_system': 'endocrine_system',
    'adrenal_gland': 'endocrine_system',
    'lymphatic_system': 'lymphatic_system',

    # cardiovascular_system
    'cardiovascular_system': 'cardiovascular_system',

    # nervous_system hierarchy
    'nervous_system': 'nervous_system_cat',
    'central_nervous_system': 'central_nervous_system',

    # body_cavities
    'body_cavities': 'body_cavities',
}

# Category colors aligned with label_definitions.yaml hierarchy
CATEGORY_COLORS = {
    # root / body_cavities
    'root': '#2C3E50',                 # dark gray-blue
    'body_cavities': '#808080',        # gray

    # splanchnology: digestive_system
    'digestive_system_cat': '#E67E22', # dark orange (category)
    'accessory_glands': '#FF6B6B',     # red (liver, pancreas, gallbladder)
    'alimentary_canal': '#F39C12',     # yellow-orange (stomach, intestines)

    # splanchnology: respiratory_system
    'respiratory_system': '#FFA502',   # orange

    # splanchnology: urinary_system
    'urinary_system': '#4ECDC4',       # teal

    # splanchnology: reproductive_system
    'reproductive_system': '#45B7D1',  # light blue

    # splanchnology: endocrine_system
    'endocrine_system': '#E74C3C',     # dark red

    # splanchnology: lymphatic_system
    'lymphatic_system': '#FF9FF3',     # pink (spleen)

    # splanchnology category
    'splanchnology_cat': '#D35400',    # burnt orange

    # cardiovascular_system
    'cardiovascular_system': '#FF4757',  # bright red

    # nervous_system: central_nervous_system
    'nervous_system_cat': '#8E44AD',   # dark purple (category)
    'central_nervous_system': '#9B59B6',  # purple

    # skeletal_system: axial_skeleton
    'skeletal_system_cat': '#2980B9',  # darker blue (category)
    'axial_skeleton': '#3498DB',       # blue (skull, spine)
    'thoracic_cage': '#5DADE2',        # light blue (sternum, costal_cartilages)
    'ribs': '#85C1E9',                 # lighter blue

    # skeletal_system: appendicular_skeleton
    'appendicular_skeleton_cat': '#17A589',  # green-teal (category)
    'upper_limb_girdle': '#1ABC9C',    # green-teal (scapula, clavicula)
    'free_upper_limb': '#16A085',      # dark teal (humerus)
    'pelvic_girdle': '#27AE60',        # green (hip)
    'free_lower_limb': '#2ECC71',      # light green (femur)

    # muscular_system
    'muscular_system_cat': '#C0392B',  # dark red (category)
    'trunk_muscles_cat': '#A93226',    # darker red (category)
    'lower_limb_muscles_cat': '#922B21',  # even darker red (category)
    'gluteal_region': '#E91E63',       # pink (gluteus muscles)
    'back_muscles_deep': '#9C27B0',    # dark purple (autochthon)
    'iliopsoas': '#673AB7',            # deep purple

    # fallback for category labels
    'category': '#95A5A6',             # gray
}


def load_embeddings(embedding_path: Path) -> Dict:
    """Load embeddings from a .pt file."""
    data = torch.load(embedding_path, map_location='cpu', weights_only=False)
    return data


def get_category_for_label(label: str) -> str:
    """Get the anatomical category for a label.

    Checks both labeled class mappings and category node mappings.
    """
    # First check labeled class mapping
    if label in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[label]
    # Then check category node mapping
    if label in CATEGORY_NODE_MAPPING:
        return CATEGORY_NODE_MAPPING[label]
    # Fallback
    return 'category'


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    random_state: int = 42
) -> np.ndarray:
    """Compute t-SNE embedding."""
    # Adjust perplexity if too high for the number of samples
    n_samples = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(5, n_samples // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        max_iter=1000,
        learning_rate='auto',
        init='pca'
    )
    return tsne.fit_transform(embeddings)


def plot_single_embedding(
    embeddings_2d: np.ndarray,
    label_names: List[str],
    title: str,
    ax: plt.Axes,
    show_labels: bool = True,
    fontsize: int = 6
) -> None:
    """Plot a single t-SNE embedding."""
    categories = [get_category_for_label(name) for name in label_names]
    colors = [CATEGORY_COLORS.get(cat, '#808080') for cat in categories]

    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        s=50,
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )

    if show_labels:
        for i, name in enumerate(label_names):
            # Simplify label for display
            short_name = name.replace('_left', '_L').replace('_right', '_R')
            short_name = short_name.replace('rib_L_', 'rL').replace('rib_R_', 'rR')
            ax.annotate(
                short_name,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=fontsize,
                alpha=0.8,
                ha='center',
                va='bottom'
            )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')


def create_comparison_plot(
    embeddings_dict: Dict[str, Dict],
    output_path: Path,
    show_labels: bool = True
) -> None:
    """Create a comparison plot of all embedding types."""
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, embeddings_dict.items()):
        embeddings = data['embeddings'].numpy()
        label_names = data['label_names']

        # Compute t-SNE
        print(f"Computing t-SNE for {model_name}...")
        embeddings_2d = compute_tsne(embeddings)

        # Plot
        plot_single_embedding(
            embeddings_2d,
            label_names,
            model_name.upper(),
            ax,
            show_labels=show_labels,
            fontsize=5
        )

    # Add legend
    legend_elements = []
    unique_categories = sorted(set(CATEGORY_COLORS.keys()) - {'category'})
    for cat in unique_categories:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=CATEGORY_COLORS[cat], label=cat.replace('_', ' '))
        )

    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(1.15, 0.5),
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def create_individual_plots(
    embeddings_dict: Dict[str, Dict],
    output_dir: Path,
    suffix: str = ''
) -> None:
    """Create individual t-SNE plots for each embedding type."""
    for model_name, data in embeddings_dict.items():
        embeddings = data['embeddings'].numpy()
        label_names = data['label_names']

        fig, ax = plt.subplots(figsize=(14, 12))

        print(f"Computing t-SNE for {model_name}...")
        embeddings_2d = compute_tsne(embeddings)

        # Adjust fontsize based on number of labels
        fontsize = 5 if len(label_names) > 80 else 7

        plot_single_embedding(
            embeddings_2d,
            label_names,
            f'{model_name.upper()} Label Embeddings (t-SNE)',
            ax,
            show_labels=True,
            fontsize=fontsize
        )

        # Add legend
        legend_elements = []
        unique_categories = sorted(set(
            get_category_for_label(name) for name in label_names
        ))
        for cat in unique_categories:
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor=CATEGORY_COLORS.get(cat, '#808080'),
                      label=cat.replace('_', ' '))
            )

        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7
        )

        plt.tight_layout()
        output_path = output_dir / f'embedding_tsne_{model_name}{suffix}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved individual plot to: {output_path}")


def create_similarity_heatmap(
    embeddings_dict: Dict[str, Dict],
    output_path: Path,
    select_labels: Optional[List[str]] = None
) -> None:
    """Create cosine similarity heatmaps for each embedding type."""
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, embeddings_dict.items()):
        embeddings = data['embeddings'].numpy()
        label_names = data['label_names']

        # Select subset of labels if specified
        if select_labels is not None:
            indices = [i for i, name in enumerate(label_names) if name in select_labels]
            embeddings = embeddings[indices]
            label_names = [label_names[i] for i in indices]

        # Compute cosine similarity
        similarity = cosine_similarity(embeddings)

        # Plot heatmap
        im = ax.imshow(similarity, cmap='RdYlBu_r', vmin=-1, vmax=1)

        # Add labels if not too many
        if len(label_names) <= 30:
            short_names = [
                name.replace('_left', '_L').replace('_right', '_R')
                for name in label_names
            ]
            ax.set_xticks(range(len(label_names)))
            ax.set_yticks(range(len(label_names)))
            ax.set_xticklabels(short_names, rotation=90, fontsize=6)
            ax.set_yticklabels(short_names, fontsize=6)

        ax.set_title(f'{model_name.upper()} Cosine Similarity', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved similarity heatmap to: {output_path}")


def create_organ_similarity_heatmap(
    embeddings_dict: Dict[str, Dict],
    output_path: Path
) -> None:
    """Create similarity heatmap for main organ labels only."""
    # Select main organ labels (excluding ribs and some muscles)
    # Aligned with label_definitions.yaml structure
    main_organs = [
        # body_cavities
        'inside_body_empty',
        # splanchnology: digestive_system
        'liver', 'pancreas', 'gallbladder',
        'stomach', 'esophagus', 'small_bowel', 'duodenum', 'colon',
        # splanchnology: respiratory_system
        'lung', 'trachea',
        # splanchnology: urinary_system
        'kidney_left', 'kidney_right', 'urinary_bladder',
        # splanchnology: reproductive_system
        'prostate',
        # splanchnology: endocrine_system
        'thyroid_gland', 'adrenal_gland_left', 'adrenal_gland_right',
        # splanchnology: lymphatic_system
        'spleen',
        # cardiovascular_system
        'heart',
        # nervous_system
        'brain', 'spinal_cord',
        # skeletal_system: axial_skeleton
        'skull', 'spine', 'sternum',
        # skeletal_system: appendicular_skeleton
        'hip_left', 'hip_right', 'femur_left', 'femur_right',
    ]

    create_similarity_heatmap(embeddings_dict, output_path, select_labels=main_organs)


def print_embedding_stats(embeddings_dict: Dict[str, Dict]) -> None:
    """Print statistics about the embeddings."""
    print("\n" + "=" * 60)
    print("Embedding Statistics")
    print("=" * 60)

    for model_name, data in embeddings_dict.items():
        embeddings = data['embeddings']
        print(f"\n{model_name.upper()}:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Norm range: [{embeddings.norm(dim=-1).min():.4f}, {embeddings.norm(dim=-1).max():.4f}]")
        print(f"  Mean: {embeddings.mean():.6f}")
        print(f"  Std: {embeddings.std():.6f}")

        # Compute average cosine similarity
        sim = cosine_similarity(embeddings.numpy())
        np.fill_diagonal(sim, 0)  # Exclude self-similarity
        print(f"  Avg cosine similarity (excluding self): {sim.mean():.4f}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Visualize label embeddings from different text encoders.'
    )
    parser.add_argument(
        '--include-categories',
        action='store_true',
        help='Include category nodes (virtual parent nodes) in visualization'
    )
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = project_root / 'Dataset' / 'text_embeddings'
    output_dir = Path(__file__).resolve().parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    print(f"Include categories: {args.include_categories}")

    # Determine which embedding files to load
    if args.include_categories:
        # Load label_embeddings.pt (includes both labeled + categories)
        embedding_files = {
            'clip': dataset_dir / 'clip_label_embeddings.pt',
            'biomedclip': dataset_dir / 'biomedclip_label_embeddings.pt',
            'sat': dataset_dir / 'sat_label_embeddings.pt',
        }
        suffix = '_with_categories'
    else:
        # Load labeled_embeddings.pt (labeled classes only)
        embedding_files = {
            'clip': dataset_dir / 'clip_labeled_embeddings.pt',
            'biomedclip': dataset_dir / 'biomedclip_labeled_embeddings.pt',
            'sat': dataset_dir / 'sat_labeled_embeddings.pt',
        }
        suffix = ''

    embeddings_dict = {}
    for model_name, file_path in embedding_files.items():
        if file_path.exists():
            print(f"Loading {model_name} embeddings from: {file_path}")
            embeddings_dict[model_name] = load_embeddings(file_path)
        else:
            print(f"Warning: {file_path} not found, skipping {model_name}")

    if not embeddings_dict:
        print("No embedding files found!")
        return

    # Print statistics
    print_embedding_stats(embeddings_dict)

    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    # 1. Comparison plot (all models side by side)
    create_comparison_plot(
        embeddings_dict,
        output_dir / f'embedding_comparison{suffix}.png',
        show_labels=False  # Too crowded with labels
    )

    # 2. Individual plots with labels
    create_individual_plots(embeddings_dict, output_dir, suffix=suffix)

    # 3. Similarity heatmap for main organs (only for labeled-only mode)
    if not args.include_categories:
        create_organ_similarity_heatmap(
            embeddings_dict,
            output_dir / 'embedding_similarity_heatmap.png'
        )
    else:
        # For category mode, create full similarity heatmap
        create_similarity_heatmap(
            embeddings_dict,
            output_dir / f'embedding_similarity_heatmap{suffix}.png'
        )

    print("\n" + "=" * 60)
    print("Done! Visualizations saved to:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
