"""
Visual comparison test for random vs semantic direction initialization.

Generates side-by-side Poincare disk visualizations to compare:
1. Random direction initialization
2. Semantic direction initialization (from BioLORD text embeddings)

Output: docs/visualizations/direction_mode_comparison/
"""
import json
import os
import sys

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.hyperbolic.label_embedding import LorentzLabelEmbedding
from models.hyperbolic.lorentz_ops import lorentz_to_poincare
from models.hyperbolic.embedding_tracker import SYSTEM_COLORS
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system


def run_comparison_test():
    """
    Compare random vs semantic initialization on Poincare disk.

    Outputs:
    - docs/visualizations/direction_mode_comparison/
        - comparison.html (interactive side-by-side plot)
        - random_init.html (random mode only)
        - semantic_init.html (semantic mode only)
        - similarity_analysis.txt (cosine similarity analysis)
    """
    print("=" * 60)
    print("Direction Mode Comparison Test")
    print("=" * 60)

    # Setup output directory
    output_dir = "docs/visualizations/direction_mode_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Load class names and hierarchy
    with open("Dataset/dataset_info.json") as f:
        class_names = json.load(f)["class_names"]

    class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)
    class_to_system = load_class_to_system("Dataset/tree.json", class_names)
    text_embedding_path = "Dataset/text_embeddings/sat_label_embeddings.pt"

    print(f"\n[1] Loaded {len(class_names)} classes")

    # Create random initialization
    print("\n[2] Creating random direction embedding...")
    torch.manual_seed(42)
    random_emb = LorentzLabelEmbedding(
        num_classes=70,
        embed_dim=32,
        class_depths=class_depths,
        min_radius=0.1,
        max_radius=2.0,
        direction_mode="random",
    )

    # Create semantic initialization
    print("[3] Creating semantic direction embedding...")
    semantic_emb = LorentzLabelEmbedding(
        num_classes=70,
        embed_dim=32,
        class_depths=class_depths,
        min_radius=0.1,
        max_radius=2.0,
        direction_mode="semantic",
        text_embedding_path=text_embedding_path,
    )

    # Get Lorentz embeddings and project to Poincare
    print("\n[4] Projecting to Poincare disk...")
    with torch.no_grad():
        random_lorentz = random_emb()
        semantic_lorentz = semantic_emb()

        random_poincare = lorentz_to_poincare(random_lorentz).numpy()
        semantic_poincare = lorentz_to_poincare(semantic_lorentz).numpy()

    print(f"    Random Poincare shape: {random_poincare.shape}")
    print(f"    Semantic Poincare shape: {semantic_poincare.shape}")

    # Analyze clustering by organ system
    print("\n[5] Analyzing clustering by organ system...")
    analysis_lines = analyze_clustering(
        random_poincare, semantic_poincare,
        class_names, class_to_system
    )

    # Save analysis
    analysis_path = os.path.join(output_dir, "similarity_analysis.txt")
    with open(analysis_path, "w") as f:
        f.write("\n".join(analysis_lines))
    print(f"    Saved analysis to: {analysis_path}")

    # Create visualizations
    print("\n[6] Creating visualizations...")

    # Side-by-side comparison
    fig_comparison = create_comparison_plot(
        random_poincare, semantic_poincare,
        class_names, class_to_system
    )
    comparison_path = os.path.join(output_dir, "comparison.html")
    fig_comparison.write_html(comparison_path)
    print(f"    Saved comparison to: {comparison_path}")

    # Individual plots
    fig_random = create_single_plot(
        random_poincare, class_names, class_to_system,
        title="Random Direction Initialization"
    )
    random_path = os.path.join(output_dir, "random_init.html")
    fig_random.write_html(random_path)
    print(f"    Saved random plot to: {random_path}")

    fig_semantic = create_single_plot(
        semantic_poincare, class_names, class_to_system,
        title="Semantic Direction Initialization (BioLORD)"
    )
    semantic_path = os.path.join(output_dir, "semantic_init.html")
    fig_semantic.write_html(semantic_path)
    print(f"    Saved semantic plot to: {semantic_path}")

    print("\n" + "=" * 60)
    print("Comparison test completed!")
    print(f"Open {comparison_path} in a browser to view the comparison.")
    print("=" * 60)

    return True


def analyze_clustering(random_pos, semantic_pos, class_names, class_to_system):
    """Analyze intra-system clustering for both modes."""
    lines = [
        "=" * 60,
        "Direction Mode Clustering Analysis",
        "=" * 60,
        "",
        "Intra-system average pairwise distance (lower = tighter clustering):",
        "-" * 60,
        f"{'System':<20} {'Random':>12} {'Semantic':>12} {'Improvement':>12}",
        "-" * 60,
    ]

    # Group classes by system
    system_to_indices = {}
    for idx, system in class_to_system.items():
        if system not in system_to_indices:
            system_to_indices[system] = []
        system_to_indices[system].append(idx)

    improvements = []

    for system, indices in sorted(system_to_indices.items()):
        if len(indices) < 2:
            continue

        # Compute average pairwise distance within system
        random_dist = compute_avg_pairwise_dist(random_pos[indices])
        semantic_dist = compute_avg_pairwise_dist(semantic_pos[indices])

        improvement = (random_dist - semantic_dist) / random_dist * 100 if random_dist > 0 else 0
        improvements.append(improvement)

        lines.append(f"{system:<20} {random_dist:>12.4f} {semantic_dist:>12.4f} {improvement:>+11.1f}%")

    lines.extend([
        "-" * 60,
        f"{'Average improvement:':<20} {'':<12} {'':<12} {np.mean(improvements):>+11.1f}%",
        "",
        "Positive improvement means semantic mode has tighter clustering.",
        "",
    ])

    # Analyze specific organ pairs
    lines.extend([
        "=" * 60,
        "Cosine Similarity Analysis (selected organ pairs)",
        "=" * 60,
        "",
    ])

    pairs = [
        ("kidney_left", "kidney_right"),
        ("lung_left", "lung_right"),
        ("rib_left_1", "rib_left_2"),
        ("liver", "spleen"),
        ("brain", "spinal_cord"),
    ]

    lines.append(f"{'Pair':<35} {'Random':>10} {'Semantic':>10}")
    lines.append("-" * 60)

    for name1, name2 in pairs:
        if name1 in class_names and name2 in class_names:
            idx1 = class_names.index(name1)
            idx2 = class_names.index(name2)

            random_sim = cosine_similarity(random_pos[idx1], random_pos[idx2])
            semantic_sim = cosine_similarity(semantic_pos[idx1], semantic_pos[idx2])

            lines.append(f"{name1} - {name2:<20} {random_sim:>10.4f} {semantic_sim:>10.4f}")

    return lines


def compute_avg_pairwise_dist(positions):
    """Compute average pairwise Euclidean distance."""
    n = len(positions)
    if n < 2:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.linalg.norm(positions[i] - positions[j])
            count += 1

    return total / count if count > 0 else 0.0


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def create_comparison_plot(random_pos, semantic_pos, class_names, class_to_system):
    """Create side-by-side comparison plot."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Random Direction", "Semantic Direction (BioLORD)"),
        horizontal_spacing=0.1
    )

    # Group by system for coloring
    system_to_indices = {}
    for idx, system in class_to_system.items():
        if system not in system_to_indices:
            system_to_indices[system] = []
        system_to_indices[system].append(idx)

    # Add traces for each system
    for system, indices in system_to_indices.items():
        color = SYSTEM_COLORS.get(system, "#808080")

        # Random plot
        fig.add_trace(
            go.Scatter(
                x=random_pos[indices, 0],
                y=random_pos[indices, 1],
                mode='markers',
                marker=dict(size=8, color=color),
                name=system,
                text=[class_names[i] for i in indices],
                hovertemplate='%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                legendgroup=system,
                showlegend=True,
            ),
            row=1, col=1
        )

        # Semantic plot
        fig.add_trace(
            go.Scatter(
                x=semantic_pos[indices, 0],
                y=semantic_pos[indices, 1],
                mode='markers',
                marker=dict(size=8, color=color),
                name=system,
                text=[class_names[i] for i in indices],
                hovertemplate='%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
                legendgroup=system,
                showlegend=False,
            ),
            row=1, col=2
        )

    # Add unit circles
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=circle_x, y=circle_y,
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=col
        )

    fig.update_layout(
        title="Random vs Semantic Direction Initialization on Poincaré Disk",
        height=600,
        width=1200,
    )

    # Set equal aspect ratio
    fig.update_xaxes(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-1.1, 1.1])

    return fig


def create_single_plot(positions, class_names, class_to_system, title):
    """Create a single Poincare disk plot."""
    fig = go.Figure()

    # Group by system
    system_to_indices = {}
    for idx, system in class_to_system.items():
        if system not in system_to_indices:
            system_to_indices[system] = []
        system_to_indices[system].append(idx)

    # Add traces for each system
    for system, indices in system_to_indices.items():
        color = SYSTEM_COLORS.get(system, "#808080")

        fig.add_trace(
            go.Scatter(
                x=positions[indices, 0],
                y=positions[indices, 1],
                mode='markers',
                marker=dict(size=10, color=color),
                name=system,
                text=[class_names[i] for i in indices],
                hovertemplate='%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
            )
        )

    # Add unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines',
            line=dict(color='black', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        )
    )

    fig.update_layout(
        title=title,
        height=700,
        width=700,
        xaxis=dict(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1.1, 1.1]),
    )

    return fig


if __name__ == "__main__":
    success = run_comparison_test()
    sys.exit(0 if success else 1)
