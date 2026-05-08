"""Tune graph distance hyperparameters and visualize results interactively.

Usage:
    python scripts/tune_graph_distance.py --lambda 0.5 --epsilon 0.005
    python scripts/tune_graph_distance.py --lambda 0.5 --epsilon 0.005 --save
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.organ_hierarchy import compute_tree_distance_matrix, load_class_to_system
from data.spatial_adjacency import (
    compute_graph_distance_matrix,
    infer_ignored_spatial_class_indices,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "Dataset"
CONTACT_MATRIX_PATH = DATASET_DIR / "contact_matrix.pt"
TREE_PATH = DATASET_DIR / "tree.json"
DATASET_INFO_PATH = DATASET_DIR / "dataset_info.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune graph distance hyperparameters and visualize results"
    )
    parser.add_argument(
        "--lambda", dest="lambda_", type=float, default=1.0,
        help="Scale factor for spatial distance: D_spatial = lambda / (C + eps)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Epsilon for numerical stability in D_spatial = lambda / (C + eps)",
    )
    parser.add_argument(
        "--contact-matrix", type=str, default=str(CONTACT_MATRIX_PATH),
        help="Path to contact_matrix.pt",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save graph_distance_matrix.pt to Dataset/ (overwrite existing)",
    )
    return parser.parse_args()


def load_data(contact_matrix_path: str):
    """Load contact matrix, tree distance, class names, and metadata."""
    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)["class_names"]

    contact_matrix = torch.load(contact_matrix_path, map_location="cpu").float()
    D_tree = compute_tree_distance_matrix(str(TREE_PATH), class_names).float()
    class_to_system = load_class_to_system(str(TREE_PATH), class_names)
    ignored_indices = infer_ignored_spatial_class_indices(class_names)

    return class_names, contact_matrix, D_tree, class_to_system, ignored_indices


def print_contact_stats(contact_matrix: torch.Tensor):
    """Print contact matrix statistics to help choose hyperparameters."""
    # Symmetrize for analysis
    contact_sym = torch.max(contact_matrix, contact_matrix.T)
    flat = contact_sym.flatten()
    nonzero = flat[flat > 0]

    print("=" * 60)
    print("CONTACT MATRIX STATISTICS (symmetrized)")
    print("=" * 60)
    print(f"  Shape:          {tuple(contact_matrix.shape)}")
    print(f"  Non-zero:       {nonzero.numel()} / {flat.numel()}")
    print(f"  Max:            {flat.max().item():.6f}")
    if nonzero.numel() > 0:
        print(f"  Mean (non-zero):{nonzero.mean().item():.6f}")
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(nonzero.numpy(), p)
            print(f"  P{p:02d}:            {val:.6f}")
    print()


def print_graph_distance_stats(
    D_graph: torch.Tensor,
    D_tree: torch.Tensor,
    lambda_: float,
    epsilon: float,
):
    """Print graph distance statistics and shortcut summary."""
    mask = D_tree > 0
    shortened = int(((D_graph < D_tree) & mask).sum().item())
    total_pairs = int(mask.sum().item())
    vals = D_graph[mask]

    print("=" * 60)
    print(f"GRAPH DISTANCE  (lambda={lambda_}, epsilon={epsilon})")
    print("=" * 60)
    print(f"  Shortened pairs: {shortened} / {total_pairs} "
          f"({100 * shortened / total_pairs:.1f}%)")
    print(f"  Mean:            {vals.mean().item():.4f}")
    print(f"  Std:             {vals.std().item():.4f}")
    print(f"  Min (non-zero):  {vals[vals > 0].min().item():.4f}")
    print(f"  Max:             {vals.max().item():.4f}")
    print()


def print_top_shortcuts(
    D_graph: torch.Tensor,
    D_tree: torch.Tensor,
    contact_matrix: torch.Tensor,
    class_names: list,
    class_to_system: dict,
    top_k: int = 20,
):
    """Print the top shortcut pairs sorted by magnitude."""
    contact_sym = torch.max(contact_matrix, contact_matrix.T)
    diff = D_tree - D_graph
    n = D_tree.shape[0]

    shortcuts = []
    for i in range(n):
        for j in range(i + 1, n):
            if diff[i, j] > 0:
                shortcuts.append((
                    i, j,
                    float(diff[i, j]),
                    float(D_tree[i, j]),
                    float(D_graph[i, j]),
                    float(contact_sym[i, j]),
                ))
    shortcuts.sort(key=lambda x: -x[2])

    print(f"Top {top_k} shortcuts:")
    header = (f"  {'Class i':<25} {'Class j':<25} {'Sys_i':<12} {'Sys_j':<12} "
              f"{'D_tree':>6} {'D_graph':>7} {'Cut':>5} {'Contact':>8}")
    print(header)
    print(f"  {'-' * len(header)}")
    for i, j, cut, dt, dg, c in shortcuts[:top_k]:
        sys_i = class_to_system.get(i, "?")
        sys_j = class_to_system.get(j, "?")
        print(f"  {class_names[i]:<25} {class_names[j]:<25} "
              f"{sys_i:<12} {sys_j:<12} "
              f"{dt:>6.1f} {dg:>7.2f} {cut:>5.2f} {c:>8.4f}")
    print()


def generate_heatmap(
    D_graph: torch.Tensor,
    D_tree: torch.Tensor,
    class_names: list,
    lambda_: float,
    epsilon: float,
):
    """Generate interactive HTML heatmap with graph distance and diff view."""
    n = D_graph.shape[0]
    labels = [f"{i}: {class_names[i]}" for i in range(n)]
    diff = (D_tree - D_graph).numpy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Graph Distance (λ={lambda_}, ε={epsilon})",
            "D_tree − D_graph (shortcuts)",
        ],
        horizontal_spacing=0.08,
    )

    # Graph distance heatmap
    fig.add_trace(
        go.Heatmap(
            z=D_graph.numpy(),
            x=labels, y=labels,
            colorscale="Viridis",
            colorbar=dict(title="Distance", x=0.45),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Distance: %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Diff heatmap (shortcuts)
    fig.add_trace(
        go.Heatmap(
            z=diff,
            x=labels, y=labels,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Shortcut", x=1.0),
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Shortcut: %{z:.3f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title=f"Graph Distance Tuning  (λ={lambda_}, ε={epsilon})",
        width=2200, height=1000,
    )
    for i in [1, 2]:
        fig.update_xaxes(tickangle=45, tickfont=dict(size=6), row=1, col=i)
        fig.update_yaxes(tickfont=dict(size=6), autorange="reversed", row=1, col=i)

    output_path = OUTPUT_DIR / "graph_distance_heatmap.html"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved heatmap to {output_path}")


def main():
    args = parse_args()

    print(f"Loading data from {args.contact_matrix} ...")
    class_names, contact_matrix, D_tree, class_to_system, ignored_indices = load_data(
        args.contact_matrix
    )
    print(f"Classes: {len(class_names)}\n")

    # --- Contact matrix stats ---
    print_contact_stats(contact_matrix)

    # --- Compute graph distance ---
    D_graph = compute_graph_distance_matrix(
        D_tree=D_tree,
        contact_matrix=contact_matrix,
        lambda_=args.lambda_,
        epsilon=args.epsilon,
        ignored_class_indices=ignored_indices,
    )

    # --- Graph distance stats ---
    print_graph_distance_stats(D_graph, D_tree, args.lambda_, args.epsilon)

    # --- Top shortcuts ---
    print_top_shortcuts(D_graph, D_tree, contact_matrix, class_names, class_to_system)

    # --- Visualization ---
    generate_heatmap(D_graph, D_tree, class_names, args.lambda_, args.epsilon)

    # --- Optionally save ---
    if args.save:
        save_path = DATASET_DIR / "graph_distance_matrix.pt"
        torch.save(D_graph.float(), save_path)
        print(f"Saved graph_distance_matrix.pt to {save_path}")
    else:
        print("(Use --save to write graph_distance_matrix.pt to Dataset/)")


if __name__ == "__main__":
    main()
