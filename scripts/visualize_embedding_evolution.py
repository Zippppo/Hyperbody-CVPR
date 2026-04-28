"""
Visualize embedding evolution across training checkpoints.

Uses MDS on geodesic distances (not PCA) for 2D projection, with Procrustes
alignment between frames for smooth animation. Extracts tangent embeddings
directly from checkpoint state_dicts without model instantiation.

Outputs:
    - Interactive HTML animation with Plotly slider and play button

Usage:
    python scripts/visualize_embedding_evolution.py \
        --checkpoint-dir checkpoints/021002 \
        --config configs/021002.yaml
"""
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.manifold import MDS, smacof

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.hyperbolic.lorentz_ops import exp_map0, pairwise_dist
from models.hyperbolic.embedding_tracker import SYSTEM_COLORS


def discover_checkpoints(checkpoint_dir: str) -> List[Tuple[str, str]]:
    """
    Find and sort checkpoint files in a directory.

    Finds epoch_*.pth and best.pth, excludes latest.pth.
    Sorts epoch files numerically, best.pth at the end.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        List of (label, path) tuples sorted by epoch number
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return []

    epoch_files = []
    best_file = None

    for f in ckpt_dir.glob("*.pth"):
        name = f.stem
        if name == "latest":
            continue
        if name == "best":
            best_file = ("best", str(f))
            continue
        match = re.match(r"epoch_(\d+)", name)
        if match:
            epoch_num = int(match.group(1))
            epoch_files.append((epoch_num, name, str(f)))

    # Sort epoch files by number
    epoch_files.sort(key=lambda x: x[0])
    result = [(name, path) for _, name, path in epoch_files]

    if best_file is not None:
        result.append(best_file)

    return result


def extract_tangent_embeddings(
    checkpoint_path: str,
) -> Tuple[torch.Tensor, Dict]:
    """
    Extract tangent vectors and metadata from a checkpoint file.

    Handles both plain and DataParallel (module. prefix) state dicts.

    Args:
        checkpoint_path: Path to .pth checkpoint file

    Returns:
        Tuple of (tangent_embeddings [N, D], metadata_dict)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Try both with and without module. prefix
    key = "label_emb.tangent_embeddings"
    key_module = "module.label_emb.tangent_embeddings"

    if key in state_dict:
        tangent = state_dict[key]
    elif key_module in state_dict:
        tangent = state_dict[key_module]
    else:
        raise KeyError(
            f"Neither '{key}' nor '{key_module}' found in state_dict. "
            f"Available keys: {[k for k in state_dict if 'label_emb' in k]}"
        )

    metadata = {
        "epoch": ckpt.get("epoch", None),
        "best_dice": ckpt.get("best_dice", None),
        "train_loss": ckpt.get("train_loss", None),
        "val_loss": ckpt.get("val_loss", None),
        "mean_dice": ckpt.get("mean_dice", None),
    }

    return tangent, metadata


def compute_geodesic_distance_matrix(
    lorentz: torch.Tensor, curv: float = 1.0
) -> np.ndarray:
    """
    Compute pairwise geodesic distance matrix on the Lorentz manifold.

    Args:
        lorentz: Points on Lorentz manifold [N, D]
        curv: Curvature parameter

    Returns:
        Distance matrix [N, N] as numpy array with zero diagonal
    """
    with torch.no_grad():
        D = pairwise_dist(lorentz, lorentz, curv=curv).cpu().numpy()
    np.fill_diagonal(D, 0.0)
    return D


def mds_project(
    dist_matrix: np.ndarray,
    init: np.ndarray = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project distance matrix to 2D using Metric MDS.

    When init is provided, warm-starts from that solution (n_init=1)
    so MDS converges to the nearest local minimum, preserving orientation.
    Without init, uses n_init=4 random initializations for best fit.

    Args:
        dist_matrix: Precomputed distance matrix [N, N]
        init: Optional initial 2D positions [N, 2] for warm-start
        random_state: Random seed for reproducibility

    Returns:
        2D projection [N, 2]
    """
    if init is not None:
        # Warm-start: use previous frame as initialization
        proj, stress = smacof(
            dist_matrix,
            n_components=2,
            metric=True,
            init=init,
            n_init=1,
            random_state=random_state,
            normalized_stress="auto",
        )
        return proj
    else:
        # Cold-start: multiple random initializations
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            normalized_stress="auto",
            n_init=4,
        )
        return mds.fit_transform(dist_matrix)


def procrustes_align(
    ref: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    Align target to reference using rotation/reflection only (no scaling).

    Uses SVD-based Procrustes: finds orthogonal matrix R that minimizes
    ||target @ R - ref||_F.

    Args:
        ref: Reference points [N, 2]
        target: Points to align [N, 2]

    Returns:
        Aligned target [N, 2]
    """
    # Center both point sets
    ref_mean = ref.mean(axis=0)
    target_mean = target.mean(axis=0)
    ref_centered = ref - ref_mean
    target_centered = target - target_mean

    # SVD of cross-covariance matrix
    M = target_centered.T @ ref_centered  # [2, 2]
    U, S, Vt = np.linalg.svd(M)

    # Optimal rotation (allowing reflection)
    R = U @ Vt  # [2, 2]

    # Apply rotation and translate to match reference centroid
    aligned = (target_centered @ R) + ref_mean

    return aligned


def normalize_to_poincare_ball(
    projections: List[np.ndarray], margin: float = 0.95
) -> List[np.ndarray]:
    """
    Normalize all projections to fit inside the unit Poincare ball.

    Uses a single global scale factor across all frames so relative
    magnitudes between epochs are preserved.

    Args:
        projections: List of [N, 2] projection arrays
        margin: Scale to this fraction of the unit ball (default 0.95)

    Returns:
        List of normalized [N, 2] arrays
    """
    # Find global maximum norm across all frames
    global_max_norm = 0.0
    for proj in projections:
        norms = np.linalg.norm(proj, axis=1)
        frame_max = norms.max()
        if frame_max > global_max_norm:
            global_max_norm = frame_max

    if global_max_norm < 1e-10:
        return [proj.copy() for proj in projections]

    scale = margin / global_max_norm
    return [proj * scale for proj in projections]


def build_animation_html(
    projections: List[np.ndarray],
    labels: List[str],
    metadata_list: List[Dict],
    class_names: List[str],
    class_to_system: Dict[int, str],
    output_path: str,
) -> None:
    """
    Build interactive Plotly HTML animation of embedding evolution.

    Args:
        projections: List of [N, 2] arrays (one per checkpoint)
        labels: List of frame labels (e.g., "epoch_10", "best")
        metadata_list: List of metadata dicts per checkpoint
        class_names: List of class names
        class_to_system: Mapping from class index to organ system name
        output_path: Path to write HTML file
    """
    import plotly.graph_objects as go

    num_classes = len(class_names)

    # Colors per class
    colors = [
        SYSTEM_COLORS.get(class_to_system.get(i, "other"), "#95A5A6")
        for i in range(num_classes)
    ]

    # Build hover text
    hover_texts = [
        f"{i}: {class_names[i]} ({class_to_system.get(i, 'other')})"
        for i in range(num_classes)
    ]

    # Build display labels with metadata
    def _frame_title(label, meta):
        parts = [label]
        if meta.get("epoch") is not None:
            parts[0] = f"Epoch {meta['epoch'] + 1}"
        if label == "best":
            parts[0] = "Best"
        if meta.get("mean_dice") is not None:
            parts.append(f"Dice={meta['mean_dice']:.4f}")
        if meta.get("train_loss") is not None:
            parts.append(f"Loss={meta['train_loss']:.4f}")
        return " | ".join(parts)

    frame_titles = [
        _frame_title(label, meta)
        for label, meta in zip(labels, metadata_list)
    ]

    # Poincare ball boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    boundary_x = np.cos(theta)
    boundary_y = np.sin(theta)

    # Create frames
    frames = []
    for i, (proj, title) in enumerate(zip(projections, frame_titles)):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=boundary_x,
                    y=boundary_y,
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                go.Scatter(
                    x=proj[:, 0],
                    y=proj[:, 1],
                    mode="markers+text",
                    marker=dict(size=8, color=colors,
                                line=dict(width=1, color="white")),
                    text=[str(j) for j in range(num_classes)],
                    textposition="top center",
                    textfont=dict(size=6),
                    hovertext=hover_texts,
                    hoverinfo="text",
                    showlegend=False,
                ),
            ],
            name=labels[i],
            layout=go.Layout(title=dict(text=title, x=0.5)),
        )
        frames.append(frame)

    # Initial figure (first frame)
    first_proj = projections[0]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=boundary_x,
                y=boundary_y,
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=first_proj[:, 0],
                y=first_proj[:, 1],
                mode="markers+text",
                marker=dict(size=8, color=colors,
                            line=dict(width=1, color="white")),
                text=[str(j) for j in range(num_classes)],
                textposition="top center",
                textfont=dict(size=6),
                hovertext=hover_texts,
                hoverinfo="text",
                showlegend=False,
            ),
        ],
        frames=frames,
    )

    # Add legend traces for organ systems
    for system, color in SYSTEM_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=system,
            showlegend=True,
        ))

    # Layout with slider and play/pause buttons
    fig.update_layout(
        title=dict(text=frame_titles[0], x=0.5),
        xaxis=dict(range=[-1.3, 1.3], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1.3, 1.3]),
        width=1000,
        height=900,
        legend=dict(title="Organ System"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 400},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=14),
                    prefix="",
                    visible=True,
                    xanchor="right",
                ),
                transition=dict(duration=400),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [label],
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 400},
                            },
                        ],
                        label=frame_titles[i],
                        method="animate",
                    )
                    for i, label in enumerate(labels)
                ],
            )
        ],
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved animation: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize embedding evolution across training checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True,
        help="Directory containing epoch_*.pth and best.pth"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML path (default: <checkpoint-dir>/embedding_evolution.html)"
    )
    return parser.parse_args()


def main():
    import json
    args = parse_args()

    from config import Config
    from data.organ_hierarchy import load_class_to_system

    # Load config
    cfg = Config.from_yaml(args.config)
    curv = cfg.hyp_curv

    # Load class names and system mapping
    with open(cfg.dataset_info_file) as f:
        info = json.load(f)
    class_names = info["class_names"]
    class_to_system = load_class_to_system(cfg.tree_file, class_names)

    # Step 1: Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    print(f"Found {len(checkpoints)} checkpoints: {[l for l, _ in checkpoints]}")

    # Step 2: Extract tangent embeddings and project via MDS (warm-started)
    projections = []
    metadata_list = []
    labels = []
    prev_proj = None

    for label, path in checkpoints:
        print(f"  Processing {label}...")
        tangent, metadata = extract_tangent_embeddings(path)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)
        proj = mds_project(D, init=prev_proj)
        projections.append(proj)
        metadata_list.append(metadata)
        labels.append(label)
        prev_proj = proj

    # Step 3: Procrustes alignment chain (each frame aligned to previous)
    # With warm-start MDS, frames are already roughly aligned.
    # Procrustes cleans up residual rotation/reflection.
    aligned = [projections[0]]
    for i in range(1, len(projections)):
        aligned_frame = procrustes_align(aligned[i - 1], projections[i])
        aligned.append(aligned_frame)

    # Step 4: Normalize to Poincare ball
    normalized = normalize_to_poincare_ball(aligned)

    # Step 5: Build animation HTML
    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.checkpoint_dir) / "embedding_evolution.html")

    build_animation_html(
        projections=normalized,
        labels=labels,
        metadata_list=metadata_list,
        class_names=class_names,
        class_to_system=class_to_system,
        output_path=output_path,
    )

    print(f"Done! Open {output_path} in a browser to view the animation.")


if __name__ == "__main__":
    main()
