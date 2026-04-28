"""
Section 5.6: Learned Embedding Analysis
========================================
Analyze what the model has learned by comparing D_graph (prior) vs D_learned (Lorentz).

Outputs (saved to docs/visualizations/embedding_analysis/):
1. spearman_summary.txt          - Global correlation + top-k discoveries
2. distance_heatmaps.html        - Side-by-side 70x70 heatmaps (D_graph vs D_learned)
3. residual_heatmap.html         - Residual rank matrix
4. poincare_projection.html      - Poincare ball with organ system colors
5. hidden_connections_table.html  - Interactive table of discovered pairs

Usage:
    python scripts/analyze_learned_embeddings.py \
        --config configs/LR-GD-M05-cosineLR.yaml \
        --checkpoint checkpoints/LR-CD-M05-consineLR/best.pth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system
from models.body_net import BodyNet
from models.hyperbolic.lorentz_ops import pairwise_dist
from models.hyperbolic.embedding_tracker import SYSTEM_COLORS


def parse_args():
    parser = argparse.ArgumentParser(description="Learned Embedding Analysis (Section 5.6)")
    parser.add_argument("--config", type=str, default="configs/LR-GD-M05-cosineLR.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/LR-CD-M05-cosineLR/best.pth")
    parser.add_argument("--output-dir", type=str, default="docs/visualizations/embedding_analysis")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top pairs to report")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model_and_embeddings(cfg, checkpoint_path, device):
    """Load trained model and extract label embeddings."""
    with open(cfg.dataset_info_file) as f:
        info = json.load(f)
    class_names = info["class_names"]
    class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

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

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    with torch.no_grad():
        label_emb_lorentz = model.label_emb()  # [num_classes, embed_dim]

    epoch = ckpt.get("epoch", "?")
    best_dice = ckpt.get("best_dice", "?")
    print(f"Loaded checkpoint: epoch={epoch}, best_dice={best_dice}")
    print(f"Label embedding shape: {label_emb_lorentz.shape}")

    return label_emb_lorentz, class_names


def compute_analysis(D_graph, D_learned, class_names, class_to_system, top_k=20):
    """Core analysis: correlation, residuals, discoveries."""
    n = len(class_names)

    # Symmetrize D_graph (contact matrix is asymmetric)
    D_graph_sym = torch.min(D_graph, D_graph.T)

    # Upper triangle mask excluding class 0 (inside_body_empty)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mask[0, :] = False
    mask[:, 0] = False

    d_graph_flat = D_graph_sym[mask].numpy()
    d_learned_flat = D_learned[mask].numpy()

    # 1. Spearman rank correlation
    rho, pval = spearmanr(d_graph_flat, d_learned_flat)

    # 2. Rank-based residual (use rankdata for proper tie handling)
    rank_graph = rankdata(d_graph_flat, method='average') - 1
    rank_learned = rankdata(d_learned_flat, method='average') - 1
    residual_flat = rank_learned - rank_graph

    # Reconstruct residual matrix
    residual_matrix = np.zeros((n, n))
    indices = np.where(mask.numpy())
    for idx in range(len(indices[0])):
        i, j = indices[0][idx], indices[1][idx]
        residual_matrix[i, j] = residual_flat[idx]
        residual_matrix[j, i] = -residual_flat[idx]  # Anti-symmetric

    # 3. Top-k discoveries
    # Hidden connections: prior=far, model=close (most negative residual)
    hidden_idx = residual_flat.argsort()[:top_k]
    hidden_pairs = []
    for idx in hidden_idx:
        i, j = indices[0][idx], indices[1][idx]
        hidden_pairs.append({
            "class_i": class_names[i],
            "class_j": class_names[j],
            "system_i": class_to_system.get(i, "other"),
            "system_j": class_to_system.get(j, "other"),
            "residual": float(residual_flat[idx]),
            "d_graph": float(D_graph_sym[i, j]),
            "d_learned": float(D_learned[i, j]),
        })

    # Over-connections: prior=close, model=far (most positive residual)
    over_idx = residual_flat.argsort()[-top_k:][::-1]
    over_pairs = []
    for idx in over_idx:
        i, j = indices[0][idx], indices[1][idx]
        over_pairs.append({
            "class_i": class_names[i],
            "class_j": class_names[j],
            "system_i": class_to_system.get(i, "other"),
            "system_j": class_to_system.get(j, "other"),
            "residual": float(residual_flat[idx]),
            "d_graph": float(D_graph_sym[i, j]),
            "d_learned": float(D_learned[i, j]),
        })

    return {
        "rho": rho,
        "pval": pval,
        "residual_matrix": residual_matrix,
        "D_graph_sym": D_graph_sym.numpy(),
        "D_learned": D_learned.numpy(),
        "hidden_pairs": hidden_pairs,
        "over_pairs": over_pairs,
    }


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def make_heatmap_comparison(D_graph, D_learned, class_names, output_path):
    """Side-by-side 70x70 distance matrix heatmaps."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["D_graph (Prior)", "D_learned (Lorentz)"],
        horizontal_spacing=0.08,
    )

    # Shared colorscale range
    vmax = max(D_graph.max(), D_learned.max())

    hover_template = "row: %{y}<br>col: %{x}<br>distance: %{z:.2f}<extra></extra>"

    fig.add_trace(go.Heatmap(
        z=D_graph,
        x=class_names, y=class_names,
        colorscale="Viridis",
        zmin=0, zmax=vmax,
        hovertemplate=hover_template,
        colorbar=dict(x=0.45, len=0.8),
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=D_learned,
        x=class_names, y=class_names,
        colorscale="Viridis",
        zmin=0, zmax=vmax,
        hovertemplate=hover_template,
        colorbar=dict(x=1.02, len=0.8),
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Distance Matrix Comparison: D_graph vs D_learned", x=0.5),
        width=1800, height=900,
        font=dict(size=8),
    )
    fig.update_xaxes(tickangle=90, tickfont=dict(size=6))
    fig.update_yaxes(tickfont=dict(size=6))

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def make_residual_heatmap(residual_matrix, class_names, output_path):
    """Residual rank matrix heatmap."""
    import plotly.graph_objects as go

    abs_max = np.abs(residual_matrix).max()

    fig = go.Figure(go.Heatmap(
        z=residual_matrix,
        x=class_names, y=class_names,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-abs_max, zmax=abs_max,
        hovertemplate="row: %{y}<br>col: %{x}<br>residual: %{z:.0f}<extra></extra>",
        colorbar=dict(title="rank(D_learned) - rank(D_graph)"),
    ))

    fig.update_layout(
        title=dict(
            text="Residual Matrix: rank(D_learned) - rank(D_graph)<br>"
                 "<sub>Blue = model says closer than prior | Red = model says farther than prior</sub>",
            x=0.5,
        ),
        width=1000, height=900,
        font=dict(size=8),
    )
    fig.update_xaxes(tickangle=90, tickfont=dict(size=6))
    fig.update_yaxes(tickfont=dict(size=6))

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def make_poincare_projection(label_emb_lorentz, class_names, class_to_system, curv, output_path):
    """Poincare ball projection with organ system colors.

    Uses MDS on geodesic distances instead of PCA, since PCA (linear, Euclidean)
    distorts the hyperbolic distance structure. MDS preserves pairwise geodesic
    distances much better when projecting to 2D.
    """
    import plotly.graph_objects as go
    from sklearn.manifold import MDS

    # Compute geodesic distance matrix for MDS
    D_geo = pairwise_dist(label_emb_lorentz, label_emb_lorentz, curv=curv).cpu().numpy()
    np.fill_diagonal(D_geo, 0.0)

    # MDS on geodesic distances -> 2D
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=42, normalized_stress='auto', n_init=4)
    proj_2d = mds.fit_transform(D_geo)

    # Normalize to fit inside unit circle for Poincare ball visualization
    max_norm = np.linalg.norm(proj_2d, axis=1).max()
    if max_norm > 0:
        proj_2d = proj_2d / max_norm * 0.95  # scale to 95% of unit ball

    fig = go.Figure()

    # Poincare ball boundary
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines",
        line=dict(color="black", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # Plot by system for legend grouping
    systems = sorted(set(class_to_system.values()))
    for system in systems:
        idxs = [i for i in range(len(class_names)) if class_to_system.get(i, "other") == system and i > 0]
        if not idxs:
            continue
        color = SYSTEM_COLORS.get(system, "#95A5A6")
        fig.add_trace(go.Scatter(
            x=proj_2d[idxs, 0],
            y=proj_2d[idxs, 1],
            mode="markers+text",
            marker=dict(size=10, color=color, line=dict(width=1, color="white")),
            text=[class_names[i] for i in idxs],
            textposition="top center",
            textfont=dict(size=7),
            name=system,
            hovertext=[f"{i}: {class_names[i]}" for i in idxs],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(
            text=f"Poincare Ball Projection (MDS on geodesic distances, "
                 f"stress={mds.stress_:.4f})",
            x=0.5,
        ),
        xaxis=dict(range=[-1.3, 1.3], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1.3, 1.3]),
        width=1100, height=1000,
        legend=dict(title="Organ System"),
    )

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def make_discovery_table(hidden_pairs, over_pairs, output_path):
    """Interactive HTML table of discovered pairs."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Hidden Connections (prior=far, model=close)",
            "Over-Connections (prior=close, model=far)",
        ],
        specs=[[{"type": "table"}], [{"type": "table"}]],
        vertical_spacing=0.08,
    )

    def _make_table(pairs):
        return go.Table(
            header=dict(
                values=["Rank", "Class A", "System A", "Class B", "System B",
                        "D_graph", "D_learned", "Residual"],
                fill_color="paleturquoise",
                align="left",
                font=dict(size=11),
            ),
            cells=dict(
                values=[
                    list(range(1, len(pairs) + 1)),
                    [p["class_i"] for p in pairs],
                    [p["system_i"] for p in pairs],
                    [p["class_j"] for p in pairs],
                    [p["system_j"] for p in pairs],
                    [f"{p['d_graph']:.2f}" for p in pairs],
                    [f"{p['d_learned']:.2f}" for p in pairs],
                    [f"{p['residual']:.0f}" for p in pairs],
                ],
                fill_color="lavender",
                align="left",
                font=dict(size=10),
            ),
        )

    fig.add_trace(_make_table(hidden_pairs), row=1, col=1)
    fig.add_trace(_make_table(over_pairs), row=2, col=1)

    fig.update_layout(
        title=dict(text="Embedding Analysis: Discovered Organ Pairs", x=0.5),
        height=900, width=1200,
    )

    fig.write_html(str(output_path))
    print(f"Saved: {output_path}")


def write_summary(results, class_names, output_path):
    """Write text summary."""
    lines = [
        "=" * 70,
        "Section 5.6: Learned Embedding Analysis",
        "=" * 70,
        "",
        f"Spearman rank correlation (D_graph vs D_learned):",
        f"  rho = {results['rho']:.4f}",
        f"  p-value = {results['pval']:.2e}",
        "",
        "-" * 70,
        "Hidden Connections (prior says far, model says close):",
        "-" * 70,
    ]
    for i, p in enumerate(results["hidden_pairs"], 1):
        lines.append(
            f"  {i:2d}. {p['class_i']:30s} <-> {p['class_j']:30s} "
            f"[{p['system_i']:15s} | {p['system_j']:15s}] "
            f"D_graph={p['d_graph']:6.2f}  D_learned={p['d_learned']:6.2f}  "
            f"residual={p['residual']:+.0f}"
        )

    lines += [
        "",
        "-" * 70,
        "Over-Connections (prior says close, model says far):",
        "-" * 70,
    ]
    for i, p in enumerate(results["over_pairs"], 1):
        lines.append(
            f"  {i:2d}. {p['class_i']:30s} <-> {p['class_j']:30s} "
            f"[{p['system_i']:15s} | {p['system_j']:15s}] "
            f"D_graph={p['d_graph']:6.2f}  D_learned={p['d_learned']:6.2f}  "
            f"residual={p['residual']:+.0f}"
        )

    text = "\n".join(lines)
    output_path.write_text(text)
    print(f"Saved: {output_path}")
    print()
    print(text)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    cfg = Config.from_yaml(args.config)
    label_emb_lorentz, class_names = load_model_and_embeddings(
        cfg, args.checkpoint, args.device
    )

    # Load prior graph distance
    D_graph = torch.load(cfg.graph_distance_matrix, map_location="cpu")

    # Compute learned pairwise distance
    with torch.no_grad():
        D_learned = pairwise_dist(label_emb_lorentz, label_emb_lorentz, curv=cfg.hyp_curv)
        # Zero out diagonal (numerical artifact)
        D_learned.fill_diagonal_(0.0)

    # Load class-to-system mapping
    class_to_system = load_class_to_system(cfg.tree_file, class_names)

    # Core analysis
    results = compute_analysis(D_graph, D_learned, class_names, class_to_system, top_k=args.top_k)

    # Generate outputs
    write_summary(results, class_names, output_dir / "spearman_summary.txt")

    make_heatmap_comparison(
        results["D_graph_sym"], results["D_learned"],
        class_names, output_dir / "distance_heatmaps.html",
    )

    make_residual_heatmap(
        results["residual_matrix"], class_names,
        output_dir / "residual_heatmap.html",
    )

    make_poincare_projection(
        label_emb_lorentz, class_names, class_to_system,
        cfg.hyp_curv, output_dir / "poincare_projection.html",
    )

    make_discovery_table(
        results["hidden_pairs"], results["over_pairs"],
        output_dir / "discovery_table.html",
    )

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
