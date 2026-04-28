"""
Interactive t-SNE visualization from saved data.

Loads pre-computed t-SNE coordinates and generates a Plotly HTML
where hovering over any point shows its class name.

Usage:
    python scripts/vis_tsne_interactive.py
    python scripts/vis_tsne_interactive.py --data _VIS/tsne/tsne_embeddings_data.npz
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go

from data.organ_hierarchy import load_class_to_system

SYSTEM_COLORS = {
    "skeletal": "#1f77b4",
    "muscular": "#d62728",
    "digestive": "#2ca02c",
    "respiratory": "#ff7f0e",
    "urinary": "#9467bd",
    "cardiovascular": "#e377c2",
    "nervous": "#8c564b",
    "other": "#7f7f7f",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="_VIS/tsne/tsne_embeddings_data.npz")
    p.add_argument("--tree", default="Dataset/tree.json")
    p.add_argument("--output", default="_VIS/tsne/tsne_interactive.html")
    args = p.parse_args()

    d = np.load(args.data, allow_pickle=True)
    coords_2d = d["coords_2d"]
    labels = d["labels"]
    class_names = list(d["class_names"])

    class_to_system = load_class_to_system(args.tree, class_names)

    fig = go.Figure()

    for system in sorted(SYSTEM_COLORS.keys()):
        sys_classes = {c for c, s in class_to_system.items() if s == system and c > 0}
        mask = np.isin(labels, list(sys_classes))
        if not mask.any():
            continue
        hover = [class_names[l] for l in labels[mask]]
        fig.add_trace(go.Scattergl(
            x=coords_2d[mask, 0], y=coords_2d[mask, 1],
            mode="markers",
            marker=dict(size=3, color=SYSTEM_COLORS[system], opacity=0.6),
            name=system,
            text=hover,
            hoverinfo="text+name",
        ))

    fig.update_layout(
        title="t-SNE Voxel Embeddings (hover to inspect)",
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
        width=1200, height=900,
        legend=dict(itemsizing="constant"),
        template="plotly_white",
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.write_html(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
