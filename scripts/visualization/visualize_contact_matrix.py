"""Visualize a contact matrix as an interactive heatmap."""

import argparse
import json
from pathlib import Path

import torch
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=str, default="Dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/visualization")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else PROJECT_ROOT / path_obj


def main() -> None:
    args = parse_args()
    dataset_dir = _resolve(args.dataset_dir)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contact = torch.load(dataset_dir / "contact_matrix.pt", map_location="cpu").numpy()
    with open(dataset_dir / "dataset_info.json", encoding="utf-8") as f:
        info = json.load(f)

    class_names = info["class_names"]
    n = contact.shape[0]
    labels = [f"{i}: {class_names[i]}" for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=contact,
        x=labels,
        y=labels,
        colorscale="Hot_r",
        colorbar=dict(title="Contact Ratio"),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Contact: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Contact Matrix ({n} x {n})",
        xaxis=dict(title="Class", tickangle=45, tickfont=dict(size=7)),
        yaxis=dict(title="Class", tickfont=dict(size=7), autorange="reversed"),
        width=1200,
        height=1050,
    )

    output_path = output_dir / f"{dataset_dir.name}_contact_matrix_heatmap.html"
    fig.write_html(str(output_path))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
