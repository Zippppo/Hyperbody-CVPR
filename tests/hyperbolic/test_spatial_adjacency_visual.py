"""Visual validation of spatial adjacency graph."""

import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import torch
from plotly.subplots import make_subplots
from torch.utils.data import Subset

OUT_DIR = Path("docs/visualizations/spatial_adjacency")


@pytest.fixture(scope="module")
def class_names() -> list[str]:
    info_path = Path("Dataset/dataset_info.json")
    if not info_path.exists():
        pytest.skip("Dataset/dataset_info.json not found")
    with info_path.open() as f:
        return json.load(f)["class_names"]


@pytest.fixture(scope="module")
def tree_path() -> str:
    path = Path("Dataset/tree.json")
    if not path.exists():
        pytest.skip("Dataset/tree.json not found")
    return str(path)


@pytest.fixture(scope="module")
def contact_and_trees(class_names, tree_path):
    """Compute contact matrix from a small subset of training data."""
    split_path = Path("Dataset/dataset_split.json")
    voxel_dir = Path("Dataset/voxel_data")
    if not split_path.exists() or not voxel_dir.exists():
        pytest.skip("Dataset split or voxel directory is missing")

    from data.dataset import HyperBodyDataset
    from data.organ_hierarchy import compute_tree_distance_matrix
    from data.spatial_adjacency import (
        compute_contact_matrix_from_dataset,
        compute_graph_distance_matrix,
        infer_ignored_spatial_class_indices,
    )

    dataset = HyperBodyDataset(
        data_dir=str(voxel_dir),
        split_file=str(split_path),
        split="train",
        volume_size=(144, 128, 268),
    )

    sample_limit = int(os.environ.get("SPATIAL_ADJ_VIZ_SAMPLES", "50"))
    subset = Subset(dataset, range(min(sample_limit, len(dataset))))

    num_classes = len(class_names)
    class_batch_size = int(os.environ.get("SPATIAL_ADJ_CLASS_BATCH", "0"))
    ignored_class_indices = infer_ignored_spatial_class_indices(class_names)
    contact = compute_contact_matrix_from_dataset(
        subset,
        num_classes=num_classes,
        dilation_radius=3,
        class_batch_size=class_batch_size,
        ignored_class_indices=ignored_class_indices,
    )
    D_tree = compute_tree_distance_matrix(tree_path, class_names)
    D_final = compute_graph_distance_matrix(
        D_tree,
        contact,
        lambda_=1.0,
        epsilon=0.01,
        ignored_class_indices=ignored_class_indices,
    )

    return contact, D_tree, D_final


class TestVisualization1ContactHeatmap:
    def test_generate_contact_matrix_heatmap(self, contact_and_trees, class_names):
        """Generate interactive heatmap of asymmetric contact matrix."""
        contact, _, _ = contact_and_trees
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        matrix = contact.cpu().numpy()
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=class_names,
                y=class_names,
                colorscale="Hot",
                reversescale=True,
                text=np.round(matrix, 3),
                texttemplate="%{text}",
                hovertemplate="From: %{y}<br>To: %{x}<br>Contact: %{z:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Asymmetric Contact Matrix: Contact(row→col)",
            xaxis_title="Target Organ (v)",
            yaxis_title="Source Organ (u) [dilated]",
            width=1400,
            height=1200,
        )

        out_path = OUT_DIR / "contact_matrix_heatmap.html"
        fig.write_html(str(out_path))
        assert out_path.exists()


class TestVisualization2DistanceDiff:
    def test_generate_distance_diff_heatmap(self, contact_and_trees, class_names):
        """Generate D_diff = D_tree - D_final showing spatial shortcuts."""
        _, D_tree, D_final = contact_and_trees
        D_diff = (D_tree - D_final).cpu().numpy()
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        fig = go.Figure(
            data=go.Heatmap(
                z=D_diff,
                x=class_names,
                y=class_names,
                colorscale="RdBu",
                zmid=0,
                text=np.round(D_diff, 2),
                texttemplate="%{text}",
                hovertemplate="From: %{y}<br>To: %{x}<br>Shortened by: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Distance Difference: D_tree - D_final (positive = shortened by spatial edge)",
            xaxis_title="Target Organ (v)",
            yaxis_title="Source Organ (u)",
            width=1400,
            height=1200,
        )

        out_path = OUT_DIR / "distance_diff_heatmap.html"
        fig.write_html(str(out_path))
        assert out_path.exists()


class TestVisualization3SamplingShift:
    def test_generate_sampling_shift_plot(self, contact_and_trees, class_names):
        """Compare normalized sampling probabilities between tree and graph distances."""
        _, D_tree, D_final = contact_and_trees
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        preferred = ["rib_left_1", "liver", "spine", "gallbladder"]
        anchors = [name for name in preferred if name in class_names]
        for name in class_names:
            if len(anchors) >= 4:
                break
            if name not in anchors:
                anchors.append(name)

        temperature = 0.5
        fig = make_subplots(
            rows=len(anchors),
            cols=1,
            subplot_titles=[f"Anchor: {name}" for name in anchors],
            vertical_spacing=0.06,
        )

        for row_idx, anchor_name in enumerate(anchors):
            u = class_names.index(anchor_name)

            w_tree = torch.exp(-D_tree[u] / temperature)
            w_tree[u] = 0
            p_tree = w_tree / w_tree.sum().clamp(min=1e-12)

            w_graph = torch.exp(-D_final[u] / temperature)
            w_graph[u] = 0
            p_graph = w_graph / w_graph.sum().clamp(min=1e-12)

            shift = p_graph - p_tree
            sorted_idx = torch.argsort(shift, descending=True)

            top_k = 15
            top_idx = sorted_idx[:top_k]
            names_top = [class_names[i] for i in top_idx.tolist()]

            fig.add_trace(
                go.Bar(
                    name="Tree P(v|u)",
                    x=names_top,
                    y=p_tree[top_idx].cpu().numpy(),
                    marker_color="steelblue",
                    showlegend=(row_idx == 0),
                ),
                row=row_idx + 1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    name="Graph P(v|u)",
                    x=names_top,
                    y=p_graph[top_idx].cpu().numpy(),
                    marker_color="coral",
                    showlegend=(row_idx == 0),
                ),
                row=row_idx + 1,
                col=1,
            )

        fig.update_layout(
            title=f"Sampling Probability Shift (T={temperature})",
            barmode="group",
            height=max(1, len(anchors)) * 400,
            width=1200,
        )

        out_path = OUT_DIR / "sampling_shift_plot.html"
        fig.write_html(str(out_path))
        assert out_path.exists()
