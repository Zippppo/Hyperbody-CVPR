"""Stage-by-stage validation tests for spatial adjacency graph rollout.

These tests mirror Tasks 1/2/4/5 in the implementation plan and emphasize
visual artifacts for quick human inspection in code review.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from data.spatial_adjacency import (
    _compute_single_sample_overlap,
    compute_contact_matrix_from_dataset,
    compute_graph_distance_matrix,
)

OUT_DIR = Path("docs/visualizations/spatial_adjacency/stage_validation")


def _load_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError:
        return None, None
    return go, make_subplots


class _LabelDataset(Dataset):
    """Minimal dataset yielding (input, label) pairs."""

    def __init__(self, labels_list: list[torch.Tensor]):
        self.labels_list = labels_list

    def __len__(self) -> int:
        return len(self.labels_list)

    def __getitem__(self, idx: int):
        labels = self.labels_list[idx]
        inp = torch.zeros(1, *labels.shape, dtype=torch.float32)
        return inp, labels


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    assert path.exists()
    assert path.stat().st_size > 2000


class TestTask1SingleSampleOverlapVisual:
    def test_task1_overlap_visual_and_chunk_consistency(self):
        """Task 1: single-sample overlap has expected geometry and stable chunk path."""
        labels = torch.zeros(24, 24, 24, dtype=torch.long)
        labels[4:10, 6:16, 6:16] = 1
        labels[10:16, 6:16, 6:16] = 2
        labels[16:22, 16:22, 16:22] = 3

        overlap_full, volume_full = _compute_single_sample_overlap(
            labels=labels,
            num_classes=4,
            dilation_radius=2,
            class_batch_size=0,
        )
        overlap_chunk, volume_chunk = _compute_single_sample_overlap(
            labels=labels,
            num_classes=4,
            dilation_radius=2,
            class_batch_size=2,
        )

        assert torch.allclose(overlap_full, overlap_chunk)
        assert torch.allclose(volume_full, volume_chunk)

        contact = overlap_full / volume_full.unsqueeze(1).clamp(min=1.0)
        contact.fill_diagonal_(0)

        assert contact[1, 2].item() > 0.2
        assert contact[2, 1].item() > 0.2
        assert contact[1, 3].item() == 0.0
        assert contact[3, 1].item() == 0.0

        mask_1 = (labels == 1).float().unsqueeze(0).unsqueeze(0)
        dilated_1 = F.max_pool3d(mask_1, kernel_size=5, stride=1, padding=2).squeeze(0).squeeze(0)
        overlap_1_to_2 = dilated_1 * (labels == 2).float()

        z = labels.shape[2] // 2
        go, make_subplots = _load_plotly()

        if go is None:
            return

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Label slice",
                "Dilated class-1 slice",
                "Overlap slice (1 -> 2)",
            ],
        )
        fig.add_trace(go.Heatmap(z=labels[:, :, z].numpy(), colorscale="Viridis"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=dilated_1[:, :, z].numpy(), colorscale="Blues"), row=1, col=2)
        fig.add_trace(go.Heatmap(z=overlap_1_to_2[:, :, z].numpy(), colorscale="Reds"), row=1, col=3)
        fig.update_layout(
            title="Task 1 Validation: Single-Sample Overlap Geometry",
            width=1400,
            height=460,
            showlegend=False,
        )

        out_path = OUT_DIR / "task1_single_sample_overlap.html"
        _save_fig(fig, out_path)


class TestTask2DatasetContactAndGraphFusionVisual:
    def test_task2_contact_aggregation_and_graph_distance_visual(self):
        """Task 2: dataset-level contact aggregation and graph-distance fusion are coherent."""
        lbl_a = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl_a[3:9, 3:12, 3:12] = 1
        lbl_a[9:15, 3:12, 3:12] = 2

        lbl_b = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl_b[3:17, 3:17, 3:17] = 2
        lbl_b[8:12, 8:12, 8:12] = 3

        lbl_c = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl_c[2:10, 2:10, 2:10] = 1

        dataset = _LabelDataset([lbl_a, lbl_b, lbl_c])
        contact = compute_contact_matrix_from_dataset(
            dataset=dataset,
            num_classes=4,
            dilation_radius=2,
            class_batch_size=2,
        )

        assert contact.shape == (4, 4)
        assert contact[1, 2].item() > 0.0
        assert contact[3, 2].item() > contact[2, 3].item()
        assert contact[1, 3].item() == 0.0

        D_tree = torch.tensor(
            [
                [0.0, 6.0, 9.0, 10.0],
                [6.0, 0.0, 6.0, 8.0],
                [9.0, 6.0, 0.0, 5.0],
                [10.0, 8.0, 5.0, 0.0],
            ],
            dtype=torch.float32,
        )
        D_final = compute_graph_distance_matrix(
            D_tree=D_tree,
            contact_matrix=contact,
            lambda_=1.0,
            epsilon=0.01,
        )
        D_diff = D_tree - D_final

        assert D_final[1, 2].item() < D_tree[1, 2].item()
        assert D_final[3, 2].item() != D_final[2, 3].item()
        assert D_diff.max().item() > 0.0

        go, make_subplots = _load_plotly()

        if go is None:
            return

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Aggregated Contact Matrix", "Distance Shortening (D_tree - D_final)"],
        )
        fig.add_trace(
            go.Heatmap(z=contact.numpy(), colorscale="Hot", text=np.round(contact.numpy(), 3), texttemplate="%{text}"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(z=D_diff.numpy(), colorscale="RdBu", zmid=0, text=np.round(D_diff.numpy(), 2), texttemplate="%{text}"),
            row=1,
            col=2,
        )
        fig.update_layout(
            title="Task 2 Validation: Contact Aggregation + Graph Distance Fusion",
            width=1300,
            height=520,
        )

        out_path = OUT_DIR / "task2_contact_and_graph_distance.html"
        _save_fig(fig, out_path)


class TestTask4TrainConfigGraphIntegration:
    def test_task4_train_graph_mode_builds_expected_matrix(self, monkeypatch, tmp_path):
        """Task 4: graph mode in train/config wires expected matrix and cache behavior."""
        import train

        class_names = ["background", "organ_a", "organ_b"]
        dataset_info = tmp_path / "dataset_info.json"
        dataset_info.write_text(json.dumps({"class_names": class_names}), encoding="utf-8")

        tree_file = tmp_path / "tree.json"
        tree_file.write_text("{}", encoding="utf-8")

        checkpoint_dir = tmp_path / "checkpoints"
        log_dir = tmp_path / "runs"
        cfg_path = tmp_path / "graph_mode.yaml"
        cfg_overrides = {
            "num_classes": 3,
            "epochs": 0,
            "batch_size": 1,
            "num_workers": 0,
            "use_amp": False,
            "track_embeddings": False,
            "gpu_ids": [0],
            "checkpoint_dir": str(checkpoint_dir),
            "log_dir": str(log_dir),
            "dataset_info_file": str(dataset_info),
            "tree_file": str(tree_file),
            "hyp_distance_mode": "graph",
            "spatial_lambda": 1.5,
            "spatial_epsilon": 0.01,
            "hyp_embed_dim": 4,
            "save_every": 1,
        }
        cfg_path.write_text(yaml.safe_dump(cfg_overrides, sort_keys=False), encoding="utf-8")

        D_tree = torch.tensor(
            [
                [0.0, 8.0, 10.0],
                [8.0, 0.0, 6.0],
                [10.0, 6.0, 0.0],
            ],
            dtype=torch.float32,
        )
        contact = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4],
                [0.0, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )
        expected_graph = compute_graph_distance_matrix(
            D_tree,
            contact,
            lambda_=cfg_overrides["spatial_lambda"],
            epsilon=cfg_overrides["spatial_epsilon"],
        )

        calls = {"contact_compute": 0}
        captured = {"tree_dist_matrix": None}

        class TinyDataset(Dataset):
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int):
                labels = torch.zeros(8, 8, 8, dtype=torch.long)
                labels[1:4, 1:4, 1:4] = 1
                labels[4:7, 1:4, 1:4] = 2
                return torch.zeros(1, 8, 8, 8), labels

        class DummyLabelEmbedding(nn.Module):
            def __init__(self, num_classes: int, embed_dim: int):
                super().__init__()
                self.tangent_embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))

        class DummyBodyNet(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.backbone_weight = nn.Parameter(torch.ones(1))
                self.label_emb = DummyLabelEmbedding(
                    num_classes=kwargs["num_classes"],
                    embed_dim=kwargs["embed_dim"],
                )

            def forward(self, x: torch.Tensor):
                b, _, d, h, w = x.shape
                num_classes = self.label_emb.tangent_embeddings.shape[0]
                embed_dim = self.label_emb.tangent_embeddings.shape[1]
                logits = torch.zeros((b, num_classes, d, h, w), device=x.device)
                voxel_emb = torch.zeros((b, embed_dim, d, h, w), device=x.device)
                return logits, voxel_emb, self.label_emb.tangent_embeddings

        class DummyCombinedLoss(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, logits, targets):
                return torch.tensor(0.0, device=targets.device)

        class DummyTreeLoss(nn.Module):
            def __init__(self, tree_dist_matrix: torch.Tensor, **kwargs):
                super().__init__()
                matrix = tree_dist_matrix.detach().clone().float()
                self.register_buffer("tree_dist_matrix", matrix)
                captured["tree_dist_matrix"] = matrix

            def set_epoch(self, epoch: int):
                return None

            def forward(self, voxel_emb, labels, label_emb):
                return torch.tensor(0.0, device=voxel_emb.device, requires_grad=True)

        class DummyMetric:
            def __init__(self, num_classes: int):
                self.num_classes = num_classes

            def reset(self):
                return None

            def update(self, logits, targets):
                return None

            def sync_across_processes(self):
                return None

            def compute(self):
                return torch.zeros(self.num_classes), 0.0, {}

        class DummySummaryWriter:
            def __init__(self, *args, **kwargs):
                return None

            def add_scalar(self, *args, **kwargs):
                return None

            def close(self):
                return None

        def fake_contact_from_dataset(*args, **kwargs):
            calls["contact_compute"] += 1
            return contact.clone()

        monkeypatch.setattr(
            train,
            "parse_args",
            lambda: argparse.Namespace(
                config=str(cfg_path),
                resume="",
                batch_size=None,
                epochs=None,
                gpuids=None,
                seed=123,
            ),
        )
        monkeypatch.setattr(train, "setup_logging", lambda *args, **kwargs: logging.getLogger("task4_test"))
        monkeypatch.setattr(train, "HyperBodyDataset", lambda *args, **kwargs: TinyDataset())
        monkeypatch.setattr(train, "compute_class_weights", lambda *args, **kwargs: torch.ones(3))
        monkeypatch.setattr(train, "load_organ_hierarchy", lambda *args, **kwargs: {0: 1, 1: 2, 2: 2})
        monkeypatch.setattr(train, "load_class_to_system", lambda *args, **kwargs: {0: "other", 1: "other", 2: "other"})
        monkeypatch.setattr(train, "BodyNet", DummyBodyNet)
        monkeypatch.setattr(train, "CombinedLoss", DummyCombinedLoss)
        monkeypatch.setattr(train, "compute_tree_distance_matrix", lambda *args, **kwargs: D_tree.clone())
        monkeypatch.setattr(train, "compute_contact_matrix_from_dataset", fake_contact_from_dataset)
        monkeypatch.setattr(train, "LorentzTreeRankingLoss", DummyTreeLoss)
        monkeypatch.setattr(train, "DiceMetric", DummyMetric)
        monkeypatch.setattr(train, "SummaryWriter", DummySummaryWriter)

        train.main()

        assert calls["contact_compute"] == 1
        assert captured["tree_dist_matrix"] is not None
        assert torch.allclose(captured["tree_dist_matrix"], expected_graph, atol=1e-6)

        cache_path = checkpoint_dir / "contact_matrix.pt"
        assert cache_path.exists()
        cached = torch.load(cache_path)
        assert torch.allclose(cached, contact)

        go, make_subplots = _load_plotly()

        if go is None:
            return

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["D_tree", "Contact", "Graph distance (fed to loss)"],
        )
        fig.add_trace(go.Heatmap(z=D_tree.numpy(), colorscale="Greys"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=contact.numpy(), colorscale="Hot"), row=1, col=2)
        fig.add_trace(
            go.Heatmap(z=captured["tree_dist_matrix"].cpu().numpy(), colorscale="Viridis"),
            row=1,
            col=3,
        )
        fig.update_layout(
            title="Task 4 Validation: train.py graph-mode integration",
            width=1500,
            height=480,
        )

        out_path = OUT_DIR / "task4_train_graph_integration.html"
        _save_fig(fig, out_path)


class TestTask5VisualizationPipeline:
    def test_task5_synthetic_visualizations_show_expected_shift(self):
        """Task 5: visualization pipeline highlights directed shortcut effects."""
        class_names = ["background", "rib_like", "lung_like", "liver_like"]

        D_tree = torch.tensor(
            [
                [0.0, 6.0, 8.0, 9.0],
                [6.0, 0.0, 8.0, 10.0],
                [8.0, 8.0, 0.0, 7.0],
                [9.0, 10.0, 7.0, 0.0],
            ],
            dtype=torch.float32,
        )
        contact = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 0.08, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)
        D_diff = D_tree - D_final

        assert D_diff[1, 2].item() > 0.0
        assert D_final[1, 2].item() < D_final[2, 1].item()

        temperature = 0.5
        anchor = class_names.index("rib_like")

        p_tree = torch.exp(-D_tree[anchor] / temperature)
        p_tree[anchor] = 0
        p_tree = p_tree / p_tree.sum().clamp(min=1e-12)

        p_graph = torch.exp(-D_final[anchor] / temperature)
        p_graph[anchor] = 0
        p_graph = p_graph / p_graph.sum().clamp(min=1e-12)

        assert p_graph[class_names.index("lung_like")].item() > p_tree[class_names.index("lung_like")].item()

        go, _ = _load_plotly()
        if go is None:
            return

        out_contact = OUT_DIR / "task5_contact_heatmap.html"
        out_diff = OUT_DIR / "task5_distance_diff_heatmap.html"
        out_shift = OUT_DIR / "task5_sampling_shift.html"

        fig_contact = go.Figure(
            data=go.Heatmap(
                z=contact.numpy(),
                x=class_names,
                y=class_names,
                text=np.round(contact.numpy(), 3),
                texttemplate="%{text}",
                colorscale="Hot",
            )
        )
        fig_contact.update_layout(title="Task 5: Contact Matrix (Synthetic)")
        _save_fig(fig_contact, out_contact)

        fig_diff = go.Figure(
            data=go.Heatmap(
                z=D_diff.numpy(),
                x=class_names,
                y=class_names,
                text=np.round(D_diff.numpy(), 2),
                texttemplate="%{text}",
                colorscale="RdBu",
                zmid=0,
            )
        )
        fig_diff.update_layout(title="Task 5: D_tree - D_final (Synthetic)")
        _save_fig(fig_diff, out_diff)

        fig_shift = go.Figure()
        fig_shift.add_trace(go.Bar(name="Tree P(v|u)", x=class_names, y=p_tree.numpy()))
        fig_shift.add_trace(go.Bar(name="Graph P(v|u)", x=class_names, y=p_graph.numpy()))
        fig_shift.update_layout(
            title="Task 5: Sampling Probability Shift (anchor = rib_like)",
            barmode="group",
        )
        _save_fig(fig_shift, out_shift)

        html = out_shift.read_text(encoding="utf-8")
        assert "Sampling Probability Shift" in html
        assert "Graph P(v|u)" in html


@pytest.mark.parametrize(
    "bad_shape",
    [
        torch.zeros(2, 3),
        torch.zeros(4),
    ],
)
def test_graph_distance_shape_guard_raises(bad_shape):
    """Review guard: mismatched shapes should fail loudly with ValueError."""
    D_tree = torch.zeros(3, 3)
    with pytest.raises(ValueError):
        compute_graph_distance_matrix(D_tree, bad_shape)
