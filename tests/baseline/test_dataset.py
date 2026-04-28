"""TDD tests for data/dataset.py - Step 3"""
import numpy as np
import pytest
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VOLUME_SIZE = (128, 96, 256)


class TestHyperBodyDataset:
    def _make_dataset(self, split="val"):
        from data.dataset import HyperBodyDataset
        return HyperBodyDataset(
            data_dir="Dataset/voxel_data",
            split_file="Dataset/dataset_split.json",
            split=split,
            volume_size=VOLUME_SIZE,
        )

    def test_create_dataset(self):
        ds = self._make_dataset("val")
        assert ds is not None

    def test_length(self):
        ds = self._make_dataset("val")
        assert len(ds) == 500

    def test_train_length(self):
        ds = self._make_dataset("train")
        assert len(ds) == 9779

    def test_getitem_returns_tuple(self):
        ds = self._make_dataset("val")
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_input_tensor_shape(self):
        """Input should be (1, 128, 96, 256) float32."""
        ds = self._make_dataset("val")
        inp, _ = ds[0]
        assert isinstance(inp, torch.Tensor)
        assert inp.shape == (1, *VOLUME_SIZE)
        assert inp.dtype == torch.float32

    def test_label_tensor_shape(self):
        """Label should be (128, 96, 256) int64."""
        ds = self._make_dataset("val")
        _, lbl = ds[0]
        assert isinstance(lbl, torch.Tensor)
        assert lbl.shape == VOLUME_SIZE
        assert lbl.dtype == torch.int64

    def test_input_binary(self):
        """Input occupancy grid should only contain 0 and 1."""
        ds = self._make_dataset("val")
        inp, _ = ds[0]
        unique = torch.unique(inp)
        assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_label_class_range(self):
        """Labels should be in [0, 69]."""
        ds = self._make_dataset("val")
        _, lbl = ds[0]
        assert lbl.min() >= 0
        assert lbl.max() <= 69

    def test_dataloader_batch(self):
        """DataLoader can produce a batch."""
        from torch.utils.data import DataLoader
        ds = self._make_dataset("val")
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        batch_inp, batch_lbl = next(iter(loader))
        assert batch_inp.shape == (2, 1, *VOLUME_SIZE)
        assert batch_lbl.shape == (2, *VOLUME_SIZE)

    def test_invalid_split_raises(self):
        from data.dataset import HyperBodyDataset
        with pytest.raises((ValueError, KeyError)):
            HyperBodyDataset(
                data_dir="Dataset/voxel_data",
                split_file="Dataset/dataset_split.json",
                split="nonexistent",
                volume_size=VOLUME_SIZE,
            )


class TestDatasetVisualization:
    def test_visualize_dataset_samples(self):
        """Visualize a few dataset samples to verify pipeline correctness."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from data.dataset import HyperBodyDataset

        ds = HyperBodyDataset(
            data_dir="Dataset/voxel_data",
            split_file="Dataset/dataset_split.json",
            split="val",
            volume_size=VOLUME_SIZE,
        )

        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "scatter3d"}] * 3, [{"type": "scatter3d"}] * 3],
            subplot_titles=[
                f"Sample 0 - Input", f"Sample 0 - Label",  f"Sample 0 - Overlay",
                f"Sample 1 - Input", f"Sample 1 - Label",  f"Sample 1 - Overlay",
            ],
        )

        for row_idx, sample_idx in enumerate([0, 1]):
            inp, lbl = ds[sample_idx]
            inp_np = inp.squeeze(0).numpy()  # (X,Y,Z)
            lbl_np = lbl.numpy()             # (X,Y,Z)

            # Subsample for rendering
            inp_idx = np.argwhere(inp_np > 0)
            step_i = max(1, len(inp_idx) // 8000)
            inp_sub = inp_idx[::step_i]

            lbl_idx = np.argwhere(lbl_np > 0)
            step_l = max(1, len(lbl_idx) // 8000)
            lbl_sub = lbl_idx[::step_l]
            lbl_classes = lbl_np[lbl_sub[:, 0], lbl_sub[:, 1], lbl_sub[:, 2]]

            r = row_idx + 1

            # Input occupancy
            fig.add_trace(go.Scatter3d(
                x=inp_sub[:, 0], y=inp_sub[:, 1], z=inp_sub[:, 2],
                mode='markers', marker=dict(size=1, color='steelblue', opacity=0.5),
            ), row=r, col=1)

            # Label
            fig.add_trace(go.Scatter3d(
                x=lbl_sub[:, 0], y=lbl_sub[:, 1], z=lbl_sub[:, 2],
                mode='markers', marker=dict(size=1, color=lbl_classes, colorscale='Rainbow', opacity=0.5),
            ), row=r, col=2)

            # Overlay: input (blue) + label (colored)
            fig.add_trace(go.Scatter3d(
                x=inp_sub[:, 0], y=inp_sub[:, 1], z=inp_sub[:, 2],
                mode='markers', marker=dict(size=1, color='steelblue', opacity=0.3),
            ), row=r, col=3)
            fig.add_trace(go.Scatter3d(
                x=lbl_sub[:, 0], y=lbl_sub[:, 1], z=lbl_sub[:, 2],
                mode='markers', marker=dict(size=1, color=lbl_classes, colorscale='Rainbow', opacity=0.4),
            ), row=r, col=3)

        fig.update_layout(
            title="Dataset Pipeline Verification: Input → Label → Overlay",
            width=1800, height=1000,
            showlegend=False,
        )

        out_dir = "docs/visualizations"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "dataset_samples.html")
        fig.write_html(out_path)
        print(f"\nVisualization saved to: {out_path}")
        assert os.path.exists(out_path)
