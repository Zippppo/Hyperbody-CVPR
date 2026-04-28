"""Task 6 validation for scripts/precompute_contact_matrix.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

OUT_DIR = Path("docs/visualizations/spatial_adjacency/stage_validation")


def _write_npz(path: Path, voxel_labels: np.ndarray) -> None:
    sensor_pc = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        sensor_pc=sensor_pc,
        grid_world_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        grid_voxel_size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        voxel_labels=voxel_labels.astype(np.int64),
    )


def test_task6_precompute_script_end_to_end(tmp_path):
    """Script should compute, report, and save a valid contact matrix from tiny synthetic data."""
    repo_root = Path(__file__).resolve().parents[2]

    data_dir = tmp_path / "voxel_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    lbl_1 = np.zeros((12, 12, 12), dtype=np.int64)
    lbl_1[2:6, 2:8, 2:8] = 1
    lbl_1[6:10, 2:8, 2:8] = 2

    lbl_2 = np.zeros((12, 12, 12), dtype=np.int64)
    lbl_2[2:10, 2:10, 2:10] = 1
    lbl_2[4:8, 4:8, 4:8] = 2

    _write_npz(data_dir / "sample_1.npz", lbl_1)
    _write_npz(data_dir / "sample_2.npz", lbl_2)

    split_path = tmp_path / "dataset_split.json"
    split_path.write_text(
        json.dumps({"train": ["sample_1.npz", "sample_2.npz"], "val": [], "test": []}),
        encoding="utf-8",
    )

    info_path = tmp_path / "dataset_info.json"
    info_path.write_text(
        json.dumps({"class_names": ["background", "organ_a", "organ_b"]}),
        encoding="utf-8",
    )

    output_path = tmp_path / "contact_matrix.pt"

    cmd = [
        sys.executable,
        "scripts/precompute_contact_matrix.py",
        "--output",
        str(output_path),
        "--dilation-radius",
        "1",
        "--class-batch-size",
        "2",
        "--data-dir",
        str(data_dir),
        "--split-file",
        str(split_path),
        "--dataset-info",
        str(info_path),
        "--volume-size",
        "12",
        "12",
        "12",
        "--num-workers",
        "0",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "Classes: 3" in result.stdout
    assert "Non-zero contacts" in result.stdout
    assert output_path.exists()

    contact = torch.load(output_path)
    assert contact.shape == (3, 3)
    assert contact.dtype == torch.float32
    assert torch.allclose(torch.diag(contact), torch.zeros(3))
    assert contact[1, 2].item() > 0.0

    if go is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        preview_path = OUT_DIR / "task6_precompute_contact_preview.html"

        fig = go.Figure(
            data=go.Heatmap(
                z=contact.numpy(),
                x=["background", "organ_a", "organ_b"],
                y=["background", "organ_a", "organ_b"],
                text=np.round(contact.numpy(), 3),
                texttemplate="%{text}",
                colorscale="Hot",
            )
        )
        fig.update_layout(title="Task 6 Validation: Precomputed Contact Matrix Preview")
        fig.write_html(str(preview_path))

        assert preview_path.exists()
        assert preview_path.stat().st_size > 2000
