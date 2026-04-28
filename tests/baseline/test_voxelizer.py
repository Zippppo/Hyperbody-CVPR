"""TDD tests for data/voxelizer.py - Step 2"""
import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_PATH = "Dataset/voxel_data/BDMAP_00000001.npz"
VOLUME_SIZE = (128, 96, 256)
NEW_VOLUME_SIZE = (144, 128, 268)  # 新的目标尺寸


# --- Unit tests ---

class TestVoxelizePointCloud:
    def test_output_shape(self):
        """Output shape matches volume_size."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        assert result.shape == VOLUME_SIZE

    def test_output_dtype(self):
        """Output is float32."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        assert result.dtype == np.float32

    def test_binary_values(self):
        """Output contains only 0.0 and 1.0."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        unique = np.unique(result)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_occupied_voxels_reasonable(self):
        """Number of occupied voxels is reasonable (less than num points, more than 0)."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        num_occupied = int(result.sum())
        num_points = len(data['sensor_pc'])
        assert num_occupied > 0
        # Multiple points can map to the same voxel
        assert num_occupied <= num_points

    def test_synthetic_data(self):
        """Voxelize a known set of points and verify placement."""
        from data.voxelizer import voxelize_point_cloud
        volume_size = (10, 10, 10)
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # Points at (0.5, 0.5, 0.5) and (3.5, 3.5, 3.5)
        pc = np.array([[0.5, 0.5, 0.5], [3.5, 3.5, 3.5]], dtype=np.float32)
        result = voxelize_point_cloud(pc, grid_world_min, voxel_size, volume_size)
        assert result[0, 0, 0] == 1.0
        assert result[3, 3, 3] == 1.0
        assert result.sum() == 2.0


class TestPadLabels:
    def test_output_shape(self):
        """Padded labels match volume_size."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        result = pad_labels(data['voxel_labels'], VOLUME_SIZE)
        assert result.shape == VOLUME_SIZE

    def test_output_dtype(self):
        """Padded labels are int64."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        result = pad_labels(data['voxel_labels'], VOLUME_SIZE)
        assert result.dtype == np.int64

    def test_original_data_preserved(self):
        """Original label data is preserved in the padded volume."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        labels = data['voxel_labels']
        result = pad_labels(labels, VOLUME_SIZE)
        x, y, z = labels.shape
        np.testing.assert_array_equal(result[:x, :y, :z], labels.astype(np.int64))

    def test_padding_is_zero(self):
        """Padded region is filled with 0 (class 0 = inside_body_empty)."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        labels = data['voxel_labels']
        result = pad_labels(labels, VOLUME_SIZE)
        x, y, z = labels.shape
        # Check padding region along x
        if x < VOLUME_SIZE[0]:
            assert result[x:, :, :].sum() == 0


# --- Visualization test ---

@pytest.mark.parametrize("sample_name", ["BDMAP_00000001"])
def test_visualize_voxelization(sample_name, tmp_path):
    """Generate interactive HTML visualization comparing point cloud and voxelized result."""
    from data.voxelizer import voxelize_point_cloud, pad_labels
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = np.load(f"Dataset/voxel_data/{sample_name}.npz")
    pc = data['sensor_pc']
    labels = data['voxel_labels']
    grid_world_min = data['grid_world_min']
    voxel_size = data['grid_voxel_size']

    occ = voxelize_point_cloud(pc, grid_world_min, voxel_size, VOLUME_SIZE)
    padded_labels = pad_labels(labels, VOLUME_SIZE)

    # Subsample point cloud for rendering
    step = max(1, len(pc) // 10000)
    pc_sub = pc[::step]

    # Get occupied voxel centers in world coordinates
    occ_idx = np.argwhere(occ > 0)  # (N, 3)
    vox_centers = grid_world_min + (occ_idx + 0.5) * voxel_size

    # Get label voxel centers (non-zero classes only) for a slice
    label_idx = np.argwhere(padded_labels > 0)
    step_l = max(1, len(label_idx) // 15000)
    label_idx_sub = label_idx[::step_l]
    label_centers = grid_world_min + (label_idx_sub + 0.5) * voxel_size
    label_classes = padded_labels[label_idx_sub[:, 0], label_idx_sub[:, 1], label_idx_sub[:, 2]]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Input Point Cloud ({len(pc)} pts)",
            f"Voxelized Occupancy ({int(occ.sum())} voxels)",
            f"Ground Truth Labels ({len(label_idx)} voxels)"
        ],
    )

    # 1. Point cloud
    fig.add_trace(go.Scatter3d(
        x=pc_sub[:, 0], y=pc_sub[:, 1], z=pc_sub[:, 2],
        mode='markers', marker=dict(size=1, color='steelblue', opacity=0.6),
        name='Point Cloud',
    ), row=1, col=1)

    # 2. Voxelized occupancy
    fig.add_trace(go.Scatter3d(
        x=vox_centers[:, 0], y=vox_centers[:, 1], z=vox_centers[:, 2],
        mode='markers', marker=dict(size=1.5, color='tomato', opacity=0.5),
        name='Voxelized',
    ), row=1, col=2)

    # 3. Ground truth labels
    fig.add_trace(go.Scatter3d(
        x=label_centers[:, 0], y=label_centers[:, 1], z=label_centers[:, 2],
        mode='markers', marker=dict(
            size=1.5, color=label_classes, colorscale='Rainbow',
            opacity=0.5, colorbar=dict(title="Class", x=1.02),
        ),
        name='Labels',
    ), row=1, col=3)

    fig.update_layout(
        title=f"Voxelization Pipeline: {sample_name}",
        width=1800, height=600,
        showlegend=False,
    )

    out_dir = "docs/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"voxelization_{sample_name}.html")
    fig.write_html(out_path)
    print(f"\nVisualization saved to: {out_path}")
    assert os.path.exists(out_path)


class TestPadLabelsCropAndPad:
    """Test pad_labels crop and pad functionality for oversized data"""

    def test_crop_oversized_synthetic_data(self):
        """Test: oversized data should be cropped to target size"""
        from data.voxelizer import pad_labels

        # Create synthetic data exceeding target size (150, 135, 270) > (144, 128, 268)
        oversized_shape = (150, 135, 270)
        target_size = NEW_VOLUME_SIZE  # (144, 128, 268)

        # Fill with coordinate-based values for verification
        labels = np.zeros(oversized_shape, dtype=np.uint8)
        for i in range(oversized_shape[0]):
            for j in range(oversized_shape[1]):
                for k in range(oversized_shape[2]):
                    labels[i, j, k] = (i + j + k) % 70

        result = pad_labels(labels, target_size)

        # Verify output shape
        assert result.shape == target_size, f"Expected {target_size}, got {result.shape}"

        # Verify cropping: result should equal the first target_size portion of labels
        expected = labels[:target_size[0], :target_size[1], :target_size[2]].astype(np.int64)
        np.testing.assert_array_equal(result, expected)

        print(f"\nCrop test passed:")
        print(f"  Input shape: {oversized_shape}")
        print(f"  Target shape: {target_size}")
        print(f"  Output shape: {result.shape}")
        print(f"  Cropped: X={oversized_shape[0]-target_size[0]}, Y={oversized_shape[1]-target_size[1]}, Z={oversized_shape[2]-target_size[2]}")

    def test_pad_undersized_synthetic_data(self):
        """Test: undersized data should be padded"""
        from data.voxelizer import pad_labels

        # Create data smaller than target size
        undersized_shape = (100, 80, 200)
        target_size = NEW_VOLUME_SIZE  # (144, 128, 268)

        labels = np.ones(undersized_shape, dtype=np.uint8) * 5  # Fill with class 5

        result = pad_labels(labels, target_size)

        # Verify output shape
        assert result.shape == target_size

        # Verify original data preserved
        np.testing.assert_array_equal(
            result[:undersized_shape[0], :undersized_shape[1], :undersized_shape[2]],
            labels.astype(np.int64)
        )

        # Verify padded region is zero
        assert result[undersized_shape[0]:, :, :].sum() == 0
        assert result[:, undersized_shape[1]:, :].sum() == 0
        assert result[:, :, undersized_shape[2]:].sum() == 0

        print(f"\nPad test passed:")
        print(f"  Input shape: {undersized_shape}")
        print(f"  Target shape: {target_size}")
        print(f"  Padded: X={target_size[0]-undersized_shape[0]}, Y={target_size[1]-undersized_shape[1]}, Z={target_size[2]-undersized_shape[2]}")

    def test_mixed_crop_and_pad(self):
        """Test: some dimensions need cropping, others need padding"""
        from data.voxelizer import pad_labels

        # X needs crop, Y needs pad, Z exact match
        mixed_shape = (150, 100, 268)  # X>144, Y<128, Z==268
        target_size = NEW_VOLUME_SIZE  # (144, 128, 268)

        labels = np.arange(np.prod(mixed_shape), dtype=np.uint8).reshape(mixed_shape) % 70

        result = pad_labels(labels, target_size)

        assert result.shape == target_size

        # Verify crop+pad correctness
        expected_x = min(mixed_shape[0], target_size[0])  # 144
        expected_y = min(mixed_shape[1], target_size[1])  # 100
        expected_z = min(mixed_shape[2], target_size[2])  # 268

        np.testing.assert_array_equal(
            result[:expected_x, :expected_y, :expected_z],
            labels[:expected_x, :expected_y, :expected_z].astype(np.int64)
        )

        # Y padding region should be zero
        assert result[:, expected_y:, :].sum() == 0

        print(f"\nMixed crop/pad test passed:")
        print(f"  Input shape: {mixed_shape}")
        print(f"  Target shape: {target_size}")
        print(f"  X: cropped {mixed_shape[0]-target_size[0]}")
        print(f"  Y: padded {target_size[1]-mixed_shape[1]}")
        print(f"  Z: unchanged")

    def test_real_oversized_sample(self):
        """Test real oversized sample BDMAP_00003811.npz (126, 101, 132)"""
        from data.voxelizer import pad_labels

        # This sample has Y=101 > 96 (old size), but < 128 (new size)
        data = np.load("Dataset/voxel_data/BDMAP_00003811.npz")
        labels = data['voxel_labels']

        print(f"\nReal sample test:")
        print(f"  Sample: BDMAP_00003811.npz")
        print(f"  Original shape: {labels.shape}")
        print(f"  Target shape: {NEW_VOLUME_SIZE}")

        result = pad_labels(labels, NEW_VOLUME_SIZE)

        assert result.shape == NEW_VOLUME_SIZE

        # Original data should be fully preserved (126<144, 101<128, 132<268)
        np.testing.assert_array_equal(
            result[:labels.shape[0], :labels.shape[1], :labels.shape[2]],
            labels.astype(np.int64)
        )

        print(f"  Output shape: {result.shape}")
        print(f"  Test passed: original data fully preserved")


@pytest.mark.parametrize("sample_name", ["BDMAP_00003811"])
def test_visualize_crop_and_pad(sample_name, tmp_path):
    """Visualize crop/pad before and after comparison"""
    from data.voxelizer import pad_labels
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = np.load(f"Dataset/voxel_data/{sample_name}.npz")
    labels = data['voxel_labels']
    grid_world_min = data['grid_world_min']
    voxel_size = data['grid_voxel_size']

    processed = pad_labels(labels, NEW_VOLUME_SIZE)

    orig_idx = np.argwhere(labels > 0)
    proc_idx = np.argwhere(processed > 0)

    step_o = max(1, len(orig_idx) // 15000)
    step_p = max(1, len(proc_idx) // 15000)
    orig_idx_sub = orig_idx[::step_o]
    proc_idx_sub = proc_idx[::step_p]

    orig_centers = grid_world_min + (orig_idx_sub + 0.5) * voxel_size
    proc_centers = grid_world_min + (proc_idx_sub + 0.5) * voxel_size

    orig_classes = labels[orig_idx_sub[:, 0], orig_idx_sub[:, 1], orig_idx_sub[:, 2]]
    proc_classes = processed[proc_idx_sub[:, 0], proc_idx_sub[:, 1], proc_idx_sub[:, 2]]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Original Labels {labels.shape}",
            f"Processed Labels {processed.shape}"
        ],
    )

    fig.add_trace(go.Scatter3d(
        x=orig_centers[:, 0], y=orig_centers[:, 1], z=orig_centers[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=orig_classes, colorscale='Rainbow', opacity=0.6),
        name='Original',
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=proc_centers[:, 0], y=proc_centers[:, 1], z=proc_centers[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=proc_classes, colorscale='Rainbow', opacity=0.6),
        name='Processed',
    ), row=1, col=2)

    fig.update_layout(
        title=f"Crop & Pad: {sample_name}<br>Original {labels.shape} -> Target {NEW_VOLUME_SIZE}",
        width=1400, height=700,
        showlegend=False,
    )

    out_dir = "docs/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"crop_pad_{sample_name}.html")
    fig.write_html(out_path)
    print(f"\nVisualization saved to: {out_path}")
    assert os.path.exists(out_path)
