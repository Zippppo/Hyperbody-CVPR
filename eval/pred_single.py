"""
Single-file inference: run BodyNet on an arbitrary sensor_pc .npz file.

The input npz only needs a 'sensor_pc' key (N,3) float32.
Grid metadata (grid_world_min, grid_voxel_size) is computed from the point cloud.

Usage:
    python eval/pred_single.py --input zrk_ponitcloud.npz
    python eval/pred_single.py --input zrk_ponitcloud.npz --config configs/021201-final.yaml --ckpt checkpoints/021201-final/best.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
import torch
from torch.amp import autocast

from config import Config
from data.organ_hierarchy import load_organ_hierarchy
from data.voxelizer import voxelize_point_cloud
from models.body_net import BodyNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single point cloud")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npz with sensor_pc")
    parser.add_argument("--config", type=str, default="configs/021201-final.yaml")
    parser.add_argument("--ckpt", type=str, default="checkpoints/021201-final/best.pth")
    parser.add_argument("--output", type=str, default="eval/pred/single", help="Output directory")
    parser.add_argument("--gpuids", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.8, help="Scale factor around centroid (1.0=no scaling)")
    parser.add_argument("--y-shift", type=float, default=200.0, help="Y-axis shift in mm (0=no shift)")
    parser.add_argument("--z-shift", type=float, default=0.0, help="Z-axis shift in mm (0=no shift)")
    return parser.parse_args()


def compute_grid_metadata(sensor_pc, volume_size, voxel_size=4.0, margin=8.0,
                          target_center_ratio=(0.35, 0.46, 0.22)):
    """Compute grid_world_min and grid_voxel_size from raw point cloud.

    Strategy: align occupancy center to training distribution, but clamp
    grid_world_min so that ALL points fit inside the volume (no clipping).

    target_center_ratio: per-axis (D, H, W) ideal center position (0~1),
        derived from training data: D=0.35, H=0.46, W=0.22.
    """
    grid_voxel_size = np.array([voxel_size] * 3, dtype=np.float32)

    # Start with ideal alignment
    grid_world_min = np.empty(3, dtype=np.float32)
    for ax in range(3):
        target_idx = target_center_ratio[ax] * volume_size[ax]
        grid_world_min[ax] = sensor_pc[:, ax].mean() - target_idx * voxel_size

    # Clamp: ensure all points fit within [0, volume_size) after voxelization
    pc_min = sensor_pc.min(axis=0)
    pc_max = sensor_pc.max(axis=0)
    for ax in range(3):
        # grid_world_min must be <= pc_min - margin (so min point gets index >= margin/voxel_size)
        upper_bound = pc_min[ax] - margin
        # grid_world_min must be >= pc_max - (volume_size[ax]-1)*voxel_size (so max point fits)
        lower_bound = pc_max[ax] - (volume_size[ax] - 1) * voxel_size
        grid_world_min[ax] = np.clip(grid_world_min[ax], lower_bound, upper_bound)

    return grid_world_min.astype(np.float32), grid_voxel_size


def load_model(cfg, ckpt_path, device):
    with open(cfg.dataset_info_file) as f:
        class_names = json.load(f)["class_names"]
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

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "N/A")
    best_dice = checkpoint.get("best_dice", "N/A")
    print(f"Loaded checkpoint: epoch={epoch}, best_dice={best_dice:.4f}")
    return model, class_names


def visualize_prediction(pred_labels, grid_world_min, voxel_size, class_names, output_path,
                         occupancy=None):
    """Create interactive plotly visualization of the prediction.

    Args:
        pred_labels: (D, H, W) int64 predicted class per voxel
        grid_world_min: (3,) float32 world-space origin
        voxel_size: float, voxel edge length in mm
        class_names: list of class name strings
        output_path: path for output HTML
        occupancy: (D, H, W) float32 binary occupancy grid (model input)
    """
    mask = pred_labels > 0
    voxel_idx = np.argwhere(mask)
    if len(voxel_idx) == 0:
        print("Warning: no non-zero predictions")
        return

    voxel_classes = pred_labels[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]]
    centers = grid_world_min + (voxel_idx + 0.5) * voxel_size
    hover_text = [class_names[int(c)] for c in voxel_classes]

    traces = [go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
        mode="markers", name="Prediction",
        marker=dict(size=2, color=voxel_classes, colorscale="Rainbow", opacity=1.0,
                    cmin=0, cmax=len(class_names) - 1),
        text=hover_text,
        hovertemplate="Class: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
    )]

    # Overlay voxelized occupancy grid (the actual model input, no subsampling)
    if occupancy is not None:
        occ_idx = np.argwhere(occupancy > 0)
        occ_centers = grid_world_min + (occ_idx + 0.5) * voxel_size
        traces.append(go.Scatter3d(
            x=occ_centers[:, 0], y=occ_centers[:, 1], z=occ_centers[:, 2],
            mode="markers", name=f"Model input ({len(occ_idx)} voxels)",
            marker=dict(size=3, color="gray", opacity=0.3),
            hovertemplate="Input voxel<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Single Point Cloud Prediction",
        width=1200, height=900,
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                   aspectmode="data"),
    )
    fig.write_html(output_path)
    print(f"Visualization saved to {output_path}")


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    volume_size = tuple(cfg.volume_size)

    # Load input point cloud and apply calibration
    data = np.load(args.input)
    sensor_pc = data["sensor_pc"].copy()
    print(f"Input point cloud: {sensor_pc.shape[0]} points (raw)")

    # Calibration: scale around centroid, then shift Y/Z
    if args.scale != 1.0 or args.y_shift != 0.0 or args.z_shift != 0.0:
        centroid = (sensor_pc.max(axis=0) + sensor_pc.min(axis=0)) / 2
        sensor_pc = (sensor_pc - centroid) * args.scale + centroid
        sensor_pc[:, 1] += args.y_shift
        sensor_pc[:, 2] += args.z_shift
        print(f"After calibration (scale={args.scale}, Y_shift={args.y_shift}mm, Z_shift={args.z_shift}mm): {sensor_pc.shape[0]} points")
    else:
        print("No calibration applied")

    raw_pc = sensor_pc.copy()  # keep for potential future use

    # Compute grid metadata from point cloud (align H center to training distribution)
    grid_world_min, grid_voxel_size = compute_grid_metadata(
        sensor_pc, volume_size, voxel_size=cfg.voxel_size)
    print(f"Computed grid_world_min={grid_world_min}")
    print(f"grid_voxel_size={grid_voxel_size}, volume_size={volume_size}")

    # Voxelize
    occupancy = voxelize_point_cloud(sensor_pc, grid_world_min, grid_voxel_size, volume_size)
    occ_count = int(occupancy.sum())
    print(f"Occupied voxels: {occ_count} / {np.prod(volume_size)}")

    inp = torch.from_numpy(occupancy).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Device
    device = torch.device(f"cuda:{args.gpuids}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, class_names = load_model(cfg, args.ckpt, device)
    inp = inp.to(device)

    # Forward pass
    if cfg.use_amp and device.type == "cuda":
        with autocast(device_type="cuda"):
            logits, _, _ = model(inp)
    else:
        logits, _, _ = model(inp)

    pred_labels = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (D, H, W)

    # Stats
    unique, counts = np.unique(pred_labels, return_counts=True)
    print(f"\nPrediction stats: {len(unique)} classes present")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:15]:
        name = class_names[int(u)] if int(u) < len(class_names) else f"cls_{u}"
        print(f"  {name} (id={u}): {c} voxels")

    # Save
    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_npz = os.path.join(args.output, f"{stem}_pred.npz")
    np.savez_compressed(out_npz,
                        pred_labels=pred_labels.astype(np.int64),
                        grid_world_min=grid_world_min,
                        grid_voxel_size=grid_voxel_size,
                        original_filename=os.path.basename(args.input))
    print(f"\nPrediction saved to {out_npz}")

    # Visualize
    out_html = os.path.join(args.output, f"{stem}_pred.html")
    visualize_prediction(pred_labels, grid_world_min, grid_voxel_size, class_names, out_html,
                         occupancy=occupancy)


if __name__ == "__main__":
    main()
