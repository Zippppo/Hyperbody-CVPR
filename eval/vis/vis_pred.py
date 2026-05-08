"""
Visualization script for prediction results.

Usage:
    python eval/vis/vis_pred.py --pred_dir eval/pred/lorentz_semantic --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/baseline/
    python eval/vis/vis_pred.py --pred_dir eval/pred/lorentz_semantic --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/semantic_vis/
    python eval/vis/vis_pred.py --pred_dir eval/pred/lorentz_random --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/random_vis/
    python eval/vis/vis_pred.py --pred_dir eval/pred/L_R_FZ+cls0_0.1 --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/L_R_FZ+cls0_0.1/
    python eval/vis/vis_pred.py --pred_dir eval/pred/L_S_FZ+cls0_0.1 --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/L_S_FZ+cls0_0.1/

python eval/vis/vis_pred.py --pred_dir eval/pred/LR-GD-M04-LRP3 --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/0209-LR-GD-M04-LRP3/
python eval/vis/vis_pred.py --pred_dir eval/pred/vis_best --gt_dir Dataset/voxel_data --compare --output_dir docs/visualizations/pred_vis/vis_best


"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


# Organ system definitions for filtering
ORGAN_SYSTEMS = {
    "All": None,  # Show all
    "Skeletal": [
        "spine", "skull", "sternum", "costal_cartilages",
        "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
        "humerus_left", "humerus_right", "hip_left", "hip_right",
        "femur_left", "femur_right",
    ] + [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)],
    "Organs": [
        "liver", "spleen", "kidney_left", "kidney_right", "stomach", "pancreas",
        "gallbladder", "urinary_bladder", "prostate", "heart", "brain",
        "thyroid_gland", "lung", "esophagus", "trachea",
        "adrenal_gland_left", "adrenal_gland_right",
    ],
    "Digestive": [
        "liver", "stomach", "pancreas", "gallbladder", "esophagus",
        "small_bowel", "duodenum", "colon",
    ],
    "Muscles": [
        "gluteus_maximus_left", "gluteus_maximus_right",
        "gluteus_medius_left", "gluteus_medius_right",
        "gluteus_minimus_left", "gluteus_minimus_right",
        "autochthon_left", "autochthon_right",
        "iliopsoas_left", "iliopsoas_right",
    ],
    "Ribs (All)": [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)],
    "Ribs Left": [f"rib_left_{i}" for i in range(1, 13)],
    "Ribs Right": [f"rib_right_{i}" for i in range(1, 13)],
}

# Individual rib pairs
for i in range(1, 13):
    ORGAN_SYSTEMS[f"Rib Pair {i}"] = [f"rib_left_{i}", f"rib_right_{i}"]

# 24 distinct colors for individual ribs (12 left + 12 right)
RIB_COLORS = [
    # Left ribs (1-12): warm colors gradient
    "#FF0000", "#FF4500", "#FF8C00", "#FFD700",
    "#ADFF2F", "#32CD32", "#00CED1", "#1E90FF",
    "#4169E1", "#8A2BE2", "#FF1493", "#DC143C",
    # Right ribs (1-12): cool colors gradient
    "#00FFFF", "#00BFFF", "#87CEEB", "#ADD8E6",
    "#B0E0E6", "#AFEEEE", "#7FFFD4", "#66CDAA",
    "#3CB371", "#2E8B57", "#228B22", "#006400",
]

# Build rib name to color index mapping
RIB_NAME_TO_INDEX = {}
for i in range(1, 13):
    RIB_NAME_TO_INDEX[f"rib_left_{i}"] = i - 1       # 0-11
    RIB_NAME_TO_INDEX[f"rib_right_{i}"] = i + 11    # 12-23


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize prediction results")
    parser.add_argument("--pred_dir", type=str, default="eval/pred/baseline",
                        help="Directory containing prediction .npz files")
    parser.add_argument("--gt_dir", type=str, default="Dataset/voxel_data",
                        help="Directory containing ground truth .npz files")
    parser.add_argument("--output_dir", type=str, default="docs/visualizations",
                        help="Output directory for HTML visualizations")
    parser.add_argument("--sample", type=str, default=None,
                        help="Specific sample filename to visualize (e.g., BDMAP_00000053.npz)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare prediction with ground truth side by side")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of samples to visualize (when --sample is not set)")
    parser.add_argument("--max_points", type=int, default=50000,
                        help="Maximum points to render per organ system")
    parser.add_argument("--dataset_info", type=str, default="Dataset/dataset_info.json",
                        help="Path to dataset info JSON for class names")
    return parser.parse_args()


def load_class_names(dataset_info_path):
    """Load class names from dataset info."""
    with open(dataset_info_path) as f:
        info = json.load(f)
    return info["class_names"]


def get_system_class_indices(class_names, system_organs):
    """Get class indices for a given organ system."""
    if system_organs is None:
        return None  # All classes
    indices = []
    for i, name in enumerate(class_names):
        if name in system_organs:
            indices.append(i)
    return set(indices)


def subsample_voxels(indices, labels, max_points):
    """Subsample voxel indices and labels if exceeding max_points."""
    n = len(indices)
    if n <= max_points:
        return indices, labels
    step = max(1, n // max_points)
    return indices[::step], labels[::step]


def get_system_color(system_name):
    """Get a distinct color for each organ system."""
    colors = {
        "All": "Rainbow",
        "Skeletal": "Viridis",
        "Organs": "Plasma",
        "Digestive": "YlOrRd",
        "Muscles": "Reds",
        "Ribs (All)": None,  # Use discrete colors
        "Ribs Left": None,   # Use discrete colors
        "Ribs Right": None,  # Use discrete colors
    }
    # Individual ribs
    if system_name.startswith("Rib "):
        return None  # Use discrete colors
    return colors.get(system_name, "Rainbow")


def is_rib_system(system_name):
    """Check if the system is a rib-related system."""
    return system_name in ["Ribs (All)", "Ribs Left", "Ribs Right"] or system_name.startswith("Rib Pair")


def create_trace_for_system(labels_3d, grid_world_min, voxel_size, class_names,
                            system_name, system_indices, max_points, trace_name):
    """Create a Scatter3d trace for a specific organ system."""
    # Filter by system
    if system_indices is None:
        # All non-zero voxels
        mask = labels_3d > 0
    else:
        # Vectorized: check if each voxel's class is in system_indices
        mask = np.isin(labels_3d, list(system_indices))

    voxel_idx = np.argwhere(mask)
    if len(voxel_idx) == 0:
        return None, 0

    voxel_classes = labels_3d[voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]]

    # Subsample
    voxel_idx, voxel_classes = subsample_voxels(voxel_idx, voxel_classes, max_points)

    # Convert to world coordinates
    centers = grid_world_min + (voxel_idx + 0.5) * voxel_size

    # Hover text
    hover_text = [class_names[int(l)] for l in voxel_classes]

    # Check if this is a rib system - use discrete colors for each rib
    if is_rib_system(system_name):
        # Map class indices to rib color indices
        rib_color_indices = []
        for cls_idx in voxel_classes:
            cls_name = class_names[int(cls_idx)]
            if cls_name in RIB_NAME_TO_INDEX:
                rib_color_indices.append(RIB_NAME_TO_INDEX[cls_name])
            else:
                rib_color_indices.append(0)

        # Map rib indices to actual colors
        marker_colors = [RIB_COLORS[idx] for idx in rib_color_indices]

        trace = go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=marker_colors,
                opacity=1.0,
            ),
            text=hover_text,
            hovertemplate="Class: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
            name=trace_name,
            visible=(system_name == "All"),
        )
    else:
        trace = go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=voxel_classes,
                colorscale=get_system_color(system_name),
                opacity=1.0,
                cmin=0,
                cmax=len(class_names) - 1,
            ),
            text=hover_text,
            hovertemplate="Class: %{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>",
            name=trace_name,
            visible=(system_name == "All"),  # Only "All" visible by default
        )

    return trace, int(mask.sum())


def visualize_with_system_selector(labels_3d, grid_world_min, voxel_size,
                                   class_names, max_points, title_prefix):
    """Create traces for all organ systems with dropdown selector."""
    traces = []
    system_voxel_counts = {}

    # Create traces for each system
    for system_name, system_organs in ORGAN_SYSTEMS.items():
        system_indices = get_system_class_indices(class_names, system_organs)
        trace, count = create_trace_for_system(
            labels_3d, grid_world_min, voxel_size, class_names,
            system_name, system_indices, max_points, f"{title_prefix} - {system_name}"
        )
        if trace is not None:
            traces.append(trace)
            system_voxel_counts[system_name] = count

    return traces, system_voxel_counts


def create_dropdown_buttons(num_systems, offset=0):
    """Create dropdown buttons for organ system selection."""
    buttons = []
    system_names = list(ORGAN_SYSTEMS.keys())

    for i, system_name in enumerate(system_names):
        # Create visibility array: only show traces for this system
        visibility = [False] * (num_systems + offset)
        visibility[i + offset] = True
        buttons.append(dict(
            label=system_name,
            method="update",
            args=[{"visible": visibility}],
        ))

    return buttons


def visualize_prediction(pred_path, output_path, class_names, max_points):
    """Visualize a single prediction file with organ system selector."""
    pred_data = np.load(pred_path)
    pred_labels = pred_data["pred_labels"]
    grid_world_min = pred_data["grid_world_min"]
    voxel_size = pred_data["grid_voxel_size"]

    # Create traces for all systems
    traces, voxel_counts = visualize_with_system_selector(
        pred_labels, grid_world_min, voxel_size, class_names, max_points, "Prediction"
    )

    fig = go.Figure(data=traces)

    # Add dropdown menu
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=create_dropdown_buttons(len(traces)),
            direction="down",
            showactive=True,
            x=0.02,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )],
    )

    sample_name = os.path.basename(pred_path).replace(".npz", "")
    fig.update_layout(
        title=f"Prediction: {sample_name}<br>Select organ system from dropdown",
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
    )

    fig.write_html(output_path)
    return output_path


def visualize_comparison(pred_path, gt_path, output_path, class_names, max_points):
    """Visualize prediction vs ground truth with organ system selector."""
    pred_data = np.load(pred_path)
    gt_data = np.load(gt_path)

    pred_labels = pred_data["pred_labels"]
    gt_labels = gt_data["voxel_labels"]
    grid_world_min = pred_data["grid_world_min"]
    voxel_size = pred_data["grid_voxel_size"]

    # Create traces for GT and Pred for each system
    all_traces = []
    system_names = list(ORGAN_SYSTEMS.keys())

    for system_name, system_organs in ORGAN_SYSTEMS.items():
        system_indices = get_system_class_indices(class_names, system_organs)

        # GT trace
        gt_trace, gt_count = create_trace_for_system(
            gt_labels, grid_world_min, voxel_size, class_names,
            system_name, system_indices, max_points, f"GT - {system_name}"
        )
        # Pred trace
        pred_trace, pred_count = create_trace_for_system(
            pred_labels, grid_world_min, voxel_size, class_names,
            system_name, system_indices, max_points, f"Pred - {system_name}"
        )

        if gt_trace is not None:
            all_traces.append(gt_trace)
        if pred_trace is not None:
            # Offset prediction to the right for side-by-side view
            x_offset = (grid_world_min[0] + pred_labels.shape[0] * voxel_size[0]) * 1.2
            pred_trace.x = tuple(x + x_offset for x in pred_trace.x)
            all_traces.append(pred_trace)

    # Create visibility buttons (show both GT and Pred for each system)
    buttons = []
    num_systems = len(system_names)
    for i, system_name in enumerate(system_names):
        visibility = [False] * (num_systems * 2)
        visibility[i * 2] = True      # GT trace
        visibility[i * 2 + 1] = True  # Pred trace
        buttons.append(dict(
            label=system_name,
            method="update",
            args=[{"visible": visibility}],
        ))

    fig = go.Figure(data=all_traces)

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.02,
            xanchor="left",
            y=1.15,
            yanchor="top",
        )],
    )

    sample_name = os.path.basename(pred_path).replace(".npz", "")
    fig.update_layout(
        title=f"Comparison: {sample_name}<br>Left: Ground Truth | Right: Prediction<br>Select organ system from dropdown",
        width=1400,
        height=800,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        showlegend=True,
        legend=dict(x=0.85, y=0.95),
    )

    fig.write_html(output_path)
    return output_path


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = load_class_names(args.dataset_info)

    # Get list of prediction files
    if args.sample:
        pred_files = [args.sample]
    else:
        pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(".npz")])
        pred_files = pred_files[:args.max_samples]

    print(f"Visualizing {len(pred_files)} samples...")
    print(f"Output directory: {args.output_dir}")

    for filename in tqdm(pred_files, desc="Generating visualizations"):
        pred_path = os.path.join(args.pred_dir, filename)
        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} not found, skipping")
            continue

        sample_name = filename.replace(".npz", "")

        if args.compare:
            gt_path = os.path.join(args.gt_dir, filename)
            if not os.path.exists(gt_path):
                print(f"Warning: GT file {gt_path} not found, skipping comparison")
                continue
            output_path = os.path.join(args.output_dir, f"{sample_name}_compare.html")
            visualize_comparison(pred_path, gt_path, output_path, class_names, args.max_points)
        else:
            output_path = os.path.join(args.output_dir, f"{sample_name}_pred.html")
            visualize_prediction(pred_path, output_path, class_names, args.max_points)

    print(f"\nDone! Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
