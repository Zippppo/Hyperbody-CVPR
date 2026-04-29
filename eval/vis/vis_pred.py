"""
Visualization script for prediction results.

This script supports the S2I 121-label setup (class 0 = inside_body_empty, classes 1-120 = anatomical foreground).
Default --dataset_info / --gt_dir point at S2I_Dataset, matching eval/eval_120cls.py outputs.

Usage (S2I 120-class, current):
    python eval/vis/vis_pred.py --pred_dir eval/pred/s2i_120cls_epoch45 --compare --output_dir docs/visualizations/pred_vis/s2i_120cls_epoch45/
    python eval/vis/vis_pred.py --pred_dir eval/pred/s2i_120cls_epoch45 --sample BDMAP_00000001.npz --compare --output_dir docs/visualizations/pred_vis/s2i_120cls_epoch45/

Usage (legacy 70-class, override defaults):
    python eval/vis/vis_pred.py --pred_dir eval/pred/lorentz_semantic --gt_dir Dataset/voxel_data --dataset_info Dataset/dataset_info.json --compare --output_dir docs/visualizations/pred_vis/baseline/
    python eval/vis/vis_pred.py --pred_dir eval/pred/vis_best --gt_dir Dataset/voxel_data --dataset_info Dataset/dataset_info.json --compare --output_dir docs/visualizations/pred_vis/vis_best
"""
import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


# JS injected into comparison HTML so the GT and Pred scenes share a camera.
# Passed as `post_script` to fig.write_html — Plotly already wraps this in a
# <script> block guarded by `if (document.getElementById("<plot_id>"))`, so we
# only need the raw JS body (no <script> tags, no IIFE required).
_CAMERA_SYNC_JS = """
var __gd = document.getElementsByClassName('plotly-graph-div')[0];
if (__gd) {
    var __syncing = false;
    __gd.on('plotly_relayout', function(eventData) {
        if (__syncing) return;
        var keys = Object.keys(eventData || {});
        var src = null, dst = null;
        if (keys.some(function(k){ return k.indexOf('scene.camera') === 0; })) {
            src = __gd.layout.scene && __gd.layout.scene.camera; dst = 'scene2';
        } else if (keys.some(function(k){ return k.indexOf('scene2.camera') === 0; })) {
            src = __gd.layout.scene2 && __gd.layout.scene2.camera; dst = 'scene';
        }
        if (src && dst) {
            __syncing = true;
            var upd = {}; upd[dst + '.camera'] = src;
            Plotly.relayout(__gd, upd).then(function(){ __syncing = false; });
        }
    });
}
"""


# --- Anatomy groups for the S2I 121-label setup --------------------------------
# Class 0 = inside_body_empty; classes 1-120 are foreground anatomical labels.
# These lists use the exact names from S2I_Dataset/dataset_info.json. Names not
# present in the loaded dataset_info are silently dropped by
# get_system_class_indices(), so legacy 70-class runs (which use "spine"/"lung")
# still work as long as you pass --dataset_info Dataset/dataset_info.json.

_VERTEBRAE = (
    [f"vertebrae_C{i}" for i in range(1, 8)]
    + [f"vertebrae_T{i}" for i in range(1, 13)]
    + [f"vertebrae_L{i}" for i in range(1, 6)]
    + ["vertebrae_S1", "sacrum"]
)

_LUNG_LOBES = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
]

_ARTERIES = [
    "aorta", "brachiocephalic_trunk", "coronary_arteries",
    "subclavian_artery_left", "subclavian_artery_right",
    "common_carotid_artery_left", "common_carotid_artery_right",
    "iliac_artery_left", "iliac_artery_right",
]

_VEINS = [
    "pulmonary_vein", "atrial_appendage_left",
    "inferior_vena_cava", "superior_vena_cava",
    "portal_vein_and_splenic_vein",
    "brachiocephalic_vein_left", "brachiocephalic_vein_right",
    "iliac_vena_left", "iliac_vena_right",
]

_FAT_AND_MUSCLE_BULK = [
    "subcutaneous_fat", "torso_fat", "intermuscular_fat", "skeletal_muscle",
]

_RIB_LEFT = [f"rib_left_{i}" for i in range(1, 13)]
_RIB_RIGHT = [f"rib_right_{i}" for i in range(1, 13)]

# Organ system definitions for filtering
ORGAN_SYSTEMS = {
    "All": None,  # Show all
    # Skeletal: keeps legacy "spine"/"lung" names so 70-class data still groups
    # correctly, and adds the 120-class vertebrae expansion.
    "Skeletal": [
        "spine", "skull", "sternum", "costal_cartilages",
        "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
        "humerus_left", "humerus_right", "hip_left", "hip_right",
        "femur_left", "femur_right",
    ] + _VERTEBRAE + _RIB_LEFT + _RIB_RIGHT,
    "Vertebrae": _VERTEBRAE,
    "Organs": [
        "liver", "spleen", "kidney_left", "kidney_right", "stomach", "pancreas",
        "gallbladder", "urinary_bladder", "prostate", "heart", "brain",
        "thyroid_gland", "spinal_cord", "lung", "esophagus", "trachea",
        "adrenal_gland_left", "adrenal_gland_right",
    ] + _LUNG_LOBES,
    "Lungs (Lobes)": _LUNG_LOBES,
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
    "Vascular (All)": _ARTERIES + _VEINS,
    "Vascular - Arteries": _ARTERIES,
    "Vascular - Veins": _VEINS,
    "Soft Tissue / Fat": _FAT_AND_MUSCLE_BULK,
    "Ribs (All)": _RIB_LEFT + _RIB_RIGHT,
    "Ribs Left": _RIB_LEFT,
    "Ribs Right": _RIB_RIGHT,
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
    parser.add_argument("--gt_dir", type=str, default="S2I_Dataset/train",
                        help="Directory containing ground truth .npz files (S2I_Dataset/train for 120-class, Dataset/voxel_data for legacy 70-class)")
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
    parser.add_argument("--dataset_info", type=str, default="S2I_Dataset/dataset_info.json",
                        help="Path to dataset info JSON for class names (S2I_Dataset/dataset_info.json = 121 labels, Dataset/dataset_info.json = legacy 70)")
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
    # Drop labels outside the foreground range (0 = inside_body_empty,
    # 255 = outside_body_background in S2I, plus any other ignore/pad value).
    # Without this, "All" includes ignore voxels and class_names[label] crashes.
    num_classes = len(class_names)
    valid_mask = (labels_3d > 0) & (labels_3d < num_classes)

    # Filter by system
    if system_indices is None:
        # All foreground voxels with valid labels
        mask = valid_mask
    else:
        # Vectorized: check if each voxel's class is in system_indices.
        # np.isin already excludes out-of-range values since they're not in
        # system_indices, but AND with valid_mask keeps the contract explicit.
        mask = np.isin(labels_3d, list(system_indices)) & valid_mask

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


def _empty_trace(name, visible):
    """Placeholder trace so visibility-button indices stay aligned when a system is empty."""
    return go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(size=2),
        name=name,
        visible=visible,
        showlegend=False,
        hoverinfo="skip",
    )


def visualize_comparison(pred_path, gt_path, output_path, class_names, max_points):
    """Visualize prediction vs ground truth in two independent 3D scenes (side-by-side).

    GT goes into the left scene, Pred into the right scene. Each scene rotates/zooms
    independently in plotly, but a small JS snippet (_CAMERA_SYNC_JS) is appended to
    the HTML so cameras stay mirrored for easier visual comparison.
    """
    pred_data = np.load(pred_path)
    gt_data = np.load(gt_path)

    pred_labels = pred_data["pred_labels"]
    gt_labels = gt_data["voxel_labels"]
    grid_world_min = pred_data["grid_world_min"]
    voxel_size = pred_data["grid_voxel_size"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Ground Truth", "Prediction"),
        horizontal_spacing=0.02,
    )

    system_names = list(ORGAN_SYSTEMS.keys())

    # Trace order in fig.data is interleaved: [GT_0, Pred_0, GT_1, Pred_1, ...]
    # so the button at index i flips visibility[2*i] and visibility[2*i + 1].
    for system_name, system_organs in ORGAN_SYSTEMS.items():
        system_indices = get_system_class_indices(class_names, system_organs)
        default_visible = (system_name == "All")

        gt_trace, _ = create_trace_for_system(
            gt_labels, grid_world_min, voxel_size, class_names,
            system_name, system_indices, max_points, f"GT - {system_name}"
        )
        pred_trace, _ = create_trace_for_system(
            pred_labels, grid_world_min, voxel_size, class_names,
            system_name, system_indices, max_points, f"Pred - {system_name}"
        )

        if gt_trace is None:
            gt_trace = _empty_trace(f"GT - {system_name}", default_visible)
        if pred_trace is None:
            pred_trace = _empty_trace(f"Pred - {system_name}", default_visible)

        fig.add_trace(gt_trace, row=1, col=1)
        fig.add_trace(pred_trace, row=1, col=2)

    # Visibility buttons: each button shows one system's GT (scene 1) + Pred (scene 2).
    num_systems = len(system_names)
    buttons = []
    for i, system_name in enumerate(system_names):
        visibility = [False] * (num_systems * 2)
        visibility[i * 2] = True      # GT trace in scene
        visibility[i * 2 + 1] = True  # Pred trace in scene2
        buttons.append(dict(
            label=system_name,
            method="update",
            args=[{"visible": visibility}],
        ))

    sample_name = os.path.basename(pred_path).replace(".npz", "")
    scene_kwargs = dict(
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        aspectmode="data",
    )
    fig.update_layout(
        title=(
            f"Comparison: {sample_name}<br>"
            "Left: Ground Truth | Right: Prediction "
            "(cameras synchronized; select organ system from dropdown)"
        ),
        width=1600,
        height=820,
        scene=scene_kwargs,
        scene2=scene_kwargs,
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.02,
            xanchor="left",
            y=1.12,
            yanchor="top",
        )],
        showlegend=False,
    )

    # write_html accepts a post_script that runs after Plotly initializes the figure,
    # which is where the camera-sync hook attaches.
    fig.write_html(output_path, post_script=_CAMERA_SYNC_JS)
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
