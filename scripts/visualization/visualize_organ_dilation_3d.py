"""Visualize 3D organ voxels before and after dilation for ECCV figure.

Produces a single HTML with 3 side-by-side Plotly Mesh3d subplots:
  - Original organs (no dilation)
  - Dilation radius = 2
  - Dilation radius = 3

Uses marching cubes for mesh extraction per organ class.
"""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.voxelizer import pad_labels

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_FILE = "BDMAP_00000001.npz"
DATA_DIR = PROJECT_ROOT / "Dataset" / "voxel_data"
DATASET_INFO = PROJECT_ROOT / "Dataset" / "dataset_info.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "organ_dilation_comparison.html"
VOLUME_SIZE = (144, 128, 268)
DILATION_RADII = [0, 2, 3]
IGNORED_CLASS_INDEX = 0  # inside_body_empty


def load_class_names() -> list[str]:
    with open(DATASET_INFO, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def load_labels(sample_file: str) -> np.ndarray:
    path = DATA_DIR / sample_file
    data = np.load(str(path))
    return pad_labels(data["voxel_labels"], VOLUME_SIZE)


def generate_organ_colors(num_classes: int) -> list[str]:
    """Generate bright, vivid, perceptually distinct colors for each organ."""
    import colorsys

    colors = []
    for i in range(num_classes):
        if i == IGNORED_CLASS_INDEX:
            colors.append("rgba(0,0,0,0)")
            continue
        hue = (i * 137.508) % 360 / 360.0
        # High saturation, high lightness — bright and vivid
        sat = 0.85 + 0.15 * ((i % 2) / 1.0)
        light = 0.55 + 0.15 * ((i % 3) / 2.0)
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return colors


def dilate_labels(labels_np: np.ndarray, radius: int, num_classes: int) -> np.ndarray:
    """Dilate organ masks using max_pool3d, return argmax label volume.

    For radius=0, returns the original labels unchanged.
    """
    if radius == 0:
        return labels_np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_t = torch.from_numpy(labels_np).long().to(device)

    # One-hot: (D,H,W) -> (C,D,H,W)
    one_hot = (
        F.one_hot(labels_t, num_classes=num_classes)
        .permute(3, 0, 1, 2)
        .float()
    )

    kernel = 2 * radius + 1
    dilated = F.max_pool3d(
        one_hot.unsqueeze(1),  # (C,1,D,H,W)
        kernel_size=kernel,
        stride=1,
        padding=radius,
    ).squeeze(1)  # (C,D,H,W)

    # Argmax to resolve overlaps; class 0 should not win over real organs
    # Set class-0 dilated values to -inf so it never wins argmax
    dilated[IGNORED_CLASS_INDEX] = -1.0

    result = dilated.argmax(dim=0).cpu().numpy()

    del labels_t, one_hot, dilated
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def _ensure_outward_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip face winding so normals point outward (away from mesh centroid)."""
    centroid = verts.mean(axis=0)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_centers = (v0 + v1 + v2) / 3.0
    normals = np.cross(v1 - v0, v2 - v0)
    outward = face_centers - centroid
    # Flip faces where normal points inward
    dot = (normals * outward).sum(axis=1)
    flip_mask = dot < 0
    faces[flip_mask] = faces[flip_mask][:, ::-1]
    return faces


def extract_meshes(label_volume: np.ndarray, num_classes: int):
    """Extract marching-cubes meshes for each organ class.

    Returns list of (vertices, faces, class_idx) tuples.
    """
    meshes = []
    for c in range(num_classes):
        if c == IGNORED_CLASS_INDEX:
            continue
        mask = (label_volume == c).astype(np.float32)
        if mask.sum() < 10:
            continue
        try:
            verts, faces, _, _ = marching_cubes(mask, level=0.5)
            faces = _ensure_outward_normals(verts, faces)
            meshes.append((verts, faces, c))
        except Exception:
            continue
    return meshes


def build_mesh_traces(meshes, colors, col_idx, class_names):
    """Build Plotly Mesh3d traces for one subplot."""
    traces = []
    for verts, faces, c in meshes:
        trace = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=colors[c],
            opacity=1.0,
            name=class_names[c],
            showlegend=(col_idx == 0),
            legendgroup=class_names[c],
            hovertext=class_names[c],
            hoverinfo="text",
            flatshading=False,
            lighting=dict(
                ambient=0.4,
                diffuse=0.6,
                specular=0.15,
                roughness=0.6,
                fresnel=0.05,
            ),
            lightposition=dict(x=200, y=200, z=300),
        )
        traces.append(trace)
    return traces


def main():
    print("Loading class names...")
    class_names = load_class_names()
    num_classes = len(class_names)
    colors = generate_organ_colors(num_classes)

    print(f"Loading sample: {SAMPLE_FILE}")
    labels = load_labels(SAMPLE_FILE)
    unique_classes = np.unique(labels)
    print(f"  Unique classes in sample: {len(unique_classes)}")

    # Shared camera for consistent viewpoint
    camera = dict(
        eye=dict(x=1.6, y=-1.6, z=1.0),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
    )
    scene_cfg = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="white",
        camera=camera,
        aspectmode="data",
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            "Original",
            "Dilation r = 2",
            "Dilation r = 3",
        ],
        horizontal_spacing=0.01,
    )

    for col_idx, radius in enumerate(DILATION_RADII):
        print(f"Processing dilation radius={radius}...")
        dilated = dilate_labels(labels, radius, num_classes)

        print(f"  Extracting meshes...")
        meshes = extract_meshes(dilated, num_classes)
        print(f"  Extracted {len(meshes)} organ meshes")

        traces = build_mesh_traces(meshes, colors, col_idx, class_names)
        for trace in traces:
            fig.add_trace(trace, row=1, col=col_idx + 1)

    # Apply scene config to all 3 subplots
    fig.update_layout(
        scene=scene_cfg,
        scene2=scene_cfg,
        scene3=scene_cfg,
        title=dict(
            text="Organ Voxel Dilation: Original vs r=2 vs r=3",
            x=0.5,
            font=dict(size=20, family="Arial, sans-serif"),
        ),
        legend=dict(
            font=dict(size=9),
            itemsizing="constant",
            tracegroupgap=1,
            yanchor="top",
            y=0.95,
            xanchor="right",
            x=1.0,
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=60, b=0),
        width=2400,
        height=800,
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=16, family="Arial, sans-serif", color="black")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUTPUT_PATH))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
