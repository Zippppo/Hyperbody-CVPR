"""
Generate publication-quality pipeline figure components for HyperBody.

Components per subject:
  1. Voxelized full body (solid cubes, neutral color)
  2. Surface point cloud (single color, sensor input)
  3. Internal organ labels (per-organ unique colors, voxel cubes)

Outputs both interactive HTML (Plotly) and static PNG.

Key technique: only render OUTER SURFACE faces of the voxel grid to avoid
Plotly Mesh3d transparency artifacts from internal faces.

Usage:
    python scripts/visualization/visualize_pipeline_figure.py
    python scripts/visualization/visualize_pipeline_figure.py --samples BDMAP_00001264.npz
"""

import argparse
import json
import os

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "Dataset/voxel_data"
INFO_PATH = "Dataset/dataset_info.json"
OUTPUT_DIR = "outputs/pipeline_figure"

DEFAULT_SAMPLES = [
    "BDMAP_00001264.npz",
    "BDMAP_00009603.npz",
    "BDMAP_00004868.npz",
    "BDMAP_00001243.npz",
    "BDMAP_00004912.npz",
]

CAMERA_PRESETS = {
    "oblique": dict(eye=dict(x=-1.2, y=2.0, z=0.6), up=dict(x=0, y=0, z=1)),
    "front":   dict(eye=dict(x=0.0, y=2.5, z=0.0), up=dict(x=0, y=0, z=1)),
    "side":    dict(eye=dict(x=2.5, y=0.3, z=0.0), up=dict(x=0, y=0, z=1)),
}

MAX_SURFACE_QUADS = 200000  # max surface quads (2 tris each)
MAX_POINTS_PC = 80000

BODY_COLOR = "rgb(255, 224, 189)"  # bright warm beige
PC_COLOR = "rgb(50, 50, 50)"       # dark gray / near-black

FIG_WIDTH = 1200
FIG_HEIGHT = 900
PNG_SCALE = 2


# ---------------------------------------------------------------------------
# High-contrast scientific palette (perceptually uniform spacing)
# ---------------------------------------------------------------------------

def build_organ_colormap(class_names):
    """High-contrast palette with maximum perceptual distinguishability.

    Design principles:
      - Organs: maximally spaced hues across the full spectrum
      - Skeletal: bone/ivory/sandy tones (anatomically intuitive)
      - Muscles: red-magenta family (anatomically intuitive)
      - Left/right pairs: same hue, right side is lighter (+lightness shift)
      - Ribs: compact gradient per side within skeletal hue range
    """
    palettes = {
        # --- Visceral organs (max hue separation across full spectrum) ---
        "liver":                "#B03030",  # dark crimson
        "heart":                "#E02020",  # bright red
        "lung":                 "#4A90D9",  # medium blue
        "stomach":              "#D4A017",  # golden
        "spleen":               "#7B2D8E",  # deep purple
        "kidney_left":          "#CC5500",  # burnt orange
        "kidney_right":         "#E87830",  # lighter orange
        "pancreas":             "#C8A800",  # dark gold
        "gallbladder":          "#2D8E50",  # forest green
        "urinary_bladder":      "#2878B5",  # cobalt blue
        "prostate":             "#008080",  # teal
        "brain":                "#D16587",  # dusty rose
        "thyroid_gland":        "#7E57C2",  # medium purple
        "spinal_cord":          "#E07040",  # terra cotta
        "esophagus":            "#00897B",  # dark cyan
        "trachea":              "#26A69A",  # medium teal
        "small_bowel":          "#E6B44C",  # warm gold
        "duodenum":             "#C49A3C",  # bronze
        "colon":                "#8D6E44",  # warm brown
        "adrenal_gland_left":   "#5C3DAA",  # indigo
        "adrenal_gland_right":  "#7E5FCC",  # lighter indigo
        # --- Skeletal (bone/ivory/cool-gray family) ---
        "skull":                "#E8E0D0",  # ivory
        "spine":                "#C8B898",  # light bone
        "sternum":              "#A8A0B0",  # warm gray
        "costal_cartilages":    "#90B8C8",  # pale steel
        "scapula_left":         "#607890",  # slate
        "scapula_right":        "#7898B0",  # lighter slate
        "clavicula_left":       "#506888",  # dark slate
        "clavicula_right":      "#6888A8",  # lighter dark slate
        "humerus_left":         "#586898",  # muted indigo
        "humerus_right":        "#7088B0",  # lighter muted indigo
        "hip_left":             "#B8A878",  # sandy bone
        "hip_right":            "#D0C098",  # lighter sandy bone
        "femur_left":           "#988868",  # dark sand
        "femur_right":          "#B0A888",  # lighter dark sand
        # --- Muscles (red-magenta family, anatomically intuitive) ---
        "gluteus_maximus_left":  "#C41E3A",  # cardinal red
        "gluteus_maximus_right": "#DC4458",  # lighter cardinal
        "gluteus_medius_left":   "#B02066",  # deep magenta
        "gluteus_medius_right":  "#CC4888",  # lighter magenta
        "gluteus_minimus_left":  "#9C2780",  # plum
        "gluteus_minimus_right": "#B84FA0",  # lighter plum
        "autochthon_left":       "#A03050",  # dark rose
        "autochthon_right":      "#C05070",  # lighter rose
        "iliopsoas_left":        "#8B2040",  # burgundy
        "iliopsoas_right":       "#AD4060",  # lighter burgundy
        # --- Background ---
        "inside_body_empty":     "#2A2A2A",
    }

    # Ribs: compact hue gradients within skeletal tonal family
    # Left ribs:  warm bone tones (hue 30-60, low saturation)
    # Right ribs: cool bone tones (hue 190-230, low saturation)
    for i in range(1, 13):
        h_l = 30 + (i - 1) * 30 / 11       # 30..60 warm
        l_l = 50 + (i - 1) * 20 / 11        # 50..70 lightness gradient
        palettes[f"rib_left_{i}"] = f"hsl({int(h_l)}, 40%, {int(l_l)}%)"
        h_r = 190 + (i - 1) * 40 / 11       # 190..230 cool
        l_r = 50 + (i - 1) * 20 / 11
        palettes[f"rib_right_{i}"] = f"hsl({int(h_r)}, 40%, {int(l_r)}%)"

    colormap = {}
    for idx, name in enumerate(class_names):
        if name in palettes:
            colormap[idx] = palettes[name]
        else:
            # Fallback: golden-ratio hue spacing for unknown classes
            h = (idx * 137.508) % 360
            colormap[idx] = f"hsl({int(h)}, 60%, 50%)"
    return colormap


# ---------------------------------------------------------------------------
# Surface extraction: only render exposed voxel faces (no internal faces)
# ---------------------------------------------------------------------------

def extract_surface_quads(binary_mask, world_min, voxel_size):
    """Extract only the outer surface faces of a binary voxel grid.

    For each occupied voxel, check its 6 neighbors. If a neighbor is empty
    (or out of bounds), that face is a surface face. Returns vertices and
    triangle indices for all surface faces.

    This eliminates internal faces that cause Plotly's transparency artifacts.
    """
    # Pad the mask with zeros on all sides so boundary checks are trivial
    padded = np.pad(binary_mask, 1, mode='constant', constant_values=0)

    # Find all occupied voxels in padded coords
    occ = np.argwhere(padded > 0)  # (N, 3)

    # 6 face directions and their quad vertex offsets
    # Each face is defined by: direction, 4 corner offsets from voxel origin
    # Voxel origin = lower corner of the voxel
    s = voxel_size[0]  # isotropic
    face_defs = [
        # (neighbor_offset, 4 quad corners relative to voxel origin)
        # +X face: neighbor at (1,0,0)
        (np.array([1, 0, 0]), np.array([[s,0,0],[s,s,0],[s,s,s],[s,0,s]])),
        # -X face: neighbor at (-1,0,0)
        (np.array([-1, 0, 0]), np.array([[0,0,0],[0,0,s],[0,s,s],[0,s,0]])),
        # +Y face: neighbor at (0,1,0)
        (np.array([0, 1, 0]), np.array([[0,s,0],[s,s,0],[s,s,s],[0,s,s]])),
        # -Y face: neighbor at (0,-1,0)
        (np.array([0, -1, 0]), np.array([[0,0,0],[0,0,s],[s,0,s],[s,0,0]])),
        # +Z face: neighbor at (0,0,1)
        (np.array([0, 0, 1]), np.array([[0,0,s],[s,0,s],[s,s,s],[0,s,s]])),
        # -Z face: neighbor at (0,0,-1)
        (np.array([0, 0, -1]), np.array([[0,0,0],[0,s,0],[s,s,0],[s,0,0]])),
    ]

    all_verts = []
    all_faces = []
    vert_count = 0

    for neighbor_off, quad_corners in face_defs:
        # Check which occupied voxels have an empty neighbor in this direction
        ni = occ + neighbor_off  # neighbor indices in padded space
        neighbor_vals = padded[ni[:, 0], ni[:, 1], ni[:, 2]]
        exposed = neighbor_vals == 0  # this face is on the surface

        if not np.any(exposed):
            continue

        exposed_voxels = occ[exposed]  # (M, 3) in padded coords
        M = len(exposed_voxels)

        # Convert padded coords back to original coords, then to world space
        # padded coords are offset by 1 from original
        origins = world_min + (exposed_voxels - 1).astype(np.float32) * s

        # Build quad vertices: (M, 4, 3)
        quad_v = origins[:, np.newaxis, :] + quad_corners[np.newaxis, :, :]
        quad_v = quad_v.reshape(-1, 3)  # (M*4, 3)

        # Build triangle indices for each quad (2 triangles per quad)
        base = np.arange(M) * 4 + vert_count
        tri1 = np.stack([base, base + 1, base + 2], axis=1)
        tri2 = np.stack([base, base + 2, base + 3], axis=1)
        tris = np.concatenate([tri1, tri2], axis=0)

        all_verts.append(quad_v)
        all_faces.append(tris)
        vert_count += M * 4

    if not all_verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return verts, faces


def extract_surface_quads_labeled(labels_3d, label_set, world_min, voxel_size):
    """Extract surface faces for a set of labels.

    A face is exposed if the neighbor voxel is NOT in label_set.
    This gives each organ its own clean surface.
    """
    mask = np.isin(labels_3d, list(label_set))

    # Subsample if too many surface faces expected
    n_occupied = mask.sum()
    if n_occupied == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)

    return extract_surface_quads(mask, world_min, voxel_size)


def subsample_mesh(verts, faces, max_tris):
    """Subsample a triangle mesh by randomly selecting triangles."""
    n_tris = len(faces)
    if n_tris <= max_tris:
        return verts, faces

    sel = np.random.choice(n_tris, max_tris, replace=False)
    sel_faces = faces[sel]

    # Remap vertex indices
    used_verts = np.unique(sel_faces)
    remap = np.zeros(len(verts), dtype=np.int64)
    remap[used_verts] = np.arange(len(used_verts))

    new_verts = verts[used_verts]
    new_faces = remap[sel_faces]
    return new_verts, new_faces


# ---------------------------------------------------------------------------
# Component 1: Voxelized full body
# ---------------------------------------------------------------------------

def create_voxel_body_trace(labels_3d, grid_world_min, voxel_size):
    """Render outer surface of all non-zero voxels."""
    binary = (labels_3d > 0).astype(np.uint8)
    verts, faces = extract_surface_quads(binary, grid_world_min, voxel_size)
    verts, faces = subsample_mesh(verts, faces, MAX_SURFACE_QUADS * 2)

    trace = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=BODY_COLOR,
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3, roughness=0.4),
        lightposition=dict(x=500, y=500, z=800),
        hoverinfo="skip",
        name="Voxelized Body",
    )
    return trace


# ---------------------------------------------------------------------------
# Component 2: Surface point cloud
# ---------------------------------------------------------------------------

def create_surface_pc_trace(sensor_pc):
    """Render surface point cloud as dark markers."""
    n = len(sensor_pc)
    if n > MAX_POINTS_PC:
        sel = np.random.choice(n, MAX_POINTS_PC, replace=False)
        pc = sensor_pc[sel]
    else:
        pc = sensor_pc

    trace = go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=PC_COLOR, opacity=1.0),
        hoverinfo="skip",
        name="Surface Point Cloud",
    )
    return trace


# ---------------------------------------------------------------------------
# Component 3: Internal organ labels
# ---------------------------------------------------------------------------

ORGAN_SYSTEM_MAP = {}
for _n in ["liver", "spleen", "kidney_left", "kidney_right", "stomach",
           "pancreas", "gallbladder", "urinary_bladder", "prostate", "heart",
           "brain", "thyroid_gland", "spinal_cord", "lung", "esophagus",
           "trachea", "small_bowel", "duodenum", "colon",
           "adrenal_gland_left", "adrenal_gland_right"]:
    ORGAN_SYSTEM_MAP[_n] = "Organs"
for _n in (["spine", "skull", "sternum", "costal_cartilages",
            "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
            "humerus_left", "humerus_right", "hip_left", "hip_right",
            "femur_left", "femur_right"]
           + [f"rib_left_{i}" for i in range(1, 13)]
           + [f"rib_right_{i}" for i in range(1, 13)]):
    ORGAN_SYSTEM_MAP[_n] = "Skeletal"
for _n in ["gluteus_maximus_left", "gluteus_maximus_right",
           "gluteus_medius_left", "gluteus_medius_right",
           "gluteus_minimus_left", "gluteus_minimus_right",
           "autochthon_left", "autochthon_right",
           "iliopsoas_left", "iliopsoas_right"]:
    ORGAN_SYSTEM_MAP[_n] = "Muscles"


def create_organ_label_traces(labels_3d, grid_world_min, voxel_size,
                               class_names, colormap):
    """Render each organ's outer surface as colored mesh."""
    unique_labels = np.unique(labels_3d)
    unique_labels = unique_labels[unique_labels > 0]

    max_tris_per_organ = max(2000, MAX_SURFACE_QUADS * 2 // max(len(unique_labels), 1))

    traces = []
    seen_groups = set()

    for lbl in unique_labels:
        lbl = int(lbl)
        verts, faces = extract_surface_quads_labeled(
            labels_3d, {lbl}, grid_world_min, voxel_size
        )
        if len(faces) == 0:
            continue

        verts, faces = subsample_mesh(verts, faces, max_tris_per_organ)

        color = colormap.get(lbl, "rgb(200,200,200)")
        organ_name = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
        legend_group = ORGAN_SYSTEM_MAP.get(organ_name, "Other")
        show_legend = legend_group not in seen_groups
        seen_groups.add(legend_group)

        trace = go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=1.0,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.9, specular=0.2, roughness=0.5),
            lightposition=dict(x=500, y=500, z=800),
            name=legend_group,
            legendgroup=legend_group,
            showlegend=show_legend,
            hovertext=organ_name,
            hoverinfo="text",
        )
        traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# Component 4: Organ labels (bottom) + Surface point cloud (top) with gap
# ---------------------------------------------------------------------------

GAP_CM = 15  # shift point cloud upward along +Z

def create_stacked_organ_pc_traces(labels_3d, sensor_pc, grid_world_min,
                                    voxel_size, class_names, colormap):
    """Organ labels in place, surface point cloud shifted up along +Z by GAP_CM."""
    organ_traces = create_organ_label_traces(
        labels_3d, grid_world_min, voxel_size, class_names, colormap
    )

    # Auto-detect units from body Z extent
    s = voxel_size[0]
    occupied = np.argwhere(labels_3d > 0)
    if len(occupied) == 0:
        return organ_traces
    z_extent = (occupied[:, 2].max() - occupied[:, 2].min() + 1) * s
    gap = GAP_CM * 10.0 if z_extent > 100 else GAP_CM / 100.0

    # Shift point cloud along +Z
    pc = sensor_pc.copy()
    pc[:, 1] += gap

    n = len(pc)
    if n > MAX_POINTS_PC:
        sel = np.random.choice(n, MAX_POINTS_PC, replace=False)
        pc = pc[sel]

    pc_trace = go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=PC_COLOR, opacity=1.0),
        hoverinfo="skip",
        name="Surface Point Cloud",
    )
    return organ_traces + [pc_trace]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def make_layout(title, show_legend=False):
    return go.Layout(
        title=dict(text=title, font=dict(size=18, family="Arial", color="#333"), x=0.5),
        width=FIG_WIDTH, height=FIG_HEIGHT,
        paper_bgcolor="white", plot_bgcolor="white",
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data", bgcolor="white",
        ),
        showlegend=show_legend,
        legend=dict(font=dict(size=10), itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.15)", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )


def save_figure(fig, output_dir, name, camera_presets):
    html_path = os.path.join(output_dir, f"{name}.html")
    fig.write_html(html_path)
    print(f"  HTML: {html_path}")
    for angle_name, camera in camera_presets.items():
        fig.update_layout(scene_camera=camera)
        png_path = os.path.join(output_dir, f"{name}_{angle_name}.png")
        fig.write_image(png_path, scale=PNG_SCALE)
        print(f"  PNG:  {png_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--no_png", action="store_true")
    args = parser.parse_args()

    np.random.seed(42)

    with open(INFO_PATH) as f:
        info = json.load(f)
    class_names = info["class_names"]
    colormap = build_organ_colormap(class_names)

    dirs = {
        "voxel_body": os.path.join(args.output_dir, "voxel_body"),
        "surface_pointcloud": os.path.join(args.output_dir, "surface_pointcloud"),
        "organ_labels": os.path.join(args.output_dir, "organ_labels"),
        "organ_pc_stacked": os.path.join(args.output_dir, "organ_pc_stacked"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    camera = CAMERA_PRESETS if not args.no_png else {}

    for i, filename in enumerate(args.samples):
        sample_name = filename.replace(".npz", "")
        path = os.path.join(args.data_dir, filename)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue

        print(f"\n[{i+1}/{len(args.samples)}] Processing {sample_name}...")
        data = np.load(path)
        labels = data["voxel_labels"]
        sensor_pc = data["sensor_pc"]
        grid_world_min = data["grid_world_min"]
        voxel_size = data["grid_voxel_size"]

        # Component 1: Voxelized body
        print("  Component 1: Voxelized body")
        trace_body = create_voxel_body_trace(labels, grid_world_min, voxel_size)
        fig1 = go.Figure(data=[trace_body], layout=make_layout(""))
        save_figure(fig1, dirs["voxel_body"], sample_name, camera)

        # Component 2: Surface point cloud
        print("  Component 2: Surface point cloud")
        trace_pc = create_surface_pc_trace(sensor_pc)
        fig2 = go.Figure(data=[trace_pc], layout=make_layout(""))
        save_figure(fig2, dirs["surface_pointcloud"], sample_name, camera)

        # Component 3: Organ labels
        print("  Component 3: Organ labels")
        organ_traces = create_organ_label_traces(
            labels, grid_world_min, voxel_size, class_names, colormap
        )
        fig3 = go.Figure(data=organ_traces, layout=make_layout("", show_legend=True))
        save_figure(fig3, dirs["organ_labels"], sample_name, camera)

        # Component 4: Organ labels (bottom) + point cloud (top) stacked
        print("  Component 4: Stacked organ + point cloud")
        stacked_traces = create_stacked_organ_pc_traces(
            labels, sensor_pc, grid_world_min, voxel_size, class_names, colormap
        )
        fig4 = go.Figure(data=stacked_traces, layout=make_layout("", show_legend=True))
        save_figure(fig4, dirs["organ_pc_stacked"], sample_name, camera)

    print(f"\nDone! All outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
