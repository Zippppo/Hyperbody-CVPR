"""
Visualize the body surface from voxel data using plotly.
Extracts surface voxels (non-zero voxels adjacent to zero/boundary)
and renders them with semantic class colors.
"""

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

# Load sample data
data = np.load("/home/comp/25481568/code/HyperBody/Dataset/voxel_data/BDMAP_00000001.npz")
voxel_labels = data["voxel_labels"]  # (D, H, W), uint8
grid_world_min = data["grid_world_min"]  # (3,)
grid_voxel_size = data["grid_voxel_size"]  # (3,)

print(f"voxel_labels shape: {voxel_labels.shape}, unique labels: {np.unique(voxel_labels).shape[0]}")

# --- Extract surface voxels ---
# A surface voxel is a non-zero voxel that has at least one zero (air) neighbor
occupied = (voxel_labels > 0).astype(np.uint8)

# Dilate the air region by 1 voxel, then AND with occupied to get surface
air = (voxel_labels == 0).astype(np.uint8)
dilated_air = ndimage.binary_dilation(air, structure=ndimage.generate_binary_structure(3, 1))
surface_mask = occupied & dilated_air.astype(np.uint8)

surface_indices = np.argwhere(surface_mask)  # (N, 3) in voxel coords (D, H, W)
surface_labels = voxel_labels[surface_mask.astype(bool)]

print(f"Total occupied voxels: {occupied.sum()}")
print(f"Surface voxels: {surface_indices.shape[0]}")
print(f"Surface unique labels: {np.unique(surface_labels).shape[0]}")

# Convert voxel indices to world coordinates
# voxel_indices are (D, H, W), world_min is (D, H, W) aligned
world_coords = surface_indices.astype(np.float32) * grid_voxel_size + grid_world_min

# --- Build a color map for semantic classes ---
unique_labels = np.unique(surface_labels)
num_labels = len(unique_labels)

# Use a perceptually distinct colormap
# Generate colors using HSV for good separation
colors_rgb = []
for i, label in enumerate(unique_labels):
    hue = i / num_labels
    # HSV to RGB
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
    colors_rgb.append((int(r * 255), int(g * 255), int(b * 255)))

label_to_color = {label: colors_rgb[i] for i, label in enumerate(unique_labels)}

# Map each surface voxel to its color
point_colors = np.array([label_to_color[l] for l in surface_labels])
color_strings = [f"rgb({r},{g},{b})" for r, g, b in point_colors]

# --- Create plotly figure ---
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=world_coords[:, 2],  # W -> X
    y=world_coords[:, 1],  # H -> Y
    z=-world_coords[:, 0], # D -> Z (negate so head is up)
    mode="markers",
    marker=dict(
        size=1.8,
        color=color_strings,
        opacity=0.95,
    ),
    hovertemplate=(
        "X: %{x:.1f}<br>"
        "Y: %{y:.1f}<br>"
        "Z: %{z:.1f}<br>"
        "<extra></extra>"
    ),
))

fig.update_layout(
    title=dict(
        text="Human Body Surface — BDMAP_00000001",
        font=dict(size=20, color="#333"),
        x=0.5,
    ),
    scene=dict(
        xaxis=dict(title="X (W)", backgroundcolor="#f0f0f0", gridcolor="white"),
        yaxis=dict(title="Y (H)", backgroundcolor="#f0f0f0", gridcolor="white"),
        zaxis=dict(title="Z (D)", backgroundcolor="#f0f0f0", gridcolor="white"),
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.5, y=0.3, z=0.3),
            up=dict(x=0, y=0, z=1),
        ),
    ),
    paper_bgcolor="#fafafa",
    plot_bgcolor="#fafafa",
    width=1000,
    height=800,
    margin=dict(l=20, r=20, t=60, b=20),
)

output_path = "/home/comp/25481568/code/HyperBody/docs/visualizations/body_surface_BDMAP_00000001.html"
fig.write_html(output_path)
print(f"Saved to {output_path}")
