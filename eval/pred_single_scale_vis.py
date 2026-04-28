"""
Visualize zrk point cloud at different scale factors overlaid with BDMAP_00000001.
Each scale produces a separate trace; use dropdown to switch.
"""
import numpy as np
import plotly.graph_objects as go

ref_data = np.load("Dataset/voxel_data/BDMAP_00000001.npz")
ref_pc = ref_data["sensor_pc"]
zrk_pc = np.load("zrk_ponitcloud.npz")["sensor_pc"]

# Center both point clouds at origin for fair comparison
ref_center = (ref_pc.max(axis=0) + ref_pc.min(axis=0)) / 2
zrk_center = (zrk_pc.max(axis=0) + zrk_pc.min(axis=0)) / 2
ref_centered = ref_pc - ref_center
zrk_centered = zrk_pc - zrk_center

# Subsample ref for performance
step_ref = max(1, len(ref_centered) // 15000)
ref_sub = ref_centered[::step_ref]

scales = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

traces = []
for i, s in enumerate(scales):
    zrk_scaled = zrk_centered * s
    step_zrk = max(1, len(zrk_scaled) // 15000)
    zrk_sub = zrk_scaled[::step_zrk]

    # Reference trace (blue)
    traces.append(go.Scatter3d(
        x=ref_sub[:, 0], y=ref_sub[:, 1], z=ref_sub[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="dodgerblue", opacity=0.5),
        name="BDMAP_00000001",
        visible=(i == 0),
    ))
    # zrk trace (red)
    traces.append(go.Scatter3d(
        x=zrk_sub[:, 0], y=zrk_sub[:, 1], z=zrk_sub[:, 2],
        mode="markers",
        marker=dict(size=1.5, color="red", opacity=0.7),
        name="zrk x%.2f" % s,
        visible=(i == 0),
    ))

# Dropdown buttons
buttons = []
for i, s in enumerate(scales):
    vis = [False] * len(traces)
    vis[i * 2] = True      # ref
    vis[i * 2 + 1] = True  # zrk
    buttons.append(dict(
        label="scale=%.2f" % s,
        method="update",
        args=[{"visible": vis}],
    ))

fig = go.Figure(data=traces)
fig.update_layout(
    updatemenus=[dict(
        active=0, buttons=buttons, direction="down",
        showactive=True, x=0.02, xanchor="left", y=1.15, yanchor="top",
    )],
    title="Blue=BDMAP_00000001 (head~knee), Red=zrk (neck~thigh, scaled)<br>Select scale factor",
    width=1200, height=900,
    scene=dict(
        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        aspectmode="data",
    ),
    showlegend=True,
)
fig.write_html("eval/pred/single/zrk_scale_comparison.html")
print("Saved to eval/pred/single/zrk_scale_comparison.html")
