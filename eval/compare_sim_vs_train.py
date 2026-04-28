"""
Compare simulation point cloud (haoxin_sensor_pc) vs training data.
All strategies center-align first, then try different scale/flip combos.
Also compare against multiple training samples to find best match.

Output: interactive HTML with dropdown to switch alignment strategies.
"""
import glob
import numpy as np
import plotly.graph_objects as go


# ── Load data ──
sim_pc = np.load("haoxin_sensor_pc.npz")["sensor_pc"]

train_files = sorted(glob.glob("Dataset/voxel_data/BDMAP_*.npz"))[:30]
train_pcs = []
train_names = []
for f in train_files:
    d = np.load(f)
    train_pcs.append(d["sensor_pc"])
    train_names.append(f.split("/")[-1].replace(".npz", ""))


def center_pc(pc):
    """Center point cloud at origin using bounding box center."""
    ctr = (pc.max(0) + pc.min(0)) / 2
    return pc - ctr, ctr


def pc_range(pc):
    return pc.max(0) - pc.min(0)


# ── Stats ──
sim_rng = pc_range(sim_pc)
train_ranges = np.array([pc_range(pc) for pc in train_pcs])
train_mean_rng = train_ranges.mean(axis=0)

# Scale: match sim range to training mean range
scale_uniform = np.median(train_mean_rng / sim_rng)

print(f"Sim range:        X={sim_rng[0]:.1f}  Y={sim_rng[1]:.1f}  Z={sim_rng[2]:.1f}")
print(f"Train mean range: X={train_mean_rng[0]:.1f}  Y={train_mean_rng[1]:.1f}  Z={train_mean_rng[2]:.1f}")
print(f"Uniform scale factor: {scale_uniform:.4f}")

# ── Center everything ──
sim_c, _ = center_pc(sim_pc)

# Pick 3 diverse training samples (small, medium, large Z range)
z_ranges = train_ranges[:, 2]
idx_small = np.argmin(z_ranges)
idx_large = np.argmax(z_ranges)
idx_mid = np.argsort(z_ranges)[len(z_ranges) // 2]
ref_indices = [idx_small, idx_mid, idx_large]

ref_pcs_c = {}
for idx in ref_indices:
    pc_c, _ = center_pc(train_pcs[idx])
    ref_pcs_c[train_names[idx]] = pc_c

# ── Define alignment strategies for simulation ──
# Confirmed calibration: scale=0.8 around centroid, then Y+200mm
strategies = {}

# 1. Raw centered (no calibration)
strategies["Centered only (raw)"] = sim_c.copy()

# 2. Final calibration: scale 0.8 + Y+200mm
sim_scaled = sim_c * 0.8
sim_final = sim_scaled.copy()
sim_final[:, 1] += 200.0
strategies["Scale 0.8 + Y+200mm (final)"] = sim_final

# ── Build HTML ──
# Use the medium-range training sample as default reference
default_ref_name = train_names[idx_mid]
default_ref_c = ref_pcs_c[default_ref_name]

# Subsample
max_pts = 15000


def subsample(pc, n=max_pts):
    step = max(1, len(pc) // n)
    return pc[::step]


traces = []
strategy_names = list(strategies.keys())

for i, (s_label, sim_t) in enumerate(strategies.items()):
    sim_s = subsample(sim_t)

    # Show all 3 training refs for each strategy
    for j, (ref_name, ref_c) in enumerate(ref_pcs_c.items()):
        ref_s = subsample(ref_c)
        colors = ["dodgerblue", "limegreen", "orange"]
        traces.append(go.Scatter3d(
            x=ref_s[:, 0], y=ref_s[:, 1], z=ref_s[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=colors[j], opacity=0.4),
            name=ref_name,
            visible=(i == 0),
            legendgroup=ref_name,
        ))

    # Simulation (red, larger)
    traces.append(go.Scatter3d(
        x=sim_s[:, 0], y=sim_s[:, 1], z=sim_s[:, 2],
        mode="markers",
        marker=dict(size=1.8, color="red", opacity=0.8),
        name=f"Sim ({s_label})",
        visible=(i == 0),
        legendgroup="sim",
    ))

n_refs = len(ref_pcs_c)
traces_per_strategy = n_refs + 1  # refs + sim

buttons = []
for i, s_label in enumerate(strategy_names):
    vis = [False] * len(traces)
    base = i * traces_per_strategy
    for k in range(traces_per_strategy):
        vis[base + k] = True
    buttons.append(dict(label=s_label, method="update", args=[{"visible": vis}]))

fig = go.Figure(data=traces)
fig.update_layout(
    updatemenus=[dict(
        active=0, buttons=buttons, direction="down",
        showactive=True, x=0.02, xanchor="left", y=1.15, yanchor="top",
    )],
    title=(
        "Sim (red) vs Training samples (blue/green/orange) — all centered at origin<br>"
        "Switch dropdown to compare scale factors"
    ),
    width=1300, height=950,
    scene=dict(
        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        aspectmode="data",
    ),
    showlegend=True,
)

out_path = "eval/pred/single/sim_vs_train_comparison.html"
fig.write_html(out_path)
print(f"\nSaved to {out_path}")
