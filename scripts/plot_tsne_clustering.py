"""t-SNE @ epoch_48 with convex hulls + silhouette analysis per organ system."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples

EXPORT_DIR = "_VIS/embedding_export_021201"
OUT_PATH = "_VIS/trajectory_pca/tsne_epoch48_clustering.png"

SYSTEM_COLORS = {
    "skeletal": "#E74C3C", "muscular": "#3498DB", "digestive": "#2ECC71",
    "respiratory": "#9B59B6", "cardiovascular": "#E67E22", "urinary": "#1ABC9C",
    "nervous": "#F1C40F", "other": "#95A5A6",
}

data = np.load(f"{EXPORT_DIR}/embedding_data.npz")
with open(f"{EXPORT_DIR}/embedding_meta.json") as f:
    meta = json.load(f)

gdist = data["geodesic_dist_epoch_48"]
class_names = meta["class_names"]
class_to_system = meta["class_to_system"]
N = len(class_names)

system_labels = [class_to_system[str(i)] for i in range(N)]
system_ids = {s: i for i, s in enumerate(sorted(set(system_labels)))}
y = np.array([system_ids[s] for s in system_labels])

# Silhouette on geodesic distances
sil = silhouette_score(gdist, y, metric="precomputed")
sil_per = silhouette_samples(gdist, y, metric="precomputed")
print(f"Silhouette score (organ system): {sil:.4f}")

# t-SNE
tsne = TSNE(n_components=2, metric="precomputed", perplexity=20,
            random_state=42, init="random")
proj = tsne.fit_transform(gdist)
proj *= 0.95 / np.linalg.norm(proj, axis=1).max()

# Group by system
systems = {}
for i in range(N):
    systems.setdefault(system_labels[i], []).append(i)


def short_name(name):
    if name.startswith("rib_left_"):
        return "L" + name.split("_")[-1]
    if name.startswith("rib_right_"):
        return "R" + name.split("_")[-1]
    return (name.replace("_left", "(L)")
            .replace("_right", "(R)")
            .replace("_", " "))


# ---- Figure: 2 panels ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

# Panel 1: t-SNE scatter + convex hulls
for sys_name, indices in systems.items():
    c = SYSTEM_COLORS.get(sys_name, "#95A5A6")
    pts = proj[indices]

    for i in indices:
        ax1.scatter(proj[i, 0], proj[i, 1], color=c, s=35,
                    edgecolors="white", linewidths=0.3, zorder=3)
        ax1.annotate(
            short_name(class_names[i]), (proj[i, 0], proj[i, 1]),
            fontsize=3.5, ha="left", va="bottom",
            xytext=(2, 2), textcoords="offset points",
            color=c, fontweight="bold", zorder=5,
        )

    if len(indices) >= 3:
        try:
            hull = ConvexHull(pts)
            verts = np.append(hull.vertices, hull.vertices[0])
            ax1.fill(pts[verts, 0], pts[verts, 1],
                     alpha=0.08, color=c, zorder=0)
            ax1.plot(pts[verts, 0], pts[verts, 1],
                     color=c, linewidth=0.8, alpha=0.4, zorder=1)
        except Exception:
            pass

handles = [mpatches.Patch(color=c, label=s.capitalize())
           for s, c in SYSTEM_COLORS.items()]
ax1.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.85)
ax1.set_aspect("equal")
ax1.set_title(
    f"t-SNE @ Epoch 48 with System Hulls\nSilhouette = {sil:.3f}",
    fontsize=11, fontweight="bold",
)
ax1.set_xlabel("t-SNE 1", fontsize=9)
ax1.set_ylabel("t-SNE 2", fontsize=9)
ax1.tick_params(labelsize=7)
ax1.grid(True, alpha=0.12, linewidth=0.4)

# Panel 2: Per-class silhouette bar chart
sorted_sys = sorted(systems.keys(),
                    key=lambda s: np.mean(sil_per[systems[s]]))
bar_y, bar_colors, bar_vals, bar_labels = [], [], [], []
y_pos = 0
yticks, ytick_labels = [], []

for sys_name in sorted_sys:
    indices = systems[sys_name]
    idx_sorted = sorted(indices, key=lambda i: sil_per[i])
    c = SYSTEM_COLORS.get(sys_name, "#95A5A6")
    for i in idx_sorted:
        bar_y.append(y_pos)
        bar_colors.append(c)
        bar_vals.append(sil_per[i])
        bar_labels.append(short_name(class_names[i]))
        y_pos += 1
    yticks.append(y_pos - len(indices) / 2)
    ytick_labels.append(sys_name.capitalize())
    y_pos += 1  # gap between systems

ax2.barh(bar_y, bar_vals, color=bar_colors, height=0.8,
         edgecolor="white", linewidth=0.2)
ax2.axvline(x=0, color="black", linewidth=0.5)
ax2.axvline(x=sil, color="red", linewidth=1, linestyle="--",
            alpha=0.6, label=f"Mean = {sil:.3f}")

for yp, val, name in zip(bar_y, bar_vals, bar_labels):
    ha = "left" if val >= 0 else "right"
    off = 2 if val >= 0 else -2
    ax2.annotate(name, (val, yp), fontsize=3.5, ha=ha, va="center",
                 xytext=(off, 0), textcoords="offset points")

ax2.set_yticks(yticks)
ax2.set_yticklabels(ytick_labels, fontsize=7)
ax2.set_xlabel("Silhouette Coefficient", fontsize=9)
ax2.set_title("Per-Class Silhouette (by Organ System)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=7)
ax2.tick_params(labelsize=6)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_PATH}")
