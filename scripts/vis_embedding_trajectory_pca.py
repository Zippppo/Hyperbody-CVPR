"""
Visualize per-class embedding trajectories in a Poincare disk.

Creates one PNG per organ system, with all other classes shown as faint
background context inside a Poincare ball. Each class trajectory runs
from epoch_3 to epoch_48, with temporal gradient coloring.

Usage:
    python scripts/vis_embedding_trajectory_pca.py \
        --export-dir _VIS/embedding_export_021201 \
        --output-dir _VIS/trajectory_pca
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba, to_rgb
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# Style: clean, publication-ready (matching scaling-law figure)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DISK_BG = "#F8F9FA"
DISK_EDGE = "#4e4e4e"
GRID_COLOR = "#D5DBDB"
BG_TRAJ_COLOR = "#D0D3D4"

# Tableau-10 colorblind-friendly palette
SYSTEM_COLORS = {
    "skeletal": "#e15759",
    "muscular": "#4e79a7",
    "digestive": "#59a14f",
    "respiratory": "#b07aa1",
    "cardiovascular": "#f28e2b",
    "urinary": "#76b7b2",
    "nervous": "#edc948",
    "other": "#bab0ac",
}

SYSTEM_DISPLAY = {
    "skeletal": "Skeletal",
    "muscular": "Muscular",
    "digestive": "Digestive",
    "respiratory": "Respiratory",
    "cardiovascular": "Cardiovascular",
    "urinary": "Urinary",
    "nervous": "Nervous",
    "other": "Other",
}

# ---------------------------------------------------------------------------
# Data loading & PCA (unchanged)
# ---------------------------------------------------------------------------

def load_data(export_dir: str):
    data = np.load(f"{export_dir}/embedding_data.npz")
    with open(f"{export_dir}/embedding_meta.json") as f:
        meta = json.load(f)
    return data, meta


def compute_joint_pca(data, labels):
    """Tangent Joint PCA: fixed coordinate system across all epochs."""
    tangents = [data[f"tangent_{l}"] for l in labels]
    all_tang = np.concatenate(tangents, axis=0)
    mean = all_tang.mean(axis=0)
    _, _, Vt = np.linalg.svd(all_tang - mean, full_matrices=False)
    basis = Vt[:2]  # [2, D]
    projs = [(t - mean) @ basis.T for t in tangents]
    # Normalize into the unit disk with margin
    all_pts = np.concatenate(projs, axis=0)
    max_norm = np.linalg.norm(all_pts, axis=1).max()
    if max_norm > 1e-10:
        scale = 0.92 / max_norm
        projs = [p * scale for p in projs]
    return projs


def group_by_system(meta):
    """Return {system_name: [class_indices]}."""
    groups = {}
    for idx_str, sys_name in meta["class_to_system"].items():
        groups.setdefault(sys_name, []).append(int(idx_str))
    for v in groups.values():
        v.sort()
    return groups

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _shorten_name(name: str) -> str:
    """Shorten anatomical class names for compact labels."""
    if name.startswith("rib_left_"):
        return f"L{name.split('_')[-1]}"
    if name.startswith("rib_right_"):
        return f"R{name.split('_')[-1]}"
    name = name.replace("_left", " (L)").replace("_right", " (R)")
    return name.replace("_", " ")


def _make_smooth(xs, ys, n_interp=120):
    """Cubic spline interpolation for smoother curves."""
    t = np.arange(len(xs))
    if len(xs) < 4:
        return np.array(xs), np.array(ys)
    t_new = np.linspace(t[0], t[-1], n_interp)
    try:
        sx = make_interp_spline(t, xs, k=3)(t_new)
        sy = make_interp_spline(t, ys, k=3)(t_new)
    except Exception:
        return np.array(xs), np.array(ys)
    return sx, sy


def _generate_class_colors(base_hex: str, n: int):
    """Generate *n* distinguishable shades around a base hue."""
    if n <= 1:
        return [base_hex]
    import colorsys
    base_rgb = to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    colors = []
    for i in range(n):
        frac = i / (n - 1)
        li = max(0.28, min(0.62, l - 0.18 + 0.36 * frac))
        si = max(0.45, min(1.0, s - 0.1 + 0.2 * frac))
        hi = h + 0.03 * (frac - 0.5)
        rgb = colorsys.hls_to_rgb(hi % 1.0, li, si)
        colors.append(rgb)
    return colors


def draw_poincare_disk(ax):
    """Draw the Poincare disk boundary and hyperbolic-style grid."""
    # Filled disk background
    disk = plt.Circle((0, 0), 1.0, facecolor=DISK_BG, edgecolor=DISK_EDGE,
                       linewidth=1.8, zorder=0)
    ax.add_patch(disk)
    # Concentric geodesic circles
    for r in (0.25, 0.5, 0.75):
        ax.add_patch(plt.Circle((0, 0), r, fill=False,
                                edgecolor=GRID_COLOR, linewidth=0.35,
                                linestyle="-", zorder=0.5))
    # Cross-hair axes
    for angle in (0, np.pi / 2):
        dx, dy = np.cos(angle), np.sin(angle)
        ax.plot([-dx, dx], [-dy, dy], color=GRID_COLOR, lw=0.35, zorder=0.5)
    # Radius labels
    for r in (0.25, 0.5, 0.75):
        ax.text(r + 0.015, -0.025, f"{r:.2f}", fontsize=5.5, color="#AAB7B8",
                ha="left", va="top", zorder=0.6)

def _draw_gradient_line(ax, xs, ys, color, lw=1.6, zorder=2):
    """Draw a trajectory with temporal gradient (faint -> saturated)."""
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    base = to_rgba(color)
    alphas = np.linspace(0.25, 0.95, n)
    seg_colors = [(base[0], base[1], base[2], a) for a in alphas]
    lc = LineCollection(segs, colors=seg_colors, linewidths=lw,
                        capstyle="round", joinstyle="round", zorder=zorder)
    ax.add_collection(lc)


def draw_snapshot_plot(
    ax, projs, epoch_idx, system_name, fg_indices, all_indices,
    class_names, epoch_label: str, is_end: bool, show_labels: bool = True,
):
    """Draw a snapshot of class positions at a specific epoch."""
    bg_indices = [i for i in all_indices if i not in set(fg_indices)]

    draw_poincare_disk(ax)

    marker = "^" if is_end else "o"

    # Background classes
    for cls_id in bg_indices:
        x = projs[epoch_idx][cls_id, 0]
        y = projs[epoch_idx][cls_id, 1]
        bg_fc = BG_TRAJ_COLOR if is_end else "none"
        ax.scatter(x, y, s=12, marker=marker,
                   facecolors=bg_fc, edgecolors=BG_TRAJ_COLOR,
                   linewidths=0.4, zorder=1, alpha=0.45)

    # Foreground classes
    base_color = SYSTEM_COLORS.get(system_name, "#707B7C")
    n_fg = len(fg_indices)
    colors = _generate_class_colors(base_color, n_fg)

    texts_xy, texts_str, texts_color = [], [], []

    for i, cls_id in enumerate(fg_indices):
        c = colors[i]
        x = projs[epoch_idx][cls_id, 0]
        y = projs[epoch_idx][cls_id, 1]

        fc = c if is_end else "none"
        ec = "white" if is_end else c
        ms = 40 if is_end else 36
        lw = 0.5 if is_end else 0.8

        ax.scatter(x, y, s=ms, marker=marker,
                   facecolors=fc, edgecolors=ec,
                   linewidths=lw, zorder=4)

        name = _shorten_name(class_names[cls_id])
        texts_xy.append((x, y))
        texts_str.append(name)
        texts_color.append(c)

    if show_labels:
        _place_labels(ax, texts_xy, texts_str, texts_color)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")

    display = SYSTEM_DISPLAY.get(system_name, system_name)
    ax.set_title(f"{display} \u2014 {epoch_label}  ({n_fg} classes)",
                 fontsize=13, fontweight="bold", color="#333333", pad=12)


def draw_system_plot(
    ax, projs, system_name, fg_indices, all_indices, class_names, class_to_system,
    epoch_start_label: str, epoch_end_label: str,
):
    """Draw trajectories for one organ system inside a Poincare disk."""
    n_epochs = len(projs)
    bg_indices = [i for i in all_indices if i not in fg_indices]
    fg_set = set(fg_indices)

    # -- Poincare disk frame --
    draw_poincare_disk(ax)

    # -- Background trajectories (all other classes) --
    for cls_id in bg_indices:
        xs = [projs[t][cls_id, 0] for t in range(n_epochs)]
        ys = [projs[t][cls_id, 1] for t in range(n_epochs)]
        sx, sy = _make_smooth(xs, ys, n_interp=80)
        ax.plot(sx, sy, color=BG_TRAJ_COLOR, linewidth=0.3, alpha=0.45, zorder=1)

    # -- Foreground trajectories --
    base_color = SYSTEM_COLORS.get(system_name, "#707B7C")
    n_fg = len(fg_indices)
    colors = _generate_class_colors(base_color, n_fg)

    texts_xy, texts_str, texts_color = [], [], []

    for i, cls_id in enumerate(fg_indices):
        xs = [projs[t][cls_id, 0] for t in range(n_epochs)]
        ys = [projs[t][cls_id, 1] for t in range(n_epochs)]
        c = colors[i]
        sx, sy = _make_smooth(xs, ys)

        # Gradient trajectory
        _draw_gradient_line(ax, sx, sy, c, lw=1.6, zorder=2)

        # Start marker: small open circle
        ax.scatter(xs[0], ys[0], s=50, facecolors="none", edgecolors=c,
                   linewidths=1.0, zorder=4)

        # End marker: filled triangle + arrowhead
        ax.scatter(xs[-1], ys[-1], s=60, marker="^", facecolors=c, edgecolors="white",
                   linewidths=0.5, zorder=4)
        # Small arrow showing direction at the end
        if len(sx) >= 2:
            dx = sx[-1] - sx[-6] if len(sx) > 6 else sx[-1] - sx[-2]
            dy = sy[-1] - sy[-6] if len(sy) > 6 else sy[-1] - sy[-2]
            norm = np.hypot(dx, dy)
            if norm > 1e-8:
                dx, dy = dx / norm * 0.025, dy / norm * 0.025
                ax.annotate("", xy=(xs[-1] + dx, ys[-1] + dy),
                            xytext=(xs[-1] - dx, ys[-1] - dy),
                            arrowprops=dict(arrowstyle="-|>", color=c,
                                            lw=1.2, mutation_scale=7),
                            zorder=5)

        # Collect label info for adjustText
        name = _shorten_name(class_names[cls_id])
        texts_xy.append((xs[-1], ys[-1]))
        texts_str.append(name)
        texts_color.append(c)

    # -- Labels with overlap avoidance --
    _place_labels(ax, texts_xy, texts_str, texts_color)

    # -- Axes cleanup --
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- Title --
    display = SYSTEM_DISPLAY.get(system_name, system_name)
    ax.set_title(f"{display}  ({n_fg} classes)", fontsize=13,
                 fontweight="bold", color="#333333", pad=12)

    # -- Epoch legend (bottom-left, inside disk) --
    ep_s = epoch_start_label.replace("epoch_", "Epoch ")
    ep_e = epoch_end_label.replace("epoch_", "Epoch ")
    ax.text(0.03, 0.03, f"○ {ep_s}   ▲ {ep_e}",
            transform=ax.transAxes, fontsize=7, color="#5D6D7E",
            va="bottom")

def _place_labels(ax, xys, names, colors):
    """Place endpoint labels, using adjustText if available."""
    try:
        from adjustText import adjust_text
        txt_objs = []
        for (x, y), name, c in zip(xys, names, colors):
            t = ax.text(x, y, name, fontsize=6, color=c,
                        fontweight="medium", ha="center", va="center",
                        zorder=6)
            txt_objs.append(t)
        adjust_text(txt_objs, ax=ax, force_text=(0.4, 0.4),
                    force_points=(0.3, 0.3), expand=(1.3, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#AAB7B8",
                                    lw=0.4, alpha=0.6))
    except ImportError:
        # Fallback: radial offset from disk center
        for (x, y), name, c in zip(xys, names, colors):
            r = np.hypot(x, y)
            if r > 1e-6:
                ox, oy = x / r * 0.04, y / r * 0.04
            else:
                ox, oy = 0.04, 0.04
            ax.annotate(name, (x, y), xytext=(x + ox, y + oy),
                        fontsize=6, color=c,
                        fontweight="medium", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color="#AAB7B8",
                                        lw=0.3, alpha=0.5),
                        zorder=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embedding trajectory visualization")
    parser.add_argument("--export-dir", type=str,
                        default="_VIS/embedding_export_021201")
    parser.add_argument("--output-dir", type=str,
                        default="_VIS/trajectory_pca")
    args = parser.parse_args()

    data, meta = load_data(args.export_dir)
    labels = [l for l in meta["labels"] if l != "best"]
    class_names = meta["class_names"]
    class_to_system = meta["class_to_system"]
    n_classes = len(class_names)
    all_indices = list(range(n_classes))

    projs = compute_joint_pca(data, labels)
    groups = group_by_system(meta)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sorted_systems = sorted(groups.items(), key=lambda x: -len(x[1]))

    for sys_name, fg_indices in sorted_systems:
        fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
        draw_system_plot(
            ax, projs, sys_name, fg_indices, all_indices,
            class_names, class_to_system,
            epoch_start_label=labels[0], epoch_end_label=labels[-1],
        )
        fname = out / f"trajectory_{sys_name}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.15)
        plt.close(fig)
        print(f"Saved {fname}")

    print(f"Done. {len(groups)} systems saved to {out}/")


if __name__ == "__main__":
    main()