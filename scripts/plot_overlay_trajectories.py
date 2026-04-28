"""Plot all class trajectories overlaid in a Poincare disk, colored by organ system.

Generates:
  1. overlay_all_trajectories.png        -- all trajectories, no labels
  2. overlay_<system>.png (per system)   -- highlighted system with 2x labels
  3. overlay_all_trajectories_legend.png -- standalone legend

Usage:
    python scripts/plot_overlay_trajectories.py \
        --export-dir _VIS/embedding_export_021201 \
        --output-dir _VIS/trajectory_pca
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba, to_rgb
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# Style: clean, publication-ready (matching per-system figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DISK_BG = "#F8F9FA"
DISK_EDGE = "#4e4e4e"
GRID_COLOR = "#D5DBDB"

# Tableau-10 colorblind-friendly palette (8 groups)
RIB_LEFT = list(range(23, 35))
RIB_RIGHT = list(range(35, 47))

CLASS_GROUPS = [
    ("Axial",    [47, 22]),
    ("Thorax",   [48, 49] + RIB_LEFT + RIB_RIGHT),
    ("Append.",  [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]),
    ("Muscles",  [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]),
    ("Digest.",  [15, 5, 17, 18, 19, 1, 7, 6]),
    ("Resp.",    [16, 14]),
    ("Urogen.",  [3, 4, 8, 9, 12, 20, 21, 2]),
    ("Cv.",      [10]),
    ("Ns.",      [11, 13]),
]

GROUP_COLORS = {
    "Axial":    "#e15759",
    "Thorax":   "#4e79a7",
    "Append.":  "#59a14f",
    "Muscles":  "#b07aa1",
    "Digest.":  "#f28e2b",
    "Resp.":    "#76b7b2",
    "Urogen.":  "#9c755f",
    "Cv.":      "#d63d5e",
    "Ns.":      "#ff2d2d",
}

CLASS0_COLOR = "#bab0ac"  # neutral color for class 0 in "all" figure

# ---------------------------------------------------------------------------
# Shared trajectory style constants
# ---------------------------------------------------------------------------
TRAJ_LW = 3.5          # line width for trajectories
MARKER_START_S = 120    # start marker size (open circle)
MARKER_END_S = 200      # end marker size (filled triangle)
MARKER_START_LW = 1.2   # start marker edge width
MARKER_END_LW = 0.6     # end marker edge width
ARROW_LW = 1.4          # arrow line width
ARROW_SCALE = 8         # arrow head mutation scale
ARROW_OFFSET = 0.025    # arrow tip offset from endpoint

BG_ALPHA_RANGE = (0.06, 0.18)   # faded background trajectory alpha range
FG_ALPHA_RANGE = (0.20, 0.85)   # foreground trajectory alpha range

# ---------------------------------------------------------------------------
# Data loading & PCA
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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _shorten_name(name: str) -> str:
    if name.startswith("rib_left_"):
        return f"L{name.split('_')[-1]}"
    if name.startswith("rib_right_"):
        return f"R{name.split('_')[-1]}"
    name = name.replace("_left", " (L)").replace("_right", " (R)")
    return name.replace("_", " ")


def _make_smooth(xs, ys, n_interp=120):
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


def draw_poincare_disk(ax):
    """Draw the Poincare disk boundary and hyperbolic-style grid."""
    disk = plt.Circle((0, 0), 1.0, facecolor=DISK_BG, edgecolor=DISK_EDGE,
                       linewidth=1.8, zorder=0)
    ax.add_patch(disk)
    for r in (0.25, 0.5, 0.75):
        ax.add_patch(plt.Circle((0, 0), r, fill=False,
                                edgecolor=GRID_COLOR, linewidth=0.35,
                                linestyle="-", zorder=0.5))
    for angle in (0, np.pi / 2):
        dx, dy = np.cos(angle), np.sin(angle)
        ax.plot([-dx, dx], [-dy, dy], color=GRID_COLOR, lw=0.35, zorder=0.5)
    for r in (0.25, 0.5, 0.75):
        ax.text(r + 0.015, -0.025, f"{r:.2f}", fontsize=7, color="#AAB7B8",
                ha="left", va="top", zorder=0.6)


def _draw_gradient_line(ax, xs, ys, color, lw=1.0, zorder=2,
                        alpha_range=(0.20, 0.85)):
    """Draw a trajectory with temporal gradient (faint -> saturated)."""
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    base = to_rgba(color)
    alphas = np.linspace(alpha_range[0], alpha_range[1], n)
    seg_colors = [(base[0], base[1], base[2], a) for a in alphas]
    lc = LineCollection(segs, colors=seg_colors, linewidths=lw,
                        capstyle="round", joinstyle="round", zorder=zorder)
    ax.add_collection(lc)


def _place_labels(ax, xys, names, colors, label_fontsize=6):
    """Place endpoint labels, using adjustText if available."""
    try:
        from adjustText import adjust_text
        txt_objs = []
        for (x, y), name, c in zip(xys, names, colors):
            t = ax.text(x, y, name, fontsize=label_fontsize, color=c,
                        fontweight="medium", ha="center", va="center",
                        zorder=6)
            txt_objs.append(t)
        adjust_text(txt_objs, ax=ax, force_text=(0.4, 0.4),
                    force_points=(0.3, 0.3), expand=(1.3, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#AAB7B8",
                                    lw=0.3, alpha=0.5))
    except ImportError:
        for (x, y), name, c in zip(xys, names, colors):
            r = np.hypot(x, y)
            if r > 1e-6:
                ox, oy = x / r * 0.04, y / r * 0.04
            else:
                ox, oy = 0.04, 0.04
            ax.annotate(name, (x, y), xytext=(x + ox, y + oy),
                        fontsize=label_fontsize, color=c,
                        fontweight="medium", ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color="#AAB7B8",
                                        lw=0.3, alpha=0.5),
                        zorder=6)


def _draw_trajectory(ax, xs, ys, sx, sy, color, lw=TRAJ_LW,
                     alpha_range=FG_ALPHA_RANGE, draw_arrow=True):
    """Draw one trajectory: gradient line + start/end markers + arrow."""
    _draw_gradient_line(ax, sx, sy, color, lw=lw, zorder=2,
                        alpha_range=alpha_range)
    ax.scatter(xs[0], ys[0], s=MARKER_START_S, marker="o",
               facecolors="none", edgecolors=color,
               linewidths=MARKER_START_LW, zorder=4)
    ax.scatter(xs[-1], ys[-1], s=MARKER_END_S, marker="^",
               facecolors=color, edgecolors="white",
               linewidths=MARKER_END_LW, zorder=4)
    if draw_arrow and len(sx) >= 2:
        ddx = sx[-1] - sx[-6] if len(sx) > 6 else sx[-1] - sx[-2]
        ddy = sy[-1] - sy[-6] if len(sy) > 6 else sy[-1] - sy[-2]
        norm = np.hypot(ddx, ddy)
        if norm > 1e-8:
            ddx, ddy = ddx / norm * ARROW_OFFSET, ddy / norm * ARROW_OFFSET
            ax.annotate("", xy=(xs[-1] + ddx, ys[-1] + ddy),
                        xytext=(xs[-1] - ddx, ys[-1] - ddy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=ARROW_LW,
                                        mutation_scale=ARROW_SCALE),
                        zorder=5)


def _draw_trajectory_bg(ax, xs, ys, sx, sy):
    """Draw one trajectory as faded gray background."""
    _draw_gradient_line(ax, sx, sy, "#AAAAAA", lw=TRAJ_LW * 0.6, zorder=1,
                        alpha_range=BG_ALPHA_RANGE)
    ax.scatter(xs[0], ys[0], s=MARKER_START_S * 0.5, marker="o",
               facecolors="none", edgecolors="#CCCCCC",
               linewidths=0.4, zorder=1)
    ax.scatter(xs[-1], ys[-1], s=MARKER_END_S * 0.5, marker="^",
               facecolors="#CCCCCC", edgecolors="white",
               linewidths=0.3, zorder=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay all embedding trajectories in a Poincare disk")
    parser.add_argument("--export-dir", type=str,
                        default="_VIS/embedding_export_021201")
    parser.add_argument("--output-dir", type=str,
                        default="_VIS/trajectory_pca")
    args = parser.parse_args()

    data, meta = load_data(args.export_dir)
    labels = [l for l in meta["labels"] if l != "best"]
    class_names = meta["class_names"]
    n_classes = len(class_names)

    projs = compute_joint_pca(data, labels)
    n_epochs = len(projs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build cls_id -> group_name mapping
    cls_to_group = {}
    for grp_name, cls_ids in CLASS_GROUPS:
        for cid in cls_ids:
            cls_to_group[cid] = grp_name

    # Pre-compute trajectory data for all grouped classes (no class 0)
    traj_data = []  # [(grp_name, cls_id, xs, ys, sx, sy, name)]
    for grp_name, cls_ids in CLASS_GROUPS:
        for cls_id in cls_ids:
            xs = [projs[t][cls_id, 0] for t in range(n_epochs)]
            ys = [projs[t][cls_id, 1] for t in range(n_epochs)]
            sx, sy = _make_smooth(xs, ys)
            name = _shorten_name(class_names[cls_id])
            traj_data.append((grp_name, cls_id, xs, ys, sx, sy, name))

    # Pre-compute class 0 trajectory
    cls0_xs = [projs[t][0, 0] for t in range(n_epochs)]
    cls0_ys = [projs[t][0, 1] for t in range(n_epochs)]
    cls0_sx, cls0_sy = _make_smooth(cls0_xs, cls0_ys)

    # ================================================================
    # 1. All-trajectories figure: trajectories only, NO labels
    #    Each trajectory colored by its group; class 0 uses CLASS0_COLOR
    # ================================================================
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    draw_poincare_disk(ax)

    # Draw class 0
    _draw_trajectory(ax, cls0_xs, cls0_ys, cls0_sx, cls0_sy, CLASS0_COLOR)

    # Draw all grouped classes
    for grp_name, cls_id, xs, ys, sx, sy, name in traj_data:
        color = GROUP_COLORS[grp_name]
        _draw_trajectory(ax, xs, ys, sx, sy, color)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"All Trajectories  ({n_classes} classes)",
                 fontsize=18, fontweight="bold", color="#333333", pad=14)

    all_path = out / "overlay_all_trajectories.png"
    fig.savefig(all_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {all_path}")

    # ================================================================
    # 2. Per-group trajectory figures (original style)
    # ================================================================
    for target_grp, target_cls_ids in CLASS_GROUPS:
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
        draw_poincare_disk(ax)

        for grp_name, cls_id, xs, ys, sx, sy, name in traj_data:
            if grp_name != target_grp:
                _draw_trajectory_bg(ax, xs, ys, sx, sy)

        target_color = GROUP_COLORS[target_grp]
        for grp_name, cls_id, xs, ys, sx, sy, name in traj_data:
            if grp_name != target_grp:
                continue
            _draw_trajectory(ax, xs, ys, sx, sy, target_color)

        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.set_aspect("equal")
        ax.axis("off")

        n_cls = len(target_cls_ids)
        ax.set_title(f"{target_grp}  ({n_cls} classes)",
                     fontsize=18, fontweight="bold", color="#333333", pad=14)

        fname = target_grp.lower().replace(".", "").replace("+", "_")
        sys_path = out / f"overlay_{fname}.png"
        fig.savefig(sys_path, dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.15)
        plt.close(fig)
        print(f"Saved {sys_path}")

    # ================================================================
    # 2b. Ribs-only figure: 24 ribs highlighted, rest gray, no labels
    # ================================================================
    rib_ids = set(RIB_LEFT + RIB_RIGHT)
    rib_left_color = "#e15759"
    rib_right_color = "#4e79a7"

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    draw_poincare_disk(ax)

    for grp_name, cls_id, xs, ys, sx, sy, name in traj_data:
        if cls_id not in rib_ids:
            _draw_trajectory_bg(ax, xs, ys, sx, sy)

    for grp_name, cls_id, xs, ys, sx, sy, name in traj_data:
        if cls_id not in rib_ids:
            continue
        color = rib_left_color if cls_id in RIB_LEFT else rib_right_color
        _draw_trajectory(ax, xs, ys, sx, sy, color)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Ribs  (24 classes)",
                 fontsize=18, fontweight="bold", color="#333333", pad=14)

    rib_path = out / "overlay_ribs.png"
    fig.savefig(rib_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {rib_path}")

    # ================================================================
    # 3. Standalone legend
    # ================================================================
    handles = []
    for grp_name, cls_ids in CLASS_GROUPS:
        n = len(cls_ids)
        handles.append(mpatches.Patch(
            color=GROUP_COLORS[grp_name], label=f"{grp_name} ({n})"))

    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="none", markeredgecolor="#555",
                              markersize=8, markeredgewidth=1.0,
                              label="Epoch 0"))
    handles.append(plt.Line2D([0], [0], marker="^", color="w",
                              markerfacecolor="#555", markeredgecolor="white",
                              markersize=10, markeredgewidth=0.5,
                              label="Epoch 50"))

    fig_leg = plt.figure(figsize=(3, 5), facecolor="white")
    fig_leg.legend(handles=handles, loc="center", fontsize=14,
                   frameon=True, fancybox=False, edgecolor="#D5DBDB",
                   framealpha=1.0, ncol=1, handlelength=1.6,
                   handletextpad=0.4, borderpad=0.6,
                   labelspacing=0.3)
    leg_path = out / "overlay_all_trajectories_legend.png"
    fig_leg.savefig(leg_path, dpi=300,
                    facecolor="white", pad_inches=0.1)
    plt.close(fig_leg)
    print(f"Saved {leg_path}")


if __name__ == "__main__":
    main()
