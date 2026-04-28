"""Plot all class trajectories overlaid in a Poincare disk, colored by organ system.
Chinese-labeled version.

Usage:
    python scripts/plot_overlay_trajectories-zh.py \
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
from matplotlib.font_manager import FontProperties
from scipy.interpolate import make_interp_spline

# ---------------------------------------------------------------------------
# CJK font setup: use IPAex Gothic bundled with matplotlib_fontja
# ---------------------------------------------------------------------------
_CJK_FONT_PATH = (Path(__file__).resolve().parent.parent /
                   ".." / "miniconda3" / "envs" / "pasco" / "lib" /
                   "python3.10" / "site-packages" / "matplotlib_fontja" /
                   "fonts" / "ipaexg.ttf")
if not _CJK_FONT_PATH.exists():
    # Fallback: search site-packages
    import matplotlib_fontja as _mfj
    _CJK_FONT_PATH = Path(_mfj.__file__).parent / "fonts" / "ipaexg.ttf"

CJK_FONT = FontProperties(fname=str(_CJK_FONT_PATH))

# ---------------------------------------------------------------------------
# Style: clean, publication-ready (matching per-system figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "axes.unicode_minus": False,
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
    "skeletal": "\u9aa8\u9abc\u7cfb\u7d71",
    "muscular": "\u808c\u8089\u7cfb\u7d71",
    "digestive": "\u6d88\u5316\u7cfb\u7d71",
    "respiratory": "\u547c\u5438\u7cfb\u7d71",
    "cardiovascular": "\u5fc3\u8840\u7ba1\u7cfb\u7d71",
    "urinary": "\u6ccc\u5c3f\u7cfb\u7d71",
    "nervous": "\u795e\u7d93\u7cfb\u7d71",
    "other": "\u5176\u4ed6",
}

# English class name -> Chinese label
# NOTE: uses traditional forms for chars missing in IPAex Gothic font
CLASS_NAME_ZH = {
    "inside_body_empty": "\u4f53\u5185\u7a7a\u8154",
    "liver": "\u809d\u81df",
    "spleen": "\u813e\u81df",
    "kidney_left": "\u5de6\u814e",
    "kidney_right": "\u53f3\u814e",
    "stomach": "\u80c3",
    "pancreas": "\u81b5\u817a",
    "gallbladder": "\u80c6\u56ca",
    "urinary_bladder": "\u818a\u80f1",
    "prostate": "\u524d\u5217\u817a",
    "heart": "\u5fc3\u81df",
    "brain": "\u8166",
    "thyroid_gland": "\u7532\u72b6\u817a",
    "spinal_cord": "\u810a\u9ad3",
    "lung": "\u80ba",
    "esophagus": "\u98df\u7ba1",
    "trachea": "\u6c14\u7ba1",
    "small_bowel": "\u5c0f\u8178",
    "duodenum": "\u5341\u4e8c\u6307\u8178",
    "colon": "\u7d50\u8178",
    "adrenal_gland_left": "\u5de6\u814e\u4e0a\u817a",
    "adrenal_gland_right": "\u53f3\u814e\u4e0a\u817a",
    "spine": "\u810a\u67f1",
    "skull": "\u9871\u9aa8",
    "sternum": "\u80f8\u9aa8",
    "costal_cartilages": "\u808b\u8edf\u9aa8",
    "scapula_left": "\u5de6\u80a9\u80db\u9aa8",
    "scapula_right": "\u53f3\u80a9\u80db\u9aa8",
    "clavicula_left": "\u5de6\u9396\u9aa8",
    "clavicula_right": "\u53f3\u9396\u9aa8",
    "humerus_left": "\u5de6\u80b1\u9aa8",
    "humerus_right": "\u53f3\u80b1\u9aa8",
    "hip_left": "\u5de6\u80ef\u9aa8",
    "hip_right": "\u53f3\u80ef\u9aa8",
    "femur_left": "\u5de6\u80a1\u9aa8",
    "femur_right": "\u53f3\u80a1\u9aa8",
    "gluteus_maximus_left": "\u5de6\u81c0\u5927\u808c",
    "gluteus_maximus_right": "\u53f3\u81c0\u5927\u808c",
    "gluteus_medius_left": "\u5de6\u81c0\u4e2d\u808c",
    "gluteus_medius_right": "\u53f3\u81c0\u4e2d\u808c",
    "gluteus_minimus_left": "\u5de6\u81c0\u5c0f\u808c",
    "gluteus_minimus_right": "\u53f3\u81c0\u5c0f\u808c",
    "autochthon_left": "\u5de6\u8c4e\u810a\u808c",
    "autochthon_right": "\u53f3\u8c4e\u810a\u808c",
    "iliopsoas_left": "\u5de6\u8178\u8170\u808c",
    "iliopsoas_right": "\u53f3\u8178\u8170\u808c",
}
# Rib names: generate programmatically
for side, zh_side in [("left", "\u5de6"), ("right", "\u53f3")]:
    for i in range(1, 13):
        CLASS_NAME_ZH[f"rib_{side}_{i}"] = f"{zh_side}\u808b{i}"

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

def _translate_name(name: str) -> str:
    """Translate English class name to Chinese."""
    return CLASS_NAME_ZH.get(name, name.replace("_", " "))


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
        ax.text(r + 0.015, -0.025, f"{r:.2f}", fontsize=5.5, color="#AAB7B8",
                ha="left", va="top", zorder=0.6)


def _draw_gradient_line(ax, xs, ys, color, lw=1.0, zorder=2):
    """Draw a trajectory with temporal gradient (faint -> saturated)."""
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    base = to_rgba(color)
    alphas = np.linspace(0.20, 0.85, n)
    seg_colors = [(base[0], base[1], base[2], a) for a in alphas]
    lc = LineCollection(segs, colors=seg_colors, linewidths=lw,
                        capstyle="round", joinstyle="round", zorder=zorder)
    ax.add_collection(lc)


def _place_labels(ax, xys, names, colors):
    """Place endpoint labels, using adjustText if available."""
    try:
        from adjustText import adjust_text
        txt_objs = []
        for (x, y), name, c in zip(xys, names, colors):
            t = ax.text(x, y, name, fontsize=4.5, color=c,
                        fontweight="medium", ha="center", va="center",
                        fontproperties=CJK_FONT, zorder=6)
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
                        fontsize=4.5, color=c,
                        fontweight="medium", ha="center", va="center",
                        fontproperties=CJK_FONT,
                        arrowprops=dict(arrowstyle="-", color="#AAB7B8",
                                        lw=0.3, alpha=0.5),
                        zorder=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay all embedding trajectories in a Poincare disk (Chinese)")
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

    projs = compute_joint_pca(data, labels)
    groups = group_by_system(meta)
    n_epochs = len(projs)

    # -- Figure ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    draw_poincare_disk(ax)

    texts_xy, texts_str, texts_color = [], [], []

    # Draw each system's trajectories with its color
    for sys_name, cls_indices in sorted(groups.items()):
        color = SYSTEM_COLORS.get(sys_name, "#707B7C")

        for cls_id in cls_indices:
            xs = [projs[t][cls_id, 0] for t in range(n_epochs)]
            ys = [projs[t][cls_id, 1] for t in range(n_epochs)]
            sx, sy = _make_smooth(xs, ys)

            # Gradient trajectory
            _draw_gradient_line(ax, sx, sy, color, lw=1.0, zorder=2)

            # Start marker: small open circle
            ax.scatter(xs[0], ys[0], s=14, facecolors="none",
                       edgecolors=color, linewidths=0.6, zorder=4)

            # End marker: filled dot
            ax.scatter(xs[-1], ys[-1], s=16, facecolors=color,
                       edgecolors="white", linewidths=0.4, zorder=4)

            # Small arrow showing direction at the end
            if len(sx) >= 2:
                dx = sx[-1] - sx[-6] if len(sx) > 6 else sx[-1] - sx[-2]
                dy = sy[-1] - sy[-6] if len(sy) > 6 else sy[-1] - sy[-2]
                norm = np.hypot(dx, dy)
                if norm > 1e-8:
                    dx, dy = dx / norm * 0.02, dy / norm * 0.02
                    ax.annotate("", xy=(xs[-1] + dx, ys[-1] + dy),
                                xytext=(xs[-1] - dx, ys[-1] - dy),
                                arrowprops=dict(arrowstyle="-|>", color=color,
                                                lw=0.8, mutation_scale=5),
                                zorder=5)

            # Collect label info (translated to Chinese)
            name = _translate_name(class_names[cls_id])
            texts_xy.append((xs[-1], ys[-1]))
            texts_str.append(name)
            texts_color.append(color)

    # -- Labels with overlap avoidance --
    _place_labels(ax, texts_xy, texts_str, texts_color)

    # -- Axes cleanup --
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- Title --
    ax.set_title(f"\u6240\u6709\u8ecc\u8ff9  \uff08{n_classes} \u985e\uff09",
                 fontsize=14, fontweight="bold", color="#333333", pad=14,
                 fontproperties=CJK_FONT)

    # -- Legend: organ system colors --
    handles = []
    for sys_name in ["skeletal", "muscular", "digestive", "respiratory",
                     "cardiovascular", "urinary", "nervous", "other"]:
        if sys_name in groups:
            display = SYSTEM_DISPLAY.get(sys_name, sys_name)
            n = len(groups[sys_name])
            handles.append(mpatches.Patch(
                color=SYSTEM_COLORS[sys_name], label=f"{display} ({n})"))

    # Epoch markers in legend
    ep_start = labels[0].replace("epoch_", "\u5468\u671f ")
    ep_end = labels[-1].replace("epoch_", "\u5468\u671f ")
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="none", markeredgecolor="#555",
                              markersize=5, markeredgewidth=0.8,
                              label=ep_start))
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor="#555", markeredgecolor="white",
                              markersize=5, markeredgewidth=0.5,
                              label=ep_end))

    ax.legend(handles=handles, loc="lower left", fontsize=7,
              frameon=True, fancybox=False, edgecolor="#D5DBDB",
              framealpha=0.9, ncol=2, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8, borderpad=0.5,
              prop=FontProperties(fname=str(_CJK_FONT_PATH), size=7))

    # -- Save --
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "overlay_all_trajectories_zh.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
