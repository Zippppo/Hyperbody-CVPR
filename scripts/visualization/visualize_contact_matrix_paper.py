"""Generate publication-ready contact matrix visualizations.

Outputs:
  A) Block heatmap grouped by anatomical system (group labels only)
  B) Full heatmap with individual class labels
  C) Rib <-> Lung asymmetry bar chart
"""

import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Allow importing project modules
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────
contact = torch.load(ROOT / "Dataset" / "contact_matrix.pt", map_location="cpu").numpy()
with open(ROOT / "Dataset" / "dataset_info.json") as f:
    CLASS_NAMES = json.load(f)["class_names"]

# ── group definitions (0-based indices, no Muscles, no class 0) ───
RIB_LEFT = list(range(23, 35))   # rib_left_1 .. rib_left_12
RIB_RIGHT = list(range(35, 47)) # rib_right_1 .. rib_right_12

GROUPS = [
    ("Axial",    [47, 22]),                         # skull, spine
    ("Thorax",   [48, 49] + RIB_LEFT + RIB_RIGHT), # sternum, costal_cart, ribs
    ("Resp.",    [16, 14]),                          # trachea, lung
    ("Digest.",  [15, 5, 17, 18, 19, 1, 7, 6]),    # esophagus, stomach, ...
    ("Cv.",      [10]),                              # heart
    ("Urogen.",  [3, 4, 8, 9, 12, 20, 21, 2]),     # kidneys, bladder, ...
    ("Ns.",      [11, 13]),                          # brain, spinal_cord
    ("Append.",  [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]),
]

# Ordered indices for reindexing
GROUP_INDICES = []
GROUP_BOUNDARIES = []  # cumulative boundary positions
GROUP_LABELS = []
for name, indices in GROUPS:
    GROUP_BOUNDARIES.append(len(GROUP_INDICES))
    GROUP_INDICES.extend(indices)
    GROUP_LABELS.append(name)
GROUP_BOUNDARIES.append(len(GROUP_INDICES))  # final boundary

N = len(GROUP_INDICES)


def _short_name(idx: int) -> str:
    """Create abbreviated class name for axis labels."""
    name = CLASS_NAMES[idx]
    # shorten common patterns
    name = name.replace("_left", "_L").replace("_right", "_R")
    name = name.replace("adrenal_gland", "adrenal")
    name = name.replace("thyroid_gland", "thyroid")
    name = name.replace("costal_cartilages", "costal_cart")
    name = name.replace("urinary_bladder", "bladder")
    name = name.replace("_", " ")
    return name


def _reorder_matrix(mat: np.ndarray, indices: list) -> np.ndarray:
    """Reorder rows and columns of a matrix by given indices."""
    return mat[np.ix_(indices, indices)]


# ── reordered matrix ──────────────────────────────────────────────
reordered = _reorder_matrix(contact, GROUP_INDICES)


# ── common style ──────────────────────────────────────────────────
CMAP = "YlOrRd"  # default
VMIN, VMAX = 0, reordered.max()

# Alternative colormaps to try
ALT_CMAPS = {
    "v2_viridis": "viridis",
    "v3_blues":   "Blues",
    "v4_inferno": "inferno",
}


def _add_group_lines(ax, boundaries, n, color="white", lw=1.5):
    """Draw group separator lines on a heatmap."""
    for b in boundaries[1:-1]:  # skip first (0) and last (N)
        ax.axhline(b - 0.5, color=color, linewidth=lw)
        ax.axvline(b - 0.5, color=color, linewidth=lw)


# ====================================================================
# Figure A: Block Heatmap (group labels only)
# ====================================================================
def plot_block_heatmap(cmap=CMAP, suffix=""):
    line_color = "white" if cmap != "Blues" else "gray"
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(reordered, cmap=cmap, vmin=VMIN, vmax=VMAX,
                   aspect="equal", interpolation="nearest")

    _add_group_lines(ax, GROUP_BOUNDARIES, N, color=line_color)

    # Group labels at midpoints
    mid_positions = []
    for i in range(len(GROUPS)):
        mid = (GROUP_BOUNDARIES[i] + GROUP_BOUNDARIES[i + 1]) / 2 - 0.5
        mid_positions.append(mid)

    ax.set_xticks(mid_positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(mid_positions)
    ax.set_yticklabels(GROUP_LABELS, fontsize=9)

    # Minor ticks off
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Contact Matrix (grouped by anatomical system)", fontsize=16)
    ax.set_xlabel("Target organ  $j$", fontsize=14)
    ax.set_ylabel("Source organ  $i$", fontsize=14)

    ax.set_xticks(mid_positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(mid_positions)
    ax.set_yticklabels(GROUP_LABELS, fontsize=10)

    fig.tight_layout()
    path = OUT_DIR / f"fig_a_block_heatmap{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ====================================================================
# Figure B: Full Heatmap (individual class labels)
# ====================================================================
def plot_full_heatmap(cmap=CMAP, suffix=""):
    from matplotlib.colors import PowerNorm
    line_color = "white" if cmap != "Blues" else "gray"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(reordered, cmap=cmap, norm=PowerNorm(gamma=0.3, vmin=0, vmax=VMAX),
                   aspect="equal", interpolation="nearest")

    _add_group_lines(ax, GROUP_BOUNDARIES, N, color=line_color, lw=1.0)

    # Group labels at midpoints (same as Fig A)
    mid_positions = []
    for i in range(len(GROUPS)):
        mid = (GROUP_BOUNDARIES[i] + GROUP_BOUNDARIES[i + 1]) / 2 - 0.5
        mid_positions.append(mid)

    ax.set_xticks(mid_positions)
    ax.set_xticklabels(GROUP_LABELS, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(mid_positions)
    ax.set_yticklabels(GROUP_LABELS, fontsize=10)

    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Contact Matrix", fontsize=18)
    ax.set_xlabel("Target organ  $j$", fontsize=16)
    ax.set_ylabel("Source organ  $i$", fontsize=16)

    fig.tight_layout()
    path = OUT_DIR / f"fig_b_full_heatmap{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ====================================================================
# Figure C: Rib <-> Lung asymmetry
# ====================================================================
def plot_rib_lung_asymmetry():
    lung_idx = 14  # lung class index

    rib_indices = RIB_LEFT + RIB_RIGHT  # 24 ribs
    rib_to_lung = contact[rib_indices, lung_idx]
    lung_to_rib = contact[lung_idx, rib_indices]

    rib_labels = [CLASS_NAMES[i].replace("rib_", "").replace("_", " ")
                  for i in rib_indices]

    x = np.arange(len(rib_indices))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 4))
    bars1 = ax.bar(x - width / 2, rib_to_lung, width,
                   label=r"Rib $\rightarrow$ Lung", color="#2B5C8A", alpha=0.9)
    bars2 = ax.bar(x + width / 2, lung_to_rib, width,
                   label=r"Lung $\rightarrow$ Rib", color="#C44E52", alpha=0.9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(rib_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Contact ratio (log scale)", fontsize=10)
    ax.set_title("Asymmetric contact: Rib vs Lung", fontsize=11)
    ax.legend(fontsize=9)

    # Add a separator between left and right ribs
    ax.axvline(len(RIB_LEFT) - 0.5, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    path = OUT_DIR / "fig_c_rib_lung_asymmetry.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ====================================================================
# Figure D: D_sem, D_sp, D_ont side by side
# ====================================================================
def plot_distance_comparison():
    from data.organ_hierarchy import compute_tree_distance_matrix
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # D_sem: tree distance
    d_sem = compute_tree_distance_matrix(
        str(ROOT / "Dataset" / "tree.json"), CLASS_NAMES
    ).numpy()

    # D_sp: spatial distance = lambda / (contact + epsilon)
    lam, eps = 0.4, 0.01
    d_sp = (lam / (contact + eps))
    np.fill_diagonal(d_sp, 0)

    # D_ont: graph distance (precomputed)
    d_ont = torch.load(
        ROOT / "Dataset" / "graph_distance_matrix.pt", map_location="cpu"
    ).numpy()

    # Reorder all matrices by group indices
    d_sem_r = _reorder_matrix(d_sem, GROUP_INDICES)
    d_sp_r = _reorder_matrix(d_sp, GROUP_INDICES)
    d_ont_r = _reorder_matrix(d_ont, GROUP_INDICES)

    # Cap D_sp for visualization (values can be very large for zero-contact pairs)
    vmax_sp = np.percentile(d_sp_r[d_sp_r > 0], 95)
    d_sp_display = np.clip(d_sp_r, 0, vmax_sp)

    # Normalize each matrix to [0, 1] for a unified colorbar
    def _normalize(mat):
        vmax = mat.max()
        return mat / vmax if vmax > 0 else mat

    d_sem_norm = _normalize(d_sem_r)
    d_sp_norm = _normalize(d_sp_display)
    d_ont_norm = _normalize(d_ont_r)

    matrices = [
        (d_sem_norm, r"$D_{\mathrm{sem}}$", "(a)"),
        (d_sp_norm, r"$D_{\mathrm{sp}}$", "(b)"),
        (d_ont_norm, r"$D_{\mathrm{ont}}$", "(c)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (mat, title, label) in zip(axes, matrices):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1,
                       aspect="equal", interpolation="nearest")

        _add_group_lines(ax, GROUP_BOUNDARIES, N, color="gray", lw=0.8)

        # Group labels at midpoints
        mid_positions = []
        for i in range(len(GROUPS)):
            mid = (GROUP_BOUNDARIES[i] + GROUP_BOUNDARIES[i + 1]) / 2 - 0.5
            mid_positions.append(mid)

        ax.set_xticks(mid_positions)
        ax.set_xticklabels(GROUP_LABELS, fontsize=13, rotation=45, ha="right")
        ax.set_yticks(mid_positions)
        ax.set_yticklabels(GROUP_LABELS, fontsize=13)

        ax.set_title(f"{label}  {title}", fontsize=20, pad=10)

    # Unified colorbar on the right side
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.935, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized distance", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    path = OUT_DIR / "fig_d_distance_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Final version: Blues colormap
    plot_block_heatmap(cmap="Blues", suffix="_final")
    plot_full_heatmap(cmap="Blues", suffix="_final")
    plot_rib_lung_asymmetry()
    plot_distance_comparison()
    print("Done. All figures saved to:", OUT_DIR)
