"""
Compare dimensionality reduction methods on epoch_3 embeddings.

Methods compared (all using geodesic distance matrix as input):
  1. Metric MDS (current pipeline)
  2. t-SNE (perplexity sweep)
  3. Isomap
  4. Spectral Embedding
  5. Tangent PCA (uses tangent vectors directly)

Usage:
    python scripts/test_projection_methods.py \
        --export-dir _VIS/embedding_export_021201 \
        --output-dir _VIS/projection_comparison
"""
import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding


SYSTEM_COLORS = {
    "skeletal": "#E74C3C", "muscular": "#3498DB", "digestive": "#2ECC71",
    "respiratory": "#9B59B6", "cardiovascular": "#E67E22", "urinary": "#1ABC9C",
    "nervous": "#F1C40F", "other": "#95A5A6",
}

EPOCH = "epoch_3"


def load_data(export_dir):
    data = np.load(f"{export_dir}/embedding_data.npz")
    with open(f"{export_dir}/embedding_meta.json") as f:
        meta = json.load(f)
    return data, meta


def normalize(proj):
    """Scale to [-0.95, 0.95] by max norm."""
    mx = np.linalg.norm(proj, axis=1).max()
    return proj * (0.95 / mx) if mx > 1e-10 else proj


def eval_quality(proj, gdist):
    """Spearman rho between projected Euclidean dist and geodesic dist."""
    mask = np.triu_indices(len(proj), k=1)
    g = gdist[mask]
    e = squareform(pdist(proj))[mask]
    rho, _ = spearmanr(g, e)
    return rho


def run_projections(tangent, gdist):
    """Return dict of {method_name: (proj_2d, rho)}."""
    results = {}

    # 1. Metric MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              normalized_stress="auto", n_init=4)
    proj = mds.fit_transform(gdist)
    results["MDS (metric)"] = normalize(proj)

    # 2. Non-metric MDS
    mds_nm = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                 normalized_stress="auto", n_init=4, metric=False)
    proj = mds_nm.fit_transform(gdist)
    results["MDS (non-metric)"] = normalize(proj)

    # 3. t-SNE with different perplexities
    for perp in [5, 15, 30]:
        tsne = TSNE(n_components=2, metric="precomputed", perplexity=perp,
                    random_state=42, init="random")
        proj = tsne.fit_transform(gdist)
        results[f"t-SNE (perp={perp})"] = normalize(proj)

    # 4. Isomap (needs dense distance → use as precomputed graph)
    iso = Isomap(n_components=2, metric="precomputed", n_neighbors=10)
    proj = iso.fit_transform(gdist)
    results["Isomap (k=10)"] = normalize(proj)

    # 5. Spectral Embedding (convert distance to affinity)
    sigma = np.median(gdist[gdist > 0])
    affinity = np.exp(-gdist ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 0)
    se = SpectralEmbedding(n_components=2, affinity="precomputed", random_state=42)
    proj = se.fit_transform(affinity)
    results["Spectral Embedding"] = normalize(proj)

    # 6. Tangent PCA
    centered = tangent - tangent.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:2].T
    results["Tangent PCA"] = normalize(proj)

    # Compute quality for all
    scored = {}
    for name, proj in results.items():
        rho = eval_quality(proj, gdist)
        scored[name] = (proj, rho)

    return scored


def plot_comparison(results, class_names, class_to_system, output_path):
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()

    # Sort by rho descending
    sorted_items = sorted(results.items(), key=lambda x: -x[1][1])

    for idx, (name, (proj, rho)) in enumerate(sorted_items):
        ax = axes[idx]
        for i in range(len(class_names)):
            sys_name = class_to_system[str(i)]
            c = SYSTEM_COLORS.get(sys_name, "#95A5A6")
            ax.scatter(proj[i, 0], proj[i, 1], color=c, s=18,
                       edgecolors="white", linewidths=0.2, zorder=3)

            label = class_names[i]
            if label.startswith("rib_left_"):
                label = f"L{label.split('_')[-1]}"
            elif label.startswith("rib_right_"):
                label = f"R{label.split('_')[-1]}"
            else:
                label = (label.replace("_left", "(L)")
                         .replace("_right", "(R)")
                         .replace("_", " "))
            ax.annotate(label, (proj[i, 0], proj[i, 1]), fontsize=3.5,
                        ha="left", va="bottom", xytext=(1.5, 1.5),
                        textcoords="offset points", color=c)

        ax.set_title(f"{name}\n(Spearman rho={rho:.3f})", fontsize=9, fontweight="bold")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.12, linewidth=0.4)

    # Hide unused axes
    for idx in range(len(sorted_items), len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=c, label=s.capitalize())
               for s, c in SYSTEM_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(SYSTEM_COLORS),
               fontsize=7, framealpha=0.8)

    fig.suptitle(f"Projection Methods Comparison @ {EPOCH}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default="_VIS/embedding_export_021201")
    parser.add_argument("--output-dir", default="_VIS/projection_comparison")
    args = parser.parse_args()

    data, meta = load_data(args.export_dir)
    tangent = data[f"tangent_{EPOCH}"]
    gdist = data[f"geodesic_dist_{EPOCH}"]
    class_names = meta["class_names"]
    class_to_system = meta["class_to_system"]

    print(f"Running projections on {EPOCH}  (N={len(class_names)}, D={tangent.shape[1]})...")
    results = run_projections(tangent, gdist)

    # Print quality summary
    print("\n=== Geodesic Distance Preservation (Spearman rho) ===")
    for name, (_, rho) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"  {name:25s}: {rho:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_comparison(results, class_names, class_to_system, out / "comparison_epoch3.png")


if __name__ == "__main__":
    main()
