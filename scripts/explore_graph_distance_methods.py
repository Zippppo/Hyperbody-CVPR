"""Explore alternative graph distance fusion methods.

Loads the precomputed contact matrix and tree distance matrix, then evaluates
several fusion formulas with different hyperparameters. Includes per-system
breakdown, adaptive threshold method, qualitative anatomical assessment,
and comparison visualizations.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.organ_hierarchy import compute_tree_distance_matrix, load_class_to_system


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(dataset_info_path: str) -> list:
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        return json.load(f)["class_names"]


def symmetrize_max(C: torch.Tensor) -> torch.Tensor:
    return torch.max(C, C.T)


def check_triangle_inequality(D: torch.Tensor, n_samples: int = 10000) -> bool:
    """Sample random triplets and check triangle inequality."""
    n = D.shape[0]
    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        i, j, k = rng.integers(0, n, size=3)
        if D[i, j] > D[i, k] + D[k, j] + 1e-6:
            return False
    return True


def is_valid_metric(D: torch.Tensor) -> dict:
    """Check basic distance metric properties."""
    nonneg = bool((D >= -1e-9).all())
    diag_zero = bool((D.diagonal() == 0).all())
    sym = bool(torch.allclose(D, D.T, atol=1e-6))
    return {"non_negative": nonneg, "diagonal_zero": diag_zero, "symmetric": sym}


def eval_method(name: str, D_final: torch.Tensor, D_tree: torch.Tensor) -> dict:
    """Compute evaluation metrics for a fusion result."""
    mask = D_tree > 0  # off-diagonal (tree dist > 0)
    shortened = int(((D_final < D_tree) & mask).sum().item())
    total_pairs = int(mask.sum().item())
    vals = D_final[mask]
    metric_props = is_valid_metric(D_final)
    tri_ok = check_triangle_inequality(D_final)
    return {
        "name": name,
        "shortened": shortened,
        "total_pairs": total_pairs,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min_nonzero": float(vals[vals > 0].min()) if (vals > 0).any() else 0.0,
        "max": float(vals.max()),
        "non_negative": metric_props["non_negative"],
        "diagonal_zero": metric_props["diagonal_zero"],
        "symmetric": metric_props["symmetric"],
        "triangle_ok": tri_ok,
    }


# ---------------------------------------------------------------------------
# Fusion methods
# ---------------------------------------------------------------------------

def method_a(D_tree, contact_sym, alpha):
    """Multiplicative discount: D_final = D_tree * (1 - alpha * contact_sym)."""
    D_final = D_tree * (1.0 - alpha * contact_sym)
    D_final.clamp_(min=0.0)
    D_final.fill_diagonal_(0.0)
    return D_final


def method_b(D_tree, contact_sym, gamma, eps=0.01):
    """Log-scaled spatial distance: D_spatial = -gamma * log(contact_sym + eps), D_final = min(D_tree, D_spatial)."""
    D_spatial = -gamma * torch.log(contact_sym + eps)
    D_final = torch.min(D_tree, D_spatial)
    D_final.fill_diagonal_(0.0)
    return D_final


def method_c(D_tree, contact_sym):
    """Rank normalization: rank-normalize non-zero contacts to [0,1], D_final = D_tree * (1 - contact_ranked)."""
    flat = contact_sym.flatten()
    nonzero_mask = flat > 0
    ranked = torch.zeros_like(flat)
    if nonzero_mask.any():
        vals = flat[nonzero_mask]
        order = vals.argsort().argsort().float()
        if order.numel() > 1:
            ranked[nonzero_mask] = order / order.max()
        else:
            ranked[nonzero_mask] = 1.0
    contact_ranked = ranked.reshape(contact_sym.shape)
    D_final = D_tree * (1.0 - contact_ranked)
    D_final.clamp_(min=0.0)
    D_final.fill_diagonal_(0.0)
    return D_final


def method_d(D_tree, contact_sym, kappa, threshold):
    """Floyd-Warshall shortest path on combined tree + spatial edges."""
    n = D_tree.shape[0]
    D = D_tree.clone().float()
    spatial_mask = contact_sym > threshold
    D_spatial = torch.full_like(D, float("inf"))
    D_spatial[spatial_mask] = kappa / contact_sym[spatial_mask]
    D = torch.min(D, D_spatial)
    D.fill_diagonal_(0.0)
    D_np = D.numpy().copy()
    for k in range(n):
        D_np = np.minimum(D_np, D_np[:, k:k+1] + D_np[k:k+1, :])
    D_final = torch.from_numpy(D_np).float()
    D_final.fill_diagonal_(0.0)
    return D_final


def method_e_adaptive(D_tree, contact_sym, percentile_threshold):
    """Adaptive threshold: only apply spatial discount where contact > percentile threshold.

    Uses the baseline formula D_spatial = 1/(C+eps) but only for pairs above threshold.
    """
    flat_nz = contact_sym[contact_sym > 0]
    if flat_nz.numel() == 0:
        return D_tree.clone()
    thr_val = float(np.percentile(flat_nz.numpy(), percentile_threshold))
    eps = 0.01
    D_spatial = 1.0 / (contact_sym + eps)
    # Only apply where contact exceeds threshold
    mask = contact_sym > thr_val
    D_final = D_tree.clone()
    D_final[mask] = torch.min(D_tree[mask], D_spatial[mask])
    D_final.fill_diagonal_(0.0)
    return D_final, thr_val


# ---------------------------------------------------------------------------
# Qualitative anatomical assessment
# ---------------------------------------------------------------------------

def assess_shortcuts(name, D_final, D_tree, contact_sym, class_names, class_to_system):
    """Print the top shortcuts and assess anatomical plausibility."""
    diff = D_tree - D_final
    # Get pairs where shortcuts were created, sorted by magnitude
    n = D_tree.shape[0]
    shortcuts = []
    for i in range(n):
        for j in range(i + 1, n):
            if diff[i, j] > 0:
                shortcuts.append((
                    i, j,
                    float(diff[i, j]),
                    float(D_tree[i, j]),
                    float(D_final[i, j]),
                    float(contact_sym[i, j]),
                ))
    shortcuts.sort(key=lambda x: -x[2])  # sort by shortcut magnitude

    print(f"\n  Top 15 shortcuts for {name}:")
    print(f"  {'Class i':<25} {'Class j':<25} {'Sys_i':<12} {'Sys_j':<12} {'D_tree':>6} {'D_final':>7} {'Cut':>5} {'Contact':>8}")
    print(f"  {'-'*110}")
    for i, j, cut, dt, df, c in shortcuts[:15]:
        sys_i = class_to_system.get(i, "?")
        sys_j = class_to_system.get(j, "?")
        print(f"  {class_names[i]:<25} {class_names[j]:<25} {sys_i:<12} {sys_j:<12} {dt:>6.1f} {df:>7.2f} {cut:>5.2f} {c:>8.4f}")

    # Cross-system vs within-system shortcuts
    cross_sys = 0
    within_sys = 0
    for i, j, cut, dt, df, c in shortcuts:
        if class_to_system.get(i, "?") == class_to_system.get(j, "?"):
            within_sys += 1
        else:
            cross_sys += 1
    print(f"  Within-system shortcuts: {within_sys}, Cross-system shortcuts: {cross_sys}")
    return shortcuts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_info_path = str(PROJECT_ROOT / "Dataset" / "dataset_info.json")
    contact_path = str(PROJECT_ROOT / "Dataset" / "contact_matrix.pt")
    tree_path = str(PROJECT_ROOT / "Dataset" / "tree.json")

    # Load data
    class_names = load_class_names(dataset_info_path)
    contact_matrix = torch.load(contact_path, map_location="cpu").float()
    D_tree = compute_tree_distance_matrix(tree_path, class_names).float()
    class_to_system = load_class_to_system(tree_path, class_names)

    n = len(class_names)
    print(f"Number of classes: {n}")
    print(f"Contact matrix shape: {contact_matrix.shape}")
    print(f"Tree distance matrix shape: {D_tree.shape}")

    # -----------------------------------------------------------------------
    # Per-class system mapping
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASS-TO-SYSTEM MAPPING")
    print("=" * 70)
    system_counts = defaultdict(int)
    for idx, sys_name in class_to_system.items():
        system_counts[sys_name] += 1
    for sys_name, cnt in sorted(system_counts.items(), key=lambda x: -x[1]):
        print(f"  {sys_name:<20}: {cnt} classes")

    # -----------------------------------------------------------------------
    # Analyze contact matrix
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONTACT MATRIX ANALYSIS (raw, asymmetric)")
    print("=" * 70)
    flat_c = contact_matrix.flatten()
    nonzero_c = flat_c[flat_c > 0]
    print(f"  Total entries: {flat_c.numel()}")
    print(f"  Non-zero entries: {nonzero_c.numel()}")
    print(f"  Max: {flat_c.max().item():.6f}")
    print(f"  Is symmetric: {torch.allclose(contact_matrix, contact_matrix.T, atol=1e-6)}")
    if nonzero_c.numel() > 0:
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(nonzero_c.numpy(), p)
            print(f"  Percentile {p:3d}: {val:.6f}")

    # Symmetrize
    contact_sym = symmetrize_max(contact_matrix)
    print("\nAfter symmetrization (max):")
    flat_cs = contact_sym.flatten()
    nonzero_cs = flat_cs[flat_cs > 0]
    print(f"  Non-zero entries: {nonzero_cs.numel()}")
    print(f"  Max: {flat_cs.max().item():.6f}")
    if nonzero_cs.numel() > 0:
        for p in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(nonzero_cs.numpy(), p)
            print(f"  Percentile {p:3d}: {val:.6f}")

    # Histogram of non-zero contacts
    print(f"\n  Histogram of non-zero contacts (symmetrized):")
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    counts, _ = np.histogram(nonzero_cs.numpy(), bins=bins)
    for i in range(len(bins) - 1):
        print(f"    ({bins[i]:.3f}, {bins[i+1]:.3f}]: {counts[i]}")

    # -----------------------------------------------------------------------
    # Per-system contact breakdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-SYSTEM CONTACT BREAKDOWN (symmetrized)")
    print("=" * 70)
    systems = sorted(set(class_to_system.values()))
    sys_indices = {s: [i for i, v in class_to_system.items() if v == s] for s in systems}

    print(f"\n  {'System A':<15} {'System B':<15} {'Pairs':>6} {'NonZero':>8} {'MeanNZ':>10} {'MaxC':>10}")
    print(f"  {'-'*70}")
    for si, sa in enumerate(systems):
        for sj, sb in enumerate(systems):
            if sj < si:
                continue
            idx_a = sys_indices[sa]
            idx_b = sys_indices[sb]
            sub = contact_sym[np.ix_(idx_a, idx_b)]
            total = sub.numel()
            nz = int((sub > 0).sum().item())
            if nz > 0:
                mean_nz = float(sub[sub > 0].mean())
                max_c = float(sub.max())
                print(f"  {sa:<15} {sb:<15} {total:>6} {nz:>8} {mean_nz:>10.6f} {max_c:>10.6f}")

    # -----------------------------------------------------------------------
    # Analyze tree distance matrix
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TREE DISTANCE MATRIX ANALYSIS")
    print("=" * 70)
    flat_t = D_tree.flatten()
    nonzero_t = flat_t[flat_t > 0]
    unique_vals = torch.unique(D_tree)
    print(f"  Unique values: {sorted(unique_vals.tolist())}")
    print(f"  Max: {D_tree.max().item():.1f}")
    print(f"  Mean (off-diag): {nonzero_t.mean().item():.3f}")
    print(f"  Is symmetric: {torch.allclose(D_tree, D_tree.T, atol=1e-6)}")
    print(f"  Distribution of tree distances:")
    for v in sorted(unique_vals.tolist()):
        if v > 0:
            cnt = int((D_tree == v).sum().item())
            print(f"    d={v:.0f}: {cnt} pairs")

    # -----------------------------------------------------------------------
    # Baseline: original formula
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE: Original formula D_spatial = lambda/(C+eps), D_final = min(D_tree, D_spatial)")
    print("=" * 70)
    eps = 0.01
    D_spatial_orig = 1.0 / (contact_sym + eps)
    D_baseline = torch.min(D_tree, D_spatial_orig)
    D_baseline.fill_diagonal_(0.0)
    res = eval_method("Baseline (lam=1.0, eps=0.01)", D_baseline, D_tree)
    print(f"  Shortened pairs: {res['shortened']}/{res['total_pairs']}")
    print(f"  Mean: {res['mean']:.4f}, Std: {res['std']:.4f}")
    print(f"  Min non-zero: {res['min_nonzero']:.4f}")
    print(f"  Symmetric: {res['symmetric']}, Triangle OK: {res['triangle_ok']}")

    # -----------------------------------------------------------------------
    # Run all methods
    # -----------------------------------------------------------------------
    results = []
    results.append(res)
    d_finals = {}
    d_finals[res["name"]] = D_baseline.clone()

    # Method A: Multiplicative discount (additive discount in the task description)
    for alpha in [0.5, 0.8, 1.0]:
        name = f"A: Mult discount (a={alpha})"
        D = method_a(D_tree, contact_sym, alpha)
        r = eval_method(name, D, D_tree)
        results.append(r)
        d_finals[name] = D.clone()

    # Method B: Log-scaled spatial distance
    for gamma in [0.5, 1.0, 2.0]:
        name = f"B: Log-scaled (g={gamma})"
        D = method_b(D_tree, contact_sym, gamma)
        r = eval_method(name, D, D_tree)
        results.append(r)
        d_finals[name] = D.clone()

    # Method C: Rank normalization
    name = "C: Rank normalization"
    D = method_c(D_tree, contact_sym)
    r = eval_method(name, D, D_tree)
    results.append(r)
    d_finals[name] = D.clone()

    # Method D: Floyd-Warshall shortest path
    for threshold, kappa in [(0.01, 0.5), (0.01, 1.0), (0.05, 0.5)]:
        name = f"D: Floyd-W (thr={threshold},k={kappa})"
        D = method_d(D_tree, contact_sym, kappa, threshold)
        r = eval_method(name, D, D_tree)
        results.append(r)
        d_finals[name] = D.clone()

    # Method E: Adaptive threshold
    for pct in [50, 75, 90, 95]:
        result_tuple = method_e_adaptive(D_tree, contact_sym, pct)
        D, thr_val = result_tuple
        name = f"E: Adaptive thr (p{pct}={thr_val:.4f})"
        r = eval_method(name, D, D_tree)
        results.append(r)
        d_finals[name] = D.clone()

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = (
        f"{'Method':<40} {'Short':>6} {'Mean':>7} {'Std':>7} "
        f"{'MinNZ':>7} {'Max':>6} {'NN':>3} {'D0':>3} {'Sym':>3} {'Tri':>3}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['name']:<40} {r['shortened']:>6} {r['mean']:>7.3f} {r['std']:>7.3f} "
            f"{r['min_nonzero']:>7.3f} {r['max']:>6.1f} "
            f"{'Y' if r['non_negative'] else 'N':>3} "
            f"{'Y' if r['diagonal_zero'] else 'N':>3} "
            f"{'Y' if r['symmetric'] else 'N':>3} "
            f"{'Y' if r['triangle_ok'] else 'N':>3}"
        )

    # -----------------------------------------------------------------------
    # Qualitative anatomical assessment for selected methods
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("QUALITATIVE ANATOMICAL ASSESSMENT")
    print("=" * 70)

    # Assess baseline
    assess_shortcuts("Baseline", D_baseline, D_tree, contact_sym, class_names, class_to_system)

    # Assess best multiplicative
    name_a = "A: Mult discount (a=0.5)"
    if name_a in d_finals:
        assess_shortcuts(name_a, d_finals[name_a], D_tree, contact_sym, class_names, class_to_system)

    # Assess Floyd-Warshall
    name_fw = "D: Floyd-W (thr=0.01,k=0.5)"
    if name_fw in d_finals:
        assess_shortcuts(name_fw, d_finals[name_fw], D_tree, contact_sym, class_names, class_to_system)

    # Assess adaptive threshold at 75th percentile
    for name_e in d_finals:
        if name_e.startswith("E: Adaptive thr (p75"):
            assess_shortcuts(name_e, d_finals[name_e], D_tree, contact_sym, class_names, class_to_system)
            break

    # -----------------------------------------------------------------------
    # Visualization 1: heatmaps of (D_tree - D_final) for each method
    # -----------------------------------------------------------------------
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select a representative subset for the heatmap figure
    heatmap_methods = [
        "Baseline (lam=1.0, eps=0.01)",
        "A: Mult discount (a=0.5)",
        "A: Mult discount (a=1.0)",
        "B: Log-scaled (g=1.0)",
        "B: Log-scaled (g=2.0)",
        "C: Rank normalization",
        "D: Floyd-W (thr=0.01,k=0.5)",
        "D: Floyd-W (thr=0.05,k=0.5)",
    ]
    # Add adaptive threshold methods
    for name in d_finals:
        if name.startswith("E: Adaptive"):
            heatmap_methods.append(name)

    heatmap_methods = [m for m in heatmap_methods if m in d_finals]
    n_methods = len(heatmap_methods)
    cols = 4
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    for idx, mname in enumerate(heatmap_methods):
        ax = axes[idx]
        diff = (D_tree - d_finals[mname]).numpy()
        pos_vals = diff[diff > 0]
        vmax = np.percentile(pos_vals, 99) if len(pos_vals) > 0 else 2
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-1, vmax=max(vmax, 0.5))
        ax.set_title(mname, fontsize=7)
        ax.set_xlabel("Class j", fontsize=6)
        ax.set_ylabel("Class i", fontsize=6)
        ax.tick_params(labelsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("D_tree - D_final (positive = shortcut created)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(str(output_dir / "graph_distance_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to {output_dir / 'graph_distance_comparison.png'}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Visualization 2: histograms of D_final distributions
    # -----------------------------------------------------------------------
    hist_methods = [
        "Baseline (lam=1.0, eps=0.01)",
        "A: Mult discount (a=0.5)",
        "B: Log-scaled (g=2.0)",
        "C: Rank normalization",
        "D: Floyd-W (thr=0.01,k=0.5)",
    ]
    # Add one adaptive threshold
    for name in d_finals:
        if "p75" in name:
            hist_methods.append(name)
            break

    hist_methods = [m for m in hist_methods if m in d_finals]
    n_hist = len(hist_methods) + 1  # +1 for D_tree
    cols_h = 3
    rows_h = (n_hist + cols_h - 1) // cols_h

    fig2, axes2 = plt.subplots(rows_h, cols_h, figsize=(5 * cols_h, 4 * rows_h))
    axes2 = np.array(axes2).flatten()

    # D_tree histogram
    ax = axes2[0]
    tree_vals = D_tree[D_tree > 0].numpy()
    ax.hist(tree_vals, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title("D_tree (original)", fontsize=9)
    ax.set_xlabel("Distance", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.axvline(np.mean(tree_vals), color="red", linestyle="--", label=f"mean={np.mean(tree_vals):.2f}")
    ax.legend(fontsize=7)

    for idx, mname in enumerate(hist_methods):
        ax = axes2[idx + 1]
        vals = d_finals[mname][D_tree > 0].numpy()
        ax.hist(vals, bins=30, color="coral", edgecolor="black", alpha=0.8)
        ax.set_title(mname, fontsize=8)
        ax.set_xlabel("Distance", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.axvline(np.mean(vals), color="red", linestyle="--", label=f"mean={np.mean(vals):.2f}")
        ax.legend(fontsize=7)

    for idx in range(n_hist, len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle("Distribution of D_final values (off-diagonal)", fontsize=12)
    plt.tight_layout()
    fig2.savefig(str(output_dir / "graph_distance_histograms.png"), dpi=150, bbox_inches="tight")
    print(f"Histograms saved to {output_dir / 'graph_distance_histograms.png'}")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # Visualization 3: contact matrix per-system heatmap
    # -----------------------------------------------------------------------
    # Sort classes by system for a block-diagonal view
    system_order = ["skeletal", "muscular", "digestive", "respiratory",
                    "cardiovascular", "nervous", "urinary", "other"]
    sorted_indices = []
    sorted_labels = []
    system_boundaries = []
    for sys_name in system_order:
        if sys_name in sys_indices:
            start = len(sorted_indices)
            sorted_indices.extend(sys_indices[sys_name])
            sorted_labels.extend([class_names[i] for i in sys_indices[sys_name]])
            system_boundaries.append((start, len(sorted_indices), sys_name))

    contact_sorted = contact_sym[np.ix_(sorted_indices, sorted_indices)].numpy()

    fig3, ax3 = plt.subplots(1, 1, figsize=(14, 12))
    im3 = ax3.imshow(np.log10(contact_sorted + 1e-8), cmap="hot_r", vmin=-5, vmax=0)
    ax3.set_title("Contact Matrix (log10 scale, sorted by system)", fontsize=12)
    # Add system boundary lines
    for start, end, sys_name in system_boundaries:
        ax3.axhline(y=start - 0.5, color="cyan", linewidth=0.5)
        ax3.axvline(x=start - 0.5, color="cyan", linewidth=0.5)
        mid = (start + end) / 2
        ax3.text(-2, mid, sys_name, fontsize=6, ha="right", va="center", color="blue")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="log10(contact)")
    plt.tight_layout()
    fig3.savefig(str(output_dir / "contact_matrix_by_system.png"), dpi=150, bbox_inches="tight")
    print(f"Per-system contact heatmap saved to {output_dir / 'contact_matrix_by_system.png'}")
    plt.close(fig3)

    print("\nDone.")


if __name__ == "__main__":
    main()
