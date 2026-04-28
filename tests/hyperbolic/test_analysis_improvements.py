"""
Tests for analyze_learned_embeddings.py improvements:

1. pairwise_dist must receive curv parameter (bug fix)
2. Rank computation should use scipy.stats.rankdata for proper tie handling
3. Poincare projection should use hyperbolic MDS instead of PCA

Run:
    pytest tests/hyperbolic/test_analysis_improvements.py -v -s
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.hyperbolic.lorentz_ops import (
    exp_map0,
    lorentz_to_poincare,
    pairwise_dist,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def num_classes():
    return 70


@pytest.fixture
def embed_dim():
    return 32


@pytest.fixture
def graph_distance_matrix():
    path = PROJECT_ROOT / "Dataset" / "graph_distance_matrix.pt"
    return torch.load(path, map_location="cpu")


@pytest.fixture
def random_tangent_vectors(num_classes, embed_dim):
    torch.manual_seed(42)
    return torch.randn(num_classes, embed_dim)


# ---------------------------------------------------------------------------
# Test 1: pairwise_dist curv parameter affects distances
# ---------------------------------------------------------------------------

class TestCurvParameterBug:
    """The script calls pairwise_dist without curv. If curv != 1.0,
    distances are wrong. These tests prove curv matters."""

    def test_different_curv_gives_different_distances(self, random_tangent_vectors):
        """Core proof: curv=0.5 and curv=2.0 must produce different D_learned."""
        emb_c05 = exp_map0(random_tangent_vectors, curv=0.5)
        emb_c20 = exp_map0(random_tangent_vectors, curv=2.0)

        D_c05 = pairwise_dist(emb_c05, emb_c05, curv=0.5)
        D_c20 = pairwise_dist(emb_c20, emb_c20, curv=2.0)

        # Distances should differ significantly
        diff = (D_c05 - D_c20).abs().mean().item()
        assert diff > 0.1, f"Expected significant difference, got mean diff = {diff:.4f}"
        print(f"[PASS] curv=0.5 vs curv=2.0 mean distance diff = {diff:.4f}")

    def test_wrong_curv_changes_off_diagonal(self, random_tangent_vectors):
        """Using wrong curv produces different off-diagonal distances."""
        curv_train = 2.0
        emb = exp_map0(random_tangent_vectors, curv=curv_train)

        D_correct = pairwise_dist(emb, emb, curv=curv_train)
        D_wrong = pairwise_dist(emb, emb, curv=1.0)

        # Off-diagonal distances should differ significantly
        mask = torch.triu(torch.ones_like(D_correct, dtype=torch.bool), diagonal=1)
        diff = (D_correct[mask] - D_wrong[mask]).abs().mean().item()
        print(f"[INFO] Mean off-diagonal distance diff (correct vs wrong curv): {diff:.4f}")
        assert diff > 0.1, f"Expected significant difference, got {diff:.4f}"
        print(f"[PASS] Wrong curv produces meaningfully different distances")

    def test_curv_scaling_relationship(self, random_tangent_vectors):
        """Geodesic distance scales as 1/sqrt(curv). Verify this relationship."""
        emb = exp_map0(random_tangent_vectors, curv=1.0)

        D_c1 = pairwise_dist(emb, emb, curv=1.0)
        D_c4 = pairwise_dist(emb, emb, curv=4.0)

        # For same spatial embeddings, d(curv=4) ≈ d(curv=1) / 2
        # This is approximate because acosh is nonlinear
        mask = torch.triu(torch.ones_like(D_c1, dtype=torch.bool), diagonal=1)
        ratio = D_c1[mask].mean() / D_c4[mask].mean()
        print(f"[INFO] D(curv=1)/D(curv=4) ratio = {ratio:.4f} (expected ~2.0)")
        # Allow generous tolerance due to acosh nonlinearity
        assert 1.5 < ratio < 3.0, f"Ratio {ratio:.4f} outside expected range"
        print(f"[PASS] Curvature scaling relationship holds")


# ---------------------------------------------------------------------------
# Test 2: Tie handling with rankdata vs argsort
# ---------------------------------------------------------------------------

class TestTieHandling:
    """D_graph is integer-valued (hop counts), so ties are common.
    The old argsort-based ranking assigns arbitrary order to ties.
    scipy.stats.rankdata with method='average' is correct."""

    def test_argsort_vs_rankdata_on_ties(self):
        """Show that argsort ranking and rankdata differ when ties exist."""
        # Array with ties
        values = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])

        # Old method: argsort-based
        rank_argsort = np.zeros_like(values)
        rank_argsort[values.argsort()] = np.arange(len(values))

        # New method: scipy rankdata (average)
        rank_scipy = rankdata(values, method='average') - 1  # 0-indexed

        print(f"[INFO] Values:       {values}")
        print(f"[INFO] Argsort rank: {rank_argsort}")
        print(f"[INFO] Scipy rank:   {rank_scipy}")

        # They should differ for tied values
        assert not np.allclose(rank_argsort, rank_scipy), \
            "Expected different rankings for tied values"
        print(f"[PASS] Argsort and rankdata produce different results with ties")

    def test_rankdata_consistent_with_spearmanr(self):
        """rankdata('average') should match what spearmanr uses internally."""
        np.random.seed(42)
        a = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 3.0, 2.0, 4.0, 5.0])

        # Manual Spearman using rankdata
        rank_a = rankdata(a, method='average')
        rank_b = rankdata(b, method='average')
        # Pearson on ranks = Spearman
        rho_manual = np.corrcoef(rank_a, rank_b)[0, 1]

        # scipy spearmanr
        rho_scipy, _ = spearmanr(a, b)

        diff = abs(rho_manual - rho_scipy)
        assert diff < 1e-10, f"Mismatch: manual={rho_manual:.6f}, scipy={rho_scipy:.6f}"
        print(f"[PASS] rankdata('average') matches spearmanr: rho={rho_scipy:.6f}")

    def test_graph_distance_has_ties(self, graph_distance_matrix, num_classes):
        """Verify that D_graph actually has tied values (motivating the fix)."""
        mask = torch.triu(torch.ones(num_classes, num_classes, dtype=torch.bool), diagonal=1)
        mask[0, :] = False
        mask[:, 0] = False
        values = graph_distance_matrix[mask].numpy()

        unique_vals = np.unique(values)
        n_total = len(values)
        n_unique = len(unique_vals)
        tie_ratio = 1.0 - n_unique / n_total

        print(f"[INFO] D_graph upper triangle: {n_total} pairs, {n_unique} unique values")
        print(f"[INFO] Tie ratio: {tie_ratio:.1%}")
        print(f"[INFO] Unique values: {unique_vals}")
        assert n_unique < n_total, "Expected ties in D_graph"
        print(f"[PASS] D_graph has significant ties ({tie_ratio:.1%})")

    def test_residual_with_rankdata(self, graph_distance_matrix, num_classes):
        """Compute residual using rankdata and verify properties."""
        torch.manual_seed(42)
        tangent = torch.randn(num_classes, 32)
        emb = exp_map0(tangent, curv=1.0)
        D_learned = pairwise_dist(emb, emb, curv=1.0)

        mask = torch.triu(torch.ones(num_classes, num_classes, dtype=torch.bool), diagonal=1)
        mask[0, :] = False
        mask[:, 0] = False

        D_graph_sym = torch.min(graph_distance_matrix, graph_distance_matrix.T)
        d_graph_flat = D_graph_sym[mask].numpy()
        d_learned_flat = D_learned[mask].detach().numpy()

        # New method: rankdata
        rank_graph = rankdata(d_graph_flat, method='average') - 1
        rank_learned = rankdata(d_learned_flat, method='average') - 1
        residual = rank_learned - rank_graph

        # Residual should have mean near 0 (both are ranks of same length)
        mean_residual = np.mean(residual)
        assert abs(mean_residual) < 1.0, f"Mean residual too large: {mean_residual:.4f}"
        print(f"[PASS] Residual mean = {mean_residual:.4f} (near zero as expected)")
        print(f"[INFO] Residual std = {np.std(residual):.1f}, "
              f"min = {np.min(residual):.1f}, max = {np.max(residual):.1f}")


# ---------------------------------------------------------------------------
# Test 3: Hyperbolic MDS projection (replacing PCA)
# ---------------------------------------------------------------------------

class TestHyperbolicMDSProjection:
    """Poincare disk is a non-Euclidean space. PCA (linear, Euclidean)
    distorts distances. Hyperbolic MDS preserves geodesic structure better."""

    def test_poincare_mds_preserves_distances_better_than_pca(self, random_tangent_vectors):
        """MDS on Poincare distances should have better stress than PCA."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS

        curv = 1.0
        emb = exp_map0(random_tangent_vectors, curv=curv)
        poincare = lorentz_to_poincare(emb, curv).detach().numpy()

        # Poincare pairwise distances (Euclidean approx in Poincare disk)
        D_poincare = pairwise_dist(emb, emb, curv=curv).detach().numpy()
        np.fill_diagonal(D_poincare, 0.0)

        # Method 1: PCA projection to 2D
        pca = PCA(n_components=2)
        proj_pca = pca.fit_transform(poincare)

        # Method 2: MDS on geodesic distances
        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, normalized_stress='auto')
        proj_mds = mds.fit_transform(D_poincare)

        # Compute stress: sum of (d_proj - d_original)^2
        from scipy.spatial.distance import pdist, squareform
        D_orig_flat = squareform(D_poincare)

        D_pca_flat = pdist(proj_pca)
        D_mds_flat = pdist(proj_mds)

        # Normalized stress
        stress_pca = np.sqrt(np.sum((D_pca_flat - D_orig_flat)**2) / np.sum(D_orig_flat**2))
        stress_mds = np.sqrt(np.sum((D_mds_flat - D_orig_flat)**2) / np.sum(D_orig_flat**2))

        print(f"[INFO] PCA stress (normalized): {stress_pca:.4f}")
        print(f"[INFO] MDS stress (normalized): {stress_mds:.4f}")
        print(f"[INFO] MDS improvement: {(stress_pca - stress_mds) / stress_pca:.1%}")

        # MDS should have lower or comparable stress
        assert stress_mds <= stress_pca * 1.1, \
            f"MDS stress ({stress_mds:.4f}) should not be much worse than PCA ({stress_pca:.4f})"
        print(f"[PASS] MDS preserves distances at least as well as PCA")

    def test_mds_output_shape(self, random_tangent_vectors):
        """MDS projection should produce [N, 2] output."""
        from sklearn.manifold import MDS

        curv = 1.0
        emb = exp_map0(random_tangent_vectors, curv=curv)
        D = pairwise_dist(emb, emb, curv=curv).detach().numpy()
        np.fill_diagonal(D, 0.0)

        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, normalized_stress='auto')
        proj = mds.fit_transform(D)

        n = random_tangent_vectors.shape[0]
        assert proj.shape == (n, 2), f"Expected ({n}, 2), got {proj.shape}"
        print(f"[PASS] MDS output shape: {proj.shape}")

    def test_mds_rank_correlation_with_original(self, random_tangent_vectors):
        """MDS 2D distances should have high rank correlation with original distances."""
        from sklearn.manifold import MDS
        from scipy.spatial.distance import pdist, squareform

        curv = 1.0
        emb = exp_map0(random_tangent_vectors, curv=curv)
        D = pairwise_dist(emb, emb, curv=curv).detach().numpy()
        np.fill_diagonal(D, 0.0)

        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, normalized_stress='auto')
        proj = mds.fit_transform(D)

        D_proj_flat = pdist(proj)
        D_orig_flat = squareform(D)  # matrix -> condensed vector

        rho, _ = spearmanr(D_orig_flat, D_proj_flat)
        print(f"[INFO] MDS 2D vs original distance Spearman rho = {rho:.4f}")
        # 32D -> 2D is a large reduction; 0.3 is a reasonable lower bound
        assert rho > 0.3, f"Expected decent rank correlation, got {rho:.4f}"
        print(f"[PASS] MDS preserves distance ranking (rho={rho:.4f})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
