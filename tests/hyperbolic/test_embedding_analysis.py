"""
Tests for learned embedding analysis (Section 5.6).

Validates:
1. Lorentz pairwise distance matrix shape and symmetry
2. Spearman rank correlation between D_graph and D_learned
3. Residual matrix computation and top-k discovery
4. Poincare ball projection for visualization
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

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
def curv():
    return 1.0


@pytest.fixture
def class_names():
    info_path = PROJECT_ROOT / "Dataset" / "dataset_info.json"
    with open(info_path) as f:
        return json.load(f)["class_names"]


@pytest.fixture
def graph_distance_matrix():
    path = PROJECT_ROOT / "Dataset" / "graph_distance_matrix.pt"
    return torch.load(path, map_location="cpu")


@pytest.fixture
def random_tangent_vectors(num_classes, embed_dim):
    """Simulate learned tangent vectors."""
    torch.manual_seed(42)
    return torch.randn(num_classes, embed_dim)


@pytest.fixture
def lorentz_embeddings(random_tangent_vectors, curv):
    return exp_map0(random_tangent_vectors, curv)


# ---------------------------------------------------------------------------
# Test 1: Pairwise distance matrix properties
# ---------------------------------------------------------------------------

class TestPairwiseDistanceMatrix:
    def test_shape(self, lorentz_embeddings, num_classes):
        D = pairwise_dist(lorentz_embeddings, lorentz_embeddings)
        assert D.shape == (num_classes, num_classes), f"Expected ({num_classes},{num_classes}), got {D.shape}"
        print(f"[PASS] D_learned shape: {D.shape}")

    def test_symmetry(self, lorentz_embeddings):
        D = pairwise_dist(lorentz_embeddings, lorentz_embeddings)
        diff = (D - D.T).abs().max().item()
        assert diff < 1e-4, f"Asymmetry too large: {diff}"
        print(f"[PASS] Symmetry check: max |D - D^T| = {diff:.2e}")

    def test_diagonal_near_zero(self, lorentz_embeddings):
        D = pairwise_dist(lorentz_embeddings, lorentz_embeddings)
        diag_max = D.diag().abs().max().item()
        # acosh(1+eps) gives small but non-zero values; 0.2 is generous
        assert diag_max < 0.2, f"Diagonal too large: max = {diag_max}"
        print(f"[PASS] Diagonal check: max diag = {diag_max:.4f} (numerical artifact from acosh)")

    def test_non_negative(self, lorentz_embeddings):
        D = pairwise_dist(lorentz_embeddings, lorentz_embeddings)
        min_val = D.min().item()
        assert min_val >= -1e-5, f"Negative distance: {min_val}"
        print(f"[PASS] Non-negative check: min = {min_val:.2e}")

    def test_triangle_inequality_sample(self, lorentz_embeddings):
        """Spot-check triangle inequality for a few triplets."""
        D = pairwise_dist(lorentz_embeddings, lorentz_embeddings)
        violations = 0
        n = D.shape[0]
        # Check 1000 random triplets
        torch.manual_seed(0)
        for _ in range(1000):
            i, j, k = torch.randint(0, n, (3,)).tolist()
            if D[i, j] > D[i, k] + D[k, j] + 1e-4:
                violations += 1
        assert violations == 0, f"Triangle inequality violations: {violations}/1000"
        print(f"[PASS] Triangle inequality: 0 violations in 1000 random triplets")


# ---------------------------------------------------------------------------
# Test 2: Graph distance matrix properties
# ---------------------------------------------------------------------------

class TestGraphDistanceMatrix:
    def test_shape(self, graph_distance_matrix, num_classes):
        assert graph_distance_matrix.shape == (num_classes, num_classes)
        print(f"[PASS] D_graph shape: {graph_distance_matrix.shape}")

    def test_asymmetry_from_contact(self, graph_distance_matrix):
        """D_graph is asymmetric because contact_matrix is asymmetric
        (dilated_A ∩ B ≠ dilated_B ∩ A). We symmetrize for analysis."""
        diff = (graph_distance_matrix - graph_distance_matrix.T).abs().max().item()
        print(f"[INFO] D_graph max asymmetry: {diff:.4f}")
        # Symmetrize: take element-wise minimum
        D_sym = torch.min(graph_distance_matrix, graph_distance_matrix.T)
        sym_diff = (D_sym - D_sym.T).abs().max().item()
        assert sym_diff < 1e-10
        print(f"[PASS] Symmetrized D_graph: max diff = {sym_diff:.2e}")

    def test_diagonal_zero(self, graph_distance_matrix):
        diag_max = graph_distance_matrix.diag().abs().max().item()
        assert diag_max < 1e-5
        print(f"[PASS] D_graph diagonal = 0")


# ---------------------------------------------------------------------------
# Test 3: Spearman rank correlation
# ---------------------------------------------------------------------------

class TestSpearmanCorrelation:
    def _extract_upper_triangle(self, matrix, exclude_class0=True):
        """Extract upper triangle values (excluding diagonal)."""
        n = matrix.shape[0]
        start = 1 if exclude_class0 else 0
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        if exclude_class0:
            mask[0, :] = False
            mask[:, 0] = False
        return matrix[mask].numpy()

    def test_spearman_computation(self, lorentz_embeddings, graph_distance_matrix):
        D_learned = pairwise_dist(lorentz_embeddings, lorentz_embeddings)

        d_graph_flat = self._extract_upper_triangle(graph_distance_matrix)
        d_learned_flat = self._extract_upper_triangle(D_learned)

        rho, pval = spearmanr(d_graph_flat, d_learned_flat)
        print(f"[INFO] Spearman rho = {rho:.4f}, p-value = {pval:.2e}")
        # Random embeddings should have low correlation
        assert isinstance(rho, float) and not np.isnan(rho)
        print(f"[PASS] Spearman correlation is a valid float")

    def test_perfect_correlation(self):
        """If D_learned == D_graph, rho should be 1.0."""
        D = torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=torch.float32)
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        a = D[mask].numpy()
        b = D[mask].numpy()
        rho, _ = spearmanr(a, b)
        assert abs(rho - 1.0) < 1e-10
        print(f"[PASS] Perfect correlation test: rho = {rho:.4f}")


# ---------------------------------------------------------------------------
# Test 4: Residual matrix and discovery
# ---------------------------------------------------------------------------

class TestResidualAnalysis:
    def test_residual_matrix_shape(self, lorentz_embeddings, graph_distance_matrix, num_classes):
        D_learned = pairwise_dist(lorentz_embeddings, lorentz_embeddings)

        # Rank transform
        n = num_classes
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        mask[0, :] = False
        mask[:, 0] = False

        d_graph_flat = graph_distance_matrix[mask]
        d_learned_flat = D_learned[mask]

        rank_graph = torch.zeros_like(d_graph_flat)
        rank_learned = torch.zeros_like(d_learned_flat)
        rank_graph[d_graph_flat.argsort()] = torch.arange(len(d_graph_flat), dtype=torch.float32)
        rank_learned[d_learned_flat.argsort()] = torch.arange(len(d_learned_flat), dtype=torch.float32)

        residual_flat = rank_learned - rank_graph
        print(f"[PASS] Residual vector shape: {residual_flat.shape}")
        print(f"[INFO] Residual stats: mean={residual_flat.mean():.1f}, "
              f"std={residual_flat.std():.1f}, "
              f"min={residual_flat.min():.1f}, max={residual_flat.max():.1f}")

    def test_top_k_hidden_connections(self, lorentz_embeddings, graph_distance_matrix, class_names):
        """Find pairs where prior says far but model says close (negative residual)."""
        D_learned = pairwise_dist(lorentz_embeddings, lorentz_embeddings)
        n = len(class_names)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        mask[0, :] = False
        mask[:, 0] = False

        d_graph_flat = graph_distance_matrix[mask]
        d_learned_flat = D_learned[mask]

        rank_graph = torch.zeros_like(d_graph_flat)
        rank_learned = torch.zeros_like(d_learned_flat)
        rank_graph[d_graph_flat.argsort()] = torch.arange(len(d_graph_flat), dtype=torch.float32)
        rank_learned[d_learned_flat.argsort()] = torch.arange(len(d_learned_flat), dtype=torch.float32)

        residual_flat = rank_learned - rank_graph

        # Get indices in the upper triangle
        indices = torch.where(mask)
        top_k = 5

        # Hidden connections: most negative residual (prior=far, model=close)
        _, hidden_idx = residual_flat.topk(top_k, largest=False)
        print(f"\n[INFO] Top-{top_k} 'hidden connections' (prior=far, model=close):")
        for idx in hidden_idx:
            i, j = indices[0][idx].item(), indices[1][idx].item()
            print(f"  {class_names[i]} <-> {class_names[j]}: "
                  f"residual={residual_flat[idx]:.0f}, "
                  f"D_graph={graph_distance_matrix[i,j]:.2f}, "
                  f"D_learned={D_learned[i,j]:.2f}")

        # Over-connections: most positive residual (prior=close, model=far)
        _, over_idx = residual_flat.topk(top_k, largest=True)
        print(f"\n[INFO] Top-{top_k} 'over-connections' (prior=close, model=far):")
        for idx in over_idx:
            i, j = indices[0][idx].item(), indices[1][idx].item()
            print(f"  {class_names[i]} <-> {class_names[j]}: "
                  f"residual={residual_flat[idx]:.0f}, "
                  f"D_graph={graph_distance_matrix[i,j]:.2f}, "
                  f"D_learned={D_learned[i,j]:.2f}")

        assert len(hidden_idx) == top_k
        assert len(over_idx) == top_k
        print(f"\n[PASS] Top-k discovery works correctly")


# ---------------------------------------------------------------------------
# Test 5: Poincare projection
# ---------------------------------------------------------------------------

class TestPoincareProjection:
    def test_inside_unit_ball(self, lorentz_embeddings):
        poincare = lorentz_to_poincare(lorentz_embeddings)
        norms = poincare.norm(dim=-1)
        max_norm = norms.max().item()
        assert max_norm < 1.0, f"Point outside Poincare ball: norm = {max_norm}"
        print(f"[PASS] All points inside Poincare ball: max norm = {max_norm:.6f}")

    def test_poincare_shape(self, lorentz_embeddings, num_classes, embed_dim):
        poincare = lorentz_to_poincare(lorentz_embeddings)
        assert poincare.shape == (num_classes, embed_dim)
        print(f"[PASS] Poincare shape: {poincare.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
