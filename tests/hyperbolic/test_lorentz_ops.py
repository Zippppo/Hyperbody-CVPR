import torch
import pytest
import math


class TestExpLogMap:
    """Test exp_map0 and log_map0 operations."""

    def test_exp_log_inverse(self):
        """exp_map0 and log_map0 should be inverses."""
        from models.hyperbolic.lorentz_ops import exp_map0, log_map0

        torch.manual_seed(42)
        v = torch.randn(100, 32) * 0.5  # Tangent vectors
        x = exp_map0(v, curv=1.0)
        v_rec = log_map0(x, curv=1.0)
        assert torch.allclose(v, v_rec, atol=1e-5), f"Max diff: {(v - v_rec).abs().max()}"

    def test_exp_map_zero_vector(self):
        """Zero tangent vector should map to origin (zero spatial components)."""
        from models.hyperbolic.lorentz_ops import exp_map0

        v = torch.zeros(10, 32)
        x = exp_map0(v, curv=1.0)
        assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)

    def test_exp_map_output_shape(self):
        """exp_map0 should preserve shape."""
        from models.hyperbolic.lorentz_ops import exp_map0

        v = torch.randn(5, 10, 32)
        x = exp_map0(v, curv=1.0)
        assert x.shape == v.shape

    def test_exp_map_large_norm_stability(self):
        """exp_map0 should handle large norm vectors without overflow."""
        from models.hyperbolic.lorentz_ops import exp_map0

        v = torch.randn(10, 32) * 10.0  # Large vectors
        x = exp_map0(v, curv=1.0)
        assert torch.isfinite(x).all(), "Output contains inf or nan"


class TestDistanceFunctions:
    """Test pointwise_dist, pairwise_dist, and distance_to_origin."""

    def test_pointwise_dist_shape(self):
        """pointwise_dist should return element-wise distances."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        v1 = torch.randn(100, 32) * 0.5
        v2 = torch.randn(100, 32) * 0.5
        x1 = exp_map0(v1)
        x2 = exp_map0(v2)

        dist = pointwise_dist(x1, x2)
        assert dist.shape == (100,), f"Expected (100,), got {dist.shape}"

    def test_pointwise_dist_non_negative(self):
        """Distances should be non-negative."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x1 = exp_map0(torch.randn(50, 32) * 0.5)
        x2 = exp_map0(torch.randn(50, 32) * 0.5)

        dist = pointwise_dist(x1, x2)
        assert (dist >= -1e-6).all(), f"Negative distance found: {dist.min()}"

    def test_pointwise_dist_symmetry(self):
        """Distance should be symmetric: d(x,y) == d(y,x)."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x1 = exp_map0(torch.randn(50, 32) * 0.5)
        x2 = exp_map0(torch.randn(50, 32) * 0.5)

        d_xy = pointwise_dist(x1, x2)
        d_yx = pointwise_dist(x2, x1)
        assert torch.allclose(d_xy, d_yx, atol=1e-6)

    def test_pointwise_dist_self_zero(self):
        """Distance to self should be zero."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x = exp_map0(torch.randn(50, 32) * 0.5)
        dist = pointwise_dist(x, x)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-2)

    def test_triangle_inequality(self):
        """Triangle inequality: d(x,z) <= d(x,y) + d(y,z)."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x = exp_map0(torch.randn(50, 32) * 0.5)
        y = exp_map0(torch.randn(50, 32) * 0.5)
        z = exp_map0(torch.randn(50, 32) * 0.5)

        d_xy = pointwise_dist(x, y)
        d_yz = pointwise_dist(y, z)
        d_xz = pointwise_dist(x, z)

        assert (d_xz <= d_xy + d_yz + 1e-5).all(), "Triangle inequality violated"

    def test_pairwise_dist_shape(self):
        """pairwise_dist should return [N, M] matrix."""
        from models.hyperbolic.lorentz_ops import exp_map0, pairwise_dist

        torch.manual_seed(42)
        x = exp_map0(torch.randn(10, 32) * 0.5)
        y = exp_map0(torch.randn(20, 32) * 0.5)

        dist = pairwise_dist(x, y)
        assert dist.shape == (10, 20), f"Expected (10, 20), got {dist.shape}"

    def test_distance_to_origin_shape(self):
        """distance_to_origin should reduce last dimension."""
        from models.hyperbolic.lorentz_ops import exp_map0, distance_to_origin

        torch.manual_seed(42)
        x = exp_map0(torch.randn(5, 10, 32) * 0.5)

        dist = distance_to_origin(x)
        assert dist.shape == (5, 10), f"Expected (5, 10), got {dist.shape}"

    def test_origin_has_zero_distance(self):
        """Origin should have zero distance from origin."""
        from models.hyperbolic.lorentz_ops import distance_to_origin

        x = torch.zeros(10, 32)  # Origin in spatial components
        dist = distance_to_origin(x)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-2)
